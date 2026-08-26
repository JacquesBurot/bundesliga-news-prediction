"""Communicate with Ollama and validate structured model responses."""

from __future__ import annotations

import json
from typing import Any

import httpx

from ._validation import require_non_empty_text
from .config import AnnotationConfig, INDICATORS


DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "gemma4:12b-it-qat"
DEFAULT_OLLAMA_NUM_CTX = 40960
OLLAMA_NUM_PREDICT = 8192


class InvalidOllamaAnnotationError(ValueError):
    """A model response remained structurally invalid after all attempts."""

    def __init__(
        self,
        message: str,
        attempt_count: int,
        last_response_content: str | None,
        last_metadata: dict[str, Any] | None,
    ) -> None:
        super().__init__(message)
        self.attempt_count = attempt_count
        self.last_response_content = last_response_content
        self.last_metadata = last_metadata or {}


def request_ollama_annotation(
    client: httpx.Client,
    task: dict[str, Any],
    config: AnnotationConfig,
    model: str,
    num_ctx: int,
    max_attempts: int = 2,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Request one schema-constrained annotation and validate its structure."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": config.system_prompt},
            {"role": "user", "content": build_user_prompt(task, config)},
        ],
        "stream": False,
        "think": True,
        "format": config.response_json_schema,
        "options": {
            "temperature": 0,
            "seed": 42,
            "num_ctx": num_ctx,
            "num_predict": OLLAMA_NUM_PREDICT,
        },
    }
    last_error: ValueError | None = None
    last_response_content: str | None = None
    last_metadata: dict[str, Any] | None = None
    attempt_count = 0
    for attempt_count in range(1, max_attempts + 1):
        http_response = client.post("/api/chat", json=payload)
        http_response.raise_for_status()
        content: str | None = None
        try:
            envelope = http_response.json()
            if isinstance(envelope, dict):
                last_metadata = envelope
            content = get_ollama_message_content(envelope)
            last_response_content = content
            validate_ollama_completion(envelope)
            parsed = json.loads(content)
            if not isinstance(parsed, dict):
                msg = "Ollama structured response must be a JSON object."
                raise ValueError(msg)
            validate_annotation_response(parsed, config)
            return parsed, envelope
        except (json.JSONDecodeError, ValueError) as exc:
            last_error = ValueError(
                f"Invalid Ollama annotation for task {task['task_id']!r}: {exc}"
            )
            if content is not None:
                payload["messages"].append(
                    {"role": "assistant", "content": content}
                )
            payload["messages"].append(
                {
                    "role": "user",
                    "content": (
                        "Die vorherige Antwort war ungültig: "
                        f"{str(exc)[:500]} Prüfe die vier Indikatoren für "
                        f"{task['target_team']} erneut. Bei einer Bewertung "
                        "muss evidence ein Array mit ein oder zwei nicht leeren "
                        "Textbelegen sein. Bei not_mentioned muss evidence ein "
                        "leeres Array sein. Korrigiere die "
                        "Struktur beziehungsweise den Beleg "
                        "und gib ausschließlich das geforderte JSON-Objekt zurück."
                    ),
                }
            )
    if last_error is None:
        msg = "Ollama annotation did not run."
        raise ValueError(msg)
    raise InvalidOllamaAnnotationError(
        message=str(last_error),
        attempt_count=attempt_count,
        last_response_content=last_response_content,
        last_metadata=last_metadata,
    )


def build_user_prompt(task: dict[str, Any], config: AnnotationConfig) -> str:
    """Serialize only target-team identity and article text as untrusted data."""
    context = {
        "target_team": task["target_team"],
        "target_team_aliases": list(
            config.target_team_aliases.get(task["target_team"], ())
        ),
        "article_title": task["title"] or "",
        "article_body": task["body"],
    }
    serialized = json.dumps(context, ensure_ascii=False, indent=2)
    return (
        "Bewerte die sportliche Situation des angegebenen Zielteams anhand "
        "des folgenden, als Daten übergebenen Artikels.\n\n<article_context>\n"
        f"{serialized}\n</article_context>"
    )


def validate_annotation_response(
    response: dict[str, Any],
    config: AnnotationConfig,
) -> None:
    """Validate exact fields, ratings, and per-indicator evidence presence."""
    properties = config.response_json_schema["properties"]
    if set(response) != set(INDICATORS) or len(response) != len(INDICATORS):
        msg = f"Annotation response fields do not match: {list(response)}."
        raise ValueError(msg)

    for indicator in INDICATORS:
        item = response[indicator]
        if (
            not isinstance(item, dict)
            or set(item) != {"rating", "evidence"}
            or len(item) != 2
        ):
            msg = f"Indicator {indicator!r} must contain rating and evidence."
            raise ValueError(msg)
        rating = item["rating"]
        evidence = item["evidence"]
        allowed = properties[indicator]["properties"]["rating"]["enum"]
        if rating not in allowed:
            msg = f"Indicator {indicator!r} has unsupported rating {rating!r}."
            raise ValueError(msg)
        if not isinstance(evidence, list) or len(evidence) > 2:
            msg = f"Evidence for {indicator!r} must contain at most two items."
            raise ValueError(msg)
        if rating == "not_mentioned":
            if evidence:
                msg = f"Unmentioned indicator {indicator!r} needs empty evidence."
                raise ValueError(msg)
            continue
        if not evidence:
            msg = f"Assessed indicator {indicator!r} needs evidence."
            raise ValueError(msg)
        for item_number, evidence_item in enumerate(evidence, start=1):
            require_non_empty_text(
                evidence_item,
                f"Evidence item {item_number} for {indicator}",
            )


def get_ollama_model_digest(client: httpx.Client, model: str) -> str:
    """Resolve the exact locally installed model digest before inference."""
    response = client.get("/api/tags")
    response.raise_for_status()
    payload = response.json()
    models = payload.get("models") if isinstance(payload, dict) else None
    if not isinstance(models, list):
        msg = "Ollama /api/tags response has no models array."
        raise ValueError(msg)
    requested_names = {model}
    if ":" not in model:
        requested_names.add(f"{model}:latest")
    for entry in models:
        if not isinstance(entry, dict):
            continue
        names = {entry.get("name"), entry.get("model")}
        if requested_names & names:
            digest = entry.get("digest")
            return require_non_empty_text(digest, f"Ollama model {model} digest")
    msg = (
        f"Ollama model {model!r} is not installed locally. Run "
        f"'ollama pull {model}' first."
    )
    raise ValueError(msg)


def get_ollama_message_content(payload: Any) -> str:
    """Extract assistant content from an Ollama chat response envelope."""
    if not isinstance(payload, dict):
        msg = "Ollama chat response must be a JSON object."
        raise ValueError(msg)
    message = payload.get("message")
    if not isinstance(message, dict):
        msg = "Ollama chat response has no message object."
        raise ValueError(msg)
    return require_non_empty_text(message.get("content"), "Ollama message content")


def validate_ollama_completion(payload: Any) -> None:
    """Reject incomplete or length-limited non-streaming Ollama responses."""
    if not isinstance(payload, dict) or payload.get("done") is not True:
        msg = "Ollama chat response is not marked as complete."
        raise ValueError(msg)
    done_reason = payload.get("done_reason")
    if done_reason not in {None, "stop"}:
        msg = f"Ollama chat response stopped with reason {done_reason!r}."
        raise ValueError(msg)
