"""Build team-specific news tasks and annotate them through local Ollama."""

from __future__ import annotations

import hashlib
import json
import os
from contextlib import contextmanager
from bisect import bisect_right
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import httpx

from buli_news.news_articles import (
    NEWS_ARTICLE_COLUMNS,
    NEWS_ARTICLE_LINK_COLUMNS,
)
from buli_news.news_contents import (
    ARTICLE_CONTENT_LINK_COLUMNS,
    NEWS_CONTENT_COLUMNS,
    build_content_id,
    normalize_content_body,
)


NEWS_ANNOTATION_TASK_SCHEMA_VERSION = 1
NEWS_ANNOTATION_OUTPUT_SCHEMA_VERSION = 3
NEWS_ANNOTATION_FAILURE_SCHEMA_VERSION = 1
NEWS_ANNOTATION_PILOT_SELECTION_VERSION = 1
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "gemma4:12b-it-qat"
DEFAULT_OLLAMA_NUM_CTX = 40960
OLLAMA_NUM_PREDICT = 8192
DEFAULT_ANNOTATION_CONFIG_PATH = Path("config/news_annotation_schema_v3.json")
INDICATORS = (
    "sporting_form",
    "personnel_situation",
    "physical_readiness",
    "confidence_and_motivation",
)
NEWS_ANNOTATION_TASK_COLUMNS = (
    "task_id",
    "task_schema_version",
    "annotation_schema_id",
    "request_id",
    "content_id",
    "match_id",
    "season",
    "league",
    "matchday",
    "kickoff",
    "side",
    "target_team_id",
    "target_team",
    "opponent_team_id",
    "opponent_team",
    "date_start",
    "date_end",
    "selected_article_id",
    "request_article_index",
    "source_host",
    "publication_datetime",
    "title",
    "body",
    "linked_article_count",
    "linked_source_host_count",
)
NEWS_ANNOTATION_OUTPUT_COLUMNS = (
    "annotation_id",
    "output_schema_version",
    "task_id",
    "annotation_schema_id",
    "annotation_schema_version",
    "prompt_version",
    "annotation_config_sha256",
    "model",
    "model_digest",
    "request_id",
    "content_id",
    "match_id",
    "side",
    "target_team_id",
    "target_team",
    "selected_article_id",
    *INDICATORS,
    "ollama_created_at",
    "total_duration_ns",
    "load_duration_ns",
    "prompt_eval_count",
    "prompt_eval_duration_ns",
    "eval_count",
    "eval_duration_ns",
)
NEWS_ANNOTATION_FAILURE_COLUMNS = (
    "failure_event_id",
    "failure_schema_version",
    "failure_sequence",
    "annotation_id",
    "task_id",
    "annotation_schema_id",
    "annotation_schema_version",
    "prompt_version",
    "annotation_config_sha256",
    "model",
    "model_digest",
    "request_id",
    "content_id",
    "match_id",
    "side",
    "target_team_id",
    "target_team",
    "selected_article_id",
    "attempt_count",
    "error",
    "last_response_content",
    "ollama_created_at",
    "total_duration_ns",
    "prompt_eval_count",
    "eval_count",
    "recorded_at",
)


@dataclass(frozen=True)
class NewsAnnotationTasksBuild:
    """Team-specific annotation tasks and their quality metadata."""

    tasks: list[dict[str, Any]]
    quality_report: dict[str, Any]


@dataclass(frozen=True)
class AnnotationConfig:
    """Validated, versioned annotation prompt and response schema."""

    schema_id: str
    schema_version: int
    prompt_version: int
    system_prompt: str
    rating_mapping: dict[str, int | None]
    response_json_schema: dict[str, Any]
    target_team_aliases: dict[str, tuple[str, ...]]
    sha256: str


@dataclass(frozen=True)
class OllamaAnnotationRun:
    """Summary of one resume-safe local annotation run."""

    selected_count: int
    annotated_count: int
    failed_count: int
    skipped_count: int
    deferred_failure_count: int
    model_digest: str


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


def load_annotation_config(path: Path) -> AnnotationConfig:
    """Load and validate the versioned prompt and structured-output schema."""
    raw_bytes = path.read_bytes()
    try:
        raw = json.loads(raw_bytes)
    except json.JSONDecodeError as exc:
        msg = f"Annotation config {path} is not valid JSON: {exc}."
        raise ValueError(msg) from exc
    if not isinstance(raw, dict):
        msg = f"Annotation config {path} must contain a JSON object."
        raise ValueError(msg)
    expected_keys = (
        "schema_id",
        "schema_version",
        "prompt_version",
        "system_prompt",
        "rating_mapping",
        "response_json_schema",
        "target_team_aliases",
    )
    if tuple(raw) != expected_keys:
        msg = f"Annotation config schema or order does not match: {list(raw)}."
        raise ValueError(msg)

    schema_id = require_text(raw, "schema_id", "Annotation config")
    schema_version = require_positive_int(
        raw.get("schema_version"), "Annotation config schema_version"
    )
    prompt_version = require_positive_int(
        raw.get("prompt_version"), "Annotation config prompt_version"
    )
    system_prompt = require_text(raw, "system_prompt", "Annotation config")
    rating_mapping = raw.get("rating_mapping")
    response_schema = raw.get("response_json_schema")
    if not isinstance(rating_mapping, dict):
        msg = "Annotation config rating_mapping must be an object."
        raise ValueError(msg)
    if not isinstance(response_schema, dict):
        msg = "Annotation config response_json_schema must be an object."
        raise ValueError(msg)
    validate_response_json_schema(response_schema, rating_mapping)
    target_team_aliases = validate_target_team_aliases(
        raw.get("target_team_aliases")
    )

    canonical = json.dumps(
        raw,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return AnnotationConfig(
        schema_id=schema_id,
        schema_version=schema_version,
        prompt_version=prompt_version,
        system_prompt=system_prompt,
        rating_mapping=rating_mapping,
        response_json_schema=response_schema,
        target_team_aliases=target_team_aliases,
        sha256=hashlib.sha256(canonical).hexdigest(),
    )


def build_news_annotation_tasks(
    matches: list[dict[str, Any]],
    articles: list[dict[str, Any]],
    article_links: list[dict[str, Any]],
    article_content_links: list[dict[str, Any]],
    contents: list[dict[str, Any]],
    season: int,
    annotation_schema_id: str,
) -> NewsAnnotationTasksBuild:
    """Build one task per request-bound content and target team."""
    if isinstance(season, bool) or not isinstance(season, int):
        msg = "Season must be an integer start year."
        raise ValueError(msg)
    require_non_empty_text(annotation_schema_id, "Annotation schema ID")
    matches_by_id = validate_matches(matches=matches, season=season)
    articles_by_id = validate_articles(articles)
    content_ids = validate_contents(contents)
    content_id_by_article_id = validate_content_links(
        article_content_links=article_content_links,
        article_ids=set(articles_by_id),
        content_ids=content_ids,
    )

    grouped_links: dict[tuple[str, str], list[dict[str, Any]]] = {}
    request_contexts: dict[str, tuple[Any, ...]] = {}
    for link in article_links:
        validate_article_link(link)
        article_id = link["article_id"]
        if article_id not in articles_by_id:
            msg = f"Article link references unknown article {article_id!r}."
            raise ValueError(msg)
        if link["season"] != season:
            msg = (
                f"Article link {link['request_id']!r} has season "
                f"{link['season']!r}, expected {season}."
            )
            raise ValueError(msg)
        validate_link_against_match(link, matches_by_id)
        context = (
            link["match_id"],
            link["side"],
            link["team_id"],
            link["team"],
            link["date_start"],
            link["date_end"],
        )
        existing_context = request_contexts.setdefault(link["request_id"], context)
        if existing_context != context:
            msg = f"Request {link['request_id']!r} has conflicting link contexts."
            raise ValueError(msg)
        content_id = content_id_by_article_id[article_id]
        grouped_links.setdefault((link["request_id"], content_id), []).append(link)

    if not grouped_links:
        msg = "Article links do not produce any annotation tasks."
        raise ValueError(msg)

    tasks = []
    for (request_id, content_id), links in grouped_links.items():
        links.sort(
            key=lambda link: (link["request_article_index"], link["article_id"])
        )
        selected_link = links[0]
        article = articles_by_id[selected_link["article_id"]]
        match = matches_by_id[selected_link["match_id"]]
        side = selected_link["side"]
        if side == "home":
            opponent_team_id = match["away_team_id"]
            opponent_team = match["away_team"]
        else:
            opponent_team_id = match["home_team_id"]
            opponent_team = match["home_team"]
        task = {
            "task_id": build_task_id(
                annotation_schema_id=annotation_schema_id,
                request_id=request_id,
                content_id=content_id,
            ),
            "task_schema_version": NEWS_ANNOTATION_TASK_SCHEMA_VERSION,
            "annotation_schema_id": annotation_schema_id,
            "request_id": request_id,
            "content_id": content_id,
            "match_id": selected_link["match_id"],
            "season": season,
            "league": selected_link["league"],
            "matchday": match["matchday"],
            "kickoff": match["kickoff"],
            "side": side,
            "target_team_id": selected_link["team_id"],
            "target_team": selected_link["team"],
            "opponent_team_id": opponent_team_id,
            "opponent_team": opponent_team,
            "date_start": selected_link["date_start"],
            "date_end": selected_link["date_end"],
            "selected_article_id": selected_link["article_id"],
            "request_article_index": selected_link["request_article_index"],
            "source_host": article["source_host"],
            "publication_datetime": article["publication_datetime"],
            "title": article["title"],
            "body": article["body"],
            "linked_article_count": len(links),
            "linked_source_host_count": len(
                {articles_by_id[link["article_id"]]["source_host"] for link in links}
            ),
        }
        tasks.append(task)

    tasks.sort(
        key=lambda task: (
            task["match_id"],
            0 if task["side"] == "home" else 1,
            task["request_article_index"],
            task["content_id"],
        )
    )
    validate_tasks(tasks)
    linked_article_count_distribution = Counter(
        task["linked_article_count"] for task in tasks
    )
    quality_report = {
        "schema_version": NEWS_ANNOTATION_TASK_SCHEMA_VERSION,
        "season": season,
        "annotation_schema_id": annotation_schema_id,
        "task_key": ["request_id", "content_id"],
        "association_rule": (
            "A content is annotated only in the request, match, side, and target-"
            "team context in which one of its articles was collected. Identical "
            "publications inside that same request collapse to one task; no task "
            "is transferred to another request or team."
        ),
        "article_selection_rule": (
            "For multiple articles with the same content in one request, use the "
            "lowest request_article_index and then the lowest article_id. Title "
            "and body always come from that same selected article."
        ),
        "summary": {
            "match_count": len({task["match_id"] for task in tasks}),
            "request_count": len({task["request_id"] for task in tasks}),
            "article_link_count": len(article_links),
            "annotation_task_count": len(tasks),
            "collapsed_within_request_article_link_count": (
                len(article_links) - len(tasks)
            ),
            "tasks_with_multiple_linked_articles": sum(
                task["linked_article_count"] > 1 for task in tasks
            ),
        },
        "linked_article_count_distribution": {
            str(count): linked_article_count_distribution[count]
            for count in sorted(linked_article_count_distribution)
        },
    }
    return NewsAnnotationTasksBuild(tasks=tasks, quality_report=quality_report)


def select_stratified_annotation_tasks(
    tasks: list[dict[str, Any]],
    size: int,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Select a deterministic pilot balanced across context and text length."""
    validate_tasks(tasks)
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        msg = "Pilot size must be a positive integer."
        raise ValueError(msg)
    if isinstance(seed, bool) or not isinstance(seed, int):
        msg = "Pilot seed must be an integer."
        raise ValueError(msg)
    if size >= len(tasks):
        return list(tasks)

    lengths = sorted(len(task["body"]) for task in tasks)
    boundaries = tuple(
        lengths[round((len(lengths) - 1) * quantile)]
        for quantile in (0.25, 0.5, 0.75)
    )

    task_metadata = {
        task["task_id"]: (
            bisect_right(boundaries, len(task["body"])),
            hashlib.sha256(
                f"{seed}:{task['task_id']}".encode("utf-8")
            ).hexdigest(),
        )
        for task in tasks
    }

    team_counts: Counter[int] = Counter()
    side_counts: Counter[str] = Counter()
    length_counts: Counter[int] = Counter()
    source_counts: Counter[str] = Counter()
    content_counts: Counter[str] = Counter()
    matchday_tasks: dict[int, dict[str, dict[str, Any]]] = {}
    for task in tasks:
        matchday_tasks.setdefault(task["matchday"], {})[task["task_id"]] = task
    matchdays = sorted(matchday_tasks)
    quotas: Counter[int] = Counter()
    if size < len(matchdays):
        if size == 1:
            quotas[matchdays[seed % len(matchdays)]] = 1
        else:
            for index in range(size):
                position = round(index * (len(matchdays) - 1) / (size - 1))
                quotas[matchdays[position]] += 1
    else:
        base, remainder = divmod(size, len(matchdays))
        for matchday in matchdays:
            quotas[matchday] = base
        offset = seed % len(matchdays)
        for index in range(remainder):
            quotas[matchdays[(offset + index) % len(matchdays)]] += 1

    selected = []
    while any(quotas.values()):
        for matchday in matchdays:
            if quotas[matchday] <= 0:
                continue
            candidates = matchday_tasks[matchday]
            task = min(
                candidates.values(),
                key=lambda candidate: (
                    team_counts[candidate["target_team_id"]],
                    content_counts[candidate["content_id"]],
                    length_counts[task_metadata[candidate["task_id"]][0]],
                    side_counts[candidate["side"]],
                    source_counts[candidate["source_host"]],
                    task_metadata[candidate["task_id"]][1],
                ),
            )
            selected.append(task)
            candidates.pop(task["task_id"])
            quotas[matchday] -= 1
            team_counts[task["target_team_id"]] += 1
            side_counts[task["side"]] += 1
            length_counts[task_metadata[task["task_id"]][0]] += 1
            source_counts[task["source_host"]] += 1
            content_counts[task["content_id"]] += 1

    return selected


def annotate_news_tasks(
    tasks: list[dict[str, Any]],
    config: AnnotationConfig,
    output_path: Path,
    failure_output_path: Path,
    model: str = DEFAULT_OLLAMA_MODEL,
    base_url: str = DEFAULT_OLLAMA_BASE_URL,
    limit: int | None = None,
    task_id: str | None = None,
    retry_failures_only: bool = False,
    timeout_seconds: float = 300.0,
    num_ctx: int = DEFAULT_OLLAMA_NUM_CTX,
    workers: int = 1,
    append_annotation: Callable[[dict[str, Any], Path], None] | None = None,
    append_failure: Callable[[dict[str, Any], Path], None] | None = None,
) -> OllamaAnnotationRun:
    """Run annotations while exclusively owning their append-only outputs."""
    with annotation_output_lock(output_path):
        return _annotate_news_tasks_unlocked(
            tasks=tasks,
            config=config,
            output_path=output_path,
            failure_output_path=failure_output_path,
            model=model,
            base_url=base_url,
            limit=limit,
            task_id=task_id,
            retry_failures_only=retry_failures_only,
            timeout_seconds=timeout_seconds,
            num_ctx=num_ctx,
            workers=workers,
            append_annotation=append_annotation,
            append_failure=append_failure,
        )


def _annotate_news_tasks_unlocked(
    tasks: list[dict[str, Any]],
    config: AnnotationConfig,
    output_path: Path,
    failure_output_path: Path,
    model: str = DEFAULT_OLLAMA_MODEL,
    base_url: str = DEFAULT_OLLAMA_BASE_URL,
    limit: int | None = None,
    task_id: str | None = None,
    retry_failures_only: bool = False,
    timeout_seconds: float = 300.0,
    num_ctx: int = DEFAULT_OLLAMA_NUM_CTX,
    workers: int = 1,
    append_annotation: Callable[[dict[str, Any], Path], None] | None = None,
    append_failure: Callable[[dict[str, Any], Path], None] | None = None,
) -> OllamaAnnotationRun:
    """Run validated structured annotations through a local Ollama server."""
    validate_tasks(tasks)
    mismatched_schema_ids = sorted(
        {
            task["annotation_schema_id"]
            for task in tasks
            if task["annotation_schema_id"] != config.schema_id
        }
    )
    if mismatched_schema_ids:
        msg = (
            "Annotation tasks were built for different schema IDs: "
            f"{mismatched_schema_ids}; current config uses {config.schema_id!r}."
        )
        raise ValueError(msg)
    require_non_empty_text(model, "Ollama model")
    require_non_empty_text(base_url, "Ollama base URL")
    if limit is not None and (isinstance(limit, bool) or limit <= 0):
        msg = "Annotation limit must be a positive integer."
        raise ValueError(msg)
    if timeout_seconds <= 0:
        msg = "Ollama timeout must be positive."
        raise ValueError(msg)
    if isinstance(num_ctx, bool) or not isinstance(num_ctx, int) or num_ctx <= 0:
        msg = "Ollama context size must be a positive integer."
        raise ValueError(msg)
    if isinstance(workers, bool) or not isinstance(workers, int) or workers <= 0:
        msg = "Annotation worker count must be a positive integer."
        raise ValueError(msg)
    if task_id is not None:
        selected = [task for task in tasks if task["task_id"] == task_id]
        if not selected:
            msg = f"Unknown annotation task ID {task_id!r}."
            raise ValueError(msg)
    else:
        selected = list(tasks)

    existing = read_existing_annotations(output_path)
    existing_ids = {row["annotation_id"] for row in existing}
    existing_failures = read_existing_annotation_failures(failure_output_path)
    failure_sequences = Counter(
        row["annotation_id"] for row in existing_failures
    )
    deferred_annotation_ids = set(failure_sequences)
    append_record = append_annotation or append_jsonl_record
    append_failure_record = append_failure or append_jsonl_record
    client_base_url = base_url.rstrip("/")
    with httpx.Client(base_url=client_base_url, timeout=timeout_seconds) as client:
        model_digest = get_ollama_model_digest(client=client, model=model)

    pending = []
    skipped_count = 0
    deferred_failure_count = 0
    for task in selected:
        annotation_id = build_annotation_id(
            task_id=task["task_id"],
            config_sha256=config.sha256,
            model_digest=model_digest,
        )
        if annotation_id in existing_ids:
            skipped_count += 1
            continue
        is_deferred_failure = annotation_id in deferred_annotation_ids
        if retry_failures_only:
            if not is_deferred_failure:
                continue
        elif is_deferred_failure:
            deferred_failure_count += 1
            continue
        pending.append((task, annotation_id))
        if limit is not None and len(pending) >= limit:
            break

    annotated_count = 0
    failed_count = 0

    def persist_outcome(
        task: dict[str, Any],
        annotation_id: str,
        outcome: tuple[dict[str, Any], dict[str, Any]] | InvalidOllamaAnnotationError,
    ) -> None:
        nonlocal annotated_count, failed_count
        if isinstance(outcome, InvalidOllamaAnnotationError):
            failure_sequences[annotation_id] += 1
            failure_row = build_annotation_failure_row(
                task=task,
                error=outcome,
                config=config,
                model=model,
                model_digest=model_digest,
                annotation_id=annotation_id,
                failure_sequence=failure_sequences[annotation_id],
            )
            append_failure_record(failure_row, failure_output_path)
            deferred_annotation_ids.add(annotation_id)
            failed_count += 1
            return
        response, metadata = outcome
        row = build_annotation_row(
            task=task,
            response=response,
            metadata=metadata,
            config=config,
            model=model,
            model_digest=model_digest,
            annotation_id=annotation_id,
        )
        append_record(row, output_path)
        existing_ids.add(annotation_id)
        annotated_count += 1

    if workers == 1:
        with httpx.Client(base_url=client_base_url, timeout=timeout_seconds) as client:
            for task, annotation_id in pending:
                try:
                    outcome = request_ollama_annotation(
                        client=client,
                        task=task,
                        config=config,
                        model=model,
                        num_ctx=num_ctx,
                    )
                except InvalidOllamaAnnotationError as exc:
                    outcome = exc
                persist_outcome(task, annotation_id, outcome)
    else:
        def request_with_own_client(
            task: dict[str, Any],
        ) -> tuple[dict[str, Any], dict[str, Any]] | InvalidOllamaAnnotationError:
            with httpx.Client(
                base_url=client_base_url,
                timeout=timeout_seconds,
            ) as client:
                try:
                    return request_ollama_annotation(
                        client=client,
                        task=task,
                        config=config,
                        model=model,
                        num_ctx=num_ctx,
                    )
                except InvalidOllamaAnnotationError as exc:
                    return exc

        pending_iterator = iter(pending)
        in_flight: dict[Future[Any], tuple[dict[str, Any], str]] = {}
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for _ in range(min(workers, len(pending))):
                task, annotation_id = next(pending_iterator)
                future = executor.submit(request_with_own_client, task)
                in_flight[future] = (task, annotation_id)
            while in_flight:
                completed, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                for future in completed:
                    task, annotation_id = in_flight.pop(future)
                    persist_outcome(task, annotation_id, future.result())
                    try:
                        next_task, next_annotation_id = next(pending_iterator)
                    except StopIteration:
                        continue
                    next_future = executor.submit(request_with_own_client, next_task)
                    in_flight[next_future] = (next_task, next_annotation_id)

    return OllamaAnnotationRun(
        selected_count=len(pending),
        annotated_count=annotated_count,
        failed_count=failed_count,
        skipped_count=skipped_count,
        deferred_failure_count=deferred_failure_count,
        model_digest=model_digest,
    )


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


def build_annotation_row(
    task: dict[str, Any],
    response: dict[str, Any],
    metadata: dict[str, Any],
    config: AnnotationConfig,
    model: str,
    model_digest: str,
    annotation_id: str,
) -> dict[str, Any]:
    """Combine the validated response with stable provenance and timings."""
    row = {
        "annotation_id": annotation_id,
        "output_schema_version": NEWS_ANNOTATION_OUTPUT_SCHEMA_VERSION,
        "task_id": task["task_id"],
        "annotation_schema_id": config.schema_id,
        "annotation_schema_version": config.schema_version,
        "prompt_version": config.prompt_version,
        "annotation_config_sha256": config.sha256,
        "model": model,
        "model_digest": model_digest,
        "request_id": task["request_id"],
        "content_id": task["content_id"],
        "match_id": task["match_id"],
        "side": task["side"],
        "target_team_id": task["target_team_id"],
        "target_team": task["target_team"],
        "selected_article_id": task["selected_article_id"],
        **{indicator: response[indicator] for indicator in INDICATORS},
        "ollama_created_at": metadata.get("created_at"),
        "total_duration_ns": metadata.get("total_duration"),
        "load_duration_ns": metadata.get("load_duration"),
        "prompt_eval_count": metadata.get("prompt_eval_count"),
        "prompt_eval_duration_ns": metadata.get("prompt_eval_duration"),
        "eval_count": metadata.get("eval_count"),
        "eval_duration_ns": metadata.get("eval_duration"),
    }
    if tuple(row) != NEWS_ANNOTATION_OUTPUT_COLUMNS:
        msg = "Internal annotation output schema does not match."
        raise ValueError(msg)
    return row


def build_annotation_failure_row(
    task: dict[str, Any],
    error: InvalidOllamaAnnotationError,
    config: AnnotationConfig,
    model: str,
    model_digest: str,
    annotation_id: str,
    failure_sequence: int,
) -> dict[str, Any]:
    """Persist one exhausted response-validation failure without marking it done."""
    metadata = error.last_metadata
    row = {
        "failure_event_id": build_failure_event_id(
            annotation_id=annotation_id,
            failure_sequence=failure_sequence,
        ),
        "failure_schema_version": NEWS_ANNOTATION_FAILURE_SCHEMA_VERSION,
        "failure_sequence": failure_sequence,
        "annotation_id": annotation_id,
        "task_id": task["task_id"],
        "annotation_schema_id": config.schema_id,
        "annotation_schema_version": config.schema_version,
        "prompt_version": config.prompt_version,
        "annotation_config_sha256": config.sha256,
        "model": model,
        "model_digest": model_digest,
        "request_id": task["request_id"],
        "content_id": task["content_id"],
        "match_id": task["match_id"],
        "side": task["side"],
        "target_team_id": task["target_team_id"],
        "target_team": task["target_team"],
        "selected_article_id": task["selected_article_id"],
        "attempt_count": error.attempt_count,
        "error": str(error),
        "last_response_content": error.last_response_content,
        "ollama_created_at": metadata.get("created_at"),
        "total_duration_ns": metadata.get("total_duration"),
        "prompt_eval_count": metadata.get("prompt_eval_count"),
        "eval_count": metadata.get("eval_count"),
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
    }
    if tuple(row) != NEWS_ANNOTATION_FAILURE_COLUMNS:
        msg = "Internal annotation failure schema does not match."
        raise ValueError(msg)
    return row


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


def read_existing_annotations(path: Path) -> list[dict[str, Any]]:
    """Read and validate an existing append-only annotation output."""
    if not path.exists():
        return []
    rows = []
    annotation_ids = set()
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            msg = f"Invalid JSON in {path} at line {line_number}: {exc}."
            raise ValueError(msg) from exc
        if (
            not isinstance(row, dict)
            or row.get("output_schema_version")
            != NEWS_ANNOTATION_OUTPUT_SCHEMA_VERSION
            or tuple(row) != NEWS_ANNOTATION_OUTPUT_COLUMNS
        ):
            msg = f"Annotation schema does not match in {path} at line {line_number}."
            raise ValueError(msg)
        annotation_id = require_text(row, "annotation_id", "Stored annotation")
        if annotation_id in annotation_ids:
            msg = f"Duplicate annotation ID {annotation_id!r} in {path}."
            raise ValueError(msg)
        annotation_ids.add(annotation_id)
        rows.append(row)
    return rows


def read_existing_annotation_failures(path: Path) -> list[dict[str, Any]]:
    """Read and validate the append-only deferred annotation failures."""
    if not path.exists():
        return []
    rows = []
    event_ids = set()
    event_keys = set()
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            msg = f"Invalid JSON in {path} at line {line_number}: {exc}."
            raise ValueError(msg) from exc
        if not isinstance(row, dict) or tuple(row) != NEWS_ANNOTATION_FAILURE_COLUMNS:
            msg = (
                f"Annotation failure schema does not match in {path} "
                f"at line {line_number}."
            )
            raise ValueError(msg)
        event_id = require_text(row, "failure_event_id", "Stored failure")
        annotation_id = require_text(row, "annotation_id", "Stored failure")
        sequence = require_positive_int(
            row["failure_sequence"], "Stored failure sequence"
        )
        expected_event_id = build_failure_event_id(annotation_id, sequence)
        if event_id != expected_event_id:
            msg = f"Stored failure event ID {event_id!r} is not deterministic."
            raise ValueError(msg)
        event_key = (annotation_id, sequence)
        if event_id in event_ids or event_key in event_keys:
            msg = f"Duplicate annotation failure event {event_id!r} in {path}."
            raise ValueError(msg)
        event_ids.add(event_id)
        event_keys.add(event_key)
        rows.append(row)
    return rows


def validate_response_json_schema(
    schema: dict[str, Any], rating_mapping: dict[str, int | None]
) -> None:
    """Validate the exact response-schema shape relied upon by the pipeline."""
    expected_rating_mapping = {
        "strong_negative": -2,
        "negative": -1,
        "neutral": 0,
        "positive": 1,
        "strong_positive": 2,
        "not_mentioned": None,
    }
    if rating_mapping != expected_rating_mapping:
        msg = "Annotation rating mapping does not match the expected scale."
        raise ValueError(msg)
    if (
        schema.get("type") != "object"
        or schema.get("additionalProperties") is not False
    ):
        msg = "Response JSON schema must be a closed object."
        raise ValueError(msg)
    properties = schema.get("properties")
    if not isinstance(properties, dict) or tuple(properties) != INDICATORS:
        msg = "Response JSON schema properties do not match the four indicators."
        raise ValueError(msg)
    if schema.get("required") != list(properties):
        msg = "Every response JSON schema property must be required in order."
        raise ValueError(msg)
    for indicator in INDICATORS:
        indicator_schema = properties[indicator]
        if (
            not isinstance(indicator_schema, dict)
            or indicator_schema.get("type") != "object"
            or indicator_schema.get("additionalProperties") is not False
        ):
            msg = f"Schema for indicator {indicator!r} must be a closed object."
            raise ValueError(msg)
        indicator_properties = indicator_schema.get("properties")
        if (
            not isinstance(indicator_properties, dict)
            or tuple(indicator_properties) != ("rating", "evidence")
            or indicator_schema.get("required") != ["rating", "evidence"]
        ):
            msg = f"Schema for indicator {indicator!r} needs rating and evidence."
            raise ValueError(msg)
        enum = indicator_properties["rating"].get("enum")
        if not isinstance(enum, list) or "not_mentioned" not in enum:
            msg = f"Schema enum for indicator {indicator!r} is invalid."
            raise ValueError(msg)
        for rating in enum:
            if rating not in rating_mapping:
                msg = f"Rating {rating!r} has no numeric mapping."
                raise ValueError(msg)
        evidence_schema = indicator_properties["evidence"]
        if (
            evidence_schema.get("type") != "array"
            or evidence_schema.get("maxItems") != 2
            or evidence_schema.get("items", {}).get("type") != "string"
            or evidence_schema.get("items", {}).get("minLength") != 1
        ):
            msg = f"Evidence schema for indicator {indicator!r} is invalid."
            raise ValueError(msg)


def validate_matches(
    matches: list[dict[str, Any]], season: int
) -> dict[int, dict[str, Any]]:
    """Validate match fields used to construct team-specific context."""
    if not matches:
        msg = "Normalized match input is empty."
        raise ValueError(msg)
    matches_by_id = {}
    required_fields = (
        "match_id",
        "season",
        "league",
        "matchday",
        "kickoff",
        "home_team",
        "away_team",
        "home_team_id",
        "away_team_id",
    )
    for match in matches:
        if not isinstance(match, dict) or any(
            field not in match for field in required_fields
        ):
            msg = "Normalized match is missing annotation-context fields."
            raise ValueError(msg)
        match_id = require_int(match["match_id"], "Match ID")
        if match_id in matches_by_id:
            msg = f"Duplicate normalized match ID {match_id}."
            raise ValueError(msg)
        if match["season"] != season:
            msg = f"Match {match_id} belongs to season {match['season']!r}."
            raise ValueError(msg)
        require_text(match, "league", f"Match {match_id}")
        require_text(match, "kickoff", f"Match {match_id}")
        require_text(match, "home_team", f"Match {match_id}")
        require_text(match, "away_team", f"Match {match_id}")
        require_int(match["home_team_id"], f"Match {match_id} home team ID")
        require_int(match["away_team_id"], f"Match {match_id} away team ID")
        matches_by_id[match_id] = match
    return matches_by_id


def validate_articles(articles: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Validate canonical articles and recompute their content IDs."""
    if not articles:
        msg = "Canonical article input is empty."
        raise ValueError(msg)
    articles_by_id = {}
    for article in articles:
        if not isinstance(article, dict) or tuple(article) != NEWS_ARTICLE_COLUMNS:
            msg = "Canonical article schema or order does not match."
            raise ValueError(msg)
        article_id = require_text(article, "article_id", "Canonical article")
        if article_id in articles_by_id:
            msg = f"Duplicate canonical article ID {article_id!r}."
            raise ValueError(msg)
        require_text(article, "source_host", f"Article {article_id}")
        require_text(article, "publication_datetime", f"Article {article_id}")
        require_text(article, "body", f"Article {article_id}")
        title = article["title"]
        if title is not None and not isinstance(title, str):
            msg = f"Article {article_id!r} title must be a string or null."
            raise ValueError(msg)
        articles_by_id[article_id] = article
    return articles_by_id


def validate_contents(contents: list[dict[str, Any]]) -> set[str]:
    """Validate the available canonical-content IDs."""
    if not contents:
        msg = "Canonical content input is empty."
        raise ValueError(msg)
    content_ids = set()
    for content in contents:
        if not isinstance(content, dict) or tuple(content) != NEWS_CONTENT_COLUMNS:
            msg = "Canonical content schema or order does not match."
            raise ValueError(msg)
        content_id = require_text(content, "content_id", "Canonical content")
        expected_id = build_content_id(normalize_content_body(content["body"]))
        if content_id != expected_id:
            msg = f"Canonical content ID {content_id!r} does not match its body."
            raise ValueError(msg)
        if content_id in content_ids:
            msg = f"Duplicate canonical content ID {content_id!r}."
            raise ValueError(msg)
        content_ids.add(content_id)
    return content_ids


def validate_content_links(
    article_content_links: list[dict[str, Any]],
    article_ids: set[str],
    content_ids: set[str],
) -> dict[str, str]:
    """Validate complete one-to-one article-to-content mappings."""
    mappings = {}
    for link in article_content_links:
        if not isinstance(link, dict) or tuple(link) != ARTICLE_CONTENT_LINK_COLUMNS:
            msg = "Article-content link schema or order does not match."
            raise ValueError(msg)
        article_id = require_text(link, "article_id", "Article-content link")
        content_id = require_text(link, "content_id", "Article-content link")
        if article_id in mappings:
            msg = f"Article {article_id!r} has multiple content mappings."
            raise ValueError(msg)
        if content_id not in content_ids:
            msg = f"Article {article_id!r} references unknown content {content_id!r}."
            raise ValueError(msg)
        mappings[article_id] = content_id
    if set(mappings) != article_ids:
        msg = "Every canonical article must have exactly one content mapping."
        raise ValueError(msg)
    return mappings


def validate_article_link(link: dict[str, Any]) -> None:
    """Validate the fixed request-bound article-link schema."""
    if not isinstance(link, dict) or tuple(link) != NEWS_ARTICLE_LINK_COLUMNS:
        msg = "Request-bound article-link schema or order does not match."
        raise ValueError(msg)
    require_text(link, "request_id", "Article link")
    require_text(link, "article_id", "Article link")
    if link["side"] not in {"home", "away"}:
        msg = f"Article link has unsupported side {link['side']!r}."
        raise ValueError(msg)
    require_int(link["match_id"], "Article-link match ID")
    require_int(link["team_id"], "Article-link team ID")
    require_positive_int(link["request_article_index"], "Request article index")


def validate_link_against_match(
    link: dict[str, Any], matches_by_id: dict[int, dict[str, Any]]
) -> None:
    """Ensure link ownership exactly matches the normalized fixture side."""
    match_id = link["match_id"]
    match = matches_by_id.get(match_id)
    if match is None:
        msg = f"Article link references unknown match {match_id}."
        raise ValueError(msg)
    side = link["side"]
    expected_team_id = match[f"{side}_team_id"]
    expected_team = match[f"{side}_team"]
    if link["team_id"] != expected_team_id or link["team"] != expected_team:
        msg = (
            f"Article link {link['request_id']!r} does not match the {side} "
            f"team of fixture {match_id}."
        )
        raise ValueError(msg)
    if link["league"] != match["league"]:
        msg = f"Article link league does not match fixture {match_id}."
        raise ValueError(msg)


def validate_tasks(tasks: list[dict[str, Any]]) -> None:
    """Validate stable task schema, IDs, and selected title/body pairing."""
    if not tasks:
        msg = "News annotation task input is empty."
        raise ValueError(msg)
    task_ids = set()
    task_keys = set()
    for task in tasks:
        if not isinstance(task, dict) or tuple(task) != NEWS_ANNOTATION_TASK_COLUMNS:
            msg = "News annotation task schema or order does not match."
            raise ValueError(msg)
        task_id = require_text(task, "task_id", "News annotation task")
        if task["task_schema_version"] != NEWS_ANNOTATION_TASK_SCHEMA_VERSION:
            msg = f"Task {task_id!r} has an unsupported task schema version."
            raise ValueError(msg)
        expected_id = build_task_id(
            task["annotation_schema_id"], task["request_id"], task["content_id"]
        )
        if task_id != expected_id:
            msg = f"News annotation task ID {task_id!r} is not deterministic."
            raise ValueError(msg)
        if task_id in task_ids:
            msg = f"Duplicate news annotation task ID {task_id!r}."
            raise ValueError(msg)
        task_ids.add(task_id)
        task_key = (task["request_id"], task["content_id"])
        if task_key in task_keys:
            msg = f"Duplicate news annotation task key {task_key!r}."
            raise ValueError(msg)
        task_keys.add(task_key)
        if task["side"] not in {"home", "away"}:
            msg = f"Task {task_id!r} has unsupported side {task['side']!r}."
            raise ValueError(msg)
        require_text(task, "target_team", f"Task {task_id}")
        require_text(task, "opponent_team", f"Task {task_id}")
        require_text(task, "body", f"Task {task_id}")
        expected_content_id = build_content_id(normalize_content_body(task["body"]))
        if task["content_id"] != expected_content_id:
            msg = f"Task {task_id!r} body does not match its content ID."
            raise ValueError(msg)


def build_task_id(
    annotation_schema_id: str, request_id: str, content_id: str
) -> str:
    """Build a stable ID for one schema, request, and content combination."""
    value = json.dumps(
        [
            NEWS_ANNOTATION_TASK_SCHEMA_VERSION,
            annotation_schema_id,
            request_id,
            content_id,
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return f"news_annotation_task_sha256:{digest}"


def build_annotation_id(
    task_id: str, config_sha256: str, model_digest: str
) -> str:
    """Build a stable ID so reruns skip only the exact same inference setup."""
    value = json.dumps(
        [
            NEWS_ANNOTATION_OUTPUT_SCHEMA_VERSION,
            task_id,
            config_sha256,
            model_digest,
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return f"news_annotation_sha256:{digest}"


def build_failure_event_id(annotation_id: str, failure_sequence: int) -> str:
    """Build a stable ID for one deferred failure attempt group."""
    value = json.dumps(
        [
            NEWS_ANNOTATION_FAILURE_SCHEMA_VERSION,
            annotation_id,
            failure_sequence,
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return f"news_annotation_failure_sha256:{digest}"


def validate_target_team_aliases(value: object) -> dict[str, tuple[str, ...]]:
    """Validate the team aliases supplied to the model as article context."""
    if not isinstance(value, dict) or not value:
        msg = "Annotation target_team_aliases must be a non-empty object."
        raise ValueError(msg)
    aliases_by_team: dict[str, tuple[str, ...]] = {}
    for team, aliases in value.items():
        team_name = require_non_empty_text(team, "Target-team alias key")
        if not isinstance(aliases, list) or not aliases:
            msg = f"Target-team aliases for {team_name!r} must be a non-empty array."
            raise ValueError(msg)
        parsed = tuple(
            require_non_empty_text(alias, f"Alias for {team_name}")
            for alias in aliases
        )
        if len(set(alias.casefold() for alias in parsed)) != len(parsed):
            msg = f"Target-team aliases for {team_name!r} contain duplicates."
            raise ValueError(msg)
        aliases_by_team[team_name] = parsed
    return aliases_by_team


@contextmanager
def annotation_output_lock(output_path: Path):
    """Hold an OS-level lock so two processes cannot append the same output."""
    lock_path = output_path.with_name(f"{output_path.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_file = lock_path.open("a+b")
    if lock_file.seek(0, os.SEEK_END) == 0:
        lock_file.write(b"\0")
        lock_file.flush()
    lock_file.seek(0)
    try:
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        lock_file.close()
        msg = (
            f"Another annotation process already owns output {output_path}. "
            "Wait for it to finish instead of starting a concurrent resume run."
        )
        raise ValueError(msg) from exc
    try:
        yield
    finally:
        try:
            lock_file.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        finally:
            lock_file.close()


def append_jsonl_record(record: dict[str, Any], path: Path) -> None:
    """Append one UTF-8 JSON object immediately for resume safety."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def require_text(record: dict[str, Any], key: str, context: str) -> str:
    """Return one required non-empty string field."""
    return require_non_empty_text(record.get(key), f"{context} field {key!r}")


def require_non_empty_text(value: object, context: str) -> str:
    """Return a stripped non-empty string or reject it."""
    if not isinstance(value, str) or not value.strip():
        msg = f"{context} must be a non-empty string."
        raise ValueError(msg)
    return value.strip()


def require_int(value: object, context: str) -> int:
    """Return an integer that is not a boolean."""
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"{context} must be an integer."
        raise ValueError(msg)
    return value


def require_positive_int(value: object, context: str) -> int:
    """Return a strictly positive integer."""
    parsed = require_int(value, context)
    if parsed <= 0:
        msg = f"{context} must be positive."
        raise ValueError(msg)
    return parsed
