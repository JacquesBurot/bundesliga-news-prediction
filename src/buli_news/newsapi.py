"""Fetch raw article responses from Event Registry / NewsAPI.ai."""

from __future__ import annotations

import json
import os
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import httpx

from buli_news.storage import write_text


API_KEY_ENV_VAR = "NEWSAPI_KEY"


def get_api_key(env_file: Path = Path(".env")) -> str:
    """Read the API key from the environment or a local .env file."""
    api_key = os.environ.get(API_KEY_ENV_VAR)
    if api_key:
        return api_key

    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            parsed_key = parse_env_line(line, key=API_KEY_ENV_VAR)
            if parsed_key:
                return parsed_key

    msg = f"{API_KEY_ENV_VAR} is not set in the environment or .env."
    raise ValueError(msg)


def parse_env_line(line: str, key: str) -> str | None:
    """Parse a simple KEY=VALUE .env line."""
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or "=" not in stripped:
        return None

    env_key, value = stripped.split("=", 1)
    if env_key.strip() != key:
        return None

    return value.strip().strip("\"'")


def select_requests(
    requests: list[dict[str, Any]],
    request_id: str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Select planned requests by optional request ID and limit."""
    selected = requests
    if request_id is not None:
        selected = [
            request for request in requests if request.get("request_id") == request_id
        ]
        if not selected:
            msg = f"No planned request found for request ID {request_id!r}."
            raise ValueError(msg)

    if limit is not None:
        if limit < 1:
            msg = "--limit must be greater than zero."
            raise ValueError(msg)
        selected = selected[:limit]

    return selected


def fetch_news_requests(
    requests: list[dict[str, Any]],
    output_dir: Path,
    api_key: str,
    delay_seconds: float,
) -> list[dict[str, Any]]:
    """Fetch selected planned requests and store raw responses."""
    results = []
    with httpx.Client(timeout=60.0) as client:
        for index, planned_request in enumerate(requests):
            if index > 0 and delay_seconds > 0:
                time.sleep(delay_seconds)

            result = fetch_news_request(
                client=client,
                planned_request=planned_request,
                output_dir=output_dir,
                api_key=api_key,
            )
            results.append(result)

    return results


def fetch_news_request(
    client: httpx.Client,
    planned_request: dict[str, Any],
    output_dir: Path,
    api_key: str,
) -> dict[str, Any]:
    """Fetch one planned request and store its raw response."""
    request_id = get_required_str(planned_request, "request_id")
    endpoint = get_required_str(planned_request, "endpoint")
    payload = get_required_dict(planned_request, "payload")
    payload_with_key = deepcopy(payload)
    payload_with_key["apiKey"] = api_key

    response = client.post(endpoint, json=payload_with_key)
    raw_response_path = output_dir / f"{request_id}.json"
    write_text(response.text, raw_response_path)
    response.raise_for_status()

    return {
        "request_id": request_id,
        "match_id": planned_request.get("match_id"),
        "season": planned_request.get("season"),
        "league": planned_request.get("league"),
        "side": planned_request.get("side"),
        "request_type": planned_request.get("request_type"),
        "status_code": response.status_code,
        "article_count": get_article_count(response.text),
        "raw_response_path": str(raw_response_path),
    }


def get_article_count(response_text: str) -> int | None:
    """Extract the number of returned articles from an Event Registry response."""
    try:
        data = json.loads(response_text)
    except json.JSONDecodeError:
        return None

    articles = data.get("articles")
    if not isinstance(articles, dict):
        return None

    results = articles.get("results")
    if isinstance(results, list):
        return len(results)

    return None


def get_required_str(data: dict[str, Any], key: str) -> str:
    """Read a required string field."""
    value = data.get(key)
    if not isinstance(value, str) or not value:
        msg = f"Field {key!r} must be a non-empty string."
        raise ValueError(msg)
    return value


def get_required_dict(data: dict[str, Any], key: str) -> dict[str, Any]:
    """Read a required dictionary field."""
    value = data.get(key)
    if not isinstance(value, dict):
        msg = f"Field {key!r} must be an object."
        raise ValueError(msg)
    return value
