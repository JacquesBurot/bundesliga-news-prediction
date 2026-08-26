"""Build and validate request-bound, team-specific annotation tasks."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from typing import Any

from ..articles import (
    NEWS_ARTICLE_COLUMNS,
    NEWS_ARTICLE_LINK_COLUMNS,
)
from ..contents import (
    ARTICLE_CONTENT_LINK_COLUMNS,
    NEWS_CONTENT_COLUMNS,
    build_content_id,
    normalize_content_body,
)
from ._validation import (
    require_int,
    require_non_empty_text,
    require_positive_int,
    require_text,
)


NEWS_ANNOTATION_TASK_SCHEMA_VERSION = 1
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


@dataclass(frozen=True)
class NewsAnnotationTasksBuild:
    """Team-specific annotation tasks and their quality metadata."""

    tasks: list[dict[str, Any]]
    quality_report: dict[str, Any]


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
