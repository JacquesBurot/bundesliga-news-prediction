"""Group canonical news articles by deterministic normalized text content."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Any

from buli_news.news.articles import NEWS_ARTICLE_COLUMNS


NEWS_CONTENT_SCHEMA_VERSION = 1
CONTENT_NORMALIZATION_VERSION = 1
WHITESPACE_PATTERN = re.compile(r"\s+")
NEWS_CONTENT_COLUMNS = (
    "content_id",
    "representative_article_id",
    "body",
    "member_article_count",
    "source_host_count",
)
ARTICLE_CONTENT_LINK_COLUMNS = (
    "article_id",
    "content_id",
)


@dataclass(frozen=True)
class NewsContentsBuild:
    """Canonical contents, article mappings, and de-duplication quality."""

    contents: list[dict[str, Any]]
    article_content_links: list[dict[str, str]]
    quality_report: dict[str, Any]


def build_news_contents(
    articles: list[dict[str, Any]],
    season: int,
) -> NewsContentsBuild:
    """Group articles with identical normalized bodies under one content ID."""
    if isinstance(season, bool) or not isinstance(season, int):
        msg = "Season must be an integer start year."
        raise ValueError(msg)
    parsed_articles = validate_and_sort_articles(articles)

    normalized_body_by_content_id: dict[str, str] = {}
    articles_by_content_id: dict[str, list[dict[str, Any]]] = {}
    article_content_links = []
    for article in parsed_articles:
        normalized_body = normalize_content_body(article["body"])
        content_id = build_content_id(normalized_body)
        existing_body = normalized_body_by_content_id.setdefault(
            content_id,
            normalized_body,
        )
        if existing_body != normalized_body:
            msg = f"SHA-256 collision detected for content ID {content_id!r}."
            raise ValueError(msg)
        articles_by_content_id.setdefault(content_id, []).append(article)
        article_content_links.append(
            {
                "article_id": article["article_id"],
                "content_id": content_id,
            }
        )

    contents = []
    duplicate_groups = []
    for content_id in sorted(articles_by_content_id):
        members = articles_by_content_id[content_id]
        representative = members[0]
        source_hosts = {article["source_host"] for article in members}
        content = {
            "content_id": content_id,
            "representative_article_id": representative["article_id"],
            "body": representative["body"],
            "member_article_count": len(members),
            "source_host_count": len(source_hosts),
        }
        contents.append(content)
        if len(members) > 1:
            duplicate_groups.append(members)

    validate_news_content_outputs(
        articles=parsed_articles,
        contents=contents,
        article_content_links=article_content_links,
    )
    quality_report = build_quality_report(
        season=season,
        articles=parsed_articles,
        contents=contents,
        duplicate_groups=duplicate_groups,
    )
    return NewsContentsBuild(
        contents=contents,
        article_content_links=article_content_links,
        quality_report=quality_report,
    )


def validate_and_sort_articles(
    articles: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Validate the canonical-article schema and unique article IDs."""
    if not articles:
        msg = "Canonical news article input is empty."
        raise ValueError(msg)

    article_ids = []
    for article in articles:
        if not isinstance(article, dict):
            msg = "Every canonical news article must be an object."
            raise ValueError(msg)
        if tuple(article) != NEWS_ARTICLE_COLUMNS:
            msg = (
                "Canonical news article schema or order does not match: "
                f"{list(article)}."
            )
            raise ValueError(msg)
        article_id = require_text(article, "article_id", "Canonical news article")
        require_text(article, "source_host", f"Canonical article {article_id}")
        require_text(article, "body", f"Canonical article {article_id}")
        article_ids.append(article_id)

    duplicate_ids = sorted(
        article_id
        for article_id, count in Counter(article_ids).items()
        if count > 1
    )
    if duplicate_ids:
        msg = f"Canonical news articles contain duplicate IDs: {duplicate_ids}."
        raise ValueError(msg)
    return sorted(articles, key=lambda article: article["article_id"])


def normalize_content_body(body: str) -> str:
    """Normalize text for exact-content grouping without editing stored text."""
    normalized = unicodedata.normalize("NFKC", body).casefold()
    normalized = WHITESPACE_PATTERN.sub(" ", normalized).strip()
    if not normalized:
        msg = "Canonical news article body is empty after normalization."
        raise ValueError(msg)
    return normalized


def build_content_id(normalized_body: str) -> str:
    """Hash one normalized article body into a stable content identifier."""
    digest = hashlib.sha256(normalized_body.encode("utf-8")).hexdigest()
    return f"content_sha256:{digest}"


def build_quality_report(
    season: int,
    articles: list[dict[str, Any]],
    contents: list[dict[str, Any]],
    duplicate_groups: list[list[dict[str, Any]]],
) -> dict[str, Any]:
    """Describe deterministic exact-text grouping without losing provenance."""
    cross_source_groups = [
        group
        for group in duplicate_groups
        if len({article["source_host"] for article in group}) > 1
    ]
    multiple_title_groups = [
        group
        for group in duplicate_groups
        if len({normalize_optional_title(article.get("title")) for article in group})
        > 1
    ]
    group_size_counts = Counter(len(group) for group in duplicate_groups)
    report = {
        "schema_version": NEWS_CONTENT_SCHEMA_VERSION,
        "season": season,
        "normalization": {
            "version": CONTENT_NORMALIZATION_VERSION,
            "input_field": "body",
            "steps": [
                "Unicode NFKC normalization",
                "Unicode case folding",
                "collapse consecutive whitespace to one space",
                "strip leading and trailing whitespace",
                "SHA-256 over UTF-8 normalized text",
            ],
            "content_id_format": "content_sha256:{64 lowercase hex characters}",
            "near_duplicate_grouping": False,
        },
        "association_rule": (
            "Content grouping never changes request, match, side, team, source, "
            "URL, or article associations. Every article keeps exactly one "
            "article_id-to-content_id mapping."
        ),
        "summary": {
            "article_count": len(articles),
            "content_count": len(contents),
            "article_content_link_count": len(articles),
            "duplicate_content_group_count": len(duplicate_groups),
            "articles_in_duplicate_content_groups": sum(
                len(group) for group in duplicate_groups
            ),
            "collapsed_article_count": len(articles) - len(contents),
            "cross_source_duplicate_content_group_count": len(
                cross_source_groups
            ),
            "duplicate_group_with_multiple_titles_count": len(
                multiple_title_groups
            ),
            "maximum_member_article_count": max(
                content["member_article_count"] for content in contents
            ),
        },
        "duplicate_group_size_counts": {
            str(size): group_size_counts[size]
            for size in sorted(group_size_counts)
        },
    }
    validate_quality_report(report)
    return report


def normalize_optional_title(value: object) -> str | None:
    """Normalize an optional title for descriptive group statistics only."""
    if not isinstance(value, str) or not value.strip():
        return None
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return WHITESPACE_PATTERN.sub(" ", normalized).strip()


def validate_news_content_outputs(
    articles: list[dict[str, Any]],
    contents: list[dict[str, Any]],
    article_content_links: list[dict[str, str]],
) -> None:
    """Validate schemas and the complete one-to-one article mapping."""
    content_ids = []
    representative_ids = []
    for content in contents:
        if tuple(content) != NEWS_CONTENT_COLUMNS:
            msg = (
                "Canonical news content schema or order does not match: "
                f"{list(content)}."
            )
            raise ValueError(msg)
        content_ids.append(content["content_id"])
        representative_ids.append(content["representative_article_id"])
    if len(content_ids) != len(set(content_ids)):
        msg = "Canonical news contents contain duplicate content IDs."
        raise ValueError(msg)

    input_article_ids = {article["article_id"] for article in articles}
    link_article_ids = []
    linked_content_ids = set()
    for link in article_content_links:
        if tuple(link) != ARTICLE_CONTENT_LINK_COLUMNS:
            msg = (
                "Article-content link schema or order does not match: "
                f"{list(link)}."
            )
            raise ValueError(msg)
        link_article_ids.append(link["article_id"])
        linked_content_ids.add(link["content_id"])
    if len(link_article_ids) != len(set(link_article_ids)):
        msg = "An article is mapped to more than one content ID."
        raise ValueError(msg)
    if set(link_article_ids) != input_article_ids:
        msg = "Every canonical article must have exactly one content mapping."
        raise ValueError(msg)
    if linked_content_ids != set(content_ids):
        msg = "Every canonical content must be referenced by an article mapping."
        raise ValueError(msg)
    if not set(representative_ids).issubset(input_article_ids):
        msg = "Every representative article must exist in the canonical input."
        raise ValueError(msg)


def validate_quality_report(report: dict[str, Any]) -> None:
    """Validate exact-content count identities in the quality report."""
    summary = report["summary"]
    if summary["article_content_link_count"] != summary["article_count"]:
        msg = "Every article must produce exactly one article-content link."
        raise ValueError(msg)
    if (
        summary["content_count"] + summary["collapsed_article_count"]
        != summary["article_count"]
    ):
        msg = "Content grouping counts do not reconstruct the article count."
        raise ValueError(msg)
    duplicate_group_count = sum(report["duplicate_group_size_counts"].values())
    if duplicate_group_count != summary["duplicate_content_group_count"]:
        msg = "Duplicate content group-size counts do not match the summary."
        raise ValueError(msg)


def require_text(data: dict[str, Any], key: str, context: str) -> str:
    """Return one required non-empty string value."""
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        msg = f"{context} needs non-empty text field {key!r}."
        raise ValueError(msg)
    return value.strip()
