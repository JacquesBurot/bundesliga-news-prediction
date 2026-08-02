"""Normalize, filter, and de-duplicate collected news articles."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit
from zoneinfo import ZoneInfo

from buli_news.news_source_review import get_homepage_url


NEWS_ARTICLE_SCHEMA_VERSION = 1
NEWS_ARTICLE_COLUMNS = (
    "article_id",
    "event_registry_uri",
    "url",
    "source_host",
    "source_uri",
    "source_title",
    "language",
    "event_registry_date",
    "publication_datetime",
    "title",
    "body",
    "authors",
)
NEWS_ARTICLE_LINK_COLUMNS = (
    "request_id",
    "article_id",
    "match_id",
    "season",
    "league",
    "side",
    "team_id",
    "team",
    "request_type",
    "query_strategy",
    "date_start",
    "date_end",
    "request_article_index",
    "raw_response_path",
)


@dataclass(frozen=True)
class NewsArticlesBuild:
    """Policy-filtered articles, request-bound links, and quality metadata."""

    articles: list[dict[str, Any]]
    article_links: list[dict[str, Any]]
    quality_report: dict[str, Any]


@dataclass(frozen=True)
class PlannedNewsRequest:
    """Validated request metadata that owns all response associations."""

    request_id: str
    match_id: int
    season: int
    league: str
    side: str
    team_id: int
    team: str
    request_type: str
    query_strategy: str
    date_start: date
    date_end: date
    lang: str


def build_news_articles(
    requests: list[dict[str, Any]],
    raw_dir: Path,
    policy: dict[str, Any],
    season: int,
    timezone: str = "Europe/Berlin",
) -> NewsArticlesBuild:
    """Build canonical articles and links only from their source requests."""
    parsed_requests = parse_planned_requests(requests=requests, season=season)
    policy_decisions = build_policy_decisions(policy)
    local_timezone = parse_timezone(timezone)
    raw_paths = match_raw_responses(
        requests=parsed_requests,
        raw_dir=raw_dir,
    )

    canonical_articles: dict[str, dict[str, Any]] = {}
    article_links: list[dict[str, Any]] = []
    link_keys: set[tuple[str, str]] = set()
    policy_occurrence_counts: Counter[str] = Counter()
    rejection_counts: Counter[str] = Counter()
    excluded_host_counts: Counter[str] = Counter()
    unknown_host_counts: Counter[str] = Counter()
    response_occurrence_count = 0
    request_link_counts: Counter[str] = Counter()
    match_ids: set[int] = set()

    for request in parsed_requests:
        raw_path = raw_paths[request.request_id]
        articles = read_response_articles(raw_path)
        for article_index, raw_article in enumerate(articles, start=1):
            response_occurrence_count += 1
            source_host = get_article_source_host(raw_article)
            if source_host is None:
                policy_occurrence_counts["unknown"] += 1
                rejection_counts["missing_source_host"] += 1
                continue

            decision = policy_decisions.get(source_host, "unknown")
            policy_occurrence_counts[decision] += 1
            if decision == "exclude":
                excluded_host_counts[source_host] += 1
                rejection_counts["policy_exclude"] += 1
                continue
            if decision == "unknown":
                unknown_host_counts[source_host] += 1
                rejection_counts["policy_unknown_host"] += 1
                continue

            normalized = normalize_article(
                raw_article=raw_article,
                source_host=source_host,
                request=request,
                timezone=local_timezone,
            )
            if isinstance(normalized, str):
                rejection_counts[normalized] += 1
                continue

            article_id = normalized["article_id"]
            existing_article = canonical_articles.get(article_id)
            if existing_article is None:
                canonical_articles[article_id] = normalized
            elif existing_article != normalized:
                msg = (
                    f"Article {article_id!r} has conflicting canonical fields "
                    f"in raw response {raw_path}."
                )
                raise ValueError(msg)

            link_key = (request.request_id, article_id)
            if link_key in link_keys:
                rejection_counts["duplicate_article_within_request"] += 1
                continue
            link_keys.add(link_key)
            article_links.append(
                build_article_link(
                    request=request,
                    article_id=article_id,
                    article_index=article_index,
                    raw_path=raw_path,
                )
            )
            request_link_counts[request.request_id] += 1
            match_ids.add(request.match_id)

    articles = sorted(
        canonical_articles.values(),
        key=lambda article: article["article_id"],
    )
    article_links.sort(
        key=lambda link: (
            link["request_id"],
            link["request_article_index"],
            link["article_id"],
        )
    )
    validate_news_article_outputs(articles=articles, article_links=article_links)

    linked_article_ids = {link["article_id"] for link in article_links}
    requests_without_links = [
        request.request_id
        for request in parsed_requests
        if request_link_counts[request.request_id] == 0
    ]
    normalized_occurrence_count = len(article_links)
    quality_report = {
        "schema_version": NEWS_ARTICLE_SCHEMA_VERSION,
        "season": season,
        "timezone": timezone,
        "association_rule": (
            "An article is linked only to the request response in which it "
            "occurred. No article is assigned to another match or side by "
            "content, team name, or de-duplication."
        ),
        "policy": {
            "policy_id": policy["policy_id"],
            "schema_version": policy["schema_version"],
            "unknown_host_decision": "exclude",
            "configured_rule_count": len(policy_decisions),
            "source_review_workbook_sha256": policy["source_review"][
                "workbook_sha256"
            ],
        },
        "inputs": {
            "planned_request_count": len(parsed_requests),
            "raw_response_count": len(raw_paths),
            "raw_response_directory": str(raw_dir),
        },
        "summary": {
            "article_occurrence_count": response_occurrence_count,
            "policy_include_occurrence_count": policy_occurrence_counts["include"],
            "policy_exclude_occurrence_count": policy_occurrence_counts["exclude"],
            "policy_unknown_occurrence_count": policy_occurrence_counts["unknown"],
            "normalized_occurrence_count": normalized_occurrence_count,
            "canonical_article_count": len(articles),
            "article_link_count": len(article_links),
            "cross_request_deduplication_count": (
                len(article_links) - len(linked_article_ids)
            ),
            "linked_request_count": len(request_link_counts),
            "linked_match_count": len(match_ids),
            "request_without_link_count": len(requests_without_links),
        },
        "rejection_counts": ordered_counter(rejection_counts),
        "excluded_occurrence_counts_by_host": ordered_counter(
            excluded_host_counts
        ),
        "unknown_occurrence_counts_by_host": ordered_counter(
            unknown_host_counts
        ),
        "request_ids_without_links": requests_without_links,
    }
    validate_quality_report(quality_report)
    return NewsArticlesBuild(
        articles=articles,
        article_links=article_links,
        quality_report=quality_report,
    )


def parse_planned_requests(
    requests: list[dict[str, Any]],
    season: int,
) -> list[PlannedNewsRequest]:
    """Validate planned requests and return deterministic typed metadata."""
    if not requests:
        msg = "Planned news request input is empty."
        raise ValueError(msg)

    parsed_requests = []
    request_ids: set[str] = set()
    request_sides_by_match: dict[int, set[str]] = {}
    for request in requests:
        if not isinstance(request, dict):
            msg = "Every planned news request must be an object."
            raise ValueError(msg)
        request_id = require_text(request, "request_id", "Planned news request")
        context = f"Planned news request {request_id}"
        if request_id in request_ids:
            msg = f"Planned news request ID {request_id!r} occurs more than once."
            raise ValueError(msg)
        request_ids.add(request_id)

        request_season = require_integer(request, "season", context)
        if request_season != season:
            msg = (
                f"{context} belongs to season {request_season}, "
                f"but season {season} was requested."
            )
            raise ValueError(msg)
        side = require_text(request, "side", context)
        if side not in {"home", "away"}:
            msg = f"{context} has invalid side {side!r}."
            raise ValueError(msg)
        date_start = parse_iso_date(request, "date_start", context)
        date_end = parse_iso_date(request, "date_end", context)
        if date_start > date_end:
            msg = f"{context} has an inverted pre-match window."
            raise ValueError(msg)

        match_id = require_integer(request, "match_id", context)
        match_sides = request_sides_by_match.setdefault(match_id, set())
        if side in match_sides:
            msg = f"Match {match_id} has more than one {side!r} news request."
            raise ValueError(msg)
        match_sides.add(side)

        parsed_requests.append(
            PlannedNewsRequest(
                request_id=request_id,
                match_id=match_id,
                season=request_season,
                league=require_text(request, "league", context),
                side=side,
                team_id=require_integer(request, "team_id", context),
                team=require_text(request, "team", context),
                request_type=require_text(request, "request_type", context),
                query_strategy=require_text(request, "query_strategy", context),
                date_start=date_start,
                date_end=date_end,
                lang=require_text(request, "lang", context),
            )
        )

    incomplete_match_sides = {
        match_id: sorted(sides)
        for match_id, sides in request_sides_by_match.items()
        if sides != {"home", "away"}
    }
    if incomplete_match_sides:
        msg = (
            "Every match must have exactly one home and one away news request. "
            f"Invalid match sides: {incomplete_match_sides}."
        )
        raise ValueError(msg)

    return sorted(parsed_requests, key=lambda request: request.request_id)


def build_policy_decisions(policy: dict[str, Any]) -> dict[str, str]:
    """Validate the source policy fields used for exact-host filtering."""
    if not isinstance(policy, dict):
        msg = "News source policy must contain a JSON object."
        raise ValueError(msg)
    if policy.get("schema_version") != 1:
        msg = "News source policy must use schema_version 1."
        raise ValueError(msg)
    require_text(policy, "policy_id", "News source policy")
    decision_policy = policy.get("decision_policy")
    if not isinstance(decision_policy, dict):
        msg = "News source policy needs a decision_policy object."
        raise ValueError(msg)
    if decision_policy.get("unknown_host_decision") != "exclude":
        msg = "News source policy must exclude unknown hosts."
        raise ValueError(msg)
    source_review = policy.get("source_review")
    if not isinstance(source_review, dict):
        msg = "News source policy needs a source_review object."
        raise ValueError(msg)
    require_text(
        source_review,
        "workbook_sha256",
        "News source policy source_review",
    )

    rules = policy.get("rules")
    if not isinstance(rules, list) or not rules:
        msg = "News source policy must contain non-empty rules."
        raise ValueError(msg)
    decisions: dict[str, str] = {}
    for index, rule in enumerate(rules, start=1):
        if not isinstance(rule, dict):
            msg = f"News source policy rule {index} must be an object."
            raise ValueError(msg)
        context = f"News source policy rule {index}"
        host = require_text(rule, "host", context)
        if host != normalize_host(host):
            msg = f"{context} has non-normalized host {host!r}."
            raise ValueError(msg)
        if rule.get("match_type") != "exact_host":
            msg = f"{context} must use exact_host matching."
            raise ValueError(msg)
        decision = require_text(rule, "decision", context)
        if decision not in {"include", "exclude"}:
            msg = f"{context} has invalid decision {decision!r}."
            raise ValueError(msg)
        if host in decisions:
            msg = f"News source policy contains duplicate host {host!r}."
            raise ValueError(msg)
        decisions[host] = decision
    return decisions


def match_raw_responses(
    requests: list[PlannedNewsRequest],
    raw_dir: Path,
) -> dict[str, Path]:
    """Require exactly one raw response for every planned request."""
    response_paths = sorted(raw_dir.glob("*.json"))
    response_ids = {path.stem for path in response_paths}
    request_ids = {request.request_id for request in requests}
    missing_ids = sorted(request_ids.difference(response_ids))
    unexpected_ids = sorted(response_ids.difference(request_ids))
    if missing_ids or unexpected_ids:
        msg = (
            "Planned requests and raw news responses do not match one-to-one. "
            f"Missing response IDs: {missing_ids}; unexpected response IDs: "
            f"{unexpected_ids}."
        )
        raise ValueError(msg)
    return {path.stem: path for path in response_paths}


def read_response_articles(path: Path) -> list[dict[str, Any]]:
    """Read the validated article list from one raw response."""
    try:
        response = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        msg = f"Raw news response {path} is not valid JSON."
        raise ValueError(msg) from exc
    if not isinstance(response, dict):
        msg = f"Raw news response {path} must contain a JSON object."
        raise ValueError(msg)
    articles = response.get("articles")
    if not isinstance(articles, dict):
        msg = f"Raw news response {path} has no articles object."
        raise ValueError(msg)
    results = articles.get("results")
    if not isinstance(results, list):
        msg = f"Raw news response {path} has no articles.results list."
        raise ValueError(msg)
    if not all(isinstance(article, dict) for article in results):
        msg = f"Raw news response {path} contains a non-object article."
        raise ValueError(msg)
    return results


def normalize_article(
    raw_article: dict[str, Any],
    source_host: str,
    request: PlannedNewsRequest,
    timezone: ZoneInfo,
) -> dict[str, Any] | str:
    """Normalize one included occurrence or return its rejection reason."""
    language = optional_text(raw_article.get("lang"))
    if language is None:
        return "missing_language"
    if language != request.lang:
        return "language_mismatch"

    event_registry_date = parse_optional_date(raw_article.get("date"))
    if event_registry_date is None:
        return "invalid_event_registry_date"
    if not request.date_start <= event_registry_date <= request.date_end:
        return "event_registry_date_outside_request_window"

    publication_datetime = parse_optional_datetime(raw_article.get("dateTimePub"))
    if publication_datetime is None:
        return "invalid_publication_datetime"
    local_publication_date = publication_datetime.astimezone(timezone).date()
    if not request.date_start <= local_publication_date <= request.date_end:
        return "publication_datetime_outside_request_window"

    url = normalize_article_url(raw_article.get("url"))
    if url is None:
        return "invalid_article_url"
    body = optional_text(raw_article.get("body"))
    if body is None:
        return "missing_article_body"
    title_value = raw_article.get("title")
    if title_value is not None and not isinstance(title_value, str):
        return "invalid_article_title"
    title = optional_text(title_value)

    event_registry_uri = optional_text(raw_article.get("uri"))
    article_id = build_article_id(
        event_registry_uri=event_registry_uri,
        url=url,
    )
    source = raw_article.get("source")
    if not isinstance(source, dict):
        return "invalid_article_source"
    authors = normalize_authors(raw_article.get("authors"))
    if authors is None:
        return "invalid_article_authors"

    return {
        "article_id": article_id,
        "event_registry_uri": event_registry_uri,
        "url": url,
        "source_host": source_host,
        "source_uri": optional_text(source.get("uri")),
        "source_title": optional_text(source.get("title")),
        "language": language,
        "event_registry_date": event_registry_date.isoformat(),
        "publication_datetime": publication_datetime.isoformat(),
        "title": title,
        "body": body,
        "authors": authors,
    }


def build_article_link(
    request: PlannedNewsRequest,
    article_id: str,
    article_index: int,
    raw_path: Path,
) -> dict[str, Any]:
    """Retain the exact request occurrence that created an association."""
    return {
        "request_id": request.request_id,
        "article_id": article_id,
        "match_id": request.match_id,
        "season": request.season,
        "league": request.league,
        "side": request.side,
        "team_id": request.team_id,
        "team": request.team,
        "request_type": request.request_type,
        "query_strategy": request.query_strategy,
        "date_start": request.date_start.isoformat(),
        "date_end": request.date_end.isoformat(),
        "request_article_index": article_index,
        "raw_response_path": str(raw_path),
    }


def get_article_source_host(article: dict[str, Any]) -> str | None:
    """Resolve a candidate host exactly like the source-review export."""
    homepage_url = get_homepage_url(article)
    if homepage_url is None:
        return None
    parsed = urlsplit(homepage_url)
    if parsed.hostname is None:
        return None
    return normalize_host(parsed.hostname)


def normalize_host(value: str) -> str:
    """Normalize one host for exact policy matching."""
    host = value.strip().lower().rstrip(".")
    if host.startswith("www."):
        host = host[4:]
    return host


def normalize_article_url(value: object) -> str | None:
    """Normalize an HTTP(S) article URL while retaining path and query."""
    text = optional_text(value)
    if text is None:
        return None
    try:
        parsed = urlsplit(text)
        port = parsed.port
    except ValueError:
        return None
    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"} or parsed.hostname is None:
        return None
    if parsed.username or parsed.password:
        return None

    host = parsed.hostname.lower().rstrip(".")
    if not host or any(character.isspace() for character in host):
        return None
    if port is None or (scheme == "http" and port == 80) or (
        scheme == "https" and port == 443
    ):
        netloc = host
    else:
        netloc = f"{host}:{port}"
    path = parsed.path or "/"
    return urlunsplit((scheme, netloc, path, parsed.query, ""))


def build_article_id(event_registry_uri: str | None, url: str) -> str:
    """Build a stable source-prefixed de-duplication identifier."""
    if event_registry_uri is not None:
        return f"event_registry:{event_registry_uri}"
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
    return f"url_sha256:{digest}"


def normalize_authors(value: object) -> list[dict[str, Any]] | None:
    """Retain stable author identity fields in their source order."""
    if not isinstance(value, list):
        return None
    authors = []
    for author in value:
        if not isinstance(author, dict):
            return None
        is_agency = author.get("isAgency")
        if is_agency is not None and not isinstance(is_agency, bool):
            return None
        authors.append(
            {
                "uri": optional_text(author.get("uri")),
                "name": optional_text(author.get("name")),
                "type": optional_text(author.get("type")),
                "is_agency": is_agency,
            }
        )
    return authors


def parse_timezone(timezone: str) -> ZoneInfo:
    """Return one configured IANA timezone with a data-oriented error."""
    try:
        return ZoneInfo(timezone)
    except (KeyError, ValueError) as exc:
        msg = f"Unknown article publication timezone {timezone!r}."
        raise ValueError(msg) from exc


def parse_optional_date(value: object) -> date | None:
    """Parse an optional ISO date without accepting datetimes."""
    text = optional_text(value)
    if text is None:
        return None
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


def parse_optional_datetime(value: object) -> datetime | None:
    """Parse a timezone-aware ISO publication timestamp."""
    text = optional_text(value)
    if text is None:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed


def parse_iso_date(data: dict[str, Any], key: str, context: str) -> date:
    """Parse a required ISO date field."""
    value = require_text(data, key, context)
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        msg = f"{context} has invalid ISO date field {key!r}: {value!r}."
        raise ValueError(msg) from exc


def optional_text(value: object) -> str | None:
    """Return stripped non-empty text or None."""
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def require_text(data: dict[str, Any], key: str, context: str) -> str:
    """Return a required non-empty string field."""
    value = optional_text(data.get(key))
    if value is None:
        msg = f"{context} needs non-empty text field {key!r}."
        raise ValueError(msg)
    return value


def require_integer(data: dict[str, Any], key: str, context: str) -> int:
    """Return a required integer while excluding booleans."""
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"{context} needs integer field {key!r}."
        raise ValueError(msg)
    return value


def ordered_counter(counter: Counter[str]) -> dict[str, int]:
    """Serialize non-zero counter values in deterministic key order."""
    return {
        key: counter[key]
        for key in sorted(counter)
        if counter[key] != 0
    }


def validate_news_article_outputs(
    articles: list[dict[str, Any]],
    article_links: list[dict[str, Any]],
) -> None:
    """Validate schemas, unique keys, and complete link references."""
    article_ids = []
    for article in articles:
        if tuple(article) != NEWS_ARTICLE_COLUMNS:
            msg = (
                "Normalized article schema or order does not match: "
                f"{list(article)}."
            )
            raise ValueError(msg)
        article_ids.append(article["article_id"])
    if len(article_ids) != len(set(article_ids)):
        msg = "Normalized articles contain duplicate article IDs."
        raise ValueError(msg)

    article_id_set = set(article_ids)
    link_keys = []
    for link in article_links:
        if tuple(link) != NEWS_ARTICLE_LINK_COLUMNS:
            msg = (
                "News article link schema or order does not match: "
                f"{list(link)}."
            )
            raise ValueError(msg)
        if link["article_id"] not in article_id_set:
            msg = (
                f"Article link references unknown article {link['article_id']!r}."
            )
            raise ValueError(msg)
        link_keys.append((link["request_id"], link["article_id"]))
    if len(link_keys) != len(set(link_keys)):
        msg = "News article links contain duplicate request/article pairs."
        raise ValueError(msg)
    if article_id_set != {link["article_id"] for link in article_links}:
        msg = "Every canonical news article must have at least one request link."
        raise ValueError(msg)


def validate_quality_report(report: dict[str, Any]) -> None:
    """Validate the occurrence accounting in the quality report."""
    summary = report["summary"]
    decision_total = (
        summary["policy_include_occurrence_count"]
        + summary["policy_exclude_occurrence_count"]
        + summary["policy_unknown_occurrence_count"]
    )
    if decision_total != summary["article_occurrence_count"]:
        msg = "News article policy occurrence counts do not cover all occurrences."
        raise ValueError(msg)
    rejected_included = sum(
        count
        for reason, count in report["rejection_counts"].items()
        if reason
        not in {
            "missing_source_host",
            "policy_exclude",
            "policy_unknown_host",
        }
    )
    if (
        summary["normalized_occurrence_count"] + rejected_included
        != summary["policy_include_occurrence_count"]
    ):
        msg = "Included news article occurrences are not fully accounted for."
        raise ValueError(msg)
    if summary["article_link_count"] != summary["normalized_occurrence_count"]:
        msg = "Every normalized occurrence must produce exactly one article link."
        raise ValueError(msg)
