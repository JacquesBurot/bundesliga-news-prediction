"""Aggregate team-specific news annotations into match-level model features."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime
from math import isfinite
from typing import Any
from zoneinfo import ZoneInfo

from buli_news.matches.features import get_dataset_split, validate_split_order
from buli_news.news.annotations._validation import (
    require_int,
    require_positive_int,
    require_text,
)
from buli_news.news.annotations.config import AnnotationConfig, INDICATORS
from buli_news.news.annotations.ollama import validate_annotation_response
from buli_news.news.annotations.results import (
    NEWS_ANNOTATION_FAILURE_COLUMNS,
    NEWS_ANNOTATION_FAILURE_SCHEMA_VERSION,
    NEWS_ANNOTATION_OUTPUT_COLUMNS,
    NEWS_ANNOTATION_OUTPUT_SCHEMA_VERSION,
    build_annotation_id,
    build_failure_event_id,
)
from buli_news.news.annotations.tasks import validate_matches, validate_tasks


NEWS_FEATURE_SCHEMA_VERSION = 1
NEWS_FEATURE_DECIMAL_PLACES = 6
NEWS_FEATURE_TIMEZONE = ZoneInfo("Europe/Berlin")
NEWS_FEATURE_SIDES = ("home", "away")
NEWS_FEATURE_STATISTICS = ("mean_rating", "mention_share")
NEWS_FEATURE_METADATA_COLUMNS = (
    "match_id",
    "season",
    "league",
    "matchday",
    "kickoff",
    "home_team_id",
    "home_team",
    "away_team_id",
    "away_team",
    "dataset_split",
)
NEWS_FEATURE_COLUMNS = tuple(
    f"{side}_news_{indicator}_{statistic}"
    for side in NEWS_FEATURE_SIDES
    for indicator in INDICATORS
    for statistic in NEWS_FEATURE_STATISTICS
)
NEWS_FEATURE_OUTPUT_COLUMNS = (
    *NEWS_FEATURE_METADATA_COLUMNS,
    *NEWS_FEATURE_COLUMNS,
)
RESULT_TASK_CONTEXT_FIELDS = (
    "task_id",
    "annotation_schema_id",
    "request_id",
    "content_id",
    "match_id",
    "side",
    "target_team_id",
    "target_team",
    "selected_article_id",
)


@dataclass(frozen=True)
class NewsFeaturesBuild:
    """Match-level news features and their separate quality report."""

    rows: list[dict[str, Any]]
    quality_report: dict[str, Any]


@dataclass(frozen=True)
class AnnotationProvenance:
    """The one inference configuration accepted for feature aggregation."""

    annotation_schema_id: str
    annotation_schema_version: int
    prompt_version: int
    annotation_config_sha256: str
    model: str
    model_digest: str


def build_news_features(
    matches: list[dict[str, Any]],
    tasks: list[dict[str, Any]],
    annotations: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    season: int,
    config: AnnotationConfig,
) -> NewsFeaturesBuild:
    """Build 16 fixed home/away news features for every normalized match."""
    if isinstance(season, bool) or not isinstance(season, int):
        msg = "Season must be an integer start year."
        raise ValueError(msg)

    matches_by_id = validate_matches(matches=matches, season=season)
    sorted_matches = validate_and_sort_matches(matches_by_id)
    validate_tasks(tasks)
    tasks_by_id, tasks_by_context = validate_and_group_tasks(
        tasks=tasks,
        matches_by_id=matches_by_id,
        season=season,
        config=config,
    )
    (
        annotations_by_task,
        selected_annotations,
        provenance,
    ) = validate_and_index_annotations(
        annotations=annotations,
        tasks_by_id=tasks_by_id,
        config=config,
    )
    failures_by_task = validate_and_group_failures(
        failures=failures,
        tasks_by_id=tasks_by_id,
        provenance=provenance,
    )

    rows: list[dict[str, Any]] = []
    context_quality: list[dict[str, Any]] = []
    for match in sorted_matches:
        match_id = match["match_id"]
        row = build_match_metadata(match)
        for side in NEWS_FEATURE_SIDES:
            context_tasks = tasks_by_context[(match_id, side)]
            context_annotations = [
                annotations_by_task[task["task_id"]]
                for task in context_tasks
                if task["task_id"] in annotations_by_task
            ]
            if not context_annotations:
                msg = (
                    f"Match {match_id} {side} news context has no successful "
                    "annotations."
                )
                raise ValueError(msg)
            indicators = summarize_indicators(
                annotations=context_annotations,
                config=config,
            )
            for indicator in INDICATORS:
                summary = indicators[indicator]
                row[f"{side}_news_{indicator}_mean_rating"] = summary[
                    "mean_rating"
                ]
                row[f"{side}_news_{indicator}_mention_share"] = summary[
                    "mention_share"
                ]

            missing_task_ids = [
                task["task_id"]
                for task in context_tasks
                if task["task_id"] not in annotations_by_task
            ]
            expected_count = len(context_tasks)
            successful_count = len(context_annotations)
            context_quality.append(
                {
                    "match_id": match_id,
                    "matchday": match["matchday"],
                    "request_id": context_tasks[0]["request_id"],
                    "side": side,
                    "target_team_id": context_tasks[0]["target_team_id"],
                    "target_team": context_tasks[0]["target_team"],
                    "expected_task_count": expected_count,
                    "successful_annotation_count": successful_count,
                    "missing_annotation_count": len(missing_task_ids),
                    "annotation_coverage": round_ratio(
                        successful_count, expected_count
                    ),
                    "missing_task_ids": missing_task_ids,
                    "indicators": indicators,
                }
            )
        validate_feature_row(row)
        rows.append(row)

    train_rows = [row for row in rows if row["dataset_split"] == "train"]
    test_rows = [row for row in rows if row["dataset_split"] == "test"]
    validate_split_order(train_rows, test_rows)

    missing_tasks = build_missing_task_quality(
        tasks_by_id=tasks_by_id,
        annotations_by_task=annotations_by_task,
        failures_by_task=failures_by_task,
    )
    quality_report = build_quality_report(
        season=season,
        config=config,
        provenance=provenance,
        matches=sorted_matches,
        train_feature_row_count=len(train_rows),
        test_feature_row_count=len(test_rows),
        tasks=tasks,
        annotations=selected_annotations,
        stored_annotation_count=len(annotations),
        failures=failures,
        failures_by_task=failures_by_task,
        context_quality=context_quality,
        missing_tasks=missing_tasks,
    )
    return NewsFeaturesBuild(rows=rows, quality_report=quality_report)


def validate_and_sort_matches(
    matches_by_id: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Validate match metadata required in the feature output."""
    parsed_kickoffs: dict[int, datetime] = {}
    for match_id, match in matches_by_id.items():
        matchday = require_positive_int(
            match.get("matchday"), f"Match {match_id} matchday"
        )
        get_dataset_split(matchday)
        home_team_id = require_int(
            match.get("home_team_id"), f"Match {match_id} home team ID"
        )
        away_team_id = require_int(
            match.get("away_team_id"), f"Match {match_id} away team ID"
        )
        if home_team_id == away_team_id:
            msg = f"Match {match_id} has the same home and away team ID."
            raise ValueError(msg)
        kickoff = require_text(match, "kickoff", f"Match {match_id}")
        try:
            parsed_kickoff = datetime.fromisoformat(kickoff)
        except ValueError as exc:
            msg = f"Match {match_id} has invalid kickoff {kickoff!r}."
            raise ValueError(msg) from exc
        if parsed_kickoff.tzinfo is None:
            msg = f"Match {match_id} kickoff needs a timezone offset."
            raise ValueError(msg)
        parsed_kickoffs[match_id] = parsed_kickoff
    return sorted(
        matches_by_id.values(),
        key=lambda match: (parsed_kickoffs[match["match_id"]], match["match_id"]),
    )


def validate_and_group_tasks(
    tasks: list[dict[str, Any]],
    matches_by_id: dict[int, dict[str, Any]],
    season: int,
    config: AnnotationConfig,
) -> tuple[
    dict[str, dict[str, Any]],
    dict[tuple[int, str], list[dict[str, Any]]],
]:
    """Validate task ownership and group the one request for each match side."""
    tasks_by_id: dict[str, dict[str, Any]] = {}
    tasks_by_context: dict[tuple[int, str], list[dict[str, Any]]] = {}
    request_ids_by_context: dict[tuple[int, str], set[str]] = {}
    context_by_request_id: dict[str, tuple[int, str, int, str]] = {}
    for task in tasks:
        task_id = task["task_id"]
        match_id = task["match_id"]
        side = task["side"]
        match = matches_by_id.get(match_id)
        if match is None:
            msg = f"Annotation task {task_id!r} references unknown match {match_id}."
            raise ValueError(msg)
        if task["season"] != season:
            msg = f"Annotation task {task_id!r} belongs to another season."
            raise ValueError(msg)
        if task["annotation_schema_id"] != config.schema_id:
            msg = (
                f"Annotation task {task_id!r} uses schema "
                f"{task['annotation_schema_id']!r}, expected {config.schema_id!r}."
            )
            raise ValueError(msg)
        expected_context = {
            "league": match["league"],
            "matchday": match["matchday"],
            "kickoff": match["kickoff"],
            "target_team_id": match[f"{side}_team_id"],
            "target_team": match[f"{side}_team"],
            "opponent_team_id": match[
                "away_team_id" if side == "home" else "home_team_id"
            ],
            "opponent_team": match[
                "away_team" if side == "home" else "home_team"
            ],
        }
        for field, expected in expected_context.items():
            if task[field] != expected:
                msg = (
                    f"Annotation task {task_id!r} field {field!r} does not "
                    f"match fixture {match_id}."
                )
                raise ValueError(msg)
        validate_task_pre_match_window(
            task=task,
            match=match,
            task_id=task_id,
        )
        tasks_by_id[task_id] = task
        context_key = (match_id, side)
        tasks_by_context.setdefault(context_key, []).append(task)
        request_ids_by_context.setdefault(context_key, set()).add(
            task["request_id"]
        )
        request_context = (
            match_id,
            side,
            task["target_team_id"],
            task["target_team"],
        )
        existing_request_context = context_by_request_id.setdefault(
            task["request_id"], request_context
        )
        if existing_request_context != request_context:
            msg = (
                f"Request {task['request_id']!r} belongs to conflicting "
                "match-side contexts."
            )
            raise ValueError(msg)

    expected_contexts = {
        (match_id, side)
        for match_id in matches_by_id
        for side in NEWS_FEATURE_SIDES
    }
    if set(tasks_by_context) != expected_contexts:
        missing = sorted(expected_contexts - set(tasks_by_context))
        unexpected = sorted(set(tasks_by_context) - expected_contexts)
        msg = (
            "Annotation tasks do not cover exactly one home and away context "
            f"per match; missing={missing}, unexpected={unexpected}."
        )
        raise ValueError(msg)
    for context_key, request_ids in request_ids_by_context.items():
        if len(request_ids) != 1:
            msg = (
                f"Match-side context {context_key!r} has multiple request IDs: "
                f"{sorted(request_ids)}."
            )
            raise ValueError(msg)
        tasks_by_context[context_key].sort(
            key=lambda task: (task["request_article_index"], task["content_id"])
        )
    return tasks_by_id, tasks_by_context


def validate_task_pre_match_window(
    task: dict[str, Any],
    match: dict[str, Any],
    task_id: str,
) -> None:
    """Ensure every aggregated publication remains inside its pre-match window."""
    if (
        task.get("date_start") != match.get("window_start")
        or task.get("date_end") != match.get("window_end")
    ):
        msg = (
            f"Annotation task {task_id!r} window does not match its normalized "
            "fixture."
        )
        raise ValueError(msg)
    try:
        date_start = date.fromisoformat(task["date_start"])
        date_end = date.fromisoformat(task["date_end"])
        publication = datetime.fromisoformat(task["publication_datetime"])
        kickoff = datetime.fromisoformat(task["kickoff"])
    except (TypeError, ValueError) as exc:
        msg = f"Annotation task {task_id!r} has invalid temporal metadata."
        raise ValueError(msg) from exc
    if publication.tzinfo is None or kickoff.tzinfo is None:
        msg = f"Annotation task {task_id!r} timestamps need timezone offsets."
        raise ValueError(msg)
    if date_start > date_end:
        msg = f"Annotation task {task_id!r} has an inverted news window."
        raise ValueError(msg)
    local_kickoff_date = kickoff.astimezone(NEWS_FEATURE_TIMEZONE).date()
    if date_end >= local_kickoff_date:
        msg = f"Annotation task {task_id!r} news window reaches kickoff or later."
        raise ValueError(msg)
    local_publication_date = publication.astimezone(
        NEWS_FEATURE_TIMEZONE
    ).date()
    if not date_start <= local_publication_date <= date_end:
        msg = f"Annotation task {task_id!r} publication is outside its news window."
        raise ValueError(msg)


def validate_and_index_annotations(
    annotations: list[dict[str, Any]],
    tasks_by_id: dict[str, dict[str, Any]],
    config: AnnotationConfig,
) -> tuple[
    dict[str, dict[str, Any]],
    list[dict[str, Any]],
    AnnotationProvenance,
]:
    """Accept one validated result per task from one exact inference setup."""
    if not annotations:
        msg = "News annotation result input is empty."
        raise ValueError(msg)
    annotation_ids: set[str] = set()
    for annotation in annotations:
        if (
            not isinstance(annotation, dict)
            or tuple(annotation) != NEWS_ANNOTATION_OUTPUT_COLUMNS
            or annotation.get("output_schema_version")
            != NEWS_ANNOTATION_OUTPUT_SCHEMA_VERSION
        ):
            msg = "News annotation result schema or order does not match."
            raise ValueError(msg)
        annotation_id = require_text(
            annotation, "annotation_id", "News annotation result"
        )
        if annotation_id in annotation_ids:
            msg = f"Duplicate news annotation ID {annotation_id!r}."
            raise ValueError(msg)
        annotation_ids.add(annotation_id)

    selected_annotations = [
        annotation
        for annotation in annotations
        if annotation["annotation_config_sha256"] == config.sha256
    ]
    if not selected_annotations:
        msg = (
            "No news annotation results match the selected annotation "
            f"configuration {config.sha256!r}."
        )
        raise ValueError(msg)
    models = {
        require_text(annotation, "model", "News annotation result")
        for annotation in selected_annotations
    }
    model_digests = {
        require_text(annotation, "model_digest", "News annotation result")
        for annotation in selected_annotations
    }
    if len(models) != 1 or len(model_digests) != 1:
        msg = (
            "News features require results from exactly one model and digest; "
            f"models={sorted(models)}, digests={sorted(model_digests)}."
        )
        raise ValueError(msg)
    model = next(iter(models))
    model_digest = next(iter(model_digests))
    provenance = AnnotationProvenance(
        annotation_schema_id=config.schema_id,
        annotation_schema_version=config.schema_version,
        prompt_version=config.prompt_version,
        annotation_config_sha256=config.sha256,
        model=model,
        model_digest=model_digest,
    )

    annotations_by_task: dict[str, dict[str, Any]] = {}
    for annotation in selected_annotations:
        task_id = require_text(annotation, "task_id", "News annotation result")
        task = tasks_by_id.get(task_id)
        if task is None:
            msg = f"News annotation result references unknown task {task_id!r}."
            raise ValueError(msg)
        if task_id in annotations_by_task:
            msg = f"Task {task_id!r} has more than one successful result."
            raise ValueError(msg)
        annotation_id = require_text(
            annotation, "annotation_id", "News annotation result"
        )
        expected_annotation_id = build_annotation_id(
            task_id=task_id,
            config_sha256=config.sha256,
            model_digest=model_digest,
        )
        if annotation_id != expected_annotation_id:
            msg = f"News annotation ID {annotation_id!r} is not deterministic."
            raise ValueError(msg)
        expected_provenance = {
            "annotation_schema_id": config.schema_id,
            "annotation_schema_version": config.schema_version,
            "prompt_version": config.prompt_version,
            "annotation_config_sha256": config.sha256,
            "model": model,
            "model_digest": model_digest,
        }
        for field, expected in expected_provenance.items():
            if annotation[field] != expected:
                msg = (
                    f"News annotation {annotation_id!r} field {field!r} does "
                    "not match the selected inference configuration."
                )
                raise ValueError(msg)
        for field in RESULT_TASK_CONTEXT_FIELDS:
            if annotation[field] != task[field]:
                msg = (
                    f"News annotation {annotation_id!r} field {field!r} does "
                    f"not match task {task_id!r}."
                )
                raise ValueError(msg)
        response = {indicator: annotation[indicator] for indicator in INDICATORS}
        validate_annotation_response(response=response, config=config)
        annotations_by_task[task_id] = annotation
    return annotations_by_task, selected_annotations, provenance


def validate_and_group_failures(
    failures: list[dict[str, Any]],
    tasks_by_id: dict[str, dict[str, Any]],
    provenance: AnnotationProvenance,
) -> dict[str, list[dict[str, Any]]]:
    """Validate failure history and retain events for the selected setup."""
    event_ids: set[str] = set()
    event_keys: set[tuple[str, int]] = set()
    matching_by_task: dict[str, list[dict[str, Any]]] = {}
    for failure in failures:
        if (
            not isinstance(failure, dict)
            or tuple(failure) != NEWS_ANNOTATION_FAILURE_COLUMNS
            or failure.get("failure_schema_version")
            != NEWS_ANNOTATION_FAILURE_SCHEMA_VERSION
        ):
            msg = "News annotation failure schema or order does not match."
            raise ValueError(msg)
        task_id = require_text(failure, "task_id", "News annotation failure")
        annotation_id = require_text(
            failure, "annotation_id", "News annotation failure"
        )
        sequence = require_positive_int(
            failure.get("failure_sequence"), "News annotation failure sequence"
        )
        event_id = require_text(
            failure, "failure_event_id", "News annotation failure"
        )
        expected_event_id = build_failure_event_id(annotation_id, sequence)
        if event_id != expected_event_id:
            msg = f"News annotation failure event {event_id!r} is not deterministic."
            raise ValueError(msg)
        event_key = (annotation_id, sequence)
        if event_id in event_ids or event_key in event_keys:
            msg = f"Duplicate news annotation failure event {event_id!r}."
            raise ValueError(msg)
        event_ids.add(event_id)
        event_keys.add(event_key)

        is_matching = (
            failure["annotation_schema_id"] == provenance.annotation_schema_id
            and failure["annotation_schema_version"]
            == provenance.annotation_schema_version
            and failure["prompt_version"] == provenance.prompt_version
            and failure["annotation_config_sha256"]
            == provenance.annotation_config_sha256
            and failure["model"] == provenance.model
            and failure["model_digest"] == provenance.model_digest
        )
        if not is_matching:
            continue
        task = tasks_by_id.get(task_id)
        if task is None:
            msg = f"News annotation failure references unknown task {task_id!r}."
            raise ValueError(msg)
        for field in RESULT_TASK_CONTEXT_FIELDS:
            if failure[field] != task[field]:
                msg = (
                    f"News annotation failure {event_id!r} field {field!r} "
                    f"does not match task {task_id!r}."
                )
                raise ValueError(msg)
        expected_annotation_id = build_annotation_id(
            task_id=task_id,
            config_sha256=provenance.annotation_config_sha256,
            model_digest=provenance.model_digest,
        )
        if annotation_id != expected_annotation_id:
            msg = f"Failure annotation ID {annotation_id!r} is not deterministic."
            raise ValueError(msg)
        matching_by_task.setdefault(task_id, []).append(failure)
    for task_failures in matching_by_task.values():
        task_failures.sort(key=lambda failure: failure["failure_sequence"])
    return matching_by_task


def summarize_indicators(
    annotations: list[dict[str, Any]],
    config: AnnotationConfig,
) -> dict[str, dict[str, Any]]:
    """Summarize rating direction and mention prevalence per indicator."""
    if not annotations:
        msg = "Cannot summarize an empty annotation group."
        raise ValueError(msg)
    summaries: dict[str, dict[str, Any]] = {}
    for indicator in INDICATORS:
        rating_counts = Counter(
            annotation[indicator]["rating"] for annotation in annotations
        )
        mentioned_ratings = [
            config.rating_mapping[annotation[indicator]["rating"]]
            for annotation in annotations
            if config.rating_mapping[annotation[indicator]["rating"]] is not None
        ]
        mention_count = len(mentioned_ratings)
        summaries[indicator] = {
            "successful_annotation_count": len(annotations),
            "mention_count": mention_count,
            "not_mentioned_count": len(annotations) - mention_count,
            "mention_share": round_ratio(mention_count, len(annotations)),
            "mean_rating": (
                round(
                    sum(mentioned_ratings) / mention_count,
                    NEWS_FEATURE_DECIMAL_PLACES,
                )
                if mention_count
                else 0.0
            ),
            "rating_counts": {
                rating: rating_counts[rating]
                for rating in config.rating_mapping
            },
        }
    return summaries


def round_ratio(numerator: int, denominator: int) -> float:
    """Return one deterministic six-decimal ratio for features and quality."""
    if denominator <= 0:
        msg = "News-feature ratio denominator must be positive."
        raise ValueError(msg)
    return round(numerator / denominator, NEWS_FEATURE_DECIMAL_PLACES)


def build_match_metadata(match: dict[str, Any]) -> dict[str, Any]:
    """Create the exact metadata prefix for one news-feature row."""
    row = {
        "match_id": match["match_id"],
        "season": match["season"],
        "league": match["league"],
        "matchday": match["matchday"],
        "kickoff": match["kickoff"],
        "home_team_id": match["home_team_id"],
        "home_team": match["home_team"],
        "away_team_id": match["away_team_id"],
        "away_team": match["away_team"],
        "dataset_split": get_dataset_split(match["matchday"]),
    }
    if tuple(row) != NEWS_FEATURE_METADATA_COLUMNS:
        msg = "Internal news-feature metadata schema does not match."
        raise ValueError(msg)
    return row


def validate_feature_row(row: dict[str, Any]) -> None:
    """Validate the fixed output schema and finite feature ranges."""
    if tuple(row) != NEWS_FEATURE_OUTPUT_COLUMNS:
        msg = f"News-feature output schema or order does not match: {list(row)}."
        raise ValueError(msg)
    for column in NEWS_FEATURE_COLUMNS:
        value = row[column]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            msg = f"News feature {column!r} must be numeric."
            raise ValueError(msg)
        numeric_value = float(value)
        if not isfinite(numeric_value):
            msg = f"News feature {column!r} must be finite."
            raise ValueError(msg)
        if column.endswith("_mean_rating") and not -2 <= numeric_value <= 2:
            msg = f"News feature {column!r} is outside the rating range."
            raise ValueError(msg)
        if column.endswith("_mention_share") and not 0 <= numeric_value <= 1:
            msg = f"News feature {column!r} is outside the share range."
            raise ValueError(msg)


def build_missing_task_quality(
    tasks_by_id: dict[str, dict[str, Any]],
    annotations_by_task: dict[str, dict[str, Any]],
    failures_by_task: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Describe unresolved tasks without copying article or model-response text."""
    missing_rows = []
    for task_id in sorted(set(tasks_by_id) - set(annotations_by_task)):
        task = tasks_by_id[task_id]
        failure_history = failures_by_task.get(task_id, [])
        missing_rows.append(
            {
                "task_id": task_id,
                "request_id": task["request_id"],
                "content_id": task["content_id"],
                "match_id": task["match_id"],
                "matchday": task["matchday"],
                "side": task["side"],
                "target_team_id": task["target_team_id"],
                "target_team": task["target_team"],
                "matching_failure_event_count": len(failure_history),
                "latest_failure_sequence": (
                    failure_history[-1]["failure_sequence"]
                    if failure_history
                    else None
                ),
            }
        )
    return missing_rows


def build_quality_report(
    season: int,
    config: AnnotationConfig,
    provenance: AnnotationProvenance,
    matches: list[dict[str, Any]],
    train_feature_row_count: int,
    test_feature_row_count: int,
    tasks: list[dict[str, Any]],
    annotations: list[dict[str, Any]],
    stored_annotation_count: int,
    failures: list[dict[str, Any]],
    failures_by_task: dict[str, list[dict[str, Any]]],
    context_quality: list[dict[str, Any]],
    missing_tasks: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build an auditable text-free report separate from model features."""
    matching_failure_task_ids = set(failures_by_task)
    successful_task_ids = {annotation["task_id"] for annotation in annotations}
    context_by_side = {
        side: [context for context in context_quality if context["side"] == side]
        for side in NEWS_FEATURE_SIDES
    }
    indicator_summary = summarize_indicators(
        annotations=annotations,
        config=config,
    )
    for indicator in INDICATORS:
        indicator_summary[indicator]["zero_mention_context_count"] = sum(
            context["indicators"][indicator]["mention_count"] == 0
            for context in context_quality
        )
    incomplete_context_count = sum(
        context["missing_annotation_count"] > 0
        for context in context_quality
    )
    report = {
        "schema_version": NEWS_FEATURE_SCHEMA_VERSION,
        "season": season,
        "annotation_provenance": {
            "annotation_output_schema_version": (
                NEWS_ANNOTATION_OUTPUT_SCHEMA_VERSION
            ),
            "annotation_schema_id": provenance.annotation_schema_id,
            "annotation_schema_version": provenance.annotation_schema_version,
            "prompt_version": provenance.prompt_version,
            "annotation_config_sha256": provenance.annotation_config_sha256,
            "model": provenance.model,
            "model_digest": provenance.model_digest,
        },
        "aggregation": {
            "feature_schema_version": NEWS_FEATURE_SCHEMA_VERSION,
            "grouping_key": ["match_id", "side", "request_id"],
            "input_unit": (
                "one successful annotation per request-bound exact content"
            ),
            "content_weighting": "equal weight for every successful task",
            "decimal_places": NEWS_FEATURE_DECIMAL_PLACES,
            "publication_timezone": NEWS_FEATURE_TIMEZONE.key,
            "indicators": list(INDICATORS),
            "rating_mapping": config.rating_mapping,
            "not_mentioned_handling": (
                "exclude from mean_rating and include in the denominator of "
                "mention_share"
            ),
            "no_mention_values": {
                "mean_rating": 0.0,
                "mention_share": 0.0,
            },
            "missing_annotation_handling": (
                "exclude unresolved tasks from rating and mention denominators; "
                "retain expected and successful counts in this quality report"
            ),
            "feature_columns": list(NEWS_FEATURE_COLUMNS),
            "feature_count": len(NEWS_FEATURE_COLUMNS),
            "quality_fields_in_feature_csv": False,
        },
        "summary": {
            "match_count": len(matches),
            "feature_row_count": len(matches),
            "train_feature_row_count": train_feature_row_count,
            "test_feature_row_count": test_feature_row_count,
            "request_context_count": len(context_quality),
            "expected_annotation_task_count": len(tasks),
            "successful_annotation_count": len(annotations),
            "stored_annotation_count": stored_annotation_count,
            "ignored_other_configuration_annotation_count": (
                stored_annotation_count - len(annotations)
            ),
            "missing_annotation_count": len(tasks) - len(annotations),
            "annotation_coverage": round_ratio(len(annotations), len(tasks)),
            "stored_failure_event_count": len(failures),
            "matching_configuration_failure_event_count": sum(
                len(task_failures) for task_failures in failures_by_task.values()
            ),
            "ignored_other_configuration_failure_event_count": (
                len(failures)
                - sum(
                    len(task_failures)
                    for task_failures in failures_by_task.values()
                )
            ),
            "tasks_with_matching_failure_history": len(
                matching_failure_task_ids
            ),
            "resolved_failure_task_count": len(
                matching_failure_task_ids & successful_task_ids
            ),
            "unresolved_failure_task_count": len(
                matching_failure_task_ids - successful_task_ids
            ),
            "complete_context_count": (
                len(context_quality) - incomplete_context_count
            ),
            "incomplete_context_count": incomplete_context_count,
            "context_without_successful_annotation_count": 0,
        },
        "indicator_summary": indicator_summary,
        "side_summary": {
            side: build_side_quality_summary(
                contexts=context_by_side[side],
                annotations=[
                    annotation
                    for annotation in annotations
                    if annotation["side"] == side
                ],
                config=config,
            )
            for side in NEWS_FEATURE_SIDES
        },
        "missing_tasks": missing_tasks,
        "context_quality": context_quality,
    }
    validate_quality_report(report)
    return report


def build_side_quality_summary(
    contexts: list[dict[str, Any]],
    annotations: list[dict[str, Any]],
    config: AnnotationConfig,
) -> dict[str, Any]:
    """Aggregate coverage and indicator distributions for one team side."""
    expected_count = sum(context["expected_task_count"] for context in contexts)
    successful_count = sum(
        context["successful_annotation_count"] for context in contexts
    )
    return {
        "request_context_count": len(contexts),
        "expected_annotation_task_count": expected_count,
        "successful_annotation_count": successful_count,
        "missing_annotation_count": expected_count - successful_count,
        "annotation_coverage": round_ratio(successful_count, expected_count),
        "indicators": summarize_indicators(
            annotations=annotations,
            config=config,
        ),
    }


def validate_quality_report(report: dict[str, Any]) -> None:
    """Validate core report identities before it is persisted."""
    summary = report["summary"]
    if summary["feature_row_count"] != summary["match_count"]:
        msg = "News-feature quality report has inconsistent match counts."
        raise ValueError(msg)
    if (
        summary["train_feature_row_count"]
        + summary["test_feature_row_count"]
        != summary["feature_row_count"]
    ):
        msg = "News-feature quality report has inconsistent split counts."
        raise ValueError(msg)
    if summary["request_context_count"] != summary["match_count"] * 2:
        msg = "Every news-feature match needs one home and one away context."
        raise ValueError(msg)
    if (
        summary["successful_annotation_count"]
        + summary["missing_annotation_count"]
        != summary["expected_annotation_task_count"]
    ):
        msg = "News-feature quality report has inconsistent annotation counts."
        raise ValueError(msg)
    if (
        summary["successful_annotation_count"]
        + summary["ignored_other_configuration_annotation_count"]
        != summary["stored_annotation_count"]
    ):
        msg = "News-feature quality report has inconsistent stored-result counts."
        raise ValueError(msg)
    if (
        summary["matching_configuration_failure_event_count"]
        + summary["ignored_other_configuration_failure_event_count"]
        != summary["stored_failure_event_count"]
    ):
        msg = "News-feature quality report has inconsistent failure counts."
        raise ValueError(msg)
    if (
        summary["complete_context_count"]
        + summary["incomplete_context_count"]
        != summary["request_context_count"]
    ):
        msg = "News-feature quality report has inconsistent context counts."
        raise ValueError(msg)
    if len(report["missing_tasks"]) != summary["missing_annotation_count"]:
        msg = "News-feature quality report has inconsistent missing-task rows."
        raise ValueError(msg)
    if len(report["aggregation"]["feature_columns"]) != len(
        NEWS_FEATURE_COLUMNS
    ):
        msg = "News-feature quality report has an inconsistent feature schema."
        raise ValueError(msg)
