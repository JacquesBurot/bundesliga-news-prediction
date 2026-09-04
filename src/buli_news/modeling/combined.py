"""Evaluate the Combined model with numerical and news features."""

from __future__ import annotations

import json
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any

import pandas as pd

from buli_news.modeling.evaluation import (
    EXPECTED_TEST_COUNT,
    EXPECTED_TRAIN_COUNT,
    PREDICTION_METADATA_COLUMNS,
    SELECTED_LOGISTIC_C,
    SELECTED_NUMERICAL_EXCLUDED_FEATURE_COLUMNS,
    SELECTED_NUMERICAL_FEATURE_COLUMNS,
    SELECTED_NUMERICAL_FEATURE_SET,
    ClassificationData,
    ModelEvaluationArtifacts,
    evaluate_logistic_classification_data,
    load_numerical_classification_data,
    select_classification_features,
    validate_chronological_order,
    validate_matchday_splits,
    validate_numerical_logistic_selection_report,
)
from buli_news.news.annotations.results import (
    NEWS_ANNOTATION_OUTPUT_SCHEMA_VERSION,
)
from buli_news.news.features import (
    NEWS_FEATURE_COLUMNS,
    NEWS_FEATURE_OUTPUT_COLUMNS,
    NEWS_FEATURE_SCHEMA_VERSION,
)


COMBINED_FEATURE_COLUMNS = (
    *SELECTED_NUMERICAL_FEATURE_COLUMNS,
    *NEWS_FEATURE_COLUMNS,
)
ANNOTATION_PROVENANCE_INTEGER_FIELDS = (
    "annotation_output_schema_version",
    "annotation_schema_version",
    "prompt_version",
)
ANNOTATION_PROVENANCE_TEXT_FIELDS = (
    "annotation_schema_id",
    "annotation_config_sha256",
    "model",
    "model_digest",
)


@dataclass(frozen=True)
class CombinedClassificationBuild:
    """Validated combined data and its one-to-one join audit."""

    data: ClassificationData
    join_validation: dict[str, Any]


def evaluate_combined_logistic_final(
    numerical_features_path: Path,
    news_features_path: Path,
    news_features_quality_path: Path,
    season: int,
    selection_report_path: Path,
) -> ModelEvaluationArtifacts:
    """Evaluate the Combined model with its 31+16 frozen features."""
    validate_numerical_logistic_selection_report(
        selection_report_path=selection_report_path,
        season=season,
    )
    news_provenance = load_news_feature_provenance(
        quality_path=news_features_quality_path,
        season=season,
    )
    combined = load_combined_classification_data(
        numerical_features_path=numerical_features_path,
        news_features_path=news_features_path,
        season=season,
    )
    return evaluate_logistic_classification_data(
        data=combined.data,
        season=season,
        experiment="combined_logistic_final",
        C=SELECTED_LOGISTIC_C,
        input_reference={
            "numerical_features": numerical_features_path,
            "news_features": news_features_path,
            "news_features_quality": news_features_quality_path,
            "numerical_selection_report": selection_report_path,
        },
        selection_provenance={
            "method": "training_only_one_standard_error_with_parsimony",
            "transferred_from": "numerical_logistic_configuration_selection",
            "numerical_feature_set": SELECTED_NUMERICAL_FEATURE_SET,
            "excluded_numerical_feature_columns": list(
                SELECTED_NUMERICAL_EXCLUDED_FEATURE_COLUMNS
            ),
            "selection_report_path": str(selection_report_path),
            "combined_hyperparameters_reselected": False,
            "outer_test_used_for_selection": False,
        },
        report_sections={
            "feature_sources": {
                "numerical": {
                    "feature_count": len(SELECTED_NUMERICAL_FEATURE_COLUMNS),
                    "feature_columns": list(SELECTED_NUMERICAL_FEATURE_COLUMNS),
                },
                "news": {
                    "feature_count": len(NEWS_FEATURE_COLUMNS),
                    "feature_columns": list(NEWS_FEATURE_COLUMNS),
                },
                "combined_feature_count": len(COMBINED_FEATURE_COLUMNS),
                "feature_order": "selected numerical columns, then news columns",
            },
            "join_validation": combined.join_validation,
            "news_feature_provenance": news_provenance,
        },
        convergence_context="Combined model",
    )


def load_combined_classification_data(
    numerical_features_path: Path,
    news_features_path: Path,
    season: int,
) -> CombinedClassificationBuild:
    """Join validated numerical and news features one-to-one by match ID."""
    numerical_data = load_numerical_classification_data(
        features_path=numerical_features_path,
        season=season,
    )
    numerical_data = select_classification_features(
        data=numerical_data,
        feature_columns=SELECTED_NUMERICAL_FEATURE_COLUMNS,
    )
    news_frame, numeric_news_features = load_news_feature_frame(
        features_path=news_features_path,
        season=season,
    )

    numerical_metadata = pd.concat(
        [numerical_data.train_metadata, numerical_data.test_metadata],
        ignore_index=True,
    )
    numerical_match_ids = [
        int(value) for value in numerical_metadata["match_id"]
    ]
    news_match_ids = [int(value) for value in news_frame["match_id"]]
    missing_match_ids = sorted(set(numerical_match_ids) - set(news_match_ids))
    unexpected_match_ids = sorted(set(news_match_ids) - set(numerical_match_ids))
    if missing_match_ids or unexpected_match_ids:
        msg = (
            "Numerical and news feature match IDs do not match one-to-one. "
            f"Missing news IDs: {missing_match_ids}; unexpected news IDs: "
            f"{unexpected_match_ids}."
        )
        raise ValueError(msg)

    news_by_match_id = news_frame.set_index("match_id", drop=False)
    aligned_news = news_by_match_id.loc[numerical_match_ids].copy()
    aligned_news.reset_index(drop=True, inplace=True)
    validate_join_metadata(
        numerical_metadata=numerical_metadata,
        aligned_news=aligned_news,
    )

    numeric_news_by_match_id = numeric_news_features.set_index(
        news_frame["match_id"]
    )
    train_news = numeric_news_by_match_id.loc[
        numerical_data.train_metadata["match_id"].astype(int)
    ].copy()
    test_news = numeric_news_by_match_id.loc[
        numerical_data.test_metadata["match_id"].astype(int)
    ].copy()
    train_news.index = numerical_data.X_train.index
    test_news.index = numerical_data.X_test.index

    X_train = pd.concat([numerical_data.X_train, train_news], axis=1)
    X_test = pd.concat([numerical_data.X_test, test_news], axis=1)
    if tuple(X_train.columns) != COMBINED_FEATURE_COLUMNS:
        msg = "Combined training feature schema or order does not match."
        raise ValueError(msg)
    if tuple(X_test.columns) != COMBINED_FEATURE_COLUMNS:
        msg = "Combined test feature schema or order does not match."
        raise ValueError(msg)

    data = ClassificationData(
        feature_columns=COMBINED_FEATURE_COLUMNS,
        X_train=X_train,
        X_test=X_test,
        y_train=numerical_data.y_train.copy(),
        y_test=numerical_data.y_test.copy(),
        train_metadata=numerical_data.train_metadata.copy(),
        test_metadata=numerical_data.test_metadata.copy(),
    )
    return CombinedClassificationBuild(
        data=data,
        join_validation={
            "key": "match_id",
            "relationship": "one_to_one",
            "match_count": len(numerical_match_ids),
            "metadata_columns": list(PREDICTION_METADATA_COLUMNS),
            "metadata_match_count": len(numerical_match_ids),
            "news_input_row_order_matches_numerical": (
                news_match_ids == numerical_match_ids
            ),
            "output_row_order": "numerical feature table",
            "target_source": "numerical feature table only",
        },
    )


def load_news_feature_frame(
    features_path: Path,
    season: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and strictly validate the fixed match-level news feature table."""
    frame = pd.read_csv(features_path)
    expected_columns = list(NEWS_FEATURE_OUTPUT_COLUMNS)
    actual_columns = list(frame.columns)
    if actual_columns != expected_columns:
        missing = [column for column in expected_columns if column not in frame]
        extra = [column for column in actual_columns if column not in expected_columns]
        msg = (
            "News feature table schema or column order does not match. "
            f"Missing: {missing}; extra: {extra}."
        )
        raise ValueError(msg)

    expected_total_count = EXPECTED_TRAIN_COUNT + EXPECTED_TEST_COUNT
    if len(frame) != expected_total_count:
        msg = (
            f"News feature table must contain {expected_total_count} rows, "
            f"got {len(frame)}."
        )
        raise ValueError(msg)
    if frame["match_id"].isna().any():
        msg = "News feature table contains a missing match ID."
        raise ValueError(msg)
    match_ids = pd.to_numeric(frame["match_id"], errors="raise")
    if any(not float(value).is_integer() for value in match_ids):
        msg = "News feature table match IDs must be whole numbers."
        raise ValueError(msg)
    frame["match_id"] = match_ids.astype(int)
    if frame["match_id"].duplicated().any():
        msg = "News feature table contains duplicate match IDs."
        raise ValueError(msg)
    if not frame["season"].eq(season).all():
        seasons = sorted(int(value) for value in frame["season"].unique())
        msg = (
            f"News feature table must contain only season {season}, "
            f"got {seasons}."
        )
        raise ValueError(msg)

    validate_matchday_splits(frame, context="News feature table")
    validate_chronological_order(frame, context="News feature table")
    numeric_features = frame.loc[:, NEWS_FEATURE_COLUMNS].apply(
        pd.to_numeric,
        errors="raise",
    )
    for column in NEWS_FEATURE_COLUMNS:
        values = [float(value) for value in numeric_features[column]]
        if any(not isfinite(value) for value in values):
            msg = f"News feature {column!r} contains non-finite values."
            raise ValueError(msg)
        if column.endswith("_mean_rating") and any(
            value < -2.0 or value > 2.0 for value in values
        ):
            msg = f"News feature {column!r} is outside the rating range."
            raise ValueError(msg)
        if column.endswith("_mention_share") and any(
            value < 0.0 or value > 1.0 for value in values
        ):
            msg = f"News feature {column!r} is outside the share range."
            raise ValueError(msg)
    return frame, numeric_features


def validate_join_metadata(
    numerical_metadata: pd.DataFrame,
    aligned_news: pd.DataFrame,
) -> None:
    """Require all ten metadata fields to match for every joined match."""
    for column in PREDICTION_METADATA_COLUMNS:
        numerical_values = numerical_metadata[column].tolist()
        news_values = aligned_news[column].tolist()
        mismatch_ids = [
            int(match_id)
            for match_id, numerical_value, news_value in zip(
                numerical_metadata["match_id"],
                numerical_values,
                news_values,
                strict=True,
            )
            if numerical_value != news_value
        ]
        if mismatch_ids:
            msg = (
                f"Numerical and news metadata field {column!r} differs for "
                f"match IDs {mismatch_ids}."
            )
            raise ValueError(msg)


def load_news_feature_provenance(
    quality_path: Path,
    season: int,
) -> dict[str, Any]:
    """Validate and retain text-free provenance for the news feature input."""
    try:
        quality = json.loads(quality_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        msg = (
            f"News feature quality report {quality_path} is not valid JSON: "
            f"{exc.msg} at line {exc.lineno}, column {exc.colno}."
        )
        raise ValueError(msg) from exc
    if not isinstance(quality, dict):
        msg = f"News feature quality report {quality_path} must be an object."
        raise ValueError(msg)
    if quality.get("schema_version") != NEWS_FEATURE_SCHEMA_VERSION:
        msg = "News feature quality report schema version does not match."
        raise ValueError(msg)
    if quality.get("season") != season:
        msg = f"News feature quality report does not belong to season {season}."
        raise ValueError(msg)

    aggregation = quality.get("aggregation")
    summary = quality.get("summary")
    annotation_provenance = quality.get("annotation_provenance")
    if not isinstance(aggregation, dict):
        msg = "News feature quality report has no aggregation object."
        raise ValueError(msg)
    if not isinstance(summary, dict):
        msg = "News feature quality report has no summary object."
        raise ValueError(msg)
    if not isinstance(annotation_provenance, dict):
        msg = "News feature quality report has no annotation provenance."
        raise ValueError(msg)
    retained_annotation_provenance: dict[str, int | str] = {}
    for key in ANNOTATION_PROVENANCE_INTEGER_FIELDS:
        value = annotation_provenance.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            msg = (
                f"News feature annotation provenance field {key!r} "
                "must be a positive integer."
            )
            raise ValueError(msg)
        retained_annotation_provenance[key] = value
    for key in ANNOTATION_PROVENANCE_TEXT_FIELDS:
        value = annotation_provenance.get(key)
        if not isinstance(value, str) or not value.strip():
            msg = (
                f"News feature annotation provenance field {key!r} "
                "must be non-empty text."
            )
            raise ValueError(msg)
        retained_annotation_provenance[key] = value
    if (
        retained_annotation_provenance["annotation_output_schema_version"]
        != NEWS_ANNOTATION_OUTPUT_SCHEMA_VERSION
    ):
        msg = "News annotation output schema version does not match."
        raise ValueError(msg)
    if aggregation.get("feature_columns") != list(NEWS_FEATURE_COLUMNS):
        msg = "News feature quality report feature columns do not match."
        raise ValueError(msg)
    if aggregation.get("feature_count") != len(NEWS_FEATURE_COLUMNS):
        msg = "News feature quality report feature count does not match."
        raise ValueError(msg)
    if aggregation.get("quality_fields_in_feature_csv") is not False:
        msg = "News feature quality fields must remain outside the model CSV."
        raise ValueError(msg)

    expected_summary = {
        "feature_row_count": EXPECTED_TRAIN_COUNT + EXPECTED_TEST_COUNT,
        "train_feature_row_count": EXPECTED_TRAIN_COUNT,
        "test_feature_row_count": EXPECTED_TEST_COUNT,
        "context_without_successful_annotation_count": 0,
    }
    for key, expected in expected_summary.items():
        actual = summary.get(key)
        if (
            isinstance(actual, bool)
            or not isinstance(actual, int)
            or actual != expected
        ):
            msg = (
                f"News feature quality report field {key!r} must be "
                f"{expected!r}, got {actual!r}."
            )
            raise ValueError(msg)

    annotation_coverage = summary.get("annotation_coverage")
    if (
        isinstance(annotation_coverage, bool)
        or not isinstance(annotation_coverage, (int, float))
        or not isfinite(float(annotation_coverage))
        or not 0.0 <= float(annotation_coverage) <= 1.0
    ):
        msg = "News feature quality report annotation coverage is invalid."
        raise ValueError(msg)
    for key in ("missing_annotation_count", "incomplete_context_count"):
        value = summary.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            msg = f"News feature quality report field {key!r} is invalid."
            raise ValueError(msg)

    return {
        "quality_report_path": str(quality_path),
        "feature_schema_version": quality["schema_version"],
        "annotation_provenance": retained_annotation_provenance,
        "annotation_coverage": float(annotation_coverage),
        "missing_annotation_count": summary["missing_annotation_count"],
        "incomplete_context_count": summary["incomplete_context_count"],
        "context_without_successful_annotation_count": summary[
            "context_without_successful_annotation_count"
        ],
    }
