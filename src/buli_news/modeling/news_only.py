"""Evaluate a diagnostic logistic model using only news features."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from buli_news.modeling.combined import (
    load_combined_classification_data,
    load_news_feature_provenance,
)
from buli_news.modeling.evaluation import (
    SELECTED_LOGISTIC_C,
    ClassificationData,
    ModelEvaluationArtifacts,
    evaluate_logistic_classification_data,
    select_classification_features,
    validate_numerical_logistic_selection_report,
)
from buli_news.news.features import NEWS_FEATURE_COLUMNS


@dataclass(frozen=True)
class NewsClassificationBuild:
    """Validated news-only data and its one-to-one alignment audit."""

    data: ClassificationData
    join_validation: dict[str, Any]


def evaluate_news_logistic_diagnostic(
    numerical_features_path: Path,
    news_features_path: Path,
    news_features_quality_path: Path,
    season: int,
    selection_report_path: Path,
) -> ModelEvaluationArtifacts:
    """Evaluate all 16 news features with the frozen logistic settings."""
    validate_numerical_logistic_selection_report(
        selection_report_path=selection_report_path,
        season=season,
    )
    news_provenance = load_news_feature_provenance(
        quality_path=news_features_quality_path,
        season=season,
    )
    news_only = load_news_classification_data(
        numerical_features_path=numerical_features_path,
        news_features_path=news_features_path,
        season=season,
    )
    return evaluate_logistic_classification_data(
        data=news_only.data,
        season=season,
        experiment="news_only_logistic_diagnostic",
        C=SELECTED_LOGISTIC_C,
        input_reference={
            "numerical_target_and_metadata_source": numerical_features_path,
            "news_features": news_features_path,
            "news_features_quality": news_features_quality_path,
            "numerical_selection_report": selection_report_path,
        },
        selection_provenance={
            "method": "fixed_transfer_from_numerical_training_only_selection",
            "transferred_from": (
                "numerical_logistic_configuration_selection"
            ),
            "transferred_parameter": "C",
            "selection_report_path": str(selection_report_path),
            "news_feature_set": "all_fixed_news_features",
            "news_hyperparameters_reselected": False,
            "outer_test_used_for_selection": False,
        },
        report_sections={
            "analysis_role": {
                "type": "post_hoc_diagnostic_feature_source_baseline",
                "primary_experiment_variant": False,
                "purpose": (
                    "Measure standalone predictive signal in the fixed news "
                    "features without changing the primary numerical-versus-"
                    "combined experiment."
                ),
            },
            "feature_sources": {
                "news": {
                    "role": "predictors",
                    "feature_count": len(NEWS_FEATURE_COLUMNS),
                    "feature_columns": list(NEWS_FEATURE_COLUMNS),
                },
                "numerical": {
                    "role": "target_and_metadata_only",
                    "feature_count": 0,
                    "feature_columns": [],
                },
                "model_feature_count": len(NEWS_FEATURE_COLUMNS),
                "numerical_features_used_as_predictors": False,
            },
            "join_validation": news_only.join_validation,
            "news_feature_provenance": news_provenance,
        },
        convergence_context="News-only diagnostic logistic regression",
    )


def load_news_classification_data(
    numerical_features_path: Path,
    news_features_path: Path,
    season: int,
) -> NewsClassificationBuild:
    """Load aligned targets and retain only the 16 news predictors."""
    combined = load_combined_classification_data(
        numerical_features_path=numerical_features_path,
        news_features_path=news_features_path,
        season=season,
    )
    data = select_classification_features(
        data=combined.data,
        feature_columns=NEWS_FEATURE_COLUMNS,
    )
    return NewsClassificationBuild(
        data=data,
        join_validation={
            **combined.join_validation,
            "predictor_source": "news feature table only",
            "numerical_predictor_count": 0,
        },
    )
