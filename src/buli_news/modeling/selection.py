"""Chronological training-only model selection for numerical classifiers."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import datetime
from math import isfinite, sqrt
from pathlib import Path
from statistics import mean, pstdev, stdev
from typing import Any

import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import confusion_matrix

from buli_news.modeling.evaluation import (
    EXPECTED_MATCHES_PER_MATCHDAY,
    METRIC_PROBABILITY_CLASSES,
    PREDICTION_METADATA_COLUMNS,
    SELECTED_LOGISTIC_C,
    SELECTED_NUMERICAL_EXCLUDED_FEATURE_COLUMNS,
    SELECTED_NUMERICAL_FEATURE_COLUMNS,
    SELECTED_NUMERICAL_FEATURE_SET,
    TARGET_CLASSES,
    ClassificationData,
    build_logistic_pipeline,
    calculate_classification_metrics,
    count_classes,
    evaluate_classifier,
    load_numerical_classification_data,
    reorder_probabilities,
    validate_target_classes,
)
from buli_news.matches.features import (
    LAST_MATCHDAY,
    NUMERICAL_FEATURE_COLUMNS,
    TEST_START_MATCHDAY,
    TRAIN_END_MATCHDAY,
)


LOGISTIC_C_CANDIDATES = (0.01, 0.1, 1.0, 10.0)
MODEL_SELECTION_PREDICTION_OUTPUT_COLUMNS = (
    "model_id",
    "model_type",
    "feature_set",
    "C",
    "fold",
    "fit_matchday_start",
    "fit_matchday_end",
    "validation_matchday_start",
    "validation_matchday_end",
    *PREDICTION_METADATA_COLUMNS,
    "actual_result",
    "predicted_result",
    "probability_H",
    "probability_D",
    "probability_A",
)


@dataclass(frozen=True)
class ChronologicalFold:
    """One expanding-window fit and subsequent validation period."""

    name: str
    fit_start_matchday: int
    fit_end_matchday: int
    validation_start_matchday: int
    validation_end_matchday: int


@dataclass(frozen=True)
class FeatureSet:
    """One explicitly defined numerical predictor set."""

    name: str
    columns: tuple[str, ...]
    excluded_columns: tuple[str, ...]


@dataclass(frozen=True)
class ModelSelectionArtifacts:
    """Serializable selection report and fold-level prediction rows."""

    report: dict[str, Any]
    prediction_rows: list[dict[str, Any]]


CHRONOLOGICAL_FOLDS = (
    ChronologicalFold("fold_1", 1, 12, 13, 17),
    ChronologicalFold("fold_2", 1, 17, 18, 22),
    ChronologicalFold("fold_3", 1, 22, 23, 27),
)
FEATURE_SETS = (
    FeatureSet(
        name="full",
        columns=NUMERICAL_FEATURE_COLUMNS,
        excluded_columns=(),
    ),
    FeatureSet(
        name=SELECTED_NUMERICAL_FEATURE_SET,
        columns=SELECTED_NUMERICAL_FEATURE_COLUMNS,
        excluded_columns=SELECTED_NUMERICAL_EXCLUDED_FEATURE_COLUMNS,
    ),
)


def select_numerical_logistic_configuration(
    features_path: Path,
    season: int,
) -> ModelSelectionArtifacts:
    """Select a numerical logistic configuration on matchdays 1-27 only."""
    data = load_numerical_classification_data(
        features_path=features_path,
        season=season,
    )
    validate_model_selection_design(data)

    dummy_report, dummy_rows = evaluate_model_candidate(
        data=data,
        model_type="dummy_prior",
        feature_set=FEATURE_SETS[0],
        C=None,
    )

    candidate_reports = []
    prediction_rows = list(dummy_rows)
    for feature_set in FEATURE_SETS:
        for C in LOGISTIC_C_CANDIDATES:
            candidate_report, candidate_rows = evaluate_model_candidate(
                data=data,
                model_type="logistic_regression",
                feature_set=feature_set,
                C=C,
            )
            candidate_reports.append(candidate_report)
            prediction_rows.extend(candidate_rows)

    log_loss_ranked_candidates = sorted(
        candidate_reports,
        key=lambda candidate: (
            candidate["pooled_metrics"]["log_loss"],
            candidate["pooled_metrics"]["multiclass_brier_score"],
            candidate["feature_count"],
            candidate["C"],
        ),
    )
    for rank, candidate in enumerate(log_loss_ranked_candidates, start=1):
        candidate["strict_log_loss_rank"] = rank

    best_log_loss_candidate = log_loss_ranked_candidates[0]
    best_log_loss_standard_error = best_log_loss_candidate[
        "fold_metric_summary"
    ]["log_loss"]["standard_error"]
    eligible_log_loss_threshold = (
        best_log_loss_candidate["pooled_metrics"]["log_loss"]
        + best_log_loss_standard_error
    )
    eligible_candidates = [
        candidate
        for candidate in candidate_reports
        if candidate["pooled_metrics"]["log_loss"]
        <= eligible_log_loss_threshold
    ]
    for candidate in candidate_reports:
        candidate["within_one_standard_error"] = (
            candidate in eligible_candidates
        )

    eligible_ranked_candidates = sorted(
        eligible_candidates,
        key=lambda candidate: (
            candidate["feature_count"],
            candidate["pooled_metrics"]["multiclass_brier_score"],
            candidate["C"],
            candidate["pooled_metrics"]["log_loss"],
        ),
    )
    ineligible_ranked_candidates = [
        candidate
        for candidate in log_loss_ranked_candidates
        if candidate not in eligible_candidates
    ]
    ranked_candidates = (
        eligible_ranked_candidates + ineligible_ranked_candidates
    )
    for rank, candidate in enumerate(ranked_candidates, start=1):
        candidate["selection_rank"] = rank

    selected = ranked_candidates[0]
    validate_frozen_selected_configuration(selected)
    validate_model_selection_predictions(
        prediction_rows=prediction_rows,
        data=data,
        expected_model_ids={
            dummy_report["model_id"],
            *(candidate["model_id"] for candidate in candidate_reports),
        },
    )

    validation_match_ids = sorted(
        {
            int(row["match_id"])
            for row in prediction_rows
            if row["model_id"] == dummy_report["model_id"]
        }
    )
    report = {
        "experiment": "numerical_logistic_configuration_selection",
        "season": season,
        "input_path": str(features_path),
        "scikit_learn_version": sklearn.__version__,
        "target_classes": list(TARGET_CLASSES),
        "selection_scope": {
            "type": "expanding_window_training_only",
            "outer_training_matchdays": [1, TRAIN_END_MATCHDAY],
            "outer_test_matchdays": [TEST_START_MATCHDAY, LAST_MATCHDAY],
            "outer_training_row_count": len(data.y_train),
            "outer_test_rows_used_for_fitting": 0,
            "outer_test_rows_used_for_scaling": 0,
            "outer_test_rows_used_for_scoring": 0,
            "validation_row_count_per_candidate": len(validation_match_ids),
            "validation_match_ids": validation_match_ids,
            "refit_after_selection": False,
            "outer_test_evaluation_performed": False,
        },
        "folds": [fold_definition(fold, data) for fold in CHRONOLOGICAL_FOLDS],
        "search_space": {
            "feature_sets": [feature_set_definition(item) for item in FEATURE_SETS],
            "C_values": list(LOGISTIC_C_CANDIDATES),
            "logistic_candidate_count": len(candidate_reports),
            "fits_per_logistic_candidate": len(CHRONOLOGICAL_FOLDS),
            "total_logistic_fit_count": (
                len(candidate_reports) * len(CHRONOLOGICAL_FOLDS)
            ),
            "dummy_reference_fit_count": len(CHRONOLOGICAL_FOLDS),
        },
        "selection_rule": {
            "name": "one_standard_error_with_parsimony",
            "primary_metric": "pooled log_loss",
            "best_log_loss_model_id": best_log_loss_candidate["model_id"],
            "minimum_pooled_log_loss": best_log_loss_candidate[
                "pooled_metrics"
            ]["log_loss"],
            "standard_error_source": (
                "Sample standard deviation of the best-log-loss candidate's "
                "three fold Log Loss values divided by sqrt(3)."
            ),
            "best_log_loss_standard_error": best_log_loss_standard_error,
            "eligible_log_loss_threshold_inclusive": (
                eligible_log_loss_threshold
            ),
            "eligible_model_ids": [
                candidate["model_id"]
                for candidate in eligible_ranked_candidates
            ],
            "preference_order_within_threshold": [
                "lowest feature_count",
                "lowest pooled multiclass_brier_score",
                "lowest C",
                "lowest pooled log_loss",
            ],
            "dummy_affects_selection": False,
        },
        "dummy_reference": dummy_report,
        "candidates_ranked": ranked_candidates,
        "selected_configuration": {
            "model_id": selected["model_id"],
            "feature_set": selected["feature_set"],
            "feature_columns": selected["feature_columns"],
            "excluded_columns": selected["excluded_columns"],
            "feature_count": selected["feature_count"],
            "C": selected["C"],
            "pooled_metrics": selected["pooled_metrics"],
            "selection_rank": selected["selection_rank"],
            "strict_log_loss_rank": selected["strict_log_loss_rank"],
            "within_one_standard_error": selected[
                "within_one_standard_error"
            ],
        },
        "frozen_evaluation_configuration": {
            "feature_set": SELECTED_NUMERICAL_FEATURE_SET,
            "feature_columns": list(SELECTED_NUMERICAL_FEATURE_COLUMNS),
            "excluded_columns": list(
                SELECTED_NUMERICAL_EXCLUDED_FEATURE_COLUMNS
            ),
            "C": SELECTED_LOGISTIC_C,
            "matches_selected_configuration": True,
        },
        "metric_definitions": {
            "pooled_metrics": (
                "Calculated once across all 135 disjoint validation predictions "
                "so every validation match has equal weight."
            ),
            "multiclass_brier_score": (
                "Mean over validation matches of the sum across H, D, and A of "
                "(observed_one_hot - predicted_probability) squared; "
                "unscaled range 0 to 2, lower is better."
            ),
            "fold_metric_population_standard_deviation": (
                "Population standard deviation across the three fold metrics."
            ),
            "fold_metric_sample_standard_deviation": (
                "Sample standard deviation across the three fold metrics."
            ),
            "fold_metric_standard_error": (
                "Sample standard deviation across the three fold metrics "
                "divided by the square root of three."
            ),
        },
    }
    return ModelSelectionArtifacts(
        report=report,
        prediction_rows=prediction_rows,
    )


def evaluate_model_candidate(
    data: ClassificationData,
    model_type: str,
    feature_set: FeatureSet,
    C: float | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate one model configuration over all chronological folds."""
    if model_type == "logistic_regression":
        if C is None or not isfinite(C) or C <= 0.0:
            msg = f"Logistic model candidate requires a positive finite C, got {C}."
            raise ValueError(msg)
        model_id = f"logistic__{feature_set.name}__C_{C:g}"
    elif model_type == "dummy_prior":
        if C is not None:
            msg = "Dummy model candidate must not define C."
            raise ValueError(msg)
        model_id = "dummy__prior"
    else:
        msg = f"Unknown model-selection candidate type: {model_type}."
        raise ValueError(msg)

    fold_reports = []
    prediction_rows = []
    pooled_targets: list[str] = []
    pooled_predictions: list[str] = []
    pooled_metric_probabilities: list[list[float]] = []

    for fold in CHRONOLOGICAL_FOLDS:
        fold_details = fold_definition(fold=fold, data=data)
        fold_data = build_fold_data(
            data=data,
            fold=fold,
            feature_columns=feature_set.columns,
        )
        if model_type == "logistic_regression":
            classifier = build_logistic_pipeline(C=C)
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("error", category=ConvergenceWarning)
                    evaluation = evaluate_classifier(
                        classifier=classifier,
                        data=fold_data,
                    )
            except ConvergenceWarning as exc:
                msg = (
                    f"Logistic candidate {model_id} did not converge in "
                    f"{fold.name}."
                )
                raise ValueError(msg) from exc

            scaler = classifier.named_steps["scaler"]
            logistic = classifier.named_steps["classifier"]
            scaler_sample_count = int(scaler.n_samples_seen_)
            if scaler_sample_count != len(fold_data.X_train):
                msg = (
                    f"Scaler for {model_id} in {fold.name} saw "
                    f"{scaler_sample_count} rows, expected "
                    f"{len(fold_data.X_train)}."
                )
                raise ValueError(msg)
            fit_details = {
                "scaler_training_sample_count": scaler_sample_count,
                "solver_iterations": [int(value) for value in logistic.n_iter_],
            }
        else:
            classifier = DummyClassifier(strategy="prior")
            evaluation = evaluate_classifier(
                classifier=classifier,
                data=fold_data,
            )
            fit_details = {
                "training_class_prior": {
                    target_class: float(probability)
                    for target_class, probability in zip(
                        TARGET_CLASSES,
                        evaluation.probabilities[0],
                        strict=True,
                    )
                }
            }

        metric_probabilities = reorder_probabilities(
            probabilities=evaluation.probabilities,
            source_classes=TARGET_CLASSES,
            target_classes=METRIC_PROBABILITY_CLASSES,
        )
        pooled_targets.extend(str(value) for value in fold_data.y_test)
        pooled_predictions.extend(evaluation.predictions)
        pooled_metric_probabilities.extend(
            [float(value) for value in row]
            for row in metric_probabilities
        )

        fold_reports.append(
            {
                "fold": fold.name,
                "nominal_fit_matchdays": [
                    fold.fit_start_matchday,
                    fold.fit_end_matchday,
                ],
                "validation_matchdays": [
                    fold.validation_start_matchday,
                    fold.validation_end_matchday,
                ],
                "fit_cutoff_exclusive": fold_details["fit_cutoff_exclusive"],
                "fit_row_count": len(fold_data.y_train),
                "validation_row_count": len(fold_data.y_test),
                "excluded_late_nominal_fit_row_count": fold_details[
                    "excluded_late_nominal_fit_row_count"
                ],
                "excluded_late_nominal_fit_match_ids": fold_details[
                    "excluded_late_nominal_fit_match_ids"
                ],
                "fit_class_counts": count_classes(fold_data.y_train),
                "validation_class_counts": count_classes(fold_data.y_test),
                "metrics": evaluation.metrics,
                "confusion_matrix": {
                    "labels": list(TARGET_CLASSES),
                    "orientation": "rows=true classes, columns=predicted classes",
                    "values": evaluation.confusion_values,
                },
                **fit_details,
            }
        )
        prediction_rows.extend(
            build_selection_prediction_rows(
                base_rows=evaluation.prediction_rows,
                model_id=model_id,
                model_type=model_type,
                feature_set=feature_set.name,
                C=C,
                fold=fold,
            )
        )

    pooled_metrics = calculate_classification_metrics(
        targets=pd.Series(pooled_targets, dtype="string"),
        predictions=pooled_predictions,
        metric_probabilities=pooled_metric_probabilities,
    )
    pooled_confusion = confusion_matrix(
        pooled_targets,
        pooled_predictions,
        labels=list(TARGET_CLASSES),
    )
    metric_summary = {
        metric_name: {
            "mean": float(
                mean(
                    report["metrics"][metric_name]
                    for report in fold_reports
                )
            ),
            "population_standard_deviation": float(
                pstdev(report["metrics"][metric_name] for report in fold_reports)
            ),
            "sample_standard_deviation": float(
                stdev(report["metrics"][metric_name] for report in fold_reports)
            ),
            "standard_error": float(
                stdev(
                    report["metrics"][metric_name]
                    for report in fold_reports
                )
                / sqrt(len(fold_reports))
            ),
        }
        for metric_name in pooled_metrics
    }
    return (
        {
            "model_id": model_id,
            "model_type": model_type,
            "feature_set": feature_set.name,
            "feature_columns": list(feature_set.columns),
            "excluded_columns": list(feature_set.excluded_columns),
            "feature_count": len(feature_set.columns),
            "C": C,
            "uses_feature_values": model_type == "logistic_regression",
            "folds": fold_reports,
            "fold_metric_summary": metric_summary,
            "pooled_metrics": pooled_metrics,
            "pooled_confusion_matrix": {
                "labels": list(TARGET_CLASSES),
                "orientation": "rows=true classes, columns=predicted classes",
                "values": pooled_confusion.tolist(),
            },
        },
        prediction_rows,
    )


def build_fold_data(
    data: ClassificationData,
    fold: ChronologicalFold,
    feature_columns: tuple[str, ...],
) -> ClassificationData:
    """Create one fit/validation view from the fixed outer training split."""
    fit_mask, validation_mask, _, _ = build_fold_masks(
        data=data,
        fold=fold,
    )
    fold_data = ClassificationData(
        feature_columns=feature_columns,
        X_train=data.X_train.loc[
            fit_mask,
            list(feature_columns),
        ].reset_index(drop=True),
        X_test=data.X_train.loc[
            validation_mask,
            list(feature_columns),
        ].reset_index(drop=True),
        y_train=data.y_train.loc[fit_mask].reset_index(drop=True),
        y_test=data.y_train.loc[validation_mask].reset_index(drop=True),
        train_metadata=data.train_metadata.loc[fit_mask].reset_index(drop=True),
        test_metadata=data.train_metadata.loc[validation_mask].reset_index(drop=True),
    )
    validate_fold_data(fold=fold, data=fold_data)
    return fold_data


def validate_model_selection_design(data: ClassificationData) -> None:
    """Validate the fixed search space and the outer-training fold design."""
    if len(set(LOGISTIC_C_CANDIDATES)) != len(LOGISTIC_C_CANDIDATES):
        msg = "Logistic C candidates must be unique."
        raise ValueError(msg)
    if any(not isfinite(C) or C <= 0.0 for C in LOGISTIC_C_CANDIDATES):
        msg = (
            "Logistic C candidates must be positive and finite: "
            f"{LOGISTIC_C_CANDIDATES}."
        )
        raise ValueError(msg)

    numerical_columns = set(NUMERICAL_FEATURE_COLUMNS)
    if not set(SELECTED_NUMERICAL_EXCLUDED_FEATURE_COLUMNS).issubset(
        numerical_columns
    ):
        msg = "Match-count exclusions must all belong to the numerical feature schema."
        raise ValueError(msg)
    if len({feature_set.name for feature_set in FEATURE_SETS}) != len(FEATURE_SETS):
        msg = "Model-selection feature-set names must be unique."
        raise ValueError(msg)
    for feature_set in FEATURE_SETS:
        if not feature_set.columns:
            msg = f"Feature set {feature_set.name!r} must not be empty."
            raise ValueError(msg)
        if len(set(feature_set.columns)) != len(feature_set.columns):
            msg = f"Feature set {feature_set.name!r} contains duplicate columns."
            raise ValueError(msg)
        if not set(feature_set.columns).issubset(numerical_columns):
            msg = f"Feature set {feature_set.name!r} contains unknown columns."
            raise ValueError(msg)

    training_matchdays = {
        int(value)
        for value in pd.to_numeric(
            data.train_metadata["matchday"],
            errors="raise",
        )
    }
    expected_matchdays = set(range(1, TRAIN_END_MATCHDAY + 1))
    if training_matchdays != expected_matchdays:
        msg = (
            "Outer training metadata must contain matchdays 1-27 exactly, "
            f"got {sorted(training_matchdays)}."
        )
        raise ValueError(msg)

    validation_ids: set[int] = set()
    for fold in CHRONOLOGICAL_FOLDS:
        if fold.fit_end_matchday >= fold.validation_start_matchday:
            msg = f"Training must end before validation in {fold.name}."
            raise ValueError(msg)
        fold_data = build_fold_data(
            data=data,
            fold=fold,
            feature_columns=NUMERICAL_FEATURE_COLUMNS,
        )
        fold_ids = {
            int(value) for value in fold_data.test_metadata["match_id"]
        }
        overlap = validation_ids.intersection(fold_ids)
        if overlap:
            msg = (
                "Validation matches must be disjoint across folds; repeated "
                f"match IDs: {sorted(overlap)}."
            )
            raise ValueError(msg)
        validation_ids.update(fold_ids)

    expected_validation_count = sum(
        (
            fold.validation_end_matchday
            - fold.validation_start_matchday
            + 1
        )
        * EXPECTED_MATCHES_PER_MATCHDAY
        for fold in CHRONOLOGICAL_FOLDS
    )
    if len(validation_ids) != expected_validation_count:
        msg = (
            f"Expected {expected_validation_count} unique validation matches, "
            f"got {len(validation_ids)}."
        )
        raise ValueError(msg)


def validate_fold_data(
    fold: ChronologicalFold,
    data: ClassificationData,
) -> None:
    """Validate counts, classes, split labels, and chronology for one fold."""
    maximum_fit_count = (
        fold.fit_end_matchday - fold.fit_start_matchday + 1
    ) * EXPECTED_MATCHES_PER_MATCHDAY
    expected_validation_count = (
        fold.validation_end_matchday - fold.validation_start_matchday + 1
    ) * EXPECTED_MATCHES_PER_MATCHDAY
    if not 0 < len(data.y_train) <= maximum_fit_count:
        msg = (
            f"{fold.name} must contain between 1 and {maximum_fit_count} "
            "chronologically eligible fit rows, "
            f"got {len(data.y_train)}."
        )
        raise ValueError(msg)
    if len(data.y_test) != expected_validation_count:
        msg = (
            f"{fold.name} must contain {expected_validation_count} validation rows, "
            f"got {len(data.y_test)}."
        )
        raise ValueError(msg)
    validate_target_classes(data.y_train, context=f"{fold.name} fit split")

    fit_ids = {int(value) for value in data.train_metadata["match_id"]}
    validation_ids = {int(value) for value in data.test_metadata["match_id"]}
    overlap = fit_ids.intersection(validation_ids)
    if overlap:
        msg = f"{fold.name} fit and validation match IDs overlap: {sorted(overlap)}."
        raise ValueError(msg)
    if not data.train_metadata["dataset_split"].eq("train").all():
        msg = f"{fold.name} fit rows must come only from the outer training split."
        raise ValueError(msg)
    if not data.test_metadata["dataset_split"].eq("train").all():
        msg = (
            f"{fold.name} validation rows must come only from the outer "
            "training split."
        )
        raise ValueError(msg)

    fit_kickoffs = [
        datetime.fromisoformat(str(value))
        for value in data.train_metadata["kickoff"]
    ]
    validation_kickoffs = [
        datetime.fromisoformat(str(value))
        for value in data.test_metadata["kickoff"]
    ]
    if max(fit_kickoffs) >= min(validation_kickoffs):
        msg = f"{fold.name} is not strictly chronological by kickoff timestamp."
        raise ValueError(msg)


def build_selection_prediction_rows(
    base_rows: list[dict[str, Any]],
    model_id: str,
    model_type: str,
    feature_set: str,
    C: float | None,
    fold: ChronologicalFold,
) -> list[dict[str, Any]]:
    """Attach model-selection audit fields to shared prediction rows."""
    return [
        {
            "model_id": model_id,
            "model_type": model_type,
            "feature_set": feature_set,
            "C": C,
            "fold": fold.name,
            "fit_matchday_start": fold.fit_start_matchday,
            "fit_matchday_end": fold.fit_end_matchday,
            "validation_matchday_start": fold.validation_start_matchday,
            "validation_matchday_end": fold.validation_end_matchday,
            **row,
        }
        for row in base_rows
    ]


def validate_model_selection_predictions(
    prediction_rows: list[dict[str, Any]],
    data: ClassificationData,
    expected_model_ids: set[str],
) -> None:
    """Ensure every candidate has the same training-only validation matches."""
    actual_model_ids = {str(row["model_id"]) for row in prediction_rows}
    if actual_model_ids != expected_model_ids:
        msg = (
            "Model-selection prediction model IDs do not match the search space: "
            f"expected {sorted(expected_model_ids)}, got {sorted(actual_model_ids)}."
        )
        raise ValueError(msg)
    if any(row["dataset_split"] != "train" for row in prediction_rows):
        msg = "Model-selection predictions must contain outer-training rows only."
        raise ValueError(msg)

    test_match_ids = {int(value) for value in data.test_metadata["match_id"]}
    reference_ids: set[int] | None = None
    expected_count = sum(
        (
            fold.validation_end_matchday
            - fold.validation_start_matchday
            + 1
        )
        * EXPECTED_MATCHES_PER_MATCHDAY
        for fold in CHRONOLOGICAL_FOLDS
    )
    for model_id in expected_model_ids:
        model_rows = [
            row for row in prediction_rows if row["model_id"] == model_id
        ]
        model_ids = [int(row["match_id"]) for row in model_rows]
        if len(model_ids) != expected_count or len(set(model_ids)) != expected_count:
            msg = (
                f"{model_id} must contain {expected_count} unique validation "
                f"matches, got {len(model_ids)} rows and {len(set(model_ids))} IDs."
            )
            raise ValueError(msg)
        if test_match_ids.intersection(model_ids):
            msg = f"{model_id} contains outer-test match IDs."
            raise ValueError(msg)
        if reference_ids is None:
            reference_ids = set(model_ids)
        elif set(model_ids) != reference_ids:
            msg = (
                "All model-selection candidates must use identical "
                "validation matches."
            )
            raise ValueError(msg)


def fold_definition(
    fold: ChronologicalFold,
    data: ClassificationData,
) -> dict[str, Any]:
    """Serialize one validated fold definition."""
    fit_mask, validation_mask, excluded_mask, cutoff = build_fold_masks(
        data=data,
        fold=fold,
    )
    return {
        "fold": fold.name,
        "nominal_fit_matchdays": [
            fold.fit_start_matchday,
            fold.fit_end_matchday,
        ],
        "validation_matchdays": [
            fold.validation_start_matchday,
            fold.validation_end_matchday,
        ],
        "fit_cutoff_exclusive": cutoff.isoformat(),
        "fit_row_count": int(fit_mask.sum()),
        "validation_row_count": int(validation_mask.sum()),
        "excluded_late_nominal_fit_row_count": int(excluded_mask.sum()),
        "excluded_late_nominal_fit_match_ids": [
            int(value)
            for value in data.train_metadata.loc[excluded_mask, "match_id"]
        ],
    }


def build_fold_masks(
    data: ClassificationData,
    fold: ChronologicalFold,
) -> tuple[pd.Series, pd.Series, pd.Series, datetime]:
    """Build matchday windows with a strict kickoff cutoff for fit rows."""
    matchdays = pd.to_numeric(data.train_metadata["matchday"], errors="raise")
    nominal_fit_mask = matchdays.between(
        fold.fit_start_matchday,
        fold.fit_end_matchday,
    )
    validation_mask = matchdays.between(
        fold.validation_start_matchday,
        fold.validation_end_matchday,
    )
    kickoffs = pd.Series(
        [
            datetime.fromisoformat(str(value))
            for value in data.train_metadata["kickoff"]
        ],
        index=data.train_metadata.index,
    )
    validation_cutoff = min(kickoffs.loc[validation_mask])
    chronological_fit_mask = kickoffs.lt(validation_cutoff)
    fit_mask = nominal_fit_mask & chronological_fit_mask
    excluded_mask = nominal_fit_mask & ~chronological_fit_mask
    return fit_mask, validation_mask, excluded_mask, validation_cutoff


def feature_set_definition(feature_set: FeatureSet) -> dict[str, Any]:
    """Serialize one fixed feature-set definition."""
    return {
        "name": feature_set.name,
        "feature_count": len(feature_set.columns),
        "feature_columns": list(feature_set.columns),
        "excluded_columns": list(feature_set.excluded_columns),
    }


def validate_frozen_selected_configuration(
    selected: dict[str, Any],
) -> None:
    """Require model selection to reproduce the frozen final configuration."""
    expected = {
        "feature_set": SELECTED_NUMERICAL_FEATURE_SET,
        "feature_columns": list(SELECTED_NUMERICAL_FEATURE_COLUMNS),
        "excluded_columns": list(
            SELECTED_NUMERICAL_EXCLUDED_FEATURE_COLUMNS
        ),
        "feature_count": len(SELECTED_NUMERICAL_FEATURE_COLUMNS),
        "C": SELECTED_LOGISTIC_C,
    }
    actual = {
        key: selected[key]
        for key in expected
    }
    if actual != expected:
        msg = (
            "Training-only model selection no longer matches the frozen "
            f"evaluation configuration. Expected {expected}, got {actual}."
        )
        raise ValueError(msg)
