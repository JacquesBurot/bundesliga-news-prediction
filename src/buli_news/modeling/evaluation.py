"""Load model-ready features and evaluate reproducible classification baselines."""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from datetime import datetime
from math import isclose, isfinite
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from buli_news.matches.features import (
    LAST_MATCHDAY,
    NUMERICAL_FEATURE_COLUMNS,
    NUMERICAL_FEATURE_OUTPUT_COLUMNS,
    TEST_START_MATCHDAY,
    TRAIN_END_MATCHDAY,
)


TARGET_CLASSES = ("H", "D", "A")
METRIC_PROBABILITY_CLASSES = tuple(sorted(TARGET_CLASSES))
EXPECTED_MATCHES_PER_MATCHDAY = 9
EXPECTED_TRAIN_COUNT = TRAIN_END_MATCHDAY * EXPECTED_MATCHES_PER_MATCHDAY
EXPECTED_TEST_COUNT = (
    LAST_MATCHDAY - TEST_START_MATCHDAY + 1
) * EXPECTED_MATCHES_PER_MATCHDAY
PREDICTION_METADATA_COLUMNS = (
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
CLASSIFICATION_PREDICTION_OUTPUT_COLUMNS = (
    *PREDICTION_METADATA_COLUMNS,
    "actual_result",
    "predicted_result",
    "probability_H",
    "probability_D",
    "probability_A",
)
LOGISTIC_SOLVER = "lbfgs"
ORIGINAL_BASELINE_LOGISTIC_C = 1.0
SELECTED_LOGISTIC_C = 0.01
LOGISTIC_L1_RATIO = 0.0
LOGISTIC_MAX_ITER = 1000
LOGISTIC_TOLERANCE = 1e-4
SELECTED_NUMERICAL_FEATURE_SET = "without_match_counts"
SELECTED_NUMERICAL_EXCLUDED_FEATURE_COLUMNS = (
    "home_matches_played",
    "away_matches_played",
    "home_venue_matches_played",
    "away_venue_matches_played",
)
SELECTED_NUMERICAL_FEATURE_COLUMNS = tuple(
    column
    for column in NUMERICAL_FEATURE_COLUMNS
    if column not in SELECTED_NUMERICAL_EXCLUDED_FEATURE_COLUMNS
)


@dataclass(frozen=True)
class ClassificationData:
    """Validated chronological train and test data for one feature set."""

    feature_columns: tuple[str, ...]
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    train_metadata: pd.DataFrame
    test_metadata: pd.DataFrame


@dataclass(frozen=True)
class ClassifierEvaluation:
    """Shared predictions, metrics, and audit rows for one fitted classifier."""

    predictions: tuple[str, ...]
    probabilities: Any
    metrics: dict[str, float]
    confusion_values: list[list[int]]
    prediction_rows: list[dict[str, Any]]


@dataclass(frozen=True)
class ModelEvaluationArtifacts:
    """Serializable report and per-match predictions for one model run."""

    report: dict[str, Any]
    prediction_rows: list[dict[str, Any]]


def evaluate_numerical_dummy(
    features_path: Path,
    season: int,
) -> ModelEvaluationArtifacts:
    """Fit and evaluate the prior-based ZeroR baseline on the fixed split."""
    data = load_numerical_classification_data(
        features_path=features_path,
        season=season,
    )
    classifier = DummyClassifier(strategy="prior")
    evaluation = evaluate_classifier(classifier=classifier, data=data)

    predicted_classes = set(evaluation.predictions)
    if len(predicted_classes) != 1:
        msg = (
            "DummyClassifier(strategy='prior') must predict one majority class, "
            f"got {sorted(predicted_classes)}."
        )
        raise ValueError(msg)

    class_prior = {
        target_class: float(probability)
        for target_class, probability in zip(
            TARGET_CLASSES,
            evaluation.probabilities[0],
            strict=True,
        )
    }
    report = build_classification_report(
        experiment="numerical_dummy_baseline",
        season=season,
        input_reference=features_path,
        model={
            "estimator": "sklearn.dummy.DummyClassifier",
            "scikit_learn_version": sklearn.__version__,
            "strategy": "prior",
            "uses_feature_values": False,
            "predicted_class": next(iter(predicted_classes)),
            "training_class_prior": class_prior,
        },
        data=data,
        evaluation=evaluation,
    )
    return ModelEvaluationArtifacts(
        report=report,
        prediction_rows=evaluation.prediction_rows,
    )


def evaluate_numerical_logistic_reference(
    features_path: Path,
    season: int,
) -> ModelEvaluationArtifacts:
    """Evaluate the fixed Full-feature numerical reference with C=1."""
    return evaluate_numerical_logistic_configuration(
        features_path=features_path,
        season=season,
        experiment="numerical_logistic_reference",
        feature_columns=NUMERICAL_FEATURE_COLUMNS,
        C=ORIGINAL_BASELINE_LOGISTIC_C,
    )


def evaluate_numerical_logistic_final(
    features_path: Path,
    season: int,
    selection_report_path: Path,
) -> ModelEvaluationArtifacts:
    """Evaluate the frozen Selected numerical model."""
    validate_numerical_logistic_selection_report(
        selection_report_path=selection_report_path,
        season=season,
    )
    return evaluate_numerical_logistic_configuration(
        features_path=features_path,
        season=season,
        experiment="numerical_logistic_final",
        feature_columns=SELECTED_NUMERICAL_FEATURE_COLUMNS,
        C=SELECTED_LOGISTIC_C,
        selection_provenance={
            "method": "training_only_one_standard_error_with_parsimony",
            "feature_set": SELECTED_NUMERICAL_FEATURE_SET,
            "excluded_feature_columns": list(
                SELECTED_NUMERICAL_EXCLUDED_FEATURE_COLUMNS
            ),
            "selection_report_path": str(selection_report_path),
            "outer_test_used_for_selection": False,
        },
    )


def validate_numerical_logistic_selection_report(
    selection_report_path: Path,
    season: int,
) -> None:
    """Validate the training-only report behind the frozen final model."""
    try:
        selection_report = json.loads(
            selection_report_path.read_text(encoding="utf-8")
        )
    except json.JSONDecodeError as exc:
        msg = (
            f"Numerical model-selection report {selection_report_path} is not "
            f"valid JSON: {exc.msg} at line {exc.lineno}, column {exc.colno}."
        )
        raise ValueError(msg) from exc

    if not isinstance(selection_report, dict):
        msg = (
            f"Numerical model-selection report {selection_report_path} must "
            "contain a JSON object."
        )
        raise ValueError(msg)

    validate_selection_report_value(
        data=selection_report,
        key="experiment",
        expected="numerical_logistic_configuration_selection",
        context="Numerical model-selection report",
    )
    validate_selection_report_value(
        data=selection_report,
        key="season",
        expected=season,
        context="Numerical model-selection report",
    )

    expected_configuration = {
        "feature_set": SELECTED_NUMERICAL_FEATURE_SET,
        "feature_columns": list(SELECTED_NUMERICAL_FEATURE_COLUMNS),
        "excluded_columns": list(
            SELECTED_NUMERICAL_EXCLUDED_FEATURE_COLUMNS
        ),
        "feature_count": len(SELECTED_NUMERICAL_FEATURE_COLUMNS),
        "C": SELECTED_LOGISTIC_C,
    }
    selected_configuration = require_selection_report_object(
        data=selection_report,
        key="selected_configuration",
        context="Numerical model-selection report",
    )
    for key, expected in expected_configuration.items():
        validate_selection_report_value(
            data=selected_configuration,
            key=key,
            expected=expected,
            context="Selected numerical logistic configuration",
        )

    frozen_configuration = require_selection_report_object(
        data=selection_report,
        key="frozen_evaluation_configuration",
        context="Numerical model-selection report",
    )
    for key, expected in expected_configuration.items():
        if key == "feature_count":
            continue
        validate_selection_report_value(
            data=frozen_configuration,
            key=key,
            expected=expected,
            context="Frozen numerical logistic configuration",
        )
    validate_selection_report_value(
        data=frozen_configuration,
        key="matches_selected_configuration",
        expected=True,
        context="Frozen numerical logistic configuration",
    )
    frozen_feature_columns = frozen_configuration["feature_columns"]
    if len(frozen_feature_columns) != expected_configuration["feature_count"]:
        msg = (
            "Frozen numerical logistic configuration must contain exactly "
            f"{expected_configuration['feature_count']} feature columns, got "
            f"{len(frozen_feature_columns)}."
        )
        raise ValueError(msg)

    selection_scope = require_selection_report_object(
        data=selection_report,
        key="selection_scope",
        context="Numerical model-selection report",
    )
    expected_selection_scope = {
        "type": "expanding_window_training_only",
        "outer_training_matchdays": [1, TRAIN_END_MATCHDAY],
        "outer_test_matchdays": [TEST_START_MATCHDAY, LAST_MATCHDAY],
        "outer_training_row_count": EXPECTED_TRAIN_COUNT,
        "outer_test_rows_used_for_fitting": 0,
        "outer_test_rows_used_for_scaling": 0,
        "outer_test_rows_used_for_scoring": 0,
        "refit_after_selection": False,
        "outer_test_evaluation_performed": False,
    }
    for key, expected in expected_selection_scope.items():
        validate_selection_report_value(
            data=selection_scope,
            key=key,
            expected=expected,
            context="Numerical logistic selection scope",
        )


def require_selection_report_object(
    data: dict[str, Any],
    key: str,
    context: str,
) -> dict[str, Any]:
    """Return one required object from the numerical selection report."""
    value = data.get(key)
    if not isinstance(value, dict):
        msg = f"{context} field {key!r} must be a JSON object."
        raise ValueError(msg)
    return value


def validate_selection_report_value(
    data: dict[str, Any],
    key: str,
    expected: Any,
    context: str,
) -> None:
    """Require one report field to equal its frozen expected value."""
    if key not in data:
        msg = f"{context} is missing required field {key!r}."
        raise ValueError(msg)

    actual = data[key]
    if type(actual) is not type(expected) or actual != expected:
        msg = (
            f"{context} field {key!r} must be {expected!r}, "
            f"got {actual!r}."
        )
        raise ValueError(msg)


def evaluate_numerical_logistic_configuration(
    features_path: Path,
    season: int,
    experiment: str,
    feature_columns: tuple[str, ...],
    C: float,
    selection_provenance: dict[str, Any] | None = None,
) -> ModelEvaluationArtifacts:
    """Fit and evaluate one explicit numerical logistic configuration."""
    data = load_numerical_classification_data(
        features_path=features_path,
        season=season,
    )
    data = select_classification_features(
        data=data,
        feature_columns=feature_columns,
    )
    return evaluate_logistic_classification_data(
        data=data,
        season=season,
        experiment=experiment,
        C=C,
        input_reference=features_path,
        selection_provenance=selection_provenance,
        convergence_context="Numerical logistic regression",
    )


def evaluate_logistic_classification_data(
    data: ClassificationData,
    season: int,
    experiment: str,
    C: float,
    input_reference: Path | dict[str, Path],
    selection_provenance: dict[str, Any] | None = None,
    report_sections: dict[str, Any] | None = None,
    convergence_context: str = "Logistic regression",
) -> ModelEvaluationArtifacts:
    """Fit and report one logistic model through the shared evaluation path."""
    classifier = build_logistic_pipeline(C=C)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=ConvergenceWarning)
            evaluation = evaluate_classifier(classifier=classifier, data=data)
    except ConvergenceWarning as exc:
        msg = (
            f"{convergence_context} did not converge within "
            f"{LOGISTIC_MAX_ITER} iterations."
        )
        raise ValueError(msg) from exc

    scaler = classifier.named_steps["scaler"]
    logistic = classifier.named_steps["classifier"]
    training_sample_count = int(scaler.n_samples_seen_)
    if training_sample_count != len(data.X_train):
        msg = (
            "StandardScaler was not fitted on exactly the training rows: "
            f"saw {training_sample_count}, expected {len(data.X_train)}."
        )
        raise ValueError(msg)

    class_indices = {
        str(target_class): index
        for index, target_class in enumerate(logistic.classes_)
    }
    ordered_coefficients = [
        [
            float(value)
            for value in logistic.coef_[class_indices[target_class]]
        ]
        for target_class in TARGET_CLASSES
    ]
    ordered_intercepts = {
        target_class: float(logistic.intercept_[class_indices[target_class]])
        for target_class in TARGET_CLASSES
    }
    model = {
        "estimator": "sklearn.pipeline.Pipeline",
        "scikit_learn_version": sklearn.__version__,
        "uses_feature_values": True,
        "steps": [
            "sklearn.preprocessing.StandardScaler",
            "sklearn.linear_model.LogisticRegression",
        ],
        "standard_scaler": {
            "with_mean": bool(scaler.with_mean),
            "with_std": bool(scaler.with_std),
            "training_sample_count": training_sample_count,
        },
        "logistic_regression": {
            "loss": "multinomial",
            "solver": logistic.solver,
            "C": float(logistic.C),
            "l1_ratio": float(logistic.l1_ratio),
            "class_weight": logistic.class_weight,
            "fit_intercept": bool(logistic.fit_intercept),
            "max_iter": int(logistic.max_iter),
            "tol": float(logistic.tol),
            "iterations": [int(value) for value in logistic.n_iter_],
        },
    }
    extra_sections: dict[str, Any] = {
        "coefficients": {
            "class_order": list(TARGET_CLASSES),
            "feature_columns": list(data.feature_columns),
            "orientation": (
                "rows=target classes, columns=standardized feature columns"
            ),
            "values": ordered_coefficients,
            "intercepts": ordered_intercepts,
        }
    }
    if selection_provenance is not None:
        extra_sections["selection_provenance"] = selection_provenance
    if report_sections is not None:
        duplicate_sections = sorted(set(extra_sections) & set(report_sections))
        if duplicate_sections:
            msg = (
                "Logistic evaluation report sections would overwrite shared "
                f"sections: {duplicate_sections}."
            )
            raise ValueError(msg)
        extra_sections.update(report_sections)

    report = build_classification_report(
        experiment=experiment,
        season=season,
        input_reference=input_reference,
        model=model,
        data=data,
        evaluation=evaluation,
        extra_sections=extra_sections,
    )
    return ModelEvaluationArtifacts(
        report=report,
        prediction_rows=evaluation.prediction_rows,
    )


def build_logistic_pipeline(C: float) -> Pipeline:
    """Build the shared standardized multinomial logistic pipeline."""
    if not isfinite(C) or C <= 0.0:
        msg = f"Logistic-regression C must be positive and finite, got {C}."
        raise ValueError(msg)
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    solver=LOGISTIC_SOLVER,
                    C=C,
                    l1_ratio=LOGISTIC_L1_RATIO,
                    class_weight=None,
                    max_iter=LOGISTIC_MAX_ITER,
                    tol=LOGISTIC_TOLERANCE,
                ),
            ),
        ]
    )


def select_classification_features(
    data: ClassificationData,
    feature_columns: tuple[str, ...],
) -> ClassificationData:
    """Return one validated feature-column view of classification data."""
    if not feature_columns:
        msg = "Classification feature selection must not be empty."
        raise ValueError(msg)
    if len(set(feature_columns)) != len(feature_columns):
        msg = "Classification feature selection contains duplicate columns."
        raise ValueError(msg)
    unknown_columns = [
        column
        for column in feature_columns
        if column not in data.feature_columns
    ]
    if unknown_columns:
        msg = f"Classification feature selection is unknown: {unknown_columns}."
        raise ValueError(msg)

    return ClassificationData(
        feature_columns=feature_columns,
        X_train=data.X_train.loc[:, list(feature_columns)].copy(),
        X_test=data.X_test.loc[:, list(feature_columns)].copy(),
        y_train=data.y_train.copy(),
        y_test=data.y_test.copy(),
        train_metadata=data.train_metadata.copy(),
        test_metadata=data.test_metadata.copy(),
    )


def build_classification_report(
    experiment: str,
    season: int,
    input_reference: Path | dict[str, Path],
    model: dict[str, Any],
    data: ClassificationData,
    evaluation: ClassifierEvaluation,
    extra_sections: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the shared serializable report structure for one classifier."""
    if isinstance(input_reference, Path):
        input_fields = {"input_path": str(input_reference)}
    elif (
        isinstance(input_reference, dict)
        and input_reference
        and all(
            isinstance(name, str) and name and isinstance(path, Path)
            for name, path in input_reference.items()
        )
    ):
        input_fields = {
            "input_paths": {
                name: str(path) for name, path in input_reference.items()
            }
        }
    else:
        msg = "Classification report input reference is invalid."
        raise ValueError(msg)

    report = {
        "experiment": experiment,
        "season": season,
        **input_fields,
        "model": model,
        "target_classes": list(TARGET_CLASSES),
        "feature_columns": list(data.feature_columns),
        "split": {
            "type": "fixed_chronological_matchday",
            "train_matchdays": [1, TRAIN_END_MATCHDAY],
            "test_matchdays": [TEST_START_MATCHDAY, LAST_MATCHDAY],
            "train_row_count": len(data.y_train),
            "test_row_count": len(data.y_test),
            "train_class_counts": count_classes(data.y_train),
            "test_class_counts": count_classes(data.y_test),
        },
        "metrics": evaluation.metrics,
        "confusion_matrix": {
            "labels": list(TARGET_CLASSES),
            "orientation": "rows=true classes, columns=predicted classes",
            "values": evaluation.confusion_values,
        },
    }
    if extra_sections is not None:
        report.update(extra_sections)
    report["metric_definitions"] = {
        "multiclass_brier_score": (
            "Mean over test matches of the sum across H, D, and A of "
            "(observed_one_hot - predicted_probability) squared; "
            "unscaled range 0 to 2, lower is better."
        ),
    }
    return report


def evaluate_classifier(
    classifier: Any,
    data: ClassificationData,
) -> ClassifierEvaluation:
    """Fit one classifier and evaluate it through the shared model interface."""
    classifier.fit(data.X_train, data.y_train)

    predictions = tuple(str(value) for value in classifier.predict(data.X_test))
    validate_predictions(predictions, expected_row_count=len(data.y_test))

    raw_probabilities = classifier.predict_proba(data.X_test)
    probabilities = reorder_probabilities(
        probabilities=raw_probabilities,
        source_classes=classifier.classes_,
        target_classes=TARGET_CLASSES,
    )
    metric_probabilities = reorder_probabilities(
        probabilities=raw_probabilities,
        source_classes=classifier.classes_,
        target_classes=METRIC_PROBABILITY_CLASSES,
    )
    validate_probabilities(
        probabilities=probabilities,
        expected_row_count=len(data.y_test),
        expected_class_count=len(TARGET_CLASSES),
    )

    metrics = calculate_classification_metrics(
        targets=data.y_test,
        predictions=predictions,
        metric_probabilities=metric_probabilities,
    )
    confusion = confusion_matrix(
        data.y_test,
        predictions,
        labels=list(TARGET_CLASSES),
    )
    prediction_rows = build_prediction_rows(
        data=data,
        predictions=predictions,
        probabilities=probabilities,
    )
    return ClassifierEvaluation(
        predictions=predictions,
        probabilities=probabilities,
        metrics=metrics,
        confusion_values=confusion.tolist(),
        prediction_rows=prediction_rows,
    )


def calculate_classification_metrics(
    targets: pd.Series,
    predictions: Sequence[str],
    metric_probabilities: Any,
) -> dict[str, float]:
    """Calculate the fixed metrics shared by all experiment classifiers."""
    return {
        "log_loss": float(
            log_loss(
                targets,
                metric_probabilities,
                labels=list(METRIC_PROBABILITY_CLASSES),
            )
        ),
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_f1": float(
            f1_score(
                targets,
                predictions,
                labels=list(TARGET_CLASSES),
                average="macro",
                zero_division=0,
            )
        ),
        "multiclass_brier_score": float(
            brier_score_loss(
                targets,
                metric_probabilities,
                labels=list(METRIC_PROBABILITY_CLASSES),
                scale_by_half=False,
            )
        ),
    }


def build_prediction_rows(
    data: ClassificationData,
    predictions: Sequence[str],
    probabilities: Any,
) -> list[dict[str, Any]]:
    """Combine test metadata, targets, predictions, and H/D/A probabilities."""
    metadata_rows = data.test_metadata.to_dict(orient="records")
    rows = []
    for metadata, actual, predicted, probability_row in zip(
        metadata_rows,
        data.y_test,
        predictions,
        probabilities,
        strict=True,
    ):
        rows.append(
            {
                **metadata,
                "actual_result": str(actual),
                "predicted_result": str(predicted),
                "probability_H": float(probability_row[0]),
                "probability_D": float(probability_row[1]),
                "probability_A": float(probability_row[2]),
            }
        )
    return rows


def load_numerical_classification_data(
    features_path: Path,
    season: int,
) -> ClassificationData:
    """Load and strictly validate numerical features for the main experiment."""
    frame = pd.read_csv(features_path)
    expected_columns = list(NUMERICAL_FEATURE_OUTPUT_COLUMNS)
    actual_columns = list(frame.columns)
    if actual_columns != expected_columns:
        missing = [
            column for column in expected_columns if column not in actual_columns
        ]
        extra = [
            column for column in actual_columns if column not in expected_columns
        ]
        msg = (
            "Numerical feature table schema or column order does not match. "
            f"Missing: {missing}; extra: {extra}."
        )
        raise ValueError(msg)

    expected_total_count = EXPECTED_TRAIN_COUNT + EXPECTED_TEST_COUNT
    if len(frame) != expected_total_count:
        msg = (
            f"Numerical feature table must contain {expected_total_count} rows, "
            f"got {len(frame)}."
        )
        raise ValueError(msg)
    if frame["match_id"].duplicated().any():
        msg = "Numerical feature table contains duplicate match IDs."
        raise ValueError(msg)
    if not frame["season"].eq(season).all():
        seasons = sorted(int(value) for value in frame["season"].unique())
        msg = (
            f"Numerical feature table must contain only season {season}, "
            f"got {seasons}."
        )
        raise ValueError(msg)

    validate_target_classes(frame["result"], context="full dataset")
    validate_matchday_splits(frame, context="Numerical feature table")
    validate_chronological_order(frame, context="Numerical feature table")

    numeric_features = frame.loc[:, NUMERICAL_FEATURE_COLUMNS].apply(
        pd.to_numeric,
        errors="raise",
    )
    for column in NUMERICAL_FEATURE_COLUMNS:
        if not all(isfinite(float(value)) for value in numeric_features[column]):
            msg = f"Numerical feature {column!r} contains non-finite values."
            raise ValueError(msg)

    train_mask = frame["dataset_split"].eq("train")
    test_mask = frame["dataset_split"].eq("test")
    X_train = numeric_features.loc[train_mask].copy()
    X_test = numeric_features.loc[test_mask].copy()
    y_train = frame.loc[train_mask, "result"].copy()
    y_test = frame.loc[test_mask, "result"].copy()
    train_metadata = frame.loc[
        train_mask,
        list(PREDICTION_METADATA_COLUMNS),
    ].copy()
    test_metadata = frame.loc[
        test_mask,
        list(PREDICTION_METADATA_COLUMNS),
    ].copy()

    if len(X_train) != EXPECTED_TRAIN_COUNT:
        msg = (
            f"Training split must contain {EXPECTED_TRAIN_COUNT} rows, "
            f"got {len(X_train)}."
        )
        raise ValueError(msg)
    if len(X_test) != EXPECTED_TEST_COUNT:
        msg = (
            f"Test split must contain {EXPECTED_TEST_COUNT} rows, "
            f"got {len(X_test)}."
        )
        raise ValueError(msg)

    validate_target_classes(y_train, context="training split")
    validate_target_classes(y_test, context="test split")
    return ClassificationData(
        feature_columns=NUMERICAL_FEATURE_COLUMNS,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        train_metadata=train_metadata,
        test_metadata=test_metadata,
    )


def validate_matchday_splits(
    frame: pd.DataFrame,
    context: str = "Numerical feature table",
) -> None:
    """Ensure each row belongs to its fixed matchday-based split."""
    matchdays = pd.to_numeric(frame["matchday"], errors="raise")
    expected_splits = matchdays.map(get_expected_split)
    actual_splits = frame["dataset_split"]
    invalid_mask = actual_splits.ne(expected_splits)
    if invalid_mask.any():
        invalid_ids = frame.loc[invalid_mask, "match_id"].astype(int).tolist()
        msg = (
            f"{context} rows have invalid dataset_split values for "
            f"match IDs {invalid_ids}."
        )
        raise ValueError(msg)

    matchday_counts = matchdays.value_counts().to_dict()
    invalid_matchdays = {
        int(matchday): int(count)
        for matchday, count in matchday_counts.items()
        if count != EXPECTED_MATCHES_PER_MATCHDAY
    }
    expected_matchdays = set(range(1, LAST_MATCHDAY + 1))
    actual_matchdays = {int(value) for value in matchday_counts}
    missing_matchdays = sorted(expected_matchdays.difference(actual_matchdays))
    if invalid_matchdays or missing_matchdays:
        msg = (
            f"{context} must contain nine matches per matchday. "
            f"Invalid counts: {invalid_matchdays}; missing: {missing_matchdays}."
        )
        raise ValueError(msg)


def validate_chronological_order(
    frame: pd.DataFrame,
    context: str = "Numerical feature table",
) -> None:
    """Ensure every training kickoff is earlier than every test kickoff."""
    try:
        kickoffs = [
            datetime.fromisoformat(str(value))
            for value in frame["kickoff"]
        ]
    except ValueError as exc:
        msg = f"{context} contains an invalid kickoff timestamp."
        raise ValueError(msg) from exc

    if any(kickoff.tzinfo is None for kickoff in kickoffs):
        msg = f"Every kickoff in {context.lower()} needs a timezone offset."
        raise ValueError(msg)

    train_kickoffs = [
        kickoff
        for kickoff, split in zip(
            kickoffs,
            frame["dataset_split"],
            strict=True,
        )
        if split == "train"
    ]
    test_kickoffs = [
        kickoff
        for kickoff, split in zip(
            kickoffs,
            frame["dataset_split"],
            strict=True,
        )
        if split == "test"
    ]
    if not train_kickoffs or not test_kickoffs:
        msg = "Both chronological dataset splits must contain matches."
        raise ValueError(msg)
    if max(train_kickoffs) >= min(test_kickoffs):
        msg = (
            "Chronological split is invalid: a training kickoff is not earlier "
            "than every test kickoff."
        )
        raise ValueError(msg)


def get_expected_split(matchday: int | float) -> str:
    """Return the experiment split for a validated whole-number matchday."""
    numeric_matchday = float(matchday)
    if not numeric_matchday.is_integer():
        msg = f"Matchday must be a whole number, got {matchday!r}."
        raise ValueError(msg)
    parsed_matchday = int(numeric_matchday)
    if 1 <= parsed_matchday <= TRAIN_END_MATCHDAY:
        return "train"
    if TEST_START_MATCHDAY <= parsed_matchday <= LAST_MATCHDAY:
        return "test"
    msg = f"Matchday must be between 1 and {LAST_MATCHDAY}, got {parsed_matchday}."
    raise ValueError(msg)


def validate_target_classes(targets: pd.Series, context: str) -> None:
    """Require all and only the fixed H/D/A target classes."""
    actual_classes = {str(value) for value in targets.dropna().unique()}
    expected_classes = set(TARGET_CLASSES)
    if targets.isna().any() or actual_classes != expected_classes:
        msg = (
            f"{context.capitalize()} must contain exactly target classes "
            f"{list(TARGET_CLASSES)}, got {sorted(actual_classes)}."
        )
        raise ValueError(msg)


def reorder_probabilities(
    probabilities: Any,
    source_classes: Sequence[str],
    target_classes: Sequence[str],
) -> Any:
    """Reorder probability columns into the fixed reporting class order."""
    source_indices = {
        str(source_class): index
        for index, source_class in enumerate(source_classes)
    }
    if set(source_indices) != set(target_classes):
        msg = (
            f"Classifier classes {sorted(source_indices)} do not match "
            f"target classes {list(target_classes)}."
        )
        raise ValueError(msg)
    return probabilities[
        :,
        [source_indices[target_class] for target_class in target_classes],
    ]


def validate_predictions(
    predictions: Sequence[str],
    expected_row_count: int,
) -> None:
    """Reject predictions with an invalid count or unknown target classes."""
    if len(predictions) != expected_row_count:
        msg = (
            f"Classifier returned {len(predictions)} predictions, "
            f"expected {expected_row_count}."
        )
        raise ValueError(msg)

    unknown_classes = sorted(set(predictions).difference(TARGET_CLASSES))
    if unknown_classes:
        msg = f"Classifier returned unknown target classes: {unknown_classes}."
        raise ValueError(msg)


def validate_probabilities(
    probabilities: Any,
    expected_row_count: int,
    expected_class_count: int,
) -> None:
    """Reject probability matrices with invalid shape or values."""
    if probabilities.shape != (expected_row_count, expected_class_count):
        msg = (
            "Predicted probability matrix has shape "
            f"{probabilities.shape}, expected "
            f"({expected_row_count}, {expected_class_count})."
        )
        raise ValueError(msg)

    for row in probabilities:
        values = [float(value) for value in row]
        if any(not isfinite(value) or value < 0.0 or value > 1.0 for value in values):
            msg = f"Predicted probabilities are invalid: {values}."
            raise ValueError(msg)
        if not isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1e-12):
            msg = f"Predicted probabilities do not sum to one: {values}."
            raise ValueError(msg)


def count_classes(targets: pd.Series) -> dict[str, int]:
    """Count targets in fixed H/D/A order."""
    return {
        target_class: int(targets.eq(target_class).sum())
        for target_class in TARGET_CLASSES
    }
