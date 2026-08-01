"""Load model-ready features and evaluate reproducible classification baselines."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import isclose, isfinite
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
)

from buli_news.numerical_features import (
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


@dataclass(frozen=True)
class ClassificationData:
    """Validated chronological train and test data for one feature set."""

    feature_columns: tuple[str, ...]
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
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
    """Fit and evaluate a prior-based dummy classifier on the fixed split."""
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
    report = {
        "experiment": "numerical_dummy_baseline",
        "season": season,
        "input_path": str(features_path),
        "model": {
            "estimator": "sklearn.dummy.DummyClassifier",
            "scikit_learn_version": sklearn.__version__,
            "strategy": "prior",
            "uses_feature_values": False,
            "predicted_class": next(iter(predicted_classes)),
            "training_class_prior": class_prior,
        },
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
        "metric_definitions": {
            "multiclass_brier_score": (
                "Mean over test matches of the sum across H, D, and A of "
                "(observed_one_hot - predicted_probability) squared; "
                "unscaled range 0 to 2, lower is better."
            ),
        },
    }
    return ModelEvaluationArtifacts(
        report=report,
        prediction_rows=evaluation.prediction_rows,
    )


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
    validate_matchday_splits(frame)
    validate_chronological_order(frame)

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
        test_metadata=test_metadata,
    )


def validate_matchday_splits(frame: pd.DataFrame) -> None:
    """Ensure each row belongs to its fixed matchday-based split."""
    matchdays = pd.to_numeric(frame["matchday"], errors="raise")
    expected_splits = matchdays.map(get_expected_split)
    actual_splits = frame["dataset_split"]
    invalid_mask = actual_splits.ne(expected_splits)
    if invalid_mask.any():
        invalid_ids = frame.loc[invalid_mask, "match_id"].astype(int).tolist()
        msg = (
            "Numerical feature rows have invalid dataset_split values for "
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
            "Numerical feature table must contain nine matches per matchday. "
            f"Invalid counts: {invalid_matchdays}; missing: {missing_matchdays}."
        )
        raise ValueError(msg)


def validate_chronological_order(frame: pd.DataFrame) -> None:
    """Ensure every training kickoff is earlier than every test kickoff."""
    try:
        kickoffs = [
            datetime.fromisoformat(str(value))
            for value in frame["kickoff"]
        ]
    except ValueError as exc:
        msg = "Numerical feature table contains an invalid kickoff timestamp."
        raise ValueError(msg) from exc

    if any(kickoff.tzinfo is None for kickoff in kickoffs):
        msg = "Every numerical feature kickoff needs a timezone offset."
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
