"""CLI commands for numerical model selection and evaluation."""

from __future__ import annotations

import argparse

from buli_news.cli.arguments import add_season_argument
from buli_news.modeling.evaluation import (
    CLASSIFICATION_PREDICTION_OUTPUT_COLUMNS,
    evaluate_numerical_dummy,
    evaluate_numerical_logistic_final,
    evaluate_numerical_logistic_reference,
)
from buli_news.modeling.selection import (
    MODEL_SELECTION_PREDICTION_OUTPUT_COLUMNS,
    select_numerical_logistic_configuration,
)
from buli_news.paths import SeasonPaths
from buli_news.storage import write_csv, write_json


def register_modeling_commands(subparsers: argparse._SubParsersAction) -> None:
    evaluate_numerical_dummy_parser = subparsers.add_parser(
        "evaluate-numerical-dummy",
        help="Evaluate a prior-based dummy classifier on the fixed split.",
    )
    add_season_argument(evaluate_numerical_dummy_parser)
    evaluate_numerical_dummy_parser.set_defaults(
        command_handler=evaluate_numerical_dummy_command,
    )

    evaluate_numerical_logistic_reference_parser = subparsers.add_parser(
        "evaluate-numerical-logistic-reference",
        help=(
            "Evaluate the fixed full-feature C=1 numerical logistic "
            "reference on the test split."
        ),
    )
    add_season_argument(evaluate_numerical_logistic_reference_parser)
    evaluate_numerical_logistic_reference_parser.set_defaults(
        command_handler=evaluate_numerical_logistic_reference_command,
    )

    select_numerical_logistic_configuration_parser = subparsers.add_parser(
        "select-numerical-logistic-configuration",
        help=(
            "Select numerical logistic features and C with training-only "
            "expanding-window validation."
        ),
    )
    add_season_argument(select_numerical_logistic_configuration_parser)
    select_numerical_logistic_configuration_parser.set_defaults(
        command_handler=select_numerical_logistic_configuration_command,
    )

    evaluate_numerical_logistic_final_parser = subparsers.add_parser(
        "evaluate-numerical-logistic-final",
        help=(
            "Evaluate the frozen training-selected numerical logistic model "
            "on the fixed test split."
        ),
    )
    add_season_argument(evaluate_numerical_logistic_final_parser)
    evaluate_numerical_logistic_final_parser.set_defaults(
        command_handler=evaluate_numerical_logistic_final_command,
    )


def evaluate_numerical_dummy_command(args: argparse.Namespace) -> None:
    paths = SeasonPaths(args.season)
    input_path = paths.numerical_features
    evaluation = evaluate_numerical_dummy(
        features_path=input_path,
        season=args.season,
    )
    result_path = paths.numerical_model_output(
        "dummy",
        "evaluation.json",
    )
    predictions_path = paths.numerical_model_output(
        "dummy",
        "test_predictions.csv",
    )
    write_json(evaluation.report, result_path)
    write_csv(
        records=evaluation.prediction_rows,
        fieldnames=CLASSIFICATION_PREDICTION_OUTPUT_COLUMNS,
        path=predictions_path,
    )

    result = evaluation.report
    metrics = result["metrics"]
    print(f"Saved numerical dummy baseline results to {result_path}")
    print(f"Saved numerical dummy predictions to {predictions_path}")
    print(f"Predicted class: {result['model']['predicted_class']}")
    print(f"Log Loss: {metrics['log_loss']:.6f}")
    print(f"Accuracy: {metrics['accuracy']:.6f}")
    print(f"Macro-F1: {metrics['macro_f1']:.6f}")
    print(
        "Multiclass Brier Score: "
        f"{metrics['multiclass_brier_score']:.6f}"
    )


def evaluate_numerical_logistic_reference_command(
    args: argparse.Namespace,
) -> None:
    paths = SeasonPaths(args.season)
    input_path = paths.numerical_features
    evaluation = evaluate_numerical_logistic_reference(
        features_path=input_path,
        season=args.season,
    )
    result_path = paths.numerical_model_output(
        "logistic_regression",
        "reference",
        "evaluation.json",
    )
    predictions_path = paths.numerical_model_output(
        "logistic_regression",
        "reference",
        "test_predictions.csv",
    )
    write_json(evaluation.report, result_path)
    write_csv(
        records=evaluation.prediction_rows,
        fieldnames=CLASSIFICATION_PREDICTION_OUTPUT_COLUMNS,
        path=predictions_path,
    )

    result = evaluation.report
    metrics = result["metrics"]
    iterations = result["model"]["logistic_regression"]["iterations"]
    print(f"Saved numerical logistic reference results to {result_path}")
    print(
        "Saved numerical logistic reference predictions to "
        f"{predictions_path}"
    )
    print(f"Solver iterations: {iterations}")
    print(f"Log Loss: {metrics['log_loss']:.6f}")
    print(f"Accuracy: {metrics['accuracy']:.6f}")
    print(f"Macro-F1: {metrics['macro_f1']:.6f}")
    print(
        "Multiclass Brier Score: "
        f"{metrics['multiclass_brier_score']:.6f}"
    )


def select_numerical_logistic_configuration_command(
    args: argparse.Namespace,
) -> None:
    paths = SeasonPaths(args.season)
    input_path = paths.numerical_features
    selection = select_numerical_logistic_configuration(
        features_path=input_path,
        season=args.season,
    )
    result_path = paths.numerical_model_output(
        "logistic_regression",
        "selection",
        "report.json",
    )
    predictions_path = paths.numerical_model_output(
        "logistic_regression",
        "selection",
        "validation_predictions.csv",
    )
    write_json(selection.report, result_path)
    write_csv(
        records=selection.prediction_rows,
        fieldnames=MODEL_SELECTION_PREDICTION_OUTPUT_COLUMNS,
        path=predictions_path,
    )

    selected = selection.report["selected_configuration"]
    metrics = selected["pooled_metrics"]
    print(f"Saved numerical logistic model selection to {result_path}")
    print(f"Saved chronological validation predictions to {predictions_path}")
    print(
        "Selected configuration: "
        f"feature_set={selected['feature_set']}, C={selected['C']:g}"
    )
    print(f"Pooled validation Log Loss: {metrics['log_loss']:.6f}")
    print(
        "Pooled validation Multiclass Brier Score: "
        f"{metrics['multiclass_brier_score']:.6f}"
    )
    print(f"Pooled validation Accuracy: {metrics['accuracy']:.6f}")
    print(f"Pooled validation Macro-F1: {metrics['macro_f1']:.6f}")
    print("Outer test matches used for model selection: 0")


def evaluate_numerical_logistic_final_command(
    args: argparse.Namespace,
) -> None:
    paths = SeasonPaths(args.season)
    input_path = paths.numerical_features
    selection_report_path = paths.numerical_model_output(
        "logistic_regression",
        "selection",
        "report.json",
    )
    evaluation = evaluate_numerical_logistic_final(
        features_path=input_path,
        season=args.season,
        selection_report_path=selection_report_path,
    )
    result_path = paths.numerical_model_output(
        "logistic_regression",
        "final",
        "evaluation.json",
    )
    predictions_path = paths.numerical_model_output(
        "logistic_regression",
        "final",
        "test_predictions.csv",
    )
    write_json(evaluation.report, result_path)
    write_csv(
        records=evaluation.prediction_rows,
        fieldnames=CLASSIFICATION_PREDICTION_OUTPUT_COLUMNS,
        path=predictions_path,
    )

    result = evaluation.report
    metrics = result["metrics"]
    logistic = result["model"]["logistic_regression"]
    provenance = result["selection_provenance"]
    print(f"Saved final numerical logistic results to {result_path}")
    print(f"Saved final numerical logistic predictions to {predictions_path}")
    print(
        "Frozen configuration: "
        f"feature_set={provenance['feature_set']}, C={logistic['C']:g}"
    )
    print(f"Feature count: {len(result['feature_columns'])}")
    print(f"Solver iterations: {logistic['iterations']}")
    print(f"Log Loss: {metrics['log_loss']:.6f}")
    print(f"Accuracy: {metrics['accuracy']:.6f}")
    print(f"Macro-F1: {metrics['macro_f1']:.6f}")
    print(
        "Multiclass Brier Score: "
        f"{metrics['multiclass_brier_score']:.6f}"
    )
