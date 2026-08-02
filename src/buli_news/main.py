"""Command line interface for the Bundesliga news pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import httpx

from buli_news.football_data import build_bundesliga_filename, fetch_bundesliga_csv
from buli_news.matches import normalize_openligadb_matches
from buli_news.model_selection import (
    MODEL_SELECTION_PREDICTION_OUTPUT_COLUMNS,
    select_numerical_logistic_configuration,
)
from buli_news.modeling import (
    CLASSIFICATION_PREDICTION_OUTPUT_COLUMNS,
    evaluate_numerical_dummy,
    evaluate_numerical_logistic_final,
    evaluate_numerical_logistic_reference,
)
from buli_news.news_articles import build_news_articles
from buli_news.news_contents import build_news_contents
from buli_news.newsapi import (
    count_successful_existing_responses,
    fetch_news_requests,
    get_api_key,
    select_requests,
)
from buli_news.news_requests import build_news_requests
from buli_news.news_source_policy import build_news_source_policy
from buli_news.news_source_review import export_news_source_review
from buli_news.numerical_features import (
    NUMERICAL_FEATURE_OUTPUT_COLUMNS,
    build_numerical_features,
)
from buli_news.numerical_matches import build_numerical_matches
from buli_news.openligadb import fetch_matchdata
from buli_news.paths import SeasonPaths
from buli_news.storage import (
    append_jsonl,
    read_bytes,
    read_json,
    read_jsonl,
    write_bytes,
    write_csv,
    write_json,
    write_jsonl,
    write_text,
)


class NoMatchesError(Exception):
    """Raised when OpenLigaDB returns an empty match list."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="buli-news")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch_openliga = subparsers.add_parser(
        "fetch-openliga",
        help="Fetch raw match data from OpenLigaDB.",
    )
    fetch_openliga.add_argument("--league", required=True, help="League shortcut, e.g. bl1.")
    fetch_openliga.add_argument(
        "--season",
        required=True,
        type=int,
        help="Season start year, e.g. 2025 for 2025/26.",
    )

    fetch_football_data = subparsers.add_parser(
        "fetch-football-data",
        help="Fetch raw Bundesliga match statistics from Football-Data.co.uk.",
    )
    fetch_football_data.add_argument(
        "--season",
        required=True,
        type=int,
        help="Season start year, e.g. 2025 for 2025/26.",
    )

    build_matches = subparsers.add_parser(
        "build-matches",
        help="Build normalized match records from raw OpenLigaDB data.",
    )
    build_matches.add_argument("--league", required=True, help="League shortcut, e.g. bl1.")
    build_matches.add_argument(
        "--season",
        required=True,
        type=int,
        help="Season start year, e.g. 2025 for 2025/26.",
    )
    build_matches.add_argument(
        "--timezone",
        default="Europe/Berlin",
        help="Timezone for local kickoff and pre-match windows.",
    )

    build_numerical_matches_parser = subparsers.add_parser(
        "build-numerical-matches",
        help="Join OpenLigaDB metadata with Football-Data match statistics.",
    )
    build_numerical_matches_parser.add_argument(
        "--season",
        required=True,
        type=int,
        help="Season start year, e.g. 2025 for 2025/26.",
    )
    build_numerical_matches_parser.add_argument(
        "--config",
        default="config/teams.json",
        help="Path to the team mapping config.",
    )

    build_numerical_features_parser = subparsers.add_parser(
        "build-numerical-features",
        help="Build leakage-safe numerical pre-match features.",
    )
    build_numerical_features_parser.add_argument(
        "--season",
        required=True,
        type=int,
        help="Season start year, e.g. 2025 for 2025/26.",
    )

    evaluate_numerical_dummy_parser = subparsers.add_parser(
        "evaluate-numerical-dummy",
        help="Evaluate a prior-based dummy classifier on the fixed split.",
    )
    evaluate_numerical_dummy_parser.add_argument(
        "--season",
        required=True,
        type=int,
        help="Season start year, e.g. 2025 for 2025/26.",
    )

    evaluate_numerical_logistic_reference_parser = subparsers.add_parser(
        "evaluate-numerical-logistic-reference",
        help=(
            "Evaluate the fixed full-feature C=1 numerical logistic "
            "reference on the test split."
        ),
    )
    evaluate_numerical_logistic_reference_parser.add_argument(
        "--season",
        required=True,
        type=int,
        help="Season start year, e.g. 2025 for 2025/26.",
    )

    select_numerical_logistic_configuration_parser = subparsers.add_parser(
        "select-numerical-logistic-configuration",
        help=(
            "Select numerical logistic features and C with training-only "
            "expanding-window validation."
        ),
    )
    select_numerical_logistic_configuration_parser.add_argument(
        "--season",
        required=True,
        type=int,
        help="Season start year, e.g. 2025 for 2025/26.",
    )

    evaluate_numerical_logistic_final_parser = subparsers.add_parser(
        "evaluate-numerical-logistic-final",
        help=(
            "Evaluate the frozen training-selected numerical logistic model "
            "on the fixed test split."
        ),
    )
    evaluate_numerical_logistic_final_parser.add_argument(
        "--season",
        required=True,
        type=int,
        help="Season start year, e.g. 2025 for 2025/26.",
    )

    export_news_source_review_parser = subparsers.add_parser(
        "export-news-source-review",
        help="Export collected news-source homepages for manual legal review.",
    )
    export_news_source_review_parser.add_argument(
        "--season",
        required=True,
        type=int,
        help="Season start year, e.g. 2025 for 2025/26.",
    )
    export_news_source_review_parser.add_argument(
        "--raw-dir",
        default=None,
        help=(
            "Directory with raw news response JSON files. Defaults to "
            "data/raw/newsapi/{season}."
        ),
    )
    export_news_source_review_parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output XLSX. Defaults to "
            "data/review/{season}/news_sources.xlsx."
        ),
    )
    export_news_source_review_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing workbook, including any manual review data.",
    )

    build_news_source_policy_parser = subparsers.add_parser(
        "build-news-source-policy",
        help="Build a versioned JSON policy from the reviewed source XLSX.",
    )
    build_news_source_policy_parser.add_argument(
        "--season",
        required=True,
        type=int,
        help="Season start year, e.g. 2025 for 2025/26.",
    )
    build_news_source_policy_parser.add_argument(
        "--input",
        default=None,
        help=(
            "Reviewed XLSX input. Defaults to "
            "data/review/{season}/news_sources.xlsx."
        ),
    )
    build_news_source_policy_parser.add_argument(
        "--output",
        default="config/news_source_policy.json",
        help="Output path for the generated JSON policy.",
    )
    build_news_source_policy_parser.add_argument(
        "--policy-version",
        default=1,
        type=int,
        help="Positive policy version used in the generated policy ID.",
    )

    build_news_articles_parser = subparsers.add_parser(
        "build-news-articles",
        help=(
            "Build policy-filtered canonical articles and request-bound "
            "match links."
        ),
    )
    build_news_articles_parser.add_argument(
        "--season",
        required=True,
        type=int,
        help="Season start year, e.g. 2025 for 2025/26.",
    )
    build_news_articles_parser.add_argument(
        "--policy",
        default="config/news_source_policy.json",
        help="Path to the reviewed news-source policy JSON.",
    )
    build_news_articles_parser.add_argument(
        "--raw-dir",
        default=None,
        help=(
            "Directory with raw news response JSON files. Defaults to "
            "data/raw/newsapi/{season}."
        ),
    )
    build_news_articles_parser.add_argument(
        "--timezone",
        default="Europe/Berlin",
        help="Timezone used to check publication timestamps against windows.",
    )

    build_news_contents_parser = subparsers.add_parser(
        "build-news-contents",
        help="Group canonical articles by deterministic normalized text content.",
    )
    build_news_contents_parser.add_argument(
        "--season",
        required=True,
        type=int,
        help="Season start year, e.g. 2025 for 2025/26.",
    )

    build_news_requests_parser = subparsers.add_parser(
        "build-news-requests",
        help="Build planned Event Registry article requests from matches.",
    )
    build_news_requests_parser.add_argument(
        "--season",
        required=True,
        type=int,
        help="Season start year, e.g. 2025 for 2025/26.",
    )
    build_news_requests_parser.add_argument(
        "--config",
        default="config/teams.json",
        help="Path to team and league concept URI config.",
    )
    build_news_requests_parser.add_argument(
        "--lang",
        default="deu",
        help="Event Registry article language code.",
    )

    fetch_news = subparsers.add_parser(
        "fetch-news",
        help="Fetch raw Event Registry article responses from planned requests.",
    )
    fetch_news.add_argument(
        "--season",
        required=True,
        type=int,
        help="Season start year, e.g. 2025 for 2025/26.",
    )
    fetch_news.add_argument(
        "--request-id",
        help="Fetch only one planned request by request_id.",
    )
    fetch_news.add_argument(
        "--limit",
        type=int,
        help="Maximum number of planned requests to fetch. Omit to fetch all.",
    )
    fetch_news.add_argument(
        "--delay-seconds",
        type=float,
        default=1.0,
        help="Pause between API calls to reduce rate-limit risk.",
    )

    return parser


def fetch_openliga_command(league: str, season: int) -> None:
    match_data = fetch_matchdata(league=league, season=season)
    if not match_data.matches:
        msg = "No matches returned. Check league shortcut and season."
        raise NoMatchesError(msg)

    output_path = SeasonPaths(season).openligadb_raw(league)
    write_text(match_data.raw_json, output_path)
    print(f"Saved {len(match_data.matches)} matches to {output_path}")


def fetch_football_data_command(season: int) -> None:
    football_data = fetch_bundesliga_csv(season=season)
    output_path = SeasonPaths(season).football_data_raw(football_data.filename)
    write_bytes(football_data.content, output_path)
    print(
        f"Saved {football_data.row_count} Football-Data match rows to {output_path}"
    )
    print(f"Source: {football_data.source_url}")


def build_matches_command(league: str, season: int, timezone: str) -> None:
    paths = SeasonPaths(season)
    input_path = paths.openligadb_raw(league)
    raw_matches = read_json(input_path)
    if not isinstance(raw_matches, list) or not all(
        isinstance(item, dict) for item in raw_matches
    ):
        msg = f"{input_path} must contain a list of OpenLigaDB match objects."
        raise ValueError(msg)
    if not raw_matches:
        msg = "No matches returned. Check league shortcut and season."
        raise NoMatchesError(msg)

    matches = normalize_openligadb_matches(
        raw_matches=raw_matches,
        league=league,
        season=season,
        timezone=timezone,
    )
    output_path = paths.normalized_matches
    write_jsonl(matches, output_path)
    print(f"Saved {len(matches)} normalized matches to {output_path}")


def build_numerical_matches_command(season: int, config_path: str) -> None:
    paths = SeasonPaths(season)
    matches_path = paths.normalized_matches
    football_data_path = paths.football_data_raw(build_bundesliga_filename(season))
    config = read_json(Path(config_path))
    openliga_matches = read_jsonl(matches_path)
    football_data_content = read_bytes(football_data_path)
    if not isinstance(config, dict):
        msg = f"{config_path} must contain a JSON object."
        raise ValueError(msg)

    build = build_numerical_matches(
        openliga_matches=openliga_matches,
        football_data_content=football_data_content,
        config=config,
        season=season,
    )
    output_path = paths.numerical_matches
    quality_path = paths.numerical_matches_quality
    write_jsonl(build.matches, output_path)
    write_json(build.quality_report, quality_path)
    print(f"Saved {len(build.matches)} numerical matches to {output_path}")
    print(f"Saved source quality report to {quality_path}")
    print(
        "Result mismatches between sources: "
        f"{build.quality_report['result_mismatch_count']}"
    )


def build_numerical_features_command(season: int) -> None:
    paths = SeasonPaths(season)
    input_path = paths.numerical_matches
    matches = read_jsonl(input_path)
    build = build_numerical_features(matches=matches, season=season)

    output_path = paths.numerical_features
    write_csv(
        records=build.rows,
        fieldnames=NUMERICAL_FEATURE_OUTPUT_COLUMNS,
        path=output_path,
    )
    print(f"Saved {len(build.rows)} numerical feature rows to {output_path}")
    print(f"Training rows: {build.train_count}")
    print(f"Test rows: {build.test_count}")


def evaluate_numerical_dummy_command(season: int) -> None:
    paths = SeasonPaths(season)
    input_path = paths.numerical_features
    evaluation = evaluate_numerical_dummy(
        features_path=input_path,
        season=season,
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


def evaluate_numerical_logistic_reference_command(season: int) -> None:
    paths = SeasonPaths(season)
    input_path = paths.numerical_features
    evaluation = evaluate_numerical_logistic_reference(
        features_path=input_path,
        season=season,
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


def evaluate_numerical_logistic_final_command(season: int) -> None:
    paths = SeasonPaths(season)
    input_path = paths.numerical_features
    selection_report_path = paths.numerical_model_output(
        "logistic_regression",
        "selection",
        "report.json",
    )
    evaluation = evaluate_numerical_logistic_final(
        features_path=input_path,
        season=season,
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


def select_numerical_logistic_configuration_command(season: int) -> None:
    paths = SeasonPaths(season)
    input_path = paths.numerical_features
    selection = select_numerical_logistic_configuration(
        features_path=input_path,
        season=season,
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


def build_news_source_policy_command(
    season: int,
    input_path: str | None,
    output_path: str,
    policy_version: int,
) -> None:
    paths = SeasonPaths(season)
    workbook_path = (
        Path(input_path)
        if input_path is not None
        else paths.news_source_review
    )
    policy = build_news_source_policy(
        workbook_path=workbook_path,
        season=season,
        policy_version=policy_version,
    )
    destination = Path(output_path)
    write_json(policy, destination)

    summary = policy["summary"]
    print(f"Saved reviewed news source policy to {destination}")
    print(f"Reviewed sources: {policy['source_review']['reviewed_source_count']}")
    print(f"Included sources: {summary['include_source_count']}")
    print(f"Excluded sources: {summary['exclude_source_count']}")
    print("Unknown source decision: exclude")


def build_news_articles_command(
    season: int,
    policy_path: str,
    raw_dir: str | None,
    timezone: str,
) -> None:
    paths = SeasonPaths(season)
    requests = read_jsonl(paths.news_requests)
    policy = read_json(Path(policy_path))
    if not isinstance(policy, dict):
        msg = f"{policy_path} must contain a JSON object."
        raise ValueError(msg)
    source_dir = Path(raw_dir) if raw_dir is not None else paths.news_raw_dir
    build = build_news_articles(
        requests=requests,
        raw_dir=source_dir,
        policy=policy,
        season=season,
        timezone=timezone,
    )

    write_jsonl(build.articles, paths.news_articles)
    write_jsonl(build.article_links, paths.news_article_links)
    write_json(build.quality_report, paths.news_articles_quality)

    summary = build.quality_report["summary"]
    print(f"Saved {len(build.articles)} canonical articles to {paths.news_articles}")
    print(
        f"Saved {len(build.article_links)} request-bound article links to "
        f"{paths.news_article_links}"
    )
    print(f"Saved news article quality report to {paths.news_articles_quality}")
    print(
        "Policy-excluded occurrences: "
        f"{summary['policy_exclude_occurrence_count']}"
    )
    outside_window_count = build.quality_report["rejection_counts"].get(
        "publication_datetime_outside_request_window",
        0,
    )
    print(
        "Occurrences outside their request publication window: "
        f"{outside_window_count}"
    )


def build_news_contents_command(season: int) -> None:
    paths = SeasonPaths(season)
    articles = read_jsonl(paths.news_articles)
    build = build_news_contents(articles=articles, season=season)

    write_jsonl(build.contents, paths.news_contents)
    write_jsonl(
        build.article_content_links,
        paths.news_article_content_links,
    )
    write_json(build.quality_report, paths.news_contents_quality)

    summary = build.quality_report["summary"]
    print(f"Saved {len(build.contents)} canonical contents to {paths.news_contents}")
    print(
        f"Saved {len(build.article_content_links)} article-content links to "
        f"{paths.news_article_content_links}"
    )
    print(f"Saved news content quality report to {paths.news_contents_quality}")
    print(f"Collapsed duplicate article rows: {summary['collapsed_article_count']}")


def export_news_source_review_command(
    season: int,
    raw_dir: str | None,
    output_path: str | None,
    overwrite: bool,
) -> None:
    paths = SeasonPaths(season)
    source_dir = (
        Path(raw_dir)
        if raw_dir is not None
        else paths.news_raw_dir
    )
    destination = (
        Path(output_path)
        if output_path is not None
        else paths.news_source_review
    )
    homepage_counts = export_news_source_review(
        raw_dir=source_dir,
        output_path=destination,
        overwrite=overwrite,
    )
    print(f"Read raw news responses from {source_dir}")
    print(f"Exported {len(homepage_counts)} unique sources to {destination}")
    print(
        "Article occurrences before deduplication: "
        f"{sum(count for _, count in homepage_counts)}"
    )


def build_news_requests_command(season: int, config_path: str, lang: str) -> None:
    paths = SeasonPaths(season)
    matches_path = paths.normalized_matches
    config = read_json(Path(config_path))
    matches = read_jsonl(matches_path)
    requests = build_news_requests(matches=matches, config=config, lang=lang)

    output_path = paths.news_requests
    write_jsonl(requests, output_path)
    print(f"Saved {len(requests)} planned news requests to {output_path}")


def fetch_news_command(
    season: int,
    request_id: str | None,
    limit: int | None,
    delay_seconds: float,
) -> None:
    paths = SeasonPaths(season)
    requests_path = paths.news_requests
    planned_requests = read_jsonl(requests_path)
    selected_requests = select_requests(
        requests=planned_requests,
        request_id=request_id,
        limit=limit,
    )
    if not selected_requests:
        msg = f"No planned requests found in {requests_path}."
        raise ValueError(msg)

    output_dir = paths.news_raw_dir
    results_path = paths.news_fetch_results
    existing_count = count_successful_existing_responses(
        requests=selected_requests,
        output_dir=output_dir,
    )
    if existing_count == len(selected_requests):
        print(f"Skipped {existing_count} already fetched news requests.")
        print("Fetched 0 news requests.")
        print(f"Saved raw responses to {output_dir}")
        print(f"No new fetch results appended to {results_path}")
        return

    api_key = get_api_key()
    run_result = fetch_news_requests(
        requests=selected_requests,
        output_dir=output_dir,
        results_path=results_path,
        api_key=api_key,
        delay_seconds=delay_seconds,
        append_result=append_jsonl,
    )

    if run_result.skipped_count:
        print(f"Skipped {run_result.skipped_count} already fetched news requests.")
    print(f"Fetched {len(run_result.fetched_results)} news requests.")
    print(f"Saved raw responses to {output_dir}")
    if run_result.fetched_results:
        print(f"Appended fetch results to {results_path}")
    else:
        print(f"No new fetch results appended to {results_path}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.command == "fetch-openliga":
            fetch_openliga_command(league=args.league, season=args.season)
        elif args.command == "fetch-football-data":
            fetch_football_data_command(season=args.season)
        elif args.command == "build-matches":
            build_matches_command(
                league=args.league,
                season=args.season,
                timezone=args.timezone,
            )
        elif args.command == "build-numerical-matches":
            build_numerical_matches_command(
                season=args.season,
                config_path=args.config,
            )
        elif args.command == "build-numerical-features":
            build_numerical_features_command(season=args.season)
        elif args.command == "evaluate-numerical-dummy":
            evaluate_numerical_dummy_command(season=args.season)
        elif args.command == "evaluate-numerical-logistic-reference":
            evaluate_numerical_logistic_reference_command(season=args.season)
        elif args.command == "select-numerical-logistic-configuration":
            select_numerical_logistic_configuration_command(season=args.season)
        elif args.command == "evaluate-numerical-logistic-final":
            evaluate_numerical_logistic_final_command(season=args.season)
        elif args.command == "export-news-source-review":
            export_news_source_review_command(
                season=args.season,
                raw_dir=args.raw_dir,
                output_path=args.output,
                overwrite=args.overwrite,
            )
        elif args.command == "build-news-source-policy":
            build_news_source_policy_command(
                season=args.season,
                input_path=args.input,
                output_path=args.output,
                policy_version=args.policy_version,
            )
        elif args.command == "build-news-articles":
            build_news_articles_command(
                season=args.season,
                policy_path=args.policy,
                raw_dir=args.raw_dir,
                timezone=args.timezone,
            )
        elif args.command == "build-news-contents":
            build_news_contents_command(season=args.season)
        elif args.command == "build-news-requests":
            build_news_requests_command(
                season=args.season,
                config_path=args.config,
                lang=args.lang,
            )
        elif args.command == "fetch-news":
            fetch_news_command(
                season=args.season,
                request_id=args.request_id,
                limit=args.limit,
                delay_seconds=args.delay_seconds,
            )
    except httpx.HTTPStatusError as exc:
        response_text = exc.response.text.strip()
        response_detail = f": {response_text[:500]}" if response_text else ""
        parser.exit(
            status=1,
            message=(
                f"HTTP request returned status {exc.response.status_code} "
                f"for {exc.request.url}{response_detail}\n"
            ),
        )
    except httpx.HTTPError as exc:
        parser.exit(status=1, message=f"HTTP request failed: {exc}\n")
    except FileNotFoundError as exc:
        parser.exit(
            status=1,
            message=f"Input file not found: {exc.filename}.\n",
        )
    except NoMatchesError as exc:
        parser.exit(status=1, message=f"{exc}\n")
    except ValueError as exc:
        parser.exit(status=1, message=f"Invalid data: {exc}\n")


if __name__ == "__main__":
    main()
