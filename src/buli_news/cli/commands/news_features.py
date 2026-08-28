"""CLI command for match-level news-feature aggregation."""

from __future__ import annotations

import argparse
from pathlib import Path

from buli_news.cli.arguments import add_season_argument
from buli_news.news.annotations import (
    DEFAULT_ANNOTATION_CONFIG_PATH,
    load_annotation_config,
)
from buli_news.news.annotations.results import (
    read_existing_annotation_failures,
    read_existing_annotations,
)
from buli_news.news.annotations.runner import annotation_output_lock
from buli_news.news.features import (
    NEWS_FEATURE_OUTPUT_COLUMNS,
    build_news_features,
)
from buli_news.paths import SeasonPaths
from buli_news.storage import read_jsonl, write_csv, write_json


def register_news_feature_commands(
    subparsers: argparse._SubParsersAction,
) -> None:
    build_news_features_parser = subparsers.add_parser(
        "build-news-features",
        help="Aggregate validated annotations into match-level news features.",
    )
    add_season_argument(build_news_features_parser)
    build_news_features_parser.add_argument(
        "--config",
        default=str(DEFAULT_ANNOTATION_CONFIG_PATH),
        help="Path to the versioned news annotation schema and rating mapping.",
    )
    build_news_features_parser.set_defaults(
        command_handler=build_news_features_command,
    )


def build_news_features_command(args: argparse.Namespace) -> None:
    paths = SeasonPaths(args.season)
    config = load_annotation_config(Path(args.config))
    with annotation_output_lock(paths.news_annotations):
        annotations = read_existing_annotations(paths.news_annotations)
        failures = read_existing_annotation_failures(
            paths.news_annotation_failures
        )
    build = build_news_features(
        matches=read_jsonl(paths.normalized_matches),
        tasks=read_jsonl(paths.news_annotation_tasks),
        annotations=annotations,
        failures=failures,
        season=args.season,
        config=config,
    )
    write_csv(
        records=build.rows,
        fieldnames=NEWS_FEATURE_OUTPUT_COLUMNS,
        path=paths.news_features,
    )
    write_json(build.quality_report, paths.news_features_quality)

    summary = build.quality_report["summary"]
    print(f"Saved {len(build.rows)} news-feature rows to {paths.news_features}")
    print(f"Saved news-feature quality report to {paths.news_features_quality}")
    print(f"Training rows: {summary['train_feature_row_count']}")
    print(f"Test rows: {summary['test_feature_row_count']}")
    print(
        "Successful annotations: "
        f"{summary['successful_annotation_count']}/"
        f"{summary['expected_annotation_task_count']}"
    )
    print(
        "Incomplete home/away contexts: "
        f"{summary['incomplete_context_count']}"
    )
