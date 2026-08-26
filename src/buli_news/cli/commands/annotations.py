"""CLI commands for local-LLM task construction and annotation runs."""

from __future__ import annotations

import argparse
from pathlib import Path

from buli_news.cli.arguments import add_season_argument
from buli_news.news.annotations import (
    DEFAULT_ANNOTATION_CONFIG_PATH,
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OLLAMA_NUM_CTX,
    NEWS_ANNOTATION_PILOT_SELECTION_VERSION,
    annotate_news_tasks,
    build_news_annotation_tasks,
    load_annotation_config,
    select_stratified_annotation_tasks,
)
from buli_news.paths import SeasonPaths
from buli_news.storage import append_jsonl, read_jsonl, write_json, write_jsonl


def register_annotation_commands(
    subparsers: argparse._SubParsersAction,
) -> None:
    build_news_annotation_tasks_parser = subparsers.add_parser(
        "build-news-annotation-tasks",
        help="Build request-bound team-specific local-LLM annotation tasks.",
    )
    add_season_argument(build_news_annotation_tasks_parser)
    build_news_annotation_tasks_parser.add_argument(
        "--config",
        default=str(DEFAULT_ANNOTATION_CONFIG_PATH),
        help="Path to the versioned news annotation schema and prompt.",
    )
    build_news_annotation_tasks_parser.set_defaults(
        command_handler=build_news_annotation_tasks_command,
    )

    annotate_news_parser = subparsers.add_parser(
        "annotate-news",
        help="Annotate team-specific news tasks through a local Ollama server.",
    )
    add_season_argument(annotate_news_parser)
    annotate_news_parser.add_argument(
        "--config",
        default=str(DEFAULT_ANNOTATION_CONFIG_PATH),
        help="Path to the versioned news annotation schema and prompt.",
    )
    annotate_news_parser.add_argument(
        "--model",
        default=DEFAULT_OLLAMA_MODEL,
        help="Exact locally installed Ollama model tag.",
    )
    annotate_news_parser.add_argument(
        "--base-url",
        default=DEFAULT_OLLAMA_BASE_URL,
        help="Base URL of the local Ollama server.",
    )
    annotate_news_parser.add_argument(
        "--output",
        default=None,
        help=(
            "Append-only annotation JSONL. Full runs default to "
            "data/interim/{season}/news/annotations/results.jsonl; pilots "
            "default to their schema-specific pilots directory."
        ),
    )
    annotate_news_parser.add_argument(
        "--failure-output",
        default=None,
        help=(
            "Append-only deferred failures JSONL. Full runs default to "
            "data/interim/{season}/news/annotations/failures.jsonl; pilots "
            "default to their schema-specific pilots directory."
        ),
    )
    annotate_news_parser.add_argument(
        "--task-id",
        default=None,
        help="Annotate only one exact task ID.",
    )
    annotate_news_parser.add_argument(
        "--match-id",
        type=int,
        default=None,
        help=(
            "Annotate every home- and away-team task for one match. Unless "
            "explicitly overridden, artifacts are stored below "
            "annotations/pilots/v{schema_version}/matches/{match_id}."
        ),
    )
    annotate_news_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of not-yet-annotated tasks for this run.",
    )
    annotate_news_parser.add_argument(
        "--pilot-size",
        type=int,
        default=None,
        help=(
            "Select a deterministic stratified pilot across matchdays, teams, "
            "sides, sources, and article lengths. Unless explicitly overridden, "
            "pilot tasks, results, and failures are stored below "
            "annotations/pilots/v{schema_version}."
        ),
    )
    annotate_news_parser.add_argument(
        "--pilot-seed",
        type=int,
        default=42,
        help="Deterministic seed for --pilot-size selection.",
    )
    annotate_news_parser.add_argument(
        "--retry-failures-only",
        action="store_true",
        help="Retry only unresolved failures for the same config and model.",
    )
    annotate_news_parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=300.0,
        help="Per-request Ollama HTTP timeout.",
    )
    annotate_news_parser.add_argument(
        "--num-ctx",
        type=int,
        default=DEFAULT_OLLAMA_NUM_CTX,
        help=(
            "Ollama context-window size used for every article "
            f"(default: {DEFAULT_OLLAMA_NUM_CTX})."
        ),
    )
    annotate_news_parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Concurrent Ollama requests. Use 1 for serial execution; benchmark "
            "2 before using higher values."
        ),
    )
    annotate_news_parser.set_defaults(
        command_handler=annotate_news_command,
    )


def build_news_annotation_tasks_command(args: argparse.Namespace) -> None:
    paths = SeasonPaths(args.season)
    config = load_annotation_config(Path(args.config))
    build = build_news_annotation_tasks(
        matches=read_jsonl(paths.normalized_matches),
        articles=read_jsonl(paths.news_articles),
        article_links=read_jsonl(paths.news_article_links),
        article_content_links=read_jsonl(paths.news_article_content_links),
        contents=read_jsonl(paths.news_contents),
        season=args.season,
        annotation_schema_id=config.schema_id,
    )
    write_jsonl(build.tasks, paths.news_annotation_tasks)
    write_json(build.quality_report, paths.news_annotation_tasks_quality)

    summary = build.quality_report["summary"]
    print(
        f"Saved {len(build.tasks)} team-specific annotation tasks to "
        f"{paths.news_annotation_tasks}"
    )
    print(
        "Collapsed same-content article links inside the same request: "
        f"{summary['collapsed_within_request_article_link_count']}"
    )
    print(
        "Every task retains its original request, match, side, and target team."
    )
    print(
        "Saved annotation-task quality report to "
        f"{paths.news_annotation_tasks_quality}"
    )


def annotate_news_command(args: argparse.Namespace) -> None:
    paths = SeasonPaths(args.season)
    config = load_annotation_config(Path(args.config))
    tasks = read_jsonl(paths.news_annotation_tasks)
    if args.pilot_size is not None:
        if (
            args.task_id is not None
            or args.match_id is not None
            or args.limit is not None
            or args.retry_failures_only
        ):
            msg = (
                "--pilot-size cannot be combined with --task-id, --match-id, "
                "--limit, or --retry-failures-only."
            )
            raise ValueError(msg)
        tasks = select_stratified_annotation_tasks(
            tasks=tasks,
            size=args.pilot_size,
            seed=args.pilot_seed,
        )
        pilot_path = paths.news_annotation_pilot_artifact(
            schema_version=config.schema_version,
            artifact="tasks",
            selection_version=NEWS_ANNOTATION_PILOT_SELECTION_VERSION,
            size=args.pilot_size,
            seed=args.pilot_seed,
        )
        write_jsonl(tasks, pilot_path)
        print(
            f"Saved deterministic stratified pilot with {len(tasks)} tasks to "
            f"{pilot_path}"
        )
        print(
            "Pilot coverage: "
            f"{len({task['matchday'] for task in tasks})} matchdays, "
            f"{len({task['target_team_id'] for task in tasks})} target teams, "
            f"{len({task['source_host'] for task in tasks})} source hosts, "
            f"{len({task['content_id'] for task in tasks})} unique contents."
        )
    elif args.match_id is not None:
        if (
            args.task_id is not None
            or args.limit is not None
            or args.retry_failures_only
        ):
            msg = (
                "--match-id cannot be combined with --task-id, --limit, or "
                "--retry-failures-only."
            )
            raise ValueError(msg)
        tasks = [task for task in tasks if task["match_id"] == args.match_id]
        if not tasks:
            msg = f"Unknown annotation match ID {args.match_id}."
            raise ValueError(msg)
        match_tasks_path = paths.news_annotation_match_pilot_artifact(
            schema_version=config.schema_version,
            match_id=args.match_id,
            artifact="tasks",
        )
        write_jsonl(tasks, match_tasks_path)
        print(
            f"Saved all {len(tasks)} annotation tasks for match {args.match_id} to "
            f"{match_tasks_path}"
        )
        print(
            "Match coverage: "
            f"{sum(task['side'] == 'home' for task in tasks)} home-team tasks, "
            f"{sum(task['side'] == 'away' for task in tasks)} away-team tasks, "
            f"{len({task['source_host'] for task in tasks})} source hosts, "
            f"{len({task['content_id'] for task in tasks})} unique contents."
        )
    if args.output is not None:
        destination = Path(args.output)
    elif args.pilot_size is not None:
        destination = paths.news_annotation_pilot_artifact(
            schema_version=config.schema_version,
            artifact="results",
            selection_version=NEWS_ANNOTATION_PILOT_SELECTION_VERSION,
            size=args.pilot_size,
            seed=args.pilot_seed,
        )
    elif args.match_id is not None:
        destination = paths.news_annotation_match_pilot_artifact(
            schema_version=config.schema_version,
            match_id=args.match_id,
            artifact="results",
        )
    else:
        destination = paths.news_annotations
    if args.failure_output is not None:
        failure_destination = Path(args.failure_output)
    elif args.pilot_size is not None:
        failure_destination = paths.news_annotation_pilot_artifact(
            schema_version=config.schema_version,
            artifact="failures",
            selection_version=NEWS_ANNOTATION_PILOT_SELECTION_VERSION,
            size=args.pilot_size,
            seed=args.pilot_seed,
        )
    elif args.match_id is not None:
        failure_destination = paths.news_annotation_match_pilot_artifact(
            schema_version=config.schema_version,
            match_id=args.match_id,
            artifact="failures",
        )
    else:
        failure_destination = paths.news_annotation_failures
    run = annotate_news_tasks(
        tasks=tasks,
        config=config,
        output_path=destination,
        failure_output_path=failure_destination,
        model=args.model,
        base_url=args.base_url,
        limit=args.limit,
        task_id=args.task_id,
        retry_failures_only=args.retry_failures_only,
        timeout_seconds=args.timeout_seconds,
        num_ctx=args.num_ctx,
        workers=args.workers,
        append_annotation=append_jsonl,
        append_failure=append_jsonl,
    )
    print(f"Resolved local Ollama model digest: {run.model_digest}")
    if run.skipped_count:
        print(f"Skipped {run.skipped_count} existing matching annotations.")
    if run.deferred_failure_count:
        print(
            f"Deferred {run.deferred_failure_count} known failures; use "
            "--retry-failures-only to retry them."
        )
    print(f"Appended {run.annotated_count} validated annotations to {destination}")
    if run.failed_count:
        print(
            f"Appended {run.failed_count} deferred failures to "
            f"{failure_destination} and continued."
        )
