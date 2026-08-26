"""CLI commands for planning, fetching, and governing news collection."""

from __future__ import annotations

import argparse
from pathlib import Path

from buli_news.cli.arguments import add_season_argument
from buli_news.news.fetch import (
    count_successful_existing_responses,
    fetch_news_requests,
    get_api_key,
    select_requests,
)
from buli_news.news.requests import build_news_requests
from buli_news.news.source_policy import build_news_source_policy
from buli_news.news.source_review import export_news_source_review
from buli_news.paths import SeasonPaths
from buli_news.storage import (
    append_jsonl,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)


def register_news_collection_commands(
    subparsers: argparse._SubParsersAction,
) -> None:
    build_news_requests_parser = subparsers.add_parser(
        "build-news-requests",
        help="Build planned Event Registry article requests from matches.",
    )
    add_season_argument(build_news_requests_parser)
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
    build_news_requests_parser.set_defaults(
        command_handler=build_news_requests_command,
    )

    fetch_news_parser = subparsers.add_parser(
        "fetch-news",
        help="Fetch raw Event Registry article responses from planned requests.",
    )
    add_season_argument(fetch_news_parser)
    fetch_news_parser.add_argument(
        "--request-id",
        help="Fetch only one planned request by request_id.",
    )
    fetch_news_parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of planned requests to fetch. Omit to fetch all.",
    )
    fetch_news_parser.add_argument(
        "--delay-seconds",
        type=float,
        default=1.0,
        help="Pause between API calls to reduce rate-limit risk.",
    )
    fetch_news_parser.set_defaults(
        command_handler=fetch_news_command,
    )

    export_news_source_review_parser = subparsers.add_parser(
        "export-news-source-review",
        help="Export collected news-source homepages for manual legal review.",
    )
    add_season_argument(export_news_source_review_parser)
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
    export_news_source_review_parser.set_defaults(
        command_handler=export_news_source_review_command,
    )

    build_news_source_policy_parser = subparsers.add_parser(
        "build-news-source-policy",
        help="Build a versioned JSON policy from the reviewed source XLSX.",
    )
    add_season_argument(build_news_source_policy_parser)
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
    build_news_source_policy_parser.set_defaults(
        command_handler=build_news_source_policy_command,
    )


def build_news_requests_command(args: argparse.Namespace) -> None:
    paths = SeasonPaths(args.season)
    matches_path = paths.normalized_matches
    config = read_json(Path(args.config))
    matches = read_jsonl(matches_path)
    requests = build_news_requests(
        matches=matches,
        config=config,
        lang=args.lang,
    )

    output_path = paths.news_requests
    write_jsonl(requests, output_path)
    print(f"Saved {len(requests)} planned news requests to {output_path}")


def fetch_news_command(args: argparse.Namespace) -> None:
    paths = SeasonPaths(args.season)
    requests_path = paths.news_requests
    planned_requests = read_jsonl(requests_path)
    selected_requests = select_requests(
        requests=planned_requests,
        request_id=args.request_id,
        limit=args.limit,
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
        delay_seconds=args.delay_seconds,
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


def export_news_source_review_command(args: argparse.Namespace) -> None:
    paths = SeasonPaths(args.season)
    source_dir = (
        Path(args.raw_dir)
        if args.raw_dir is not None
        else paths.news_raw_dir
    )
    destination = (
        Path(args.output)
        if args.output is not None
        else paths.news_source_review
    )
    homepage_counts = export_news_source_review(
        raw_dir=source_dir,
        output_path=destination,
        overwrite=args.overwrite,
    )
    print(f"Read raw news responses from {source_dir}")
    print(f"Exported {len(homepage_counts)} unique sources to {destination}")
    print(
        "Article occurrences before deduplication: "
        f"{sum(count for _, count in homepage_counts)}"
    )


def build_news_source_policy_command(args: argparse.Namespace) -> None:
    paths = SeasonPaths(args.season)
    workbook_path = (
        Path(args.input)
        if args.input is not None
        else paths.news_source_review
    )
    policy = build_news_source_policy(
        workbook_path=workbook_path,
        season=args.season,
        policy_version=args.policy_version,
    )
    destination = Path(args.output)
    write_json(policy, destination)

    summary = policy["summary"]
    print(f"Saved reviewed news source policy to {destination}")
    print(f"Reviewed sources: {policy['source_review']['reviewed_source_count']}")
    print(f"Included sources: {summary['include_source_count']}")
    print(f"Excluded sources: {summary['exclude_source_count']}")
    print("Unknown source decision: exclude")
