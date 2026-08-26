"""CLI commands for canonical news articles and exact content grouping."""

from __future__ import annotations

import argparse
from pathlib import Path

from buli_news.cli.arguments import add_season_argument
from buli_news.news.articles import build_news_articles
from buli_news.news.contents import build_news_contents
from buli_news.paths import SeasonPaths
from buli_news.storage import read_json, read_jsonl, write_json, write_jsonl


def register_news_processing_commands(
    subparsers: argparse._SubParsersAction,
) -> None:
    build_news_articles_parser = subparsers.add_parser(
        "build-news-articles",
        help=(
            "Build policy-filtered canonical articles and request-bound "
            "match links."
        ),
    )
    add_season_argument(build_news_articles_parser)
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
    build_news_articles_parser.set_defaults(
        command_handler=build_news_articles_command,
    )

    build_news_contents_parser = subparsers.add_parser(
        "build-news-contents",
        help="Group canonical articles by deterministic normalized text content.",
    )
    add_season_argument(build_news_contents_parser)
    build_news_contents_parser.set_defaults(
        command_handler=build_news_contents_command,
    )


def build_news_articles_command(args: argparse.Namespace) -> None:
    paths = SeasonPaths(args.season)
    requests = read_jsonl(paths.news_requests)
    policy = read_json(Path(args.policy))
    if not isinstance(policy, dict):
        msg = f"{args.policy} must contain a JSON object."
        raise ValueError(msg)
    source_dir = (
        Path(args.raw_dir)
        if args.raw_dir is not None
        else paths.news_raw_dir
    )
    build = build_news_articles(
        requests=requests,
        raw_dir=source_dir,
        policy=policy,
        season=args.season,
        timezone=args.timezone,
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


def build_news_contents_command(args: argparse.Namespace) -> None:
    paths = SeasonPaths(args.season)
    articles = read_jsonl(paths.news_articles)
    build = build_news_contents(
        articles=articles,
        season=args.season,
    )

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
