"""CLI commands for match acquisition, normalization, and numerical features."""

from __future__ import annotations

import argparse
from pathlib import Path

from buli_news.cli.arguments import add_season_argument
from buli_news.cli.errors import NoMatchesError
from buli_news.matches.features import (
    NUMERICAL_FEATURE_OUTPUT_COLUMNS,
    build_numerical_features,
)
from buli_news.matches.football_data import (
    build_bundesliga_filename,
    fetch_bundesliga_csv,
)
from buli_news.matches.normalization import normalize_openligadb_matches
from buli_news.matches.numerical_matches import build_numerical_matches
from buli_news.matches.openligadb import fetch_matchdata
from buli_news.paths import SeasonPaths
from buli_news.storage import (
    read_bytes,
    read_json,
    read_jsonl,
    write_bytes,
    write_csv,
    write_json,
    write_jsonl,
    write_text,
)


def register_match_commands(subparsers: argparse._SubParsersAction) -> None:
    fetch_openliga_parser = subparsers.add_parser(
        "fetch-openliga",
        help="Fetch raw match data from OpenLigaDB.",
    )
    fetch_openliga_parser.add_argument(
        "--league",
        required=True,
        help="League shortcut, e.g. bl1.",
    )
    add_season_argument(fetch_openliga_parser)
    fetch_openliga_parser.set_defaults(
        command_handler=fetch_openliga_command,
    )

    fetch_football_data_parser = subparsers.add_parser(
        "fetch-football-data",
        help="Fetch raw Bundesliga match statistics from Football-Data.co.uk.",
    )
    add_season_argument(fetch_football_data_parser)
    fetch_football_data_parser.set_defaults(
        command_handler=fetch_football_data_command,
    )

    build_matches_parser = subparsers.add_parser(
        "build-matches",
        help="Build normalized match records from raw OpenLigaDB data.",
    )
    build_matches_parser.add_argument(
        "--league",
        required=True,
        help="League shortcut, e.g. bl1.",
    )
    add_season_argument(build_matches_parser)
    build_matches_parser.add_argument(
        "--timezone",
        default="Europe/Berlin",
        help="Timezone for local kickoff and pre-match windows.",
    )
    build_matches_parser.set_defaults(
        command_handler=build_matches_command,
    )

    build_numerical_matches_parser = subparsers.add_parser(
        "build-numerical-matches",
        help="Join OpenLigaDB metadata with Football-Data match statistics.",
    )
    add_season_argument(build_numerical_matches_parser)
    build_numerical_matches_parser.add_argument(
        "--config",
        default="config/teams.json",
        help="Path to the team mapping config.",
    )
    build_numerical_matches_parser.set_defaults(
        command_handler=build_numerical_matches_command,
    )

    build_numerical_features_parser = subparsers.add_parser(
        "build-numerical-features",
        help="Build leakage-safe numerical pre-match features.",
    )
    add_season_argument(build_numerical_features_parser)
    build_numerical_features_parser.set_defaults(
        command_handler=build_numerical_features_command,
    )


def fetch_openliga_command(args: argparse.Namespace) -> None:
    match_data = fetch_matchdata(
        league=args.league,
        season=args.season,
    )
    if not match_data.matches:
        msg = "No matches returned. Check league shortcut and season."
        raise NoMatchesError(msg)

    output_path = SeasonPaths(args.season).openligadb_raw(args.league)
    write_text(match_data.raw_json, output_path)
    print(f"Saved {len(match_data.matches)} matches to {output_path}")


def fetch_football_data_command(args: argparse.Namespace) -> None:
    football_data = fetch_bundesliga_csv(season=args.season)
    output_path = SeasonPaths(args.season).football_data_raw(
        football_data.filename,
    )
    write_bytes(football_data.content, output_path)
    print(
        f"Saved {football_data.row_count} Football-Data match rows to {output_path}"
    )
    print(f"Source: {football_data.source_url}")


def build_matches_command(args: argparse.Namespace) -> None:
    paths = SeasonPaths(args.season)
    input_path = paths.openligadb_raw(args.league)
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
        league=args.league,
        season=args.season,
        timezone=args.timezone,
    )
    output_path = paths.normalized_matches
    write_jsonl(matches, output_path)
    print(f"Saved {len(matches)} normalized matches to {output_path}")


def build_numerical_matches_command(args: argparse.Namespace) -> None:
    paths = SeasonPaths(args.season)
    matches_path = paths.normalized_matches
    football_data_path = paths.football_data_raw(
        build_bundesliga_filename(args.season),
    )
    config = read_json(Path(args.config))
    openliga_matches = read_jsonl(matches_path)
    football_data_content = read_bytes(football_data_path)
    if not isinstance(config, dict):
        msg = f"{args.config} must contain a JSON object."
        raise ValueError(msg)

    build = build_numerical_matches(
        openliga_matches=openliga_matches,
        football_data_content=football_data_content,
        config=config,
        season=args.season,
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


def build_numerical_features_command(args: argparse.Namespace) -> None:
    paths = SeasonPaths(args.season)
    input_path = paths.numerical_matches
    matches = read_jsonl(input_path)
    build = build_numerical_features(
        matches=matches,
        season=args.season,
    )

    output_path = paths.numerical_features
    write_csv(
        records=build.rows,
        fieldnames=NUMERICAL_FEATURE_OUTPUT_COLUMNS,
        path=output_path,
    )
    print(f"Saved {len(build.rows)} numerical feature rows to {output_path}")
    print(f"Training rows: {build.train_count}")
    print(f"Test rows: {build.test_count}")
