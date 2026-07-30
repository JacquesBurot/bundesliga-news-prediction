"""Command line interface for the Bundesliga news pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import httpx

from buli_news.football_data import build_bundesliga_filename, fetch_bundesliga_csv
from buli_news.matches import normalize_openligadb_matches
from buli_news.newsapi import (
    count_successful_existing_responses,
    fetch_news_requests,
    get_api_key,
    select_requests,
)
from buli_news.news_requests import build_news_requests
from buli_news.numerical_features import (
    NUMERICAL_FEATURE_OUTPUT_COLUMNS,
    build_numerical_features,
)
from buli_news.numerical_matches import build_numerical_matches
from buli_news.openligadb import fetch_matchdata
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

    output_path = Path("data") / "raw" / "openligadb" / f"{league}_{season}.json"
    write_text(match_data.raw_json, output_path)
    print(f"Saved {len(match_data.matches)} matches to {output_path}")


def fetch_football_data_command(season: int) -> None:
    football_data = fetch_bundesliga_csv(season=season)
    output_path = Path("data") / "raw" / "football_data" / football_data.filename
    write_bytes(football_data.content, output_path)
    print(
        f"Saved {football_data.row_count} Football-Data match rows to {output_path}"
    )
    print(f"Source: {football_data.source_url}")


def build_matches_command(league: str, season: int, timezone: str) -> None:
    input_path = Path("data") / "raw" / "openligadb" / f"{league}_{season}.json"
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
    output_path = Path("data") / "interim" / f"matches_{season}.jsonl"
    write_jsonl(matches, output_path)
    print(f"Saved {len(matches)} normalized matches to {output_path}")


def build_numerical_matches_command(season: int, config_path: str) -> None:
    matches_path = Path("data") / "interim" / f"matches_{season}.jsonl"
    football_data_path = (
        Path("data")
        / "raw"
        / "football_data"
        / build_bundesliga_filename(season)
    )
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
    output_path = (
        Path("data") / "interim" / f"numerical_matches_{season}.jsonl"
    )
    quality_path = (
        Path("data")
        / "interim"
        / f"numerical_matches_{season}_quality.json"
    )
    write_jsonl(build.matches, output_path)
    write_json(build.quality_report, quality_path)
    print(f"Saved {len(build.matches)} numerical matches to {output_path}")
    print(f"Saved source quality report to {quality_path}")
    print(
        "Result mismatches between sources: "
        f"{build.quality_report['result_mismatch_count']}"
    )


def build_numerical_features_command(season: int) -> None:
    input_path = (
        Path("data") / "interim" / f"numerical_matches_{season}.jsonl"
    )
    matches = read_jsonl(input_path)
    build = build_numerical_features(matches=matches, season=season)

    output_path = (
        Path("data") / "processed" / f"numerical_features_{season}.csv"
    )
    write_csv(
        records=build.rows,
        fieldnames=NUMERICAL_FEATURE_OUTPUT_COLUMNS,
        path=output_path,
    )
    print(f"Saved {len(build.rows)} numerical feature rows to {output_path}")
    print(f"Training rows: {build.train_count}")
    print(f"Test rows: {build.test_count}")


def build_news_requests_command(season: int, config_path: str, lang: str) -> None:
    matches_path = Path("data") / "interim" / f"matches_{season}.jsonl"
    config = read_json(Path(config_path))
    matches = read_jsonl(matches_path)
    requests = build_news_requests(matches=matches, config=config, lang=lang)

    output_path = Path("data") / "interim" / f"news_requests_{season}.jsonl"
    write_jsonl(requests, output_path)
    print(f"Saved {len(requests)} planned news requests to {output_path}")


def fetch_news_command(
    season: int,
    request_id: str | None,
    limit: int | None,
    delay_seconds: float,
) -> None:
    requests_path = Path("data") / "interim" / f"news_requests_{season}.jsonl"
    planned_requests = read_jsonl(requests_path)
    selected_requests = select_requests(
        requests=planned_requests,
        request_id=request_id,
        limit=limit,
    )
    if not selected_requests:
        msg = f"No planned requests found in {requests_path}."
        raise ValueError(msg)

    output_dir = Path("data") / "raw" / "newsapi" / str(season)
    results_path = Path("data") / "interim" / f"news_fetch_results_{season}.jsonl"
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
