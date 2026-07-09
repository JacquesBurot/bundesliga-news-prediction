"""Command line interface for the Bundesliga news pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import httpx

from buli_news.matches import normalize_openligadb_matches
from buli_news.news_requests import build_news_requests
from buli_news.openligadb import fetch_matchdata
from buli_news.storage import read_json, read_jsonl, write_jsonl, write_text


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

    return parser


def fetch_openliga_command(league: str, season: int) -> None:
    match_data = fetch_matchdata(league=league, season=season)
    if not match_data.matches:
        msg = "No matches returned. Check league shortcut and season."
        raise NoMatchesError(msg)

    output_path = Path("data") / "raw" / "openligadb" / f"{league}_{season}.json"
    write_text(match_data.raw_json, output_path)
    print(f"Saved {len(match_data.matches)} matches to {output_path}")


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


def build_news_requests_command(season: int, config_path: str, lang: str) -> None:
    matches_path = Path("data") / "interim" / f"matches_{season}.jsonl"
    config = read_json(Path(config_path))
    matches = read_jsonl(matches_path)
    requests = build_news_requests(matches=matches, config=config, lang=lang)

    output_path = Path("data") / "interim" / f"news_requests_{season}.jsonl"
    write_jsonl(requests, output_path)
    print(f"Saved {len(requests)} planned news requests to {output_path}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.command == "fetch-openliga":
            fetch_openliga_command(league=args.league, season=args.season)
        elif args.command == "build-matches":
            build_matches_command(
                league=args.league,
                season=args.season,
                timezone=args.timezone,
            )
        elif args.command == "build-news-requests":
            build_news_requests_command(
                season=args.season,
                config_path=args.config,
                lang=args.lang,
            )
    except httpx.HTTPStatusError as exc:
        parser.exit(
            status=1,
            message=(
                f"OpenLigaDB returned HTTP {exc.response.status_code} "
                f"for {exc.request.url}\n"
            ),
        )
    except httpx.HTTPError as exc:
        parser.exit(status=1, message=f"OpenLigaDB request failed: {exc}\n")
    except FileNotFoundError as exc:
        parser.exit(
            status=1,
            message=f"Input file not found: {exc.filename}. Run fetch-openliga first.\n",
        )
    except NoMatchesError as exc:
        parser.exit(status=1, message=f"{exc}\n")
    except ValueError as exc:
        parser.exit(status=1, message=f"Invalid data: {exc}\n")


if __name__ == "__main__":
    main()
