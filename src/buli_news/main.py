"""Command line interface for the Bundesliga news pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import httpx

from buli_news.openligadb import fetch_matchdata
from buli_news.storage import write_text


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

    return parser


def fetch_openliga_command(league: str, season: int) -> None:
    match_data = fetch_matchdata(league=league, season=season)
    if not match_data.matches:
        msg = "No matches returned. Check league shortcut and season."
        raise NoMatchesError(msg)

    output_path = Path("data") / "raw" / "openligadb" / f"{league}_{season}.json"
    write_text(match_data.raw_json, output_path)
    print(f"Saved {len(match_data.matches)} matches to {output_path}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.command == "fetch-openliga":
            fetch_openliga_command(league=args.league, season=args.season)
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
    except NoMatchesError as exc:
        parser.exit(status=1, message=f"{exc}\n")
    except ValueError as exc:
        parser.exit(status=1, message=f"Invalid OpenLigaDB response: {exc}\n")


if __name__ == "__main__":
    main()
