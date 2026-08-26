"""Shared command-line argument definitions."""

from __future__ import annotations

import argparse


def add_season_argument(parser: argparse.ArgumentParser) -> None:
    """Add the season start year shared by every pipeline command."""
    parser.add_argument(
        "--season",
        required=True,
        type=int,
        help="Season start year, e.g. 2025 for 2025/26.",
    )
