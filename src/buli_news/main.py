"""Backward-compatible entry point for the Bundesliga news pipeline."""

from buli_news.cli.application import main
from buli_news.cli.errors import NoMatchesError
from buli_news.cli.parser import build_parser

__all__ = ["NoMatchesError", "build_parser", "main"]


if __name__ == "__main__":
    main()
