"""Root parser for the Bundesliga news pipeline."""

from __future__ import annotations

import argparse

from buli_news.cli.commands.annotations import register_annotation_commands
from buli_news.cli.commands.matches import register_match_commands
from buli_news.cli.commands.modeling import register_modeling_commands
from buli_news.cli.commands.news_collection import register_news_collection_commands
from buli_news.cli.commands.news_features import register_news_feature_commands
from buli_news.cli.commands.news_processing import register_news_processing_commands


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="buli-news")
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    register_match_commands(subparsers)
    register_news_collection_commands(subparsers)
    register_news_processing_commands(subparsers)
    register_annotation_commands(subparsers)
    register_news_feature_commands(subparsers)
    register_modeling_commands(subparsers)

    return parser
