"""Central filesystem paths for season-specific pipeline artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DATA_ROOT = Path("data")
RAW_ROOT = DATA_ROOT / "raw"
INTERIM_ROOT = DATA_ROOT / "interim"
REVIEW_ROOT = DATA_ROOT / "review"
PROCESSED_ROOT = DATA_ROOT / "processed"
MODELING_OUTPUT_ROOT = Path("outputs") / "modeling"


@dataclass(frozen=True)
class SeasonPaths:
    """Resolve every standard pipeline path for one season start year."""

    season: int

    @property
    def interim_root(self) -> Path:
        return INTERIM_ROOT / str(self.season)

    @property
    def matches_interim_dir(self) -> Path:
        return self.interim_root / "matches"

    @property
    def news_interim_dir(self) -> Path:
        return self.interim_root / "news"

    @property
    def normalized_matches(self) -> Path:
        return self.matches_interim_dir / "normalized.jsonl"

    @property
    def numerical_matches(self) -> Path:
        return self.matches_interim_dir / "numerical.jsonl"

    @property
    def numerical_matches_quality(self) -> Path:
        return self.matches_interim_dir / "numerical_quality.json"

    @property
    def news_requests(self) -> Path:
        return self.news_interim_dir / "requests.jsonl"

    @property
    def news_fetch_results(self) -> Path:
        return self.news_interim_dir / "fetch_results.jsonl"

    @property
    def news_articles(self) -> Path:
        return self.news_interim_dir / "articles.jsonl"

    @property
    def news_article_links(self) -> Path:
        return self.news_interim_dir / "article_links.jsonl"

    @property
    def news_articles_quality(self) -> Path:
        return self.news_interim_dir / "articles_quality.json"

    @property
    def news_source_review(self) -> Path:
        return REVIEW_ROOT / str(self.season) / "news_sources.xlsx"

    @property
    def news_raw_dir(self) -> Path:
        return RAW_ROOT / "newsapi" / str(self.season)

    @property
    def numerical_features(self) -> Path:
        return PROCESSED_ROOT / f"numerical_features_{self.season}.csv"

    def openligadb_raw(self, league: str) -> Path:
        return RAW_ROOT / "openligadb" / f"{league}_{self.season}.json"

    def football_data_raw(self, filename: str) -> Path:
        return RAW_ROOT / "football_data" / filename

    def numerical_model_output(self, *parts: str) -> Path:
        root = MODELING_OUTPUT_ROOT / str(self.season) / "numerical"
        return root.joinpath(*parts)
