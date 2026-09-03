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
    def news_collection_dir(self) -> Path:
        return self.news_interim_dir / "collection"

    @property
    def news_articles_dir(self) -> Path:
        return self.news_interim_dir / "articles"

    @property
    def news_contents_dir(self) -> Path:
        return self.news_interim_dir / "contents"

    @property
    def news_annotations_dir(self) -> Path:
        return self.news_interim_dir / "annotations"

    @property
    def news_annotation_pilots_dir(self) -> Path:
        return self.news_annotations_dir / "pilots"

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
        return self.news_collection_dir / "requests.jsonl"

    @property
    def news_fetch_results(self) -> Path:
        return self.news_collection_dir / "fetch_results.jsonl"

    @property
    def news_articles(self) -> Path:
        return self.news_articles_dir / "articles.jsonl"

    @property
    def news_article_links(self) -> Path:
        return self.news_articles_dir / "request_links.jsonl"

    @property
    def news_articles_quality(self) -> Path:
        return self.news_articles_dir / "quality.json"

    @property
    def news_contents(self) -> Path:
        return self.news_contents_dir / "contents.jsonl"

    @property
    def news_article_content_links(self) -> Path:
        return self.news_contents_dir / "article_links.jsonl"

    @property
    def news_contents_quality(self) -> Path:
        return self.news_contents_dir / "quality.json"

    @property
    def news_annotation_tasks(self) -> Path:
        return self.news_annotations_dir / "tasks.jsonl"

    @property
    def news_annotation_tasks_quality(self) -> Path:
        return self.news_annotations_dir / "tasks_quality.json"

    @property
    def news_annotations(self) -> Path:
        return self.news_annotations_dir / "results.jsonl"

    @property
    def news_annotation_failures(self) -> Path:
        return self.news_annotations_dir / "failures.jsonl"

    def news_annotation_pilot_dir(self, schema_version: int) -> Path:
        return self.news_annotation_pilots_dir / f"v{schema_version}"

    def news_annotation_pilot_artifact(
        self,
        schema_version: int,
        artifact: str,
        selection_version: int,
        size: int,
        seed: int,
    ) -> Path:
        filename = (
            f"{artifact}_selection_v{selection_version}_{size}_seed_{seed}.jsonl"
        )
        return self.news_annotation_pilot_dir(schema_version) / filename

    def news_annotation_match_pilot_artifact(
        self,
        schema_version: int,
        match_id: int,
        artifact: str,
    ) -> Path:
        directory = (
            self.news_annotation_pilot_dir(schema_version)
            / "matches"
            / str(match_id)
        )
        return directory / f"{artifact}.jsonl"

    @property
    def news_source_review(self) -> Path:
        return REVIEW_ROOT / str(self.season) / "news_sources.xlsx"

    @property
    def news_raw_dir(self) -> Path:
        return RAW_ROOT / "newsapi" / str(self.season)

    @property
    def numerical_features(self) -> Path:
        return PROCESSED_ROOT / f"numerical_features_{self.season}.csv"

    @property
    def news_features(self) -> Path:
        return PROCESSED_ROOT / f"news_features_{self.season}.csv"

    @property
    def news_features_quality(self) -> Path:
        return PROCESSED_ROOT / f"news_features_{self.season}_quality.json"

    def openligadb_raw(self, league: str) -> Path:
        return RAW_ROOT / "openligadb" / f"{league}_{self.season}.json"

    def football_data_raw(self, filename: str) -> Path:
        return RAW_ROOT / "football_data" / filename

    def numerical_model_output(self, *parts: str) -> Path:
        root = MODELING_OUTPUT_ROOT / str(self.season) / "numerical"
        return root.joinpath(*parts)

    def news_model_output(self, *parts: str) -> Path:
        root = MODELING_OUTPUT_ROOT / str(self.season) / "news"
        return root.joinpath(*parts)

    def combined_model_output(self, *parts: str) -> Path:
        root = MODELING_OUTPUT_ROOT / str(self.season) / "combined"
        return root.joinpath(*parts)
