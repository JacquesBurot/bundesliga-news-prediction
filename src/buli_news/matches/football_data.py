"""Client and validation helpers for Football-Data.co.uk CSV files."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from io import StringIO

import httpx


BASE_URL = "https://www.football-data.co.uk/mmz4281"
BUNDESLIGA_DIVISION = "D1"
REQUIRED_COLUMNS = frozenset(
    {
        "Div",
        "Date",
        "HomeTeam",
        "AwayTeam",
        "FTHG",
        "FTAG",
        "FTR",
        "HS",
        "AS",
        "HST",
        "AST",
    }
)


@dataclass(frozen=True)
class FootballDataCsv:
    """Validated raw Football-Data CSV response."""

    content: bytes
    source_url: str
    filename: str
    columns: tuple[str, ...]
    row_count: int


@dataclass(frozen=True)
class FootballDataCsvValidation:
    """Structural information extracted while validating a CSV response."""

    columns: tuple[str, ...]
    row_count: int


def fetch_bundesliga_csv(season: int) -> FootballDataCsv:
    """Fetch and validate one Bundesliga season CSV without changing its bytes."""
    source_url = build_bundesliga_url(season)
    with httpx.Client(timeout=60.0, follow_redirects=True) as client:
        response = client.get(source_url)
        response.raise_for_status()
        validation = validate_football_data_csv(response.content)

    return FootballDataCsv(
        content=response.content,
        source_url=source_url,
        filename=build_bundesliga_filename(season),
        columns=validation.columns,
        row_count=validation.row_count,
    )


def build_bundesliga_url(season: int) -> str:
    """Build the Football-Data URL for a Bundesliga season start year."""
    season_code = build_season_code(season)
    return f"{BASE_URL}/{season_code}/{BUNDESLIGA_DIVISION}.csv"


def build_bundesliga_filename(season: int) -> str:
    """Build a local filename that retains source and season information."""
    return f"{BUNDESLIGA_DIVISION}_{build_season_code(season)}.csv"


def build_season_code(season: int) -> str:
    """Convert 2025 to the Football-Data season code 2526."""
    if isinstance(season, bool) or not isinstance(season, int):
        msg = "Season must be an integer start year."
        raise ValueError(msg)
    if season < 1900 or season > 9998:
        msg = "Season start year must be between 1900 and 9998."
        raise ValueError(msg)

    return f"{season % 100:02d}{(season + 1) % 100:02d}"


def validate_football_data_csv(content: bytes) -> FootballDataCsvValidation:
    """Check that a response is a non-empty CSV with the expected columns."""
    if not content:
        msg = "Football-Data response is empty."
        raise ValueError(msg)

    text = decode_csv(content)
    try:
        reader = csv.DictReader(StringIO(text, newline=""))
        if reader.fieldnames is None:
            msg = "Football-Data CSV has no header row."
            raise ValueError(msg)

        columns = tuple(column.strip() for column in reader.fieldnames)
        if any(not column for column in columns):
            msg = "Football-Data CSV contains an empty column name."
            raise ValueError(msg)
        if len(columns) != len(set(columns)):
            msg = "Football-Data CSV contains duplicate column names."
            raise ValueError(msg)

        missing_columns = sorted(REQUIRED_COLUMNS.difference(columns))
        if missing_columns:
            msg = (
                "Football-Data CSV is missing required columns: "
                f"{', '.join(missing_columns)}."
            )
            raise ValueError(msg)

        row_count = sum(1 for row in reader if not is_empty_row(row))
    except csv.Error as exc:
        msg = f"Football-Data response is not valid CSV: {exc}"
        raise ValueError(msg) from exc

    if row_count == 0:
        msg = "Football-Data CSV contains no match rows."
        raise ValueError(msg)

    return FootballDataCsvValidation(columns=columns, row_count=row_count)


def decode_csv(content: bytes) -> str:
    """Decode current and historical Football-Data CSV encodings."""
    try:
        return content.decode("utf-8-sig")
    except UnicodeDecodeError:
        try:
            return content.decode("cp1252")
        except UnicodeDecodeError as exc:
            msg = "Football-Data CSV is neither UTF-8 nor Windows-1252 encoded."
            raise ValueError(msg) from exc


def is_empty_row(row: dict[str | None, str | list[str] | None]) -> bool:
    """Return whether a CSV row contains no values."""
    return all(
        value is None
        or (isinstance(value, str) and not value.strip())
        or (isinstance(value, list) and not value)
        for value in row.values()
    )
