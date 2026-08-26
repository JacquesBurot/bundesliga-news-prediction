"""Build a canonical numerical match history from the two match sources."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, datetime
from io import StringIO
from typing import Any

from buli_news.matches.football_data import BUNDESLIGA_DIVISION, decode_csv


FOOTBALL_DATA_COLUMNS = frozenset(
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
        "HF",
        "AF",
        "HC",
        "AC",
        "HY",
        "AY",
        "HR",
        "AR",
    }
)
FOOTBALL_DATA_RESULT_FROM_GOALS = {
    -1: "A",
    0: "D",
    1: "H",
}
OPENLIGA_RESULT_TO_TARGET = {
    "HOME_WIN": "H",
    "DRAW": "D",
    "AWAY_WIN": "A",
}


@dataclass(frozen=True)
class FootballDataMatch:
    """One parsed Football-Data match with non-betting statistics only."""

    match_date: date
    home_team: str
    away_team: str
    home_goals: int
    away_goals: int
    result: str
    home_shots: int
    away_shots: int
    home_shots_on_target: int
    away_shots_on_target: int
    home_fouls: int
    away_fouls: int
    home_corners: int
    away_corners: int
    home_yellow_cards: int
    away_yellow_cards: int
    home_red_cards: int
    away_red_cards: int

    @property
    def key(self) -> tuple[date, str, str]:
        """Return the source join key."""
        return (self.match_date, self.home_team, self.away_team)


@dataclass(frozen=True)
class NumericalMatchesBuild:
    """Canonical match records and their source-quality report."""

    matches: list[dict[str, Any]]
    quality_report: dict[str, Any]


def build_numerical_matches(
    openliga_matches: list[dict[str, Any]],
    football_data_content: bytes,
    config: dict[str, Any],
    season: int,
) -> NumericalMatchesBuild:
    """Join both match sources and retain an explicit non-betting schema."""
    if not openliga_matches:
        msg = "OpenLigaDB match input is empty."
        raise ValueError(msg)

    football_data_matches = parse_football_data_matches(football_data_content)
    football_data_names = build_football_data_name_mapping(config)
    football_data_by_key = index_football_data_matches(football_data_matches)

    openliga_ids: set[int] = set()
    openliga_keys: set[tuple[date, str, str]] = set()
    matched_football_data_keys: set[tuple[date, str, str]] = set()
    numerical_matches: list[dict[str, Any]] = []
    result_mismatches: list[dict[str, Any]] = []
    unmatched_openliga_matches: list[dict[str, Any]] = []

    for openliga_match in openliga_matches:
        match_id = require_integer(openliga_match, "match_id", "OpenLigaDB match")
        match_context = f"OpenLigaDB match {match_id}"
        if match_id in openliga_ids:
            msg = f"OpenLigaDB match ID {match_id} occurs more than once."
            raise ValueError(msg)
        openliga_ids.add(match_id)

        match_season = require_integer(openliga_match, "season", match_context)
        if match_season != season:
            msg = (
                f"{match_context} belongs to season {match_season}, "
                f"but season {season} was requested."
            )
            raise ValueError(msg)
        require_text(openliga_match, "league", match_context)
        require_integer(openliga_match, "matchday", match_context)
        require_text(openliga_match, "home_team", match_context)
        require_text(openliga_match, "away_team", match_context)

        kickoff = parse_openliga_kickoff(openliga_match, match_id)
        home_team_id = require_integer(
            openliga_match,
            "home_team_id",
            match_context,
        )
        away_team_id = require_integer(
            openliga_match,
            "away_team_id",
            match_context,
        )
        home_football_data_name = get_football_data_name(
            home_team_id,
            football_data_names,
        )
        away_football_data_name = get_football_data_name(
            away_team_id,
            football_data_names,
        )
        match_key = (
            kickoff.date(),
            home_football_data_name,
            away_football_data_name,
        )

        if match_key in openliga_keys:
            msg = (
                "OpenLigaDB contains the same date, home team, and away team "
                f"more than once: {format_match_key(match_key)}."
            )
            raise ValueError(msg)
        openliga_keys.add(match_key)

        football_data_match = football_data_by_key.get(match_key)
        if football_data_match is None:
            unmatched_openliga_matches.append(
                {
                    "match_id": match_id,
                    "kickoff": kickoff.isoformat(timespec="seconds"),
                    "home_team": openliga_match.get("home_team"),
                    "away_team": openliga_match.get("away_team"),
                    "expected_football_data_key": format_match_key(match_key),
                }
            )
            continue

        matched_football_data_keys.add(match_key)
        mismatch = get_result_mismatch(
            openliga_match=openliga_match,
            football_data_match=football_data_match,
            match_id=match_id,
        )
        if mismatch is not None:
            result_mismatches.append(mismatch)

        numerical_matches.append(
            build_numerical_match(
                openliga_match=openliga_match,
                football_data_match=football_data_match,
            )
        )

    unmatched_football_data_matches = [
        format_match_key(match_key)
        for match_key in sorted(
            set(football_data_by_key).difference(matched_football_data_keys)
        )
    ]
    if unmatched_openliga_matches or unmatched_football_data_matches:
        msg = (
            "OpenLigaDB and Football-Data do not match one-to-one: "
            f"{len(unmatched_openliga_matches)} unmatched OpenLigaDB matches and "
            f"{len(unmatched_football_data_matches)} unmatched Football-Data matches."
        )
        raise ValueError(msg)

    numerical_matches.sort(key=lambda match: (match["kickoff"], match["match_id"]))
    quality_report = {
        "season": season,
        "openligadb_match_count": len(openliga_matches),
        "football_data_match_count": len(football_data_matches),
        "matched_match_count": len(numerical_matches),
        "team_mapping_count": len(football_data_names),
        "result_source": "Football-Data.co.uk",
        "result_mismatch_count": len(result_mismatches),
        "result_mismatches": result_mismatches,
    }
    return NumericalMatchesBuild(
        matches=numerical_matches,
        quality_report=quality_report,
    )


def parse_football_data_matches(content: bytes) -> list[FootballDataMatch]:
    """Parse and validate selected non-betting Football-Data columns."""
    text = decode_csv(content)
    try:
        reader = csv.DictReader(StringIO(text, newline=""))
        if reader.fieldnames is None:
            msg = "Football-Data CSV has no header row."
            raise ValueError(msg)

        stripped_columns = [column.strip() for column in reader.fieldnames]
        if any(not column for column in stripped_columns):
            msg = "Football-Data CSV contains an empty column name."
            raise ValueError(msg)
        if len(stripped_columns) != len(set(stripped_columns)):
            msg = "Football-Data CSV contains duplicate column names."
            raise ValueError(msg)

        columns = set(stripped_columns)
        missing_columns = sorted(FOOTBALL_DATA_COLUMNS.difference(columns))
        if missing_columns:
            msg = (
                "Football-Data CSV is missing numerical match columns: "
                f"{', '.join(missing_columns)}."
            )
            raise ValueError(msg)

        matches = [
            parse_football_data_match(row, row_number)
            for row_number, row in enumerate(reader, start=2)
            if not is_empty_csv_row(row)
        ]
    except csv.Error as exc:
        msg = f"Football-Data response is not valid CSV: {exc}"
        raise ValueError(msg) from exc

    if not matches:
        msg = "Football-Data CSV contains no match rows."
        raise ValueError(msg)
    return matches


def parse_football_data_match(
    row: dict[str | None, str | list[str] | None],
    row_number: int,
) -> FootballDataMatch:
    """Parse one Football-Data row into typed historical observations."""
    context = f"Football-Data row {row_number}"
    division = require_csv_text(row, "Div", context)
    if division != BUNDESLIGA_DIVISION:
        msg = (
            f"{context} has division {division!r}; "
            f"expected {BUNDESLIGA_DIVISION!r}."
        )
        raise ValueError(msg)

    date_text = require_csv_text(row, "Date", context)
    try:
        match_date = datetime.strptime(date_text, "%d/%m/%Y").date()
    except ValueError as exc:
        msg = f"{context} has invalid date {date_text!r}; expected DD/MM/YYYY."
        raise ValueError(msg) from exc

    home_goals = parse_csv_integer(row, "FTHG", context)
    away_goals = parse_csv_integer(row, "FTAG", context)
    result = require_csv_text(row, "FTR", context)
    if result not in {"H", "D", "A"}:
        msg = f"{context} has invalid FTR value {result!r}."
        raise ValueError(msg)

    goal_difference_sign = (home_goals > away_goals) - (home_goals < away_goals)
    expected_result = FOOTBALL_DATA_RESULT_FROM_GOALS[goal_difference_sign]
    if result != expected_result:
        msg = (
            f"{context} has FTR {result!r}, but its full-time goals imply "
            f"{expected_result!r}."
        )
        raise ValueError(msg)

    return FootballDataMatch(
        match_date=match_date,
        home_team=require_csv_text(row, "HomeTeam", context),
        away_team=require_csv_text(row, "AwayTeam", context),
        home_goals=home_goals,
        away_goals=away_goals,
        result=result,
        home_shots=parse_csv_integer(row, "HS", context),
        away_shots=parse_csv_integer(row, "AS", context),
        home_shots_on_target=parse_csv_integer(row, "HST", context),
        away_shots_on_target=parse_csv_integer(row, "AST", context),
        home_fouls=parse_csv_integer(row, "HF", context),
        away_fouls=parse_csv_integer(row, "AF", context),
        home_corners=parse_csv_integer(row, "HC", context),
        away_corners=parse_csv_integer(row, "AC", context),
        home_yellow_cards=parse_csv_integer(row, "HY", context),
        away_yellow_cards=parse_csv_integer(row, "AY", context),
        home_red_cards=parse_csv_integer(row, "HR", context),
        away_red_cards=parse_csv_integer(row, "AR", context),
    )


def build_football_data_name_mapping(config: dict[str, Any]) -> dict[int, str]:
    """Map OpenLigaDB team IDs to their Football-Data names."""
    teams = config.get("teams")
    if not isinstance(teams, list):
        msg = "Team config must contain a teams list."
        raise ValueError(msg)

    mapping: dict[int, str] = {}
    used_names: set[str] = set()
    for team in teams:
        if not isinstance(team, dict):
            msg = "Every team config entry must be an object."
            raise ValueError(msg)

        team_id = team.get("openligadb_team_id")
        football_data_name = team.get("football_data_name")
        if isinstance(team_id, bool) or not isinstance(team_id, int):
            msg = "Every team config entry needs an integer openligadb_team_id."
            raise ValueError(msg)
        if not isinstance(football_data_name, str) or not football_data_name.strip():
            msg = (
                f"Team config entry {team_id} needs a non-empty "
                "football_data_name."
            )
            raise ValueError(msg)
        football_data_name = football_data_name.strip()
        if team_id in mapping:
            msg = f"Team config contains OpenLigaDB team ID {team_id} more than once."
            raise ValueError(msg)
        if football_data_name in used_names:
            msg = (
                "Team config contains Football-Data name "
                f"{football_data_name!r} more than once."
            )
            raise ValueError(msg)

        mapping[team_id] = football_data_name
        used_names.add(football_data_name)

    return mapping


def index_football_data_matches(
    matches: list[FootballDataMatch],
) -> dict[tuple[date, str, str], FootballDataMatch]:
    """Index Football-Data matches by date and mapped team names."""
    indexed_matches: dict[tuple[date, str, str], FootballDataMatch] = {}
    for match in matches:
        if match.key in indexed_matches:
            msg = (
                "Football-Data contains the same date, home team, and away team "
                f"more than once: {format_match_key(match.key)}."
            )
            raise ValueError(msg)
        indexed_matches[match.key] = match
    return indexed_matches


def build_numerical_match(
    openliga_match: dict[str, Any],
    football_data_match: FootballDataMatch,
) -> dict[str, Any]:
    """Build one canonical row without betting-odds columns."""
    return {
        "match_id": openliga_match["match_id"],
        "season": openliga_match["season"],
        "league": openliga_match["league"],
        "matchday": openliga_match["matchday"],
        "kickoff": openliga_match["kickoff"],
        "home_team_id": openliga_match["home_team_id"],
        "home_team": openliga_match["home_team"],
        "away_team_id": openliga_match["away_team_id"],
        "away_team": openliga_match["away_team"],
        "result": football_data_match.result,
        "home_goals": football_data_match.home_goals,
        "away_goals": football_data_match.away_goals,
        "home_shots": football_data_match.home_shots,
        "away_shots": football_data_match.away_shots,
        "home_shots_on_target": football_data_match.home_shots_on_target,
        "away_shots_on_target": football_data_match.away_shots_on_target,
        "home_fouls": football_data_match.home_fouls,
        "away_fouls": football_data_match.away_fouls,
        "home_corners": football_data_match.home_corners,
        "away_corners": football_data_match.away_corners,
        "home_yellow_cards": football_data_match.home_yellow_cards,
        "away_yellow_cards": football_data_match.away_yellow_cards,
        "home_red_cards": football_data_match.home_red_cards,
        "away_red_cards": football_data_match.away_red_cards,
    }


def get_result_mismatch(
    openliga_match: dict[str, Any],
    football_data_match: FootballDataMatch,
    match_id: int,
) -> dict[str, Any] | None:
    """Describe a result disagreement without changing either source."""
    openliga_home_goals = openliga_match.get("home_goals")
    openliga_away_goals = openliga_match.get("away_goals")
    raw_openliga_result = openliga_match.get("result")
    openliga_result = (
        OPENLIGA_RESULT_TO_TARGET.get(raw_openliga_result)
        if isinstance(raw_openliga_result, str)
        else None
    )
    if (
        openliga_home_goals == football_data_match.home_goals
        and openliga_away_goals == football_data_match.away_goals
        and openliga_result == football_data_match.result
    ):
        return None

    return {
        "match_id": match_id,
        "matchday": openliga_match.get("matchday"),
        "kickoff": openliga_match.get("kickoff"),
        "home_team": openliga_match.get("home_team"),
        "away_team": openliga_match.get("away_team"),
        "openligadb": {
            "home_goals": openliga_home_goals,
            "away_goals": openliga_away_goals,
            "result": openliga_result,
        },
        "football_data": {
            "home_goals": football_data_match.home_goals,
            "away_goals": football_data_match.away_goals,
            "result": football_data_match.result,
        },
    }


def get_football_data_name(team_id: int, mapping: dict[int, str]) -> str:
    """Return the Football-Data name for one OpenLigaDB team ID."""
    try:
        return mapping[team_id]
    except KeyError as exc:
        msg = f"No Football-Data team mapping configured for team ID {team_id}."
        raise ValueError(msg) from exc


def parse_openliga_kickoff(match: dict[str, Any], match_id: int) -> datetime:
    """Parse the normalized local OpenLigaDB kickoff timestamp."""
    kickoff = match.get("kickoff")
    if not isinstance(kickoff, str) or not kickoff:
        msg = f"OpenLigaDB match {match_id} has no normalized kickoff."
        raise ValueError(msg)
    try:
        return datetime.fromisoformat(kickoff)
    except ValueError as exc:
        msg = f"OpenLigaDB match {match_id} has invalid kickoff {kickoff!r}."
        raise ValueError(msg) from exc


def require_integer(data: dict[str, Any], key: str, context: str) -> int:
    """Return a required integer while excluding booleans."""
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"{context} needs integer field {key!r}."
        raise ValueError(msg)
    return value


def require_text(data: dict[str, Any], key: str, context: str) -> str:
    """Return a required, non-empty string."""
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        msg = f"{context} needs non-empty field {key!r}."
        raise ValueError(msg)
    return value.strip()


def require_csv_text(
    row: dict[str | None, str | list[str] | None],
    key: str,
    context: str,
) -> str:
    """Return a required, stripped CSV text value."""
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        msg = f"{context} needs non-empty field {key!r}."
        raise ValueError(msg)
    return value.strip()


def parse_csv_integer(
    row: dict[str | None, str | list[str] | None],
    key: str,
    context: str,
) -> int:
    """Parse a required non-negative integer CSV value."""
    value = require_csv_text(row, key, context)
    try:
        parsed = int(value)
    except ValueError as exc:
        msg = f"{context} field {key!r} must be an integer, got {value!r}."
        raise ValueError(msg) from exc
    if parsed < 0:
        msg = f"{context} field {key!r} must not be negative."
        raise ValueError(msg)
    return parsed


def is_empty_csv_row(
    row: dict[str | None, str | list[str] | None],
) -> bool:
    """Return whether a CSV row contains no values."""
    return all(
        value is None
        or (isinstance(value, str) and not value.strip())
        or (isinstance(value, list) and not value)
        for value in row.values()
    )


def format_match_key(match_key: tuple[date, str, str]) -> dict[str, str]:
    """Format an internal match key for errors and quality information."""
    match_date, home_team, away_team = match_key
    return {
        "date": match_date.isoformat(),
        "home_team": home_team,
        "away_team": away_team,
    }
