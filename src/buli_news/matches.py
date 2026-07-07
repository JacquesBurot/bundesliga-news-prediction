"""Transform OpenLigaDB raw match data into a flat match dataset."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo


def normalize_openligadb_matches(
    raw_matches: list[dict[str, Any]],
    league: str,
    season: int,
    timezone: str = "Europe/Berlin",
) -> list[dict[str, Any]]:
    """Normalize OpenLigaDB match objects into flat records."""
    return [
        normalize_openligadb_match(
            raw_match=raw_match,
            league=league,
            season=season,
            timezone=timezone,
        )
        for raw_match in raw_matches
    ]


def normalize_openligadb_match(
    raw_match: dict[str, Any],
    league: str,
    season: int,
    timezone: str,
) -> dict[str, Any]:
    """Normalize one OpenLigaDB match object into a flat record."""
    match_id = raw_match.get("matchID")
    kickoff = parse_kickoff(raw_match, timezone=timezone)
    final_result = get_final_result(raw_match)
    home_goals, away_goals = get_goal_values(final_result)

    return {
        "match_id": match_id,
        "season": season,
        "league": league,
        "matchday": get_nested_value(raw_match, "group", "groupOrderID"),
        "kickoff": kickoff.isoformat(timespec="seconds"),
        "home_team": get_nested_value(raw_match, "team1", "teamName"),
        "away_team": get_nested_value(raw_match, "team2", "teamName"),
        "home_team_id": get_nested_value(raw_match, "team1", "teamId"),
        "away_team_id": get_nested_value(raw_match, "team2", "teamId"),
        "home_goals": home_goals,
        "away_goals": away_goals,
        "result": get_match_result(home_goals, away_goals),
        "window_start": (kickoff.date() - timedelta(days=5)).isoformat(),
        "window_end": (kickoff.date() - timedelta(days=1)).isoformat(),
    }


def parse_kickoff(raw_match: dict[str, Any], timezone: str) -> datetime:
    """Parse kickoff time and return it in the configured local timezone."""
    match_datetime_utc = raw_match.get("matchDateTimeUTC")
    if isinstance(match_datetime_utc, str) and match_datetime_utc:
        parsed_utc = datetime.fromisoformat(match_datetime_utc.replace("Z", "+00:00"))
        return parsed_utc.astimezone(ZoneInfo(timezone))

    match_datetime = raw_match.get("matchDateTime")
    if isinstance(match_datetime, str) and match_datetime:
        parsed_local = datetime.fromisoformat(match_datetime)
        return parsed_local.replace(tzinfo=ZoneInfo(timezone))

    match_id = raw_match.get("matchID", "<unknown>")
    msg = f"Match {match_id} has no kickoff timestamp."
    raise ValueError(msg)


def get_final_result(raw_match: dict[str, Any]) -> dict[str, Any] | None:
    """Return the final result object for a finished match."""
    if raw_match.get("matchIsFinished") is not True:
        return None

    match_results = raw_match.get("matchResults", [])
    if not isinstance(match_results, list):
        return None

    for result in match_results:
        if isinstance(result, dict) and result.get("resultTypeID") == 2:
            return result

    for result in match_results:
        if isinstance(result, dict) and result.get("resultName") == "Endergebnis":
            return result

    return None


def get_goal_values(result: dict[str, Any] | None) -> tuple[int | None, int | None]:
    """Extract home and away goals from a final result object."""
    if result is None:
        return None, None

    home_goals = result.get("pointsTeam1")
    away_goals = result.get("pointsTeam2")
    if not isinstance(home_goals, int) or not isinstance(away_goals, int):
        return None, None

    return home_goals, away_goals


def get_match_result(home_goals: int | None, away_goals: int | None) -> str | None:
    """Derive a categorical result label from goal values."""
    if home_goals is None or away_goals is None:
        return None
    if home_goals > away_goals:
        return "HOME_WIN"
    if home_goals < away_goals:
        return "AWAY_WIN"
    return "DRAW"


def get_nested_value(data: dict[str, Any], key: str, nested_key: str) -> Any:
    """Get a nested value from an OpenLigaDB object."""
    nested = data.get(key)
    if not isinstance(nested, dict):
        return None
    return nested.get(nested_key)
