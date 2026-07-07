"""Transform OpenLigaDB raw match data into a flat match dataset."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo


def normalize_openligadb_matches(
    raw_matches: list[dict[str, Any]],
    league: str,
    season: int,
    timezone: str = "Europe/Berlin",
) -> list[dict[str, Any]]:
    """Normalize OpenLigaDB match objects into flat records."""
    previous_match_dates = get_previous_match_dates(
        raw_matches=raw_matches,
        timezone=timezone,
    )

    return [
        normalize_openligadb_match(
            raw_match=raw_match,
            league=league,
            season=season,
            timezone=timezone,
            previous_home_match_date=previous_match_dates[index][0],
            previous_away_match_date=previous_match_dates[index][1],
        )
        for index, raw_match in enumerate(raw_matches)
    ]


def normalize_openligadb_match(
    raw_match: dict[str, Any],
    league: str,
    season: int,
    timezone: str,
    previous_home_match_date: date | None,
    previous_away_match_date: date | None,
) -> dict[str, Any]:
    """Normalize one OpenLigaDB match object into a flat record."""
    match_id = raw_match.get("matchID")
    kickoff = parse_kickoff(raw_match, timezone=timezone)
    final_result = get_final_result(raw_match)
    home_goals, away_goals = get_goal_values(final_result)
    window_start, window_end = get_pre_match_window(
        kickoff_date=kickoff.date(),
        previous_home_match_date=previous_home_match_date,
        previous_away_match_date=previous_away_match_date,
    )

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
        "previous_home_match_date": format_date(previous_home_match_date),
        "previous_away_match_date": format_date(previous_away_match_date),
        "window_start": format_date(window_start),
        "window_end": window_end.isoformat(),
        "window_days": get_window_days(window_start, window_end),
    }


def get_previous_match_dates(
    raw_matches: list[dict[str, Any]],
    timezone: str,
) -> dict[int, tuple[date | None, date | None]]:
    """Find each team's previous match date before every fixture."""
    matches_by_kickoff = sorted(
        (
            (index, raw_match, parse_kickoff(raw_match, timezone=timezone))
            for index, raw_match in enumerate(raw_matches)
        ),
        key=lambda item: item[2],
    )
    previous_match_dates: dict[int, tuple[date | None, date | None]] = {}
    last_match_date_by_team: dict[int, date] = {}

    for index, raw_match, kickoff in matches_by_kickoff:
        home_team_id = get_nested_value(raw_match, "team1", "teamId")
        away_team_id = get_nested_value(raw_match, "team2", "teamId")

        previous_home_match_date = get_previous_team_match_date(
            team_id=home_team_id,
            last_match_date_by_team=last_match_date_by_team,
        )
        previous_away_match_date = get_previous_team_match_date(
            team_id=away_team_id,
            last_match_date_by_team=last_match_date_by_team,
        )
        previous_match_dates[index] = (
            previous_home_match_date,
            previous_away_match_date,
        )

        match_date = kickoff.date()
        if isinstance(home_team_id, int):
            last_match_date_by_team[home_team_id] = match_date
        if isinstance(away_team_id, int):
            last_match_date_by_team[away_team_id] = match_date

    return previous_match_dates


def get_previous_team_match_date(
    team_id: Any,
    last_match_date_by_team: dict[int, date],
) -> date | None:
    """Return the previous match date for one team ID."""
    if not isinstance(team_id, int):
        return None
    return last_match_date_by_team.get(team_id)


def get_pre_match_window(
    kickoff_date: date,
    previous_home_match_date: date | None,
    previous_away_match_date: date | None,
    max_window_days: int = 5,
) -> tuple[date | None, date]:
    """Build a pre-match window after both teams' previous matches."""
    window_end = kickoff_date - timedelta(days=1)
    window_start_candidates = [kickoff_date - timedelta(days=max_window_days)]

    if previous_home_match_date is not None:
        window_start_candidates.append(previous_home_match_date + timedelta(days=1))
    if previous_away_match_date is not None:
        window_start_candidates.append(previous_away_match_date + timedelta(days=1))

    window_start = max(window_start_candidates)
    if window_start > window_end:
        return None, window_end

    return window_start, window_end


def get_window_days(window_start: date | None, window_end: date) -> int:
    """Return the inclusive number of days in a pre-match window."""
    if window_start is None:
        return 0
    return (window_end - window_start).days + 1


def format_date(value: date | None) -> str | None:
    """Format an optional date value for JSON output."""
    if value is None:
        return None
    return value.isoformat()


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
