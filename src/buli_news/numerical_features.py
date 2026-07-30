"""Build leakage-safe numerical pre-match features from match history."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import date, datetime
from math import isfinite
from typing import Any


FORM_WINDOW = 5
TRAIN_END_MATCHDAY = 27
TEST_START_MATCHDAY = 28
LAST_MATCHDAY = 34
ELO_INITIAL_RATING = 1500.0
ELO_K_FACTOR = 20.0
ELO_HOME_ADVANTAGE = 100.0
ELO_RATING_SCALE = 400.0

NUMERICAL_FEATURE_COLUMNS = (
    "home_matches_played",
    "away_matches_played",
    "home_points_per_game",
    "away_points_per_game",
    "home_form_points_last_5",
    "away_form_points_last_5",
    "home_goals_for_last_5_avg",
    "away_goals_for_last_5_avg",
    "home_goals_against_last_5_avg",
    "away_goals_against_last_5_avg",
    "home_shots_for_last_5_avg",
    "away_shots_for_last_5_avg",
    "home_shots_against_last_5_avg",
    "away_shots_against_last_5_avg",
    "home_shots_on_target_for_last_5_avg",
    "away_shots_on_target_for_last_5_avg",
    "home_shots_on_target_against_last_5_avg",
    "away_shots_on_target_against_last_5_avg",
    "home_corners_for_last_5_avg",
    "away_corners_for_last_5_avg",
    "home_corners_against_last_5_avg",
    "away_corners_against_last_5_avg",
    "home_fouls_committed_last_5_avg",
    "away_fouls_committed_last_5_avg",
    "home_yellow_cards_last_5_avg",
    "away_yellow_cards_last_5_avg",
    "home_red_cards_per_game",
    "away_red_cards_per_game",
    "home_venue_matches_played",
    "away_venue_matches_played",
    "home_venue_points_per_game",
    "away_venue_points_per_game",
    "home_days_since_last_match",
    "away_days_since_last_match",
    "elo_difference_before",
)

NUMERICAL_FEATURE_OUTPUT_COLUMNS = (
    "match_id",
    "season",
    "league",
    "matchday",
    "kickoff",
    "home_team_id",
    "home_team",
    "away_team_id",
    "away_team",
    "dataset_split",
    *NUMERICAL_FEATURE_COLUMNS,
    "result",
)


@dataclass(frozen=True)
class TeamMatchObservation:
    """One completed match from one team's perspective."""

    points: int
    goals_for: int
    goals_against: int
    shots_for: int
    shots_against: int
    shots_on_target_for: int
    shots_on_target_against: int
    corners_for: int
    corners_against: int
    fouls_committed: int
    yellow_cards: int
    red_cards: int


@dataclass
class TeamHistory:
    """Mutable state containing completed matches only."""

    matches_played: int = 0
    points: int = 0
    home_matches_played: int = 0
    home_points: int = 0
    away_matches_played: int = 0
    away_points: int = 0
    red_cards: int = 0
    last_match_date: date | None = None
    recent_matches: deque[TeamMatchObservation] = field(
        default_factory=lambda: deque(maxlen=FORM_WINDOW)
    )

    def update(
        self,
        observation: TeamMatchObservation,
        venue: str,
        match_date: date,
    ) -> None:
        """Add one completed match after its pre-match row was created."""
        if self.last_match_date is not None and match_date <= self.last_match_date:
            msg = (
                "A team cannot be updated twice on or before its previous "
                f"match date {self.last_match_date.isoformat()}."
            )
            raise ValueError(msg)
        if venue not in {"home", "away"}:
            msg = f"Unknown venue {venue!r}."
            raise ValueError(msg)

        self.matches_played += 1
        self.points += observation.points
        if venue == "home":
            self.home_matches_played += 1
            self.home_points += observation.points
        else:
            self.away_matches_played += 1
            self.away_points += observation.points

        self.red_cards += observation.red_cards
        self.recent_matches.append(observation)
        self.last_match_date = match_date


@dataclass(frozen=True)
class NumericalFeaturesBuild:
    """Model-ready rows and their fixed chronological split counts."""

    rows: list[dict[str, Any]]
    train_count: int
    test_count: int


def build_numerical_features(
    matches: list[dict[str, Any]],
    season: int,
) -> NumericalFeaturesBuild:
    """Create one feature row per match before updating either team."""
    if not matches:
        msg = "Numerical match history is empty."
        raise ValueError(msg)

    sorted_matches = sorted(
        matches,
        key=lambda match: (
            parse_kickoff(match, match.get("match_id")).isoformat(),
            require_integer(match, "match_id", "Numerical match"),
        ),
    )
    histories: dict[int, TeamHistory] = {}
    elo_ratings: dict[int, float] = {}
    seen_match_ids: set[int] = set()
    feature_rows: list[dict[str, Any]] = []

    for match in sorted_matches:
        parsed = parse_match(match, season=season)
        match_id = parsed["match_id"]
        if match_id in seen_match_ids:
            msg = f"Numerical match ID {match_id} occurs more than once."
            raise ValueError(msg)
        seen_match_ids.add(match_id)

        home_history = histories.setdefault(parsed["home_team_id"], TeamHistory())
        away_history = histories.setdefault(parsed["away_team_id"], TeamHistory())
        home_elo = elo_ratings.setdefault(
            parsed["home_team_id"],
            ELO_INITIAL_RATING,
        )
        away_elo = elo_ratings.setdefault(
            parsed["away_team_id"],
            ELO_INITIAL_RATING,
        )

        row = {
            "match_id": match_id,
            "season": parsed["season"],
            "league": parsed["league"],
            "matchday": parsed["matchday"],
            "kickoff": parsed["kickoff"].isoformat(timespec="seconds"),
            "home_team_id": parsed["home_team_id"],
            "home_team": parsed["home_team"],
            "away_team_id": parsed["away_team_id"],
            "away_team": parsed["away_team"],
            "dataset_split": get_dataset_split(parsed["matchday"]),
            **build_team_features(
                history=home_history,
                prefix="home",
                venue="home",
                match_date=parsed["kickoff"].date(),
            ),
            **build_team_features(
                history=away_history,
                prefix="away",
                venue="away",
                match_date=parsed["kickoff"].date(),
            ),
            "elo_difference_before": round(home_elo - away_elo, 6),
            "result": parsed["result"],
        }
        validate_feature_row(row)
        feature_rows.append(row)

        home_points, away_points = get_match_points(parsed["result"])
        home_history.update(
            observation=TeamMatchObservation(
                points=home_points,
                goals_for=parsed["home_goals"],
                goals_against=parsed["away_goals"],
                shots_for=parsed["home_shots"],
                shots_against=parsed["away_shots"],
                shots_on_target_for=parsed["home_shots_on_target"],
                shots_on_target_against=parsed["away_shots_on_target"],
                corners_for=parsed["home_corners"],
                corners_against=parsed["away_corners"],
                fouls_committed=parsed["home_fouls"],
                yellow_cards=parsed["home_yellow_cards"],
                red_cards=parsed["home_red_cards"],
            ),
            venue="home",
            match_date=parsed["kickoff"].date(),
        )
        away_history.update(
            observation=TeamMatchObservation(
                points=away_points,
                goals_for=parsed["away_goals"],
                goals_against=parsed["home_goals"],
                shots_for=parsed["away_shots"],
                shots_against=parsed["home_shots"],
                shots_on_target_for=parsed["away_shots_on_target"],
                shots_on_target_against=parsed["home_shots_on_target"],
                corners_for=parsed["away_corners"],
                corners_against=parsed["home_corners"],
                fouls_committed=parsed["away_fouls"],
                yellow_cards=parsed["away_yellow_cards"],
                red_cards=parsed["away_red_cards"],
            ),
            venue="away",
            match_date=parsed["kickoff"].date(),
        )
        updated_home_elo, updated_away_elo = update_elo_ratings(
            home_elo=home_elo,
            away_elo=away_elo,
            result=parsed["result"],
        )
        elo_ratings[parsed["home_team_id"]] = updated_home_elo
        elo_ratings[parsed["away_team_id"]] = updated_away_elo

    train_rows = [
        row for row in feature_rows if row["dataset_split"] == "train"
    ]
    test_rows = [
        row for row in feature_rows if row["dataset_split"] == "test"
    ]
    validate_split_order(train_rows, test_rows)
    return NumericalFeaturesBuild(
        rows=feature_rows,
        train_count=len(train_rows),
        test_count=len(test_rows),
    )


def build_team_features(
    history: TeamHistory,
    prefix: str,
    venue: str,
    match_date: date,
) -> dict[str, int | float]:
    """Read a team's state without adding the current match."""
    if venue == "home":
        venue_matches_played = history.home_matches_played
        venue_points = history.home_points
    elif venue == "away":
        venue_matches_played = history.away_matches_played
        venue_points = history.away_points
    else:
        msg = f"Unknown venue {venue!r}."
        raise ValueError(msg)

    recent_matches = list(history.recent_matches)
    days_since_last_match = 0
    if history.last_match_date is not None:
        days_since_last_match = (match_date - history.last_match_date).days
        if days_since_last_match <= 0:
            msg = (
                f"Current match date {match_date.isoformat()} must be after "
                f"the previous match date {history.last_match_date.isoformat()}."
            )
            raise ValueError(msg)

    return {
        f"{prefix}_matches_played": history.matches_played,
        f"{prefix}_points_per_game": safe_average(
            history.points,
            history.matches_played,
        ),
        f"{prefix}_form_points_last_5": sum(
            match.points for match in recent_matches
        ),
        f"{prefix}_goals_for_last_5_avg": safe_average(
            sum(match.goals_for for match in recent_matches),
            len(recent_matches),
        ),
        f"{prefix}_goals_against_last_5_avg": safe_average(
            sum(match.goals_against for match in recent_matches),
            len(recent_matches),
        ),
        f"{prefix}_shots_for_last_5_avg": safe_average(
            sum(match.shots_for for match in recent_matches),
            len(recent_matches),
        ),
        f"{prefix}_shots_against_last_5_avg": safe_average(
            sum(match.shots_against for match in recent_matches),
            len(recent_matches),
        ),
        f"{prefix}_shots_on_target_for_last_5_avg": safe_average(
            sum(match.shots_on_target_for for match in recent_matches),
            len(recent_matches),
        ),
        f"{prefix}_shots_on_target_against_last_5_avg": safe_average(
            sum(match.shots_on_target_against for match in recent_matches),
            len(recent_matches),
        ),
        f"{prefix}_corners_for_last_5_avg": safe_average(
            sum(match.corners_for for match in recent_matches),
            len(recent_matches),
        ),
        f"{prefix}_corners_against_last_5_avg": safe_average(
            sum(match.corners_against for match in recent_matches),
            len(recent_matches),
        ),
        f"{prefix}_fouls_committed_last_5_avg": safe_average(
            sum(match.fouls_committed for match in recent_matches),
            len(recent_matches),
        ),
        f"{prefix}_yellow_cards_last_5_avg": safe_average(
            sum(match.yellow_cards for match in recent_matches),
            len(recent_matches),
        ),
        f"{prefix}_red_cards_per_game": safe_average(
            history.red_cards,
            history.matches_played,
        ),
        f"{prefix}_venue_matches_played": venue_matches_played,
        f"{prefix}_venue_points_per_game": safe_average(
            venue_points,
            venue_matches_played,
        ),
        f"{prefix}_days_since_last_match": days_since_last_match,
    }


def parse_match(match: dict[str, Any], season: int) -> dict[str, Any]:
    """Validate fields needed by the feature builder."""
    match_id = require_integer(match, "match_id", "Numerical match")
    context = f"Numerical match {match_id}"
    match_season = require_integer(match, "season", context)
    if match_season != season:
        msg = (
            f"{context} belongs to season {match_season}, "
            f"but season {season} was requested."
        )
        raise ValueError(msg)

    home_team_id = require_integer(match, "home_team_id", context)
    away_team_id = require_integer(match, "away_team_id", context)
    if home_team_id == away_team_id:
        msg = f"{context} has the same home and away team ID."
        raise ValueError(msg)

    result = require_text(match, "result", context)
    if result not in {"H", "D", "A"}:
        msg = f"{context} has invalid result {result!r}."
        raise ValueError(msg)

    home_goals = require_non_negative_integer(match, "home_goals", context)
    away_goals = require_non_negative_integer(match, "away_goals", context)
    expected_result = get_result_from_goals(home_goals, away_goals)
    if result != expected_result:
        msg = (
            f"{context} has result {result!r}, but its goals imply "
            f"{expected_result!r}."
        )
        raise ValueError(msg)

    return {
        "match_id": match_id,
        "season": match_season,
        "league": require_text(match, "league", context),
        "matchday": require_integer(match, "matchday", context),
        "kickoff": parse_kickoff(match, match_id),
        "home_team_id": home_team_id,
        "home_team": require_text(match, "home_team", context),
        "away_team_id": away_team_id,
        "away_team": require_text(match, "away_team", context),
        "result": result,
        "home_goals": home_goals,
        "away_goals": away_goals,
        "home_shots": require_non_negative_integer(
            match,
            "home_shots",
            context,
        ),
        "away_shots": require_non_negative_integer(
            match,
            "away_shots",
            context,
        ),
        "home_shots_on_target": require_non_negative_integer(
            match,
            "home_shots_on_target",
            context,
        ),
        "away_shots_on_target": require_non_negative_integer(
            match,
            "away_shots_on_target",
            context,
        ),
        "home_corners": require_non_negative_integer(
            match,
            "home_corners",
            context,
        ),
        "away_corners": require_non_negative_integer(
            match,
            "away_corners",
            context,
        ),
        "home_fouls": require_non_negative_integer(
            match,
            "home_fouls",
            context,
        ),
        "away_fouls": require_non_negative_integer(
            match,
            "away_fouls",
            context,
        ),
        "home_yellow_cards": require_non_negative_integer(
            match,
            "home_yellow_cards",
            context,
        ),
        "away_yellow_cards": require_non_negative_integer(
            match,
            "away_yellow_cards",
            context,
        ),
        "home_red_cards": require_non_negative_integer(
            match,
            "home_red_cards",
            context,
        ),
        "away_red_cards": require_non_negative_integer(
            match,
            "away_red_cards",
            context,
        ),
    }


def parse_kickoff(match: dict[str, Any], match_id: Any) -> datetime:
    """Parse one required local kickoff timestamp."""
    kickoff = match.get("kickoff")
    if not isinstance(kickoff, str) or not kickoff:
        msg = f"Numerical match {match_id!r} has no kickoff."
        raise ValueError(msg)
    try:
        parsed = datetime.fromisoformat(kickoff)
    except ValueError as exc:
        msg = f"Numerical match {match_id!r} has invalid kickoff {kickoff!r}."
        raise ValueError(msg) from exc
    if parsed.tzinfo is None:
        msg = f"Numerical match {match_id!r} kickoff needs a timezone offset."
        raise ValueError(msg)
    return parsed


def get_dataset_split(matchday: int) -> str:
    """Return the fixed chronological experiment split."""
    if 1 <= matchday <= TRAIN_END_MATCHDAY:
        return "train"
    if TEST_START_MATCHDAY <= matchday <= LAST_MATCHDAY:
        return "test"
    msg = f"Matchday must be between 1 and {LAST_MATCHDAY}, got {matchday}."
    raise ValueError(msg)


def get_match_points(result: str) -> tuple[int, int]:
    """Convert one H/D/A target into team points."""
    if result == "H":
        return 3, 0
    if result == "D":
        return 1, 1
    if result == "A":
        return 0, 3
    msg = f"Unknown match result {result!r}."
    raise ValueError(msg)


def get_result_from_goals(home_goals: int, away_goals: int) -> str:
    """Derive the H/D/A target from final goals."""
    if home_goals > away_goals:
        return "H"
    if home_goals < away_goals:
        return "A"
    return "D"


def update_elo_ratings(
    home_elo: float,
    away_elo: float,
    result: str,
) -> tuple[float, float]:
    """Update both ratings after a completed match."""
    home_rating_with_advantage = home_elo + ELO_HOME_ADVANTAGE
    rating_difference = home_rating_with_advantage - away_elo
    expected_home = 1.0 / (
        1.0 + 10.0 ** (-rating_difference / ELO_RATING_SCALE)
    )
    if result == "H":
        actual_home = 1.0
    elif result == "D":
        actual_home = 0.5
    elif result == "A":
        actual_home = 0.0
    else:
        msg = f"Unknown match result {result!r}."
        raise ValueError(msg)

    rating_change = ELO_K_FACTOR * (actual_home - expected_home)
    return home_elo + rating_change, away_elo - rating_change


def safe_average(total: int, count: int) -> float:
    """Return a deterministic zero for an unavailable pre-season average."""
    if count == 0:
        return 0.0
    return round(total / count, 6)


def validate_feature_row(row: dict[str, Any]) -> None:
    """Reject missing, extra, non-numeric, or non-finite feature values."""
    expected_columns = set(NUMERICAL_FEATURE_OUTPUT_COLUMNS)
    actual_columns = set(row)
    if actual_columns != expected_columns:
        missing = sorted(expected_columns.difference(actual_columns))
        extra = sorted(actual_columns.difference(expected_columns))
        msg = f"Feature row schema mismatch. Missing: {missing}; extra: {extra}."
        raise ValueError(msg)

    for column in NUMERICAL_FEATURE_COLUMNS:
        value = row[column]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            msg = f"Feature {column!r} must be numeric."
            raise ValueError(msg)
        if not isfinite(value):
            msg = f"Feature {column!r} must be finite."
            raise ValueError(msg)


def validate_split_order(
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
) -> None:
    """Ensure all training kickoffs occur before all test kickoffs."""
    if not train_rows or not test_rows:
        msg = "Both chronological dataset splits must contain matches."
        raise ValueError(msg)

    latest_train_kickoff = max(
        datetime.fromisoformat(row["kickoff"]) for row in train_rows
    )
    earliest_test_kickoff = min(
        datetime.fromisoformat(row["kickoff"]) for row in test_rows
    )
    if latest_train_kickoff >= earliest_test_kickoff:
        msg = (
            "Chronological split is invalid: a training match is not earlier "
            "than every test match."
        )
        raise ValueError(msg)


def require_integer(data: dict[str, Any], key: str, context: str) -> int:
    """Return a required integer while excluding booleans."""
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"{context} needs integer field {key!r}."
        raise ValueError(msg)
    return value


def require_non_negative_integer(
    data: dict[str, Any],
    key: str,
    context: str,
) -> int:
    """Return a required non-negative integer."""
    value = require_integer(data, key, context)
    if value < 0:
        msg = f"{context} field {key!r} must not be negative."
        raise ValueError(msg)
    return value


def require_text(data: dict[str, Any], key: str, context: str) -> str:
    """Return a required, non-empty string."""
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        msg = f"{context} needs non-empty field {key!r}."
        raise ValueError(msg)
    return value.strip()
