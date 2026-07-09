"""Build planned Event Registry article requests from normalized matches."""

from __future__ import annotations

from typing import Any


EVENT_REGISTRY_ARTICLES_ENDPOINT = "https://eventregistry.org/api/v1/article/getArticles"


def build_news_requests(
    matches: list[dict[str, Any]],
    config: dict[str, Any],
    request_type: str = "team_context",
    lang: str = "deu",
) -> list[dict[str, Any]]:
    """Build planned team-context article requests for normalized matches."""
    if request_type != "team_context":
        msg = f"Unsupported request type: {request_type}"
        raise ValueError(msg)

    league_concepts = get_league_concepts(config)
    team_concepts = get_team_concepts(config)
    requests = []

    for match in matches:
        validate_match_window(match)
        league = get_required_str(match, "league")
        league_concept_uri = league_concepts.get(league)
        if league_concept_uri is None:
            msg = f"No league concept URI configured for league {league!r}."
            raise ValueError(msg)

        requests.append(
            build_team_context_request(
                match=match,
                side="home",
                team_field="home_team",
                team_id_field="home_team_id",
                league_concept_uri=league_concept_uri,
                team_concepts=team_concepts,
                lang=lang,
            )
        )
        requests.append(
            build_team_context_request(
                match=match,
                side="away",
                team_field="away_team",
                team_id_field="away_team_id",
                league_concept_uri=league_concept_uri,
                team_concepts=team_concepts,
                lang=lang,
            )
        )

    return requests


def build_team_context_request(
    match: dict[str, Any],
    side: str,
    team_field: str,
    team_id_field: str,
    league_concept_uri: str,
    team_concepts: dict[int, str],
    lang: str,
) -> dict[str, Any]:
    """Build one planned team-context request for a match side."""
    match_id = get_required_int(match, "match_id")
    season = get_required_int(match, "season")
    league = get_required_str(match, "league")
    team = get_required_str(match, team_field)
    team_id = get_required_int(match, team_id_field)
    team_concept_uri = team_concepts.get(team_id)
    if team_concept_uri is None:
        msg = f"No team concept URI configured for team ID {team_id} ({team})."
        raise ValueError(msg)

    concept_uris = [league_concept_uri, team_concept_uri]
    date_start = get_required_str(match, "window_start")
    date_end = get_required_str(match, "window_end")

    return {
        "request_id": f"{league}_{season}_{match_id}_{side}_team_context",
        "match_id": match_id,
        "season": season,
        "league": league,
        "side": side,
        "request_type": "team_context",
        "team": team,
        "team_id": team_id,
        "concept_uris": concept_uris,
        "date_start": date_start,
        "date_end": date_end,
        "lang": lang,
        "endpoint": EVENT_REGISTRY_ARTICLES_ENDPOINT,
        "payload": build_event_registry_payload(
            concept_uris=concept_uris,
            date_start=date_start,
            date_end=date_end,
            lang=lang,
        ),
    }


def build_event_registry_payload(
    concept_uris: list[str],
    date_start: str,
    date_end: str,
    lang: str,
) -> dict[str, Any]:
    """Build an Event Registry article request payload without an API key."""
    return {
        "query": {
            "$query": {
                "$and": [
                    *({"conceptUri": concept_uri} for concept_uri in concept_uris),
                    {
                        "dateStart": date_start,
                        "dateEnd": date_end,
                        "lang": lang,
                    },
                ]
            },
            "$filter": {
                "isDuplicate": "skipDuplicates",
            },
        },
        "resultType": "articles",
        "articlesCount": 100,
        "articlesSortBy": "date",
        "includeSourceDescription": True,
    }


def get_league_concepts(config: dict[str, Any]) -> dict[str, str]:
    """Extract league concept URIs from the request config."""
    leagues = config.get("leagues")
    if not isinstance(leagues, dict):
        msg = "Config must contain a 'leagues' object."
        raise ValueError(msg)

    concepts = {}
    for league, values in leagues.items():
        if isinstance(league, str) and isinstance(values, dict):
            concept_uri = values.get("concept_uri")
            if isinstance(concept_uri, str):
                concepts[league] = concept_uri
    return concepts


def get_team_concepts(config: dict[str, Any]) -> dict[int, str]:
    """Extract team concept URIs keyed by OpenLigaDB team ID."""
    teams = config.get("teams")
    if not isinstance(teams, list):
        msg = "Config must contain a 'teams' list."
        raise ValueError(msg)

    concepts = {}
    for team in teams:
        if not isinstance(team, dict):
            continue
        team_id = team.get("openligadb_team_id")
        concept_uri = team.get("concept_uri")
        if isinstance(team_id, int) and isinstance(concept_uri, str):
            concepts[team_id] = concept_uri
    return concepts


def validate_match_window(match: dict[str, Any]) -> None:
    """Ensure a match has a usable pre-match date window."""
    if match.get("window_start") is None or match.get("window_end") is None:
        match_id = match.get("match_id", "<unknown>")
        msg = f"Match {match_id} has no usable pre-match window."
        raise ValueError(msg)


def get_required_int(data: dict[str, Any], key: str) -> int:
    """Read a required integer field."""
    value = data.get(key)
    if not isinstance(value, int):
        msg = f"Field {key!r} must be an integer."
        raise ValueError(msg)
    return value


def get_required_str(data: dict[str, Any], key: str) -> str:
    """Read a required string field."""
    value = data.get(key)
    if not isinstance(value, str) or not value:
        msg = f"Field {key!r} must be a non-empty string."
        raise ValueError(msg)
    return value
