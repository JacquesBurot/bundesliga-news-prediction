"""Client functions for the OpenLigaDB API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx


BASE_URL = "https://api.openligadb.de"


@dataclass(frozen=True)
class OpenLigaDBMatchData:
    matches: list[dict[str, Any]]
    raw_json: str


def fetch_matchdata(league: str, season: int) -> OpenLigaDBMatchData:
    """Fetch all matches for a league season from OpenLigaDB."""
    url = f"{BASE_URL}/getmatchdata/{league}/{season}"

    with httpx.Client(timeout=30.0) as client:
        response = client.get(url)
        response.raise_for_status()
        data = response.json()

    if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
        msg = "OpenLigaDB response must be a list of match objects."
        raise ValueError(msg)

    return OpenLigaDBMatchData(matches=data, raw_json=response.text)
