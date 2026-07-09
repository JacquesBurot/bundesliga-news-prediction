# bundesliga-news-prediction

Python pipeline for collecting Bundesliga match data and German pre-match news articles as the data basis for a master's thesis in business informatics.

The project investigates whether pre-match news before Bundesliga matches is related to later match outcomes and whether this data can later be used for a prediction model.

## Current Scope

The pipeline currently covers:

1. Fetching raw Bundesliga match data from OpenLigaDB.
2. Normalizing OpenLigaDB matches into a flat JSONL match table.
3. Building planned Event Registry / NewsAPI.ai article requests.
4. Fetching raw Event Registry article responses for selected planned requests.

Feature engineering, model training, LLM scoring, and prediction logic are not part of the current implementation.

## Project Structure

```text
bundesliga-news-prediction/
├── config/
│   └── teams.json
├── data/
│   ├── raw/
│   │   ├── openligadb/
│   │   │   └── .gitkeep
│   │   └── newsapi/
│   │       └── .gitkeep
│   ├── interim/
│   │   └── .gitkeep
│   └── processed/
│       └── .gitkeep
├── src/
│   └── buli_news/
│       ├── __init__.py
│       ├── main.py
│       ├── matches.py
│       ├── news_requests.py
│       ├── newsapi.py
│       ├── openligadb.py
│       └── storage.py
├── .env.example
├── .gitignore
├── .python-version
├── pyproject.toml
├── README.md
└── uv.lock
```

The data directory structure is tracked with `.gitkeep` files. Real data files under `data/raw`, `data/interim`, and `data/processed` are ignored by Git.

## Setup

This project uses `uv`.

```powershell
uv sync
```

Run commands from the repository root.

## Commands

All commands can be run either with the module form:

```powershell
uv run python -m buli_news.main ...
```

or with the console script:

```powershell
uv run buli-news ...
```

### Fetch OpenLigaDB Match Data

```powershell
uv run python -m buli_news.main fetch-openliga --league bl1 --season 2025
```

This calls `https://api.openligadb.de/getmatchdata/bl1/2025` and writes raw JSON to `data/raw/openligadb/bl1_2025.json`.

OpenLigaDB uses the season start year:

```text
2025 = season 2025/26
2024 = season 2024/25
```

If OpenLigaDB returns an empty match list, the command exits with:

```text
No matches returned. Check league shortcut and season.
```

### Build Normalized Matches

```powershell
uv run python -m buli_news.main build-matches --league bl1 --season 2025
```

Input: `data/raw/openligadb/bl1_2025.json`

Output: `data/interim/matches_2025.jsonl`

The normalized match records contain fields such as `match_id`, `kickoff`, `home_team`, `away_team`, `result`, `window_start`, `window_end`, and `window_days`.

The pre-match window is dynamic:

```text
window_end = kickoff_date - 1 day
window_start = max(
    kickoff_date - 5 days,
    previous_home_match_date + 1 day,
    previous_away_match_date + 1 day
)
```

This keeps the window at a maximum of five days and avoids including articles from the day of either team's previous match.

### Build Planned News Requests

```powershell
uv run python -m buli_news.main build-news-requests --season 2025
```

Inputs:

```text
data/interim/matches_2025.jsonl
config/teams.json
```

Output: `data/interim/news_requests_2025.jsonl`

For each match, the command creates two planned requests:

- home team context
- away team context

### Fetch News Responses

Fetch exactly one planned request:

```powershell
uv run python -m buli_news.main fetch-news --season 2025 --request-id bl1_2025_77261_home_team_context
```

Fetch only the first planned request:

```powershell
uv run python -m buli_news.main fetch-news --season 2025 --limit 1
```

Fetch all planned requests:

```powershell
uv run python -m buli_news.main fetch-news --season 2025
```

The command waits one second between requests by default to reduce rate-limit risk:

```text
--delay-seconds 1.0
```

For a slower full run:

```powershell
uv run python -m buli_news.main fetch-news --season 2025 --delay-seconds 2
```

Fetch behavior:

- existing successful raw responses are skipped for resume-safe runs
- failed raw responses are not treated as successful and can be retried
- each successful request is appended immediately to the fetch result log
- HTTP 429 responses are retried up to three times with a 10 second wait

Outputs:

```text
data/raw/newsapi/2025/{request_id}.json
data/interim/news_fetch_results_2025.jsonl
```

Raw API responses are stored unchanged. The fetch result file records technical metadata such as request ID, HTTP status, article count, and raw response path.

## Event Registry Request Design

The current article request payload has this structure:

```json
{
  "query": {
    "$query": {
      "$and": [
        {
          "conceptUri": "http://en.wikipedia.org/wiki/Bundesliga"
        },
        {
          "conceptUri": "http://en.wikipedia.org/wiki/FC_Bayern_Munich"
        },
        {
          "dateStart": "2025-08-17",
          "dateEnd": "2025-08-21",
          "lang": "deu"
        }
      ]
    },
    "$filter": {
      "isDuplicate": "skipDuplicates"
    }
  },
  "resultType": "articles",
  "articlesCount": 100,
  "articlesSortBy": "date",
  "includeSourceDescription": true
}
```

The `apiKey` is added only at request time from `NEWSAPI_KEY`.

The pipeline intentionally fetches only page 1 with up to 100 articles per planned request to stay within the available API request budget.
