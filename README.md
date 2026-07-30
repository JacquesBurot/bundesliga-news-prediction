# bundesliga-news-prediction

Python pipeline for a master's thesis investigating whether structured
pre-match sports-news features improve Bundesliga match-outcome predictions
beyond numerical pre-match data alone.

## Current Scope

The implemented pipeline currently supports:

1. downloading raw Bundesliga match metadata from OpenLigaDB
2. downloading raw match statistics from Football-Data.co.uk
3. normalizing OpenLigaDB match metadata
4. joining both match sources into a canonical numerical match history
5. calculating leakage-safe numerical pre-match features
6. planning and fetching German pre-match news from Event Registry / NewsAPI.ai

Numerical model training, local LLM annotation, news-feature aggregation, and
the final model comparison are the next planned stages.

## Experiment Design

The main experiment compares two variants on exactly the same matches:

1. numerical pre-match features only
2. the same numerical features plus structured news features extracted by a
   local LLM

The target classes are:

```text
H = home win
D = draw
A = away win
```

The fixed chronological split for season 2025/26 is:

```text
Training: matchdays 1-27 (243 matches)
Test:     matchdays 28-34 (63 matches)
```

No random split is used. Planned models are a `DummyClassifier` reference and
multinomial logistic regression. Evaluation uses Log Loss, Accuracy, Macro-F1,
multiclass Brier Score, and a Confusion Matrix.

The local LLM will act as a feature extractor, not as the match predictor:

```text
news article -> local LLM -> structured article annotation
             -> aggregated pre-match news features
```

Both logistic-regression variants will use the same preprocessing, target,
training rows, test rows, and evaluation code. Only the feature columns differ.

## Data Flow

```text
OpenLigaDB raw --------> normalized match metadata --+
                                                     +--> numerical match history
Football-Data raw -----> non-betting match stats ----+             |
                                                                   v
                                               numerical pre-match features

NewsAPI.ai raw --------> normalized articles -> local LLM
                                             -> news features

numerical features -------------------------> model A
numerical features + news features ---------> model B
```

The data directories represent processing stages:

- `data/raw`: unchanged responses from external sources
- `data/interim`: normalized, joined, or annotated source data
- `data/processed`: model-ready pre-match feature tables

Real data files are ignored by Git. Only `.gitkeep` files preserve the directory
structure in the repository.

## Leakage Rules

Every prediction row must contain only information available before that
fixture. Numerical features are created in chronological order:

```text
1. read both teams' existing histories
2. create the current match's pre-match features
3. store the current result as the target
4. update both histories with the completed match
```

Current-match goals, shots, cards, and the result never enter that match's
predictors. They are retained in the interim history only to calculate features
for later fixtures.

For test matchdays, earlier completed test matches may inform later test
fixtures, because those results would be known at the later kickoff. The model
itself is not retrained on test rows.

News articles must be published inside their pre-match windows. Results,
post-match articles, and other information unavailable at prediction time must
never be passed to the local LLM or the prediction model.

## Project Structure

```text
bundesliga-news-prediction/
├── config/
│   └── teams.json
├── data/
│   ├── raw/
│   │   ├── football_data/
│   │   ├── openligadb/
│   │   └── newsapi/
│   ├── interim/
│   └── processed/
├── scripts/
│   └── export_news_source_homepages.py
├── src/
│   └── buli_news/
│       ├── __init__.py
│       ├── football_data.py
│       ├── main.py
│       ├── matches.py
│       ├── news_requests.py
│       ├── newsapi.py
│       ├── numerical_features.py
│       ├── numerical_matches.py
│       ├── openligadb.py
│       └── storage.py
├── .env.example
├── .gitignore
├── .python-version
├── pyproject.toml
├── README.md
└── uv.lock
```

## Setup

The project uses `uv` and Python 3.13:

```console
uv sync --locked
```

Run all commands from the repository root. Commands are available in module
form:

```console
uv run python -m buli_news.main ...
```

or through the console script:

```console
uv run buli-news ...
```

## Numerical Data Pipeline

### 1. Fetch OpenLigaDB Match Data

```console
uv run python -m buli_news.main fetch-openliga --league bl1 --season 2025
```

Source:

```text
https://api.openligadb.de/getmatchdata/bl1/2025
```

Output:

```text
data/raw/openligadb/bl1_2025.json
```

OpenLigaDB uses the season start year: `2025` represents season 2025/26.

### 2. Fetch Football-Data Match Statistics

```console
uv run python -m buli_news.main fetch-football-data --season 2025
```

Source:

```text
https://www.football-data.co.uk/mmz4281/2526/D1.csv
```

Output:

```text
data/raw/football_data/D1_2526.csv
```

The response bytes are stored unchanged. Download validation is deliberately
structural: the response must be a non-empty UTF-8 or Windows-1252 CSV with a
valid header, at least one data row, unique column names, and the required core
columns.

The raw CSV also contains betting odds. They remain in the unchanged raw file
for provenance but are excluded from every derived numerical table through
explicit column lists.

### 3. Build Normalized OpenLigaDB Matches

```console
uv run python -m buli_news.main build-matches --league bl1 --season 2025
```

Input:

```text
data/raw/openligadb/bl1_2025.json
```

Output:

```text
data/interim/matches_2025.jsonl
```

This table contains match IDs, matchdays, local kickoff timestamps, canonical
team names, team IDs, results, and news-window metadata.

The pre-match news window is:

```text
window_end = kickoff_date - 1 day
window_start = max(
    kickoff_date - 5 days,
    previous_home_match_date + 1 day,
    previous_away_match_date + 1 day
)
```

### 4. Build Numerical Match History

```console
uv run python -m buli_news.main build-numerical-matches --season 2025
```

Inputs:

```text
data/interim/matches_2025.jsonl
data/raw/football_data/D1_2526.csv
config/teams.json
```

Outputs:

```text
data/interim/numerical_matches_2025.jsonl
data/interim/numerical_matches_2025_quality.json
```

Team mappings connect exact Football-Data names to OpenLigaDB team IDs. The
sources must match one-to-one by local date, home team, and away team.

Source responsibilities:

- OpenLigaDB: match ID, matchday, kickoff, canonical team names, and team IDs
- Football-Data: target, goals, shots, shots on target, fouls, corners, and
  cards

Football-Data is the canonical result source. Differences from OpenLigaDB are
recorded in the quality report rather than silently hidden. In the current
2025/26 data, OpenLigaDB reports match ID `77546` as 0-1, while Football-Data
and the
[official Bundesliga result](https://www.bundesliga.com/de/bundesliga/news/1-fsv-mainz-05-1-fc-union-berlin-spieltag-33-spielbericht-highlights-37326)
are 1-3.

The interim statistics describe completed matches. They are historical
observations, not direct model inputs for the same fixture.

### 5. Build Numerical Pre-Match Features

```console
uv run python -m buli_news.main build-numerical-features --season 2025
```

Input:

```text
data/interim/numerical_matches_2025.jsonl
```

Output:

```text
data/processed/numerical_features_2025.csv
```

Each output row contains match metadata, the fixed `dataset_split`, the target
`result`, and 35 numerical predictors.

Feature-name conventions:

- `home_*` describes the current home team before kickoff.
- `away_*` describes the current away team before kickoff.
- `for` is the observed team's value; `against` is its opponent's value.
- `last_5` uses up to the five most recent completed Bundesliga matches,
  regardless of their venues.
- `avg` divides by the number of available prior matches, not always by five.
- `per_game` uses all prior Bundesliga matches in the current season.

### Numerical Feature Catalog

| Feature columns | Meaning and calculation |
| --- | --- |
| `home_matches_played`, `away_matches_played` | Number of completed season matches before the current fixture. |
| `home_points_per_game`, `away_points_per_game` | All prior season points divided by prior matches, using 3 points for a win, 1 for a draw, and 0 for a loss. |
| `home_form_points_last_5`, `away_form_points_last_5` | Sum of points from up to five most recent matches. After five matches, the range is 0 to 15. |
| `home_goals_for_last_5_avg`, `away_goals_for_last_5_avg` | Average goals scored by the team over its last five matches. |
| `home_goals_against_last_5_avg`, `away_goals_against_last_5_avg` | Average goals scored by the opponents over the team's last five matches. |
| `home_shots_for_last_5_avg`, `away_shots_for_last_5_avg` | Average total shots taken by the team over its last five matches. |
| `home_shots_against_last_5_avg`, `away_shots_against_last_5_avg` | Average total shots allowed to opponents over the team's last five matches. |
| `home_shots_on_target_for_last_5_avg`, `away_shots_on_target_for_last_5_avg` | Average shots on target by the team over its last five matches. |
| `home_shots_on_target_against_last_5_avg`, `away_shots_on_target_against_last_5_avg` | Average opponent shots on target over the team's last five matches. |
| `home_corners_for_last_5_avg`, `away_corners_for_last_5_avg` | Average corners won by the team over its last five matches. |
| `home_corners_against_last_5_avg`, `away_corners_against_last_5_avg` | Average corners won by opponents over the team's last five matches. |
| `home_fouls_committed_last_5_avg`, `away_fouls_committed_last_5_avg` | Average fouls committed by the team over its last five matches. |
| `home_yellow_cards_last_5_avg`, `away_yellow_cards_last_5_avg` | Average yellow cards received by the team over its last five matches. |
| `home_red_cards_per_game`, `away_red_cards_per_game` | All prior red cards in the current season divided by prior matches. A season-to-date rate is used because red cards are rare. |
| `home_venue_matches_played`, `away_venue_matches_played` | Prior home matches of the current home team and prior away matches of the current away team. |
| `home_venue_points_per_game`, `away_venue_points_per_game` | Prior home points divided by home matches for the current home team, and prior away points divided by away matches for the current away team. |
| `home_days_since_last_match`, `away_days_since_last_match` | Calendar-date difference between the current fixture and the team's previous Bundesliga match. Cup and international matches are not included. |
| `elo_difference_before` | Home-team Elo minus away-team Elo immediately before kickoff. The detailed update is described below. |

Only the explicitly defined numerical feature columns will later form `X`.
Match IDs, team IDs, team names, kickoff, matchday, `dataset_split`, and
`result` are retained for assignment and auditing but are not predictors.

All rolling averages are rounded to six decimal places. When no prior
observation exists, the affected history value is `0`; the accompanying match
count is also `0`.

### Elo Calculation

Every team starts the season with an Elo rating of 1500. The constants are fixed
before model evaluation and are not tuned against the test set:

```text
initial rating: 1500
K-factor:       20
home advantage: 100 rating points
rating scale:   400
actual score:   1.0 home win, 0.5 draw, 0.0 away win
```

Before a match, the expected home-team score is:

```text
expected_home =
    1 / (1 + 10 ** (-((home_elo + 100 - away_elo) / 400)))
```

The home advantage of 100 rating points is used only in this expectation. The
actual home-team score is `1.0` for a home win, `0.5` for a draw, and `0.0` for
an away win. After the match:

```text
rating_change = 20 * (actual_home - expected_home)

new_home_elo = home_elo + rating_change
new_away_elo = away_elo - rating_change
```

The feature row stores:

```text
elo_difference_before = home_elo - away_elo
```

This value is created before the current match update. A positive value means
that the home team has the higher pre-match rating, while a negative value means
that the away team has the higher rating. The 100-point home advantage is not
added to the stored difference.

Both team ratings are updated only after all current pre-match features have
been stored. The current result can therefore affect future fixtures but never
its own feature row. Absolute Elo ratings are omitted because their difference
already represents relative team strength and avoids redundant model columns.

The first match for every team uses deterministic zero values and a
`matches_played` value of zero. This makes unavailable early-season history
explicit without learning an imputation value from future matches.

The current implementation uses Bundesliga fixtures only. Therefore
`days_since_last_match` does not yet account for cup or international matches.

## News Pipeline

### Build Planned News Requests

```console
uv run python -m buli_news.main build-news-requests --season 2025
```

Inputs:

```text
data/interim/matches_2025.jsonl
config/teams.json
```

Output:

```text
data/interim/news_requests_2025.jsonl
```

For every match, the command creates separate home-team and away-team requests.
Most teams use Event Registry concept URIs. `1. FSV Mainz 05` uses the keyword
strategy `mainz 05` because its concept URI returned no articles in initial API
tests.

### Fetch News Responses

Fetch one planned request:

```console
uv run python -m buli_news.main fetch-news --season 2025 --request-id bl1_2025_77261_home_team_context
```

Fetch a limited number:

```console
uv run python -m buli_news.main fetch-news --season 2025 --limit 1
```

Fetch all planned requests:

```console
uv run python -m buli_news.main fetch-news --season 2025
```

Optional request delay:

```text
--delay-seconds 1.0
```

Outputs:

```text
data/raw/newsapi/2025/{request_id}.json
data/interim/news_fetch_results_2025.jsonl
```

Fetch behavior:

- successful existing raw responses are skipped
- failed responses can be retried
- successful fetch results are appended immediately
- HTTP 429 responses are retried up to three times with a 10-second wait
- only page 1 with up to 100 articles is fetched per planned request

The API key is read from `NEWSAPI_KEY` in the environment or `.env`. It must
never be hardcoded, logged, or committed.

## Planned Next Stages

1. verify the numerical feature table and train the numerical baseline
2. normalize and deduplicate raw articles
3. define a versioned structured local-LLM annotation schema
4. annotate only pre-match article text and persist the responses
5. aggregate annotations into home- and away-team news features per `match_id`
6. train the identical model with numerical plus news features
7. compare both variants on the same 63 test matches
