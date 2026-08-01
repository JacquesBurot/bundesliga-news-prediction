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
7. evaluating a prior-based numerical `DummyClassifier` reference
8. evaluating standardized multinomial logistic regression on the numerical
   features
9. selecting numerical logistic-regression regularization and one of two fixed
   feature sets with training-only expanding-window validation
10. evaluating the frozen training-selected numerical logistic model on the
    fixed test split without overwriting the original baseline

Local LLM annotation, news-feature aggregation, and the final model comparison
are the next planned stages.

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

No random split is used. A `DummyClassifier` reference and multinomial logistic
regression are implemented. Evaluation uses Log Loss, Accuracy, Macro-F1,
multiclass Brier Score, and a Confusion Matrix.

The local LLM will act as a feature extractor, not as the match predictor:

```text
news article -> local LLM -> structured article annotation
             -> aggregated pre-match news features
```

Both logistic-regression variants will use the same preprocessing, target,
training rows, test rows, and evaluation code. Only the feature columns differ.
Hyperparameter and feature-set selection is performed inside matchdays 1-27;
matchdays 28-34 are excluded from fitting, scaling, scoring, and selection.

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
│       ├── modeling.py
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

The implementation uses the `ELO-Goals` variant evaluated by
[Wunderlich and Memmert (2018)](https://doi.org/10.1371/journal.pone.0198668).
It adopts their published update parameters and starting value. The project
uses only Bundesliga matches from 2025/26, so unlike the study it has no
multi-season rating warm-up or parameter-calibration period.

Every team starts the season with the same Elo rating. The constants are fixed
from the literature before model evaluation and are not tuned against the test
set:

```text
initial rating:            1000
base K-factor:             4
goal-difference exponent:  1.6
home advantage:            80 rating points
rating scale:              400
actual score:              1.0 home win, 0.5 draw, 0.0 away win
```

Before a match, the expected home-team score is:

```text
expected_home =
    1 / (1 + 10 ** (-((home_elo + 80 - away_elo) / 400)))
```

The home advantage of 80 rating points is used only in this expectation. The
actual home-team score is `1.0` for a home win, `0.5` for a draw, and `0.0` for
an away win. The absolute goal difference determines the match-specific
K-factor:

```text
goal_difference = abs(home_goals - away_goals)
match_k_factor = 4 * (1 + goal_difference) ** 1.6
rating_change = match_k_factor * (actual_home - expected_home)

new_home_elo = home_elo + rating_change
new_away_elo = away_elo - rating_change
```

A larger goal difference increases the match-specific K-factor. The final
rating change also depends on how strongly the actual result differs from the
expected result. The update remains zero-sum: one team gains exactly the rating
points lost by the other.

The feature row stores:

```text
elo_difference_before = home_elo - away_elo
```

This value is created before the current match update. A positive value means
that the home team has the higher pre-match rating, while a negative value means
that the away team has the higher rating. The 80-point home advantage is not
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

## Numerical Dummy Baseline

```console
uv run python -m buli_news.main evaluate-numerical-dummy --season 2025
```

The console-script equivalent is:

```console
uv run buli-news evaluate-numerical-dummy --season 2025
```

Input:

```text
data/processed/numerical_features_2025.csv
```

Outputs:

```text
data/processed/numerical_dummy_baseline_2025.json
data/processed/numerical_dummy_predictions_2025.csv
```

The command validates the exact feature schema, all 306 unique match IDs, nine
matches per matchday, the fixed 243/63 chronological split, finite numerical
values, and the presence of all three target classes in both splits.

The reference model is:

```text
DummyClassifier(strategy="prior")
```

It learns only the H/D/A class proportions from matchdays 1-27. It always
predicts the most frequent training class, while its predicted probabilities
equal the training class proportions. The numerical feature values are passed
through the common model-data interface but deliberately ignored by the dummy
estimator. Standardization is therefore neither needed nor applied at this
stage.

The result JSON records the input, explicit feature list, model configuration,
class counts, learned class probabilities, Log Loss, Accuracy, Macro-F1,
multiclass Brier Score, and the Confusion Matrix. Reports use the fixed class
order `H`, `D`, `A`.

The prediction CSV retains one row per test match with match metadata, the
actual and predicted result, and the predicted H/D/A probabilities. This makes
later model comparisons auditable per `match_id`, while the test labels remain
excluded from model fitting and are attached only for evaluation.

Classifier-independent evaluation code fits an estimator, validates and
reorders its probability columns, calculates the fixed metrics and Confusion
Matrix, and builds the per-match prediction rows. The dummy-specific wrapper
only configures `DummyClassifier(strategy="prior")` and adds its model metadata
to the shared report structure. The logistic regression already reuses this
evaluation path, and the news-extended model will do the same.

The multiclass Brier Score uses its original unscaled definition:

```text
mean over matches(
    sum over H, D, A(
        observed_one_hot - predicted_probability
    ) ** 2
)
```

Its range is 0 to 2, and lower values are better. This definition will be
reused unchanged for the logistic-regression and news-extended models.

## Numerical Logistic Regression

```console
uv run python -m buli_news.main evaluate-numerical-logistic --season 2025
```

The console-script equivalent is:

```console
uv run buli-news evaluate-numerical-logistic --season 2025
```

Input:

```text
data/processed/numerical_features_2025.csv
```

Outputs:

```text
data/processed/numerical_logistic_regression_2025.json
data/processed/numerical_logistic_predictions_2025.csv
```

The numerical model is a scikit-learn pipeline:

```text
StandardScaler()
-> LogisticRegression(
       solver="lbfgs",
       C=1.0,
       l1_ratio=0.0,
       class_weight=None,
       max_iter=1000,
       tol=0.0001,
   )
```

With scikit-learn 1.9, `l1_ratio=0.0` specifies L2 regularization. For the three
H/D/A classes, `lbfgs` optimizes the multinomial loss directly. The
configuration is fixed before test evaluation: no feature selection,
regularization search, scaling decision, or other model choice uses matchdays
28-34.

Because the scaler is inside the pipeline, `fit` calculates its mean and scale
from the 243 training rows only. The fitted scaler is then used unchanged to
transform the 63 test rows. The command rejects convergence warnings and
verifies that the scaler saw exactly the training-row count.

The result JSON uses the same split, metrics, target order, and Confusion Matrix
definition as the dummy report. It additionally records the fixed pipeline
configuration, solver iterations, scaler training-row count, standardized
class coefficients, and intercepts. The prediction CSV uses the same per-match
schema as the dummy predictions so both outputs can be joined by `match_id`.

This fixed `C=1.0` evaluation remains the original numerical test baseline. The
separate training-only model-selection stage below does not overwrite its
report or predictions.

## Training-Only Numerical Logistic Model Selection

```console
uv run python -m buli_news.main select-numerical-logistic --season 2025
```

The console-script equivalent is:

```console
uv run buli-news select-numerical-logistic --season 2025
```

Input:

```text
data/processed/numerical_features_2025.csv
```

Outputs:

```text
data/processed/numerical_logistic_model_selection_2025.json
data/processed/numerical_logistic_validation_predictions_2025.csv
```

The command evaluates two predefined numerical feature sets:

- `full`: all 35 numerical features
- `without_match_counts`: 31 features after excluding
  `home_matches_played`, `away_matches_played`,
  `home_venue_matches_played`, and `away_venue_matches_played`

Each feature set is evaluated with:

```text
C = 0.01, 0.1, 1.0, 10.0
```

This produces eight logistic candidates. A fold-specific
`DummyClassifier(strategy="prior")` is evaluated on the same validation matches
as a reference but cannot be selected. Every logistic fit uses a new
`StandardScaler` and estimator. The scaler is fitted only on that fold's fit
rows.

The expanding-window design uses these nominal matchday ranges:

| Fold | Nominal fit matchdays | Validation matchdays | Actual fit rows | Validation rows |
| --- | --- | --- | ---: | ---: |
| 1 | 1-12 | 13-17 | 108 | 45 |
| 2 | 1-17 | 18-22 | 150 | 45 |
| 3 | 1-22 | 23-27 | 197 | 45 |

Actual kickoff timestamps take precedence over nominal matchday numbers. For
each fold, the earliest validation kickoff is an exclusive fit cutoff. This
excludes three postponed nominal fit matches from fold 2 (`77395`, `77399`,
`77409`) and one from fold 3 (`77409`). Without this cutoff, later-played
matches from earlier numbered matchdays would leak future information into the
validation fit. The report records the cutoff, exclusions, match IDs, row
counts, and scaler sample count for every fold.

The 45 validation predictions from each fold are disjoint and are pooled into
135 predictions per candidate. Selection uses a one-standard-error rule:

1. identify the candidate with the lowest pooled Log Loss
2. calculate that candidate's Log-Loss standard error as the sample standard
   deviation across its three fold values divided by the square root of three
3. retain every candidate whose pooled Log Loss is no greater than the minimum
   plus that standard error
4. prefer fewer features inside this eligible set, followed by lower pooled
   multiclass Brier Score, lower `C`, and lower pooled Log Loss

The complete feature set at `C=0.01` has the minimum pooled Log Loss of
`0.986955`. Its fold-based standard error is `0.020599`, producing an inclusive
eligibility threshold of `1.007554`. Both `full/C=0.01` and
`without_match_counts/C=0.01` fall inside that range. The parsimony rule selects
`without_match_counts` with `C=0.01` because it uses 31 instead of 35 features.
Its pooled validation metrics are Log Loss `0.988059`, multiclass Brier Score
`0.585754`, Accuracy `0.548148`, and Macro-F1 `0.394554`. The pooled dummy
reference has Log Loss `1.072735` and multiclass Brier Score `0.647433`.

The prediction CSV contains the eight logistic candidates plus the dummy
reference for all 135 validation matches. The JSON contains fold metrics,
pooled metrics, confusion matrices, population and sample dispersion, standard
errors, strict Log-Loss ranks, selection ranks, the eligibility threshold, and
the selected configuration. This command does not refit the selected model and
does not evaluate matchdays 28-34.

## Selected Numerical Logistic Regression

```console
uv run python -m buli_news.main evaluate-selected-numerical-logistic --season 2025
```

The console-script equivalent is:

```console
uv run buli-news evaluate-selected-numerical-logistic --season 2025
```

Input:

```text
data/processed/numerical_features_2025.csv
```

Outputs:

```text
data/processed/selected_numerical_logistic_regression_2025.json
data/processed/selected_numerical_logistic_predictions_2025.csv
```

This command freezes the result of the training-only model-selection stage:

```text
feature set = without_match_counts
feature count = 31
excluded features = home_matches_played, away_matches_played,
                    home_venue_matches_played,
                    away_venue_matches_played
C = 0.01
```

The pipeline fits a new `StandardScaler` and logistic regression on all 243
matches from matchdays 1-27, then transforms and predicts the unchanged 63
matches from matchdays 28-34. The report records the frozen feature list,
selection provenance, scaler training-row count, model configuration,
coefficients, predictions, and evaluation metrics.

Current fixed-test results are:

| Model | Log Loss | Multiclass Brier Score | Accuracy | Macro-F1 |
| --- | ---: | ---: | ---: | ---: |
| Prior dummy | 1.095512 | 0.666028 | 0.380952 | 0.183908 |
| Original full-feature logistic, `C=1.0` | 1.388468 | 0.813405 | 0.333333 | 0.285714 |
| Selected 31-feature logistic, `C=0.01` | 1.061313 | 0.636447 | 0.492063 | 0.366667 |

The selected model improves all four reported metrics over both the original
logistic baseline and the prior dummy. It predicts 46 home wins, no draws, and
17 away wins on the test set; this class behavior remains visible in the stored
Confusion Matrix and must be considered when interpreting the aggregate
metrics.

The original `evaluate-numerical-logistic` command remains fixed at the full
35-feature schema and `C=1.0`. Its reports are retained as an auditable original
baseline and are not overwritten by the selected-model command.

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

1. normalize and deduplicate raw articles
2. define a versioned structured local-LLM annotation schema
3. annotate only pre-match article text and persist the responses
4. aggregate annotations into home- and away-team news features per `match_id`
5. train the identical selected model with numerical plus news features
6. compare both variants on the same 63 test matches
