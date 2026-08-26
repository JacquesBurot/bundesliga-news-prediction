# bundesliga-news-prediction

Python pipeline for a master's thesis investigating whether structured
pre-match sports-news features improve Bundesliga match-outcome predictions
beyond numerical pre-match data alone.

## Contents

- [Current Scope](#current-scope)
- [Experiment Design](#experiment-design)
- [Data Flow](#data-flow)
- [Leakage Rules](#leakage-rules)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Numerical Data Pipeline](#numerical-data-pipeline)
- [Numerical Dummy Baseline](#numerical-dummy-baseline)
- [Numerical Logistic Reference](#numerical-logistic-reference)
- [Training-Only Numerical Logistic Configuration Selection](#training-only-numerical-logistic-configuration-selection)
- [Final Numerical Logistic Regression](#final-numerical-logistic-regression)
- [News Pipeline](#news-pipeline)
- [Planned Next Stages](#planned-next-stages)

## Current Scope

The implemented pipeline currently supports:

1. downloading raw Bundesliga match metadata from OpenLigaDB
2. downloading raw match statistics from Football-Data.co.uk
3. normalizing OpenLigaDB match metadata
4. joining both match sources into a canonical numerical match history
5. calculating leakage-safe numerical pre-match features
6. planning and fetching German pre-match news from Event Registry / NewsAPI.ai
7. exporting collected source homepages into a workbook for manual legal review
8. converting the reviewed news-source XLSX into a versioned JSON policy
9. applying the source policy and building canonical articles with
   request-bound match and team-side links
10. grouping text-identical publications under stable content IDs while
    retaining every article and request association
11. building request-bound, team-specific local-LLM annotation tasks
12. extracting four structured news indicators through a local Ollama server
    with a versioned system prompt and strict JSON output
13. evaluating a prior-based numerical `DummyClassifier` reference
14. evaluating standardized multinomial logistic regression on the numerical
   features
15. selecting numerical logistic-regression regularization and one of two fixed
   feature sets with training-only expanding-window validation
16. evaluating the frozen training-selected numerical logistic model on the
    fixed test split without overwriting the original baseline

News-feature aggregation and the final numerical-versus-news model comparison
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

NewsAPI.ai raw -----> source-review XLSX -----> source policy
       |                                          |
       +------------------------------------------+
                                                  v
                      canonical articles + request-bound links
                                                  |
                                                  v
                                     content IDs -> team-specific tasks
                                                   -> local Ollama annotations
                                                   -> news features

numerical features -------------------------> model A
numerical features + news features ---------> model B
```

The data directories represent processing stages:

- `data/raw`: unchanged responses from external sources
- `data/interim/{season}/matches`: normalized and joined match data
- `data/interim/{season}/news/collection`: planned requests and fetch ledger
- `data/interim/{season}/news/articles`: canonical articles and request links
- `data/interim/{season}/news/contents`: exact-content groups and article links
- `data/interim/{season}/news/annotations`: local-LLM tasks, results, and failures
- `data/review/{season}`: manually maintained review workbooks
- `data/processed`: model-ready pre-match feature tables
- `outputs/modeling`: generated model reports and prediction tables

Real data files and generated model outputs are ignored by Git. Only `.gitkeep`
files preserve the directory structure in the repository.

All standard season-specific locations are defined centrally in
`buli_news.paths.SeasonPaths`. Pipeline commands must use these path definitions
instead of reconstructing interim, review, processed, or modeling paths locally.

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
│   ├── news_annotation_schema_v1.json
│   ├── news_annotation_schema_v2.json
│   ├── news_annotation_schema_v3.json
│   ├── news_source_policy.json
│   └── teams.json
├── data/
│   ├── raw/
│   │   ├── football_data/
│   │   ├── openligadb/
│   │   └── newsapi/
│   ├── interim/
│   │   └── {season}/
│   │       ├── matches/
│   │       └── news/
│   │           ├── collection/
│   │           ├── articles/
│   │           ├── contents/
│   │           └── annotations/
│   │               └── pilots/
│   │                   └── v{schema_version}/
│   ├── review/
│   │   └── {season}/
│   └── processed/
├── outputs/
│   └── modeling/
├── src/
│   └── buli_news/
│       ├── __init__.py
│       ├── main.py
│       ├── paths.py
│       ├── storage.py
│       ├── cli/
│       │   ├── application.py
│       │   ├── arguments.py
│       │   ├── errors.py
│       │   ├── parser.py
│       │   └── commands/
│       │       ├── annotations.py
│       │       ├── matches.py
│       │       ├── modeling.py
│       │       ├── news_collection.py
│       │       └── news_processing.py
│       ├── matches/
│       │   ├── features.py
│       │   ├── football_data.py
│       │   ├── history.py
│       │   ├── normalization.py
│       │   └── openligadb.py
│       ├── modeling/
│       │   ├── evaluation.py
│       │   └── selection.py
│       └── news/
│           ├── articles.py
│           ├── contents.py
│           ├── fetch.py
│           ├── requests.py
│           ├── source_policy.py
│           ├── source_review.py
│           └── annotations/
│               ├── _validation.py
│               ├── config.py
│               ├── ollama.py
│               ├── results.py
│               ├── runner.py
│               ├── selection.py
│               └── tasks.py
├── .env.example
├── .gitattributes
├── .gitignore
├── .python-version
├── pyproject.toml
├── README.md
└── uv.lock
```

The source tree follows the pipeline domains. `matches` owns match acquisition
and numerical feature construction, `news` owns collection and article
processing, and `modeling` owns model selection and evaluation. `cli` contains
command registration, top-level artifact orchestration, console output, and the
shared error boundary; domain modules never import it. Format-bound and
resume-sensitive I/O remains with the responsible domain stage, including raw
response persistence, source-review XLSX handling, and locked append-only
annotation outputs. The small top-level `main.py` keeps both
`python -m buli_news.main` and the `buli-news` console script stable.

The annotation package separates deterministic configuration and task
construction from Ollama communication, append-only result handling, and
concurrent run orchestration. This keeps schema- and ID-defining code isolated
from infrastructure concerns while preserving the existing artifact formats.

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

The project currently has no automated test suite. Pipeline stages perform
their own input and output validation; dedicated tests may be added during a
later cleanup phase.

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
data/interim/2025/matches/normalized.jsonl
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
data/interim/2025/matches/normalized.jsonl
data/raw/football_data/D1_2526.csv
config/teams.json
```

Outputs:

```text
data/interim/2025/matches/numerical.jsonl
data/interim/2025/matches/numerical_quality.json
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
data/interim/2025/matches/numerical.jsonl
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
outputs/modeling/2025/numerical/dummy/evaluation.json
outputs/modeling/2025/numerical/dummy/test_predictions.csv
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

## Numerical Logistic Reference

```console
uv run python -m buli_news.main evaluate-numerical-logistic-reference --season 2025
```

The console-script equivalent is:

```console
uv run buli-news evaluate-numerical-logistic-reference --season 2025
```

Input:

```text
data/processed/numerical_features_2025.csv
```

Outputs:

```text
outputs/modeling/2025/numerical/logistic_regression/reference/evaluation.json
outputs/modeling/2025/numerical/logistic_regression/reference/test_predictions.csv
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

## Training-Only Numerical Logistic Configuration Selection

```console
uv run python -m buli_news.main select-numerical-logistic-configuration --season 2025
```

The console-script equivalent is:

```console
uv run buli-news select-numerical-logistic-configuration --season 2025
```

Input:

```text
data/processed/numerical_features_2025.csv
```

Outputs:

```text
outputs/modeling/2025/numerical/logistic_regression/selection/report.json
outputs/modeling/2025/numerical/logistic_regression/selection/validation_predictions.csv
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

## Final Numerical Logistic Regression

```console
uv run python -m buli_news.main evaluate-numerical-logistic-final --season 2025
```

The console-script equivalent is:

```console
uv run buli-news evaluate-numerical-logistic-final --season 2025
```

Inputs:

```text
data/processed/numerical_features_2025.csv
outputs/modeling/2025/numerical/logistic_regression/selection/report.json
```

Outputs:

```text
outputs/modeling/2025/numerical/logistic_regression/final/evaluation.json
outputs/modeling/2025/numerical/logistic_regression/final/test_predictions.csv
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

Before fitting, the command loads the training-only selection report and
validates its season, selected feature set, retained and excluded columns, and
`C` against this frozen source configuration. A missing, stale, or mismatching
selection report is rejected; the report is provenance input and never causes
the outer test split to participate in model selection.

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

The `evaluate-numerical-logistic-reference` command remains fixed at the full
35-feature schema and `C=1.0`. Its reports are retained as an auditable original
reference and are not overwritten by the final-model command.

## News Pipeline

### Build Planned News Requests

```console
uv run python -m buli_news.main build-news-requests --season 2025
```

Inputs:

```text
data/interim/2025/matches/normalized.jsonl
config/teams.json
```

Output:

```text
data/interim/2025/news/collection/requests.jsonl
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
data/interim/2025/news/collection/fetch_results.jsonl
```

Fetch behavior:

- successful existing raw responses are skipped
- failed responses can be retried
- each new response must first pass its HTTP-status and expected article-result
  structure checks
- a structurally valid empty `articles.results` list is stored as a successful
  zero-result response with `article_count = 0`
- after validation, the exact `response.content` bytes are stored unchanged
- successful fetch metadata is appended immediately
- HTTP 429 responses are retried up to three times with a 10-second wait
- only page 1 with up to 100 articles is fetched per planned request

`fetch_results.jsonl` is an append-only observation ledger, not an inventory of
all raw files. A row records only the request and HTTP metadata observed when
that fetch was actually performed. The current 612 legacy raw response files
remain valid pipeline input even where no historical ledger row exists, because
their stored article-result structure is validated directly. The pipeline must
not invent retrospective ledger rows for those files or present reconstructed
metadata as a historical fetch observation.

The API key is read from `NEWSAPI_KEY` in the environment or `.env`. It must
never be hardcoded, logged, or committed.

### Export Sources for Manual Legal Review

After fetching the raw responses, create the review workbook:

```console
uv run python -m buli_news.main export-news-source-review --season 2025
```

Input:

```text
data/raw/newsapi/2025/*.json
```

Output:

```text
data/review/2025/news_sources.xlsx
```

The command normalizes every article or source URI to an HTTPS homepage,
counts its occurrences before article deduplication, and sorts sources by
descending count and then URL. It creates the `sources` worksheet with the
complete review schema:

```text
website_url, article_count, legal_text, review_status, review_date
```

Only `website_url` and `article_count` are filled automatically. The three
remaining columns are reserved for the manual legal review. To protect that
work, the command refuses to replace an existing output file. The explicit
`--overwrite` option should only be used when all existing manual review data
may be discarded.

### Build Reviewed News Source Policy

After reviewing the exported source workbook, convert it reproducibly into the
versioned JSON policy:

```console
uv run python -m buli_news.main build-news-source-policy --season 2025
```

Input:

```text
data/review/2025/news_sources.xlsx
```

Output:

```text
config/news_source_policy.json
```

The `sources` worksheet must contain exactly these columns in row 1:

```text
website_url, article_count, legal_text, review_status, review_date
```

The converter maps `complete_no_ml_clause` to `include` and
`complete_explicit_ml_clause` to `exclude`. Every excluded source needs legal
text containing at least one evidence URL. Homepage URLs are normalized to
lowercase hosts with one leading `www.` label removed, matching is exact, and
unknown hosts are excluded by default. The generated policy records the source
workbook's SHA-256 and is sorted by host for deterministic diffs.

The current workbook contains 333 reviewed sources: 282 included and 51
excluded. Raw Event Registry responses remain unchanged.

### Build Policy-Filtered News Articles

After generating the reviewed source policy, normalize and de-duplicate the
collected articles:

```console
uv run python -m buli_news.main build-news-articles --season 2025
```

Inputs:

```text
data/interim/2025/news/collection/requests.jsonl
data/raw/newsapi/2025/*.json
config/news_source_policy.json
```

Outputs:

```text
data/interim/2025/news/articles/articles.jsonl
data/interim/2025/news/articles/request_links.jsonl
data/interim/2025/news/articles/quality.json
```

`articles.jsonl` contains each policy-approved canonical article once. The
Event Registry article URI is the primary de-duplication key; a SHA-256 of the
normalized article URL is the deterministic fallback. Article text from
excluded or unknown hosts never enters this output.

`request_links.jsonl` preserves the exact request occurrence that created each
association. An article returned by a home-team request is linked only to that
request's `match_id`, `side`, and `team_id`. It is not assigned to another
match or side based on its content. If the same canonical article occurs in a
second raw response, it receives a separate link to that second request while
its text remains stored only once.

The stage requires exactly one home and one away request per match and exactly
one raw response per planned request. It validates the Event Registry query
date and converts `dateTimePub` to `Europe/Berlin`; only occurrences whose
actual publication date is inside that request's pre-match window receive a
link. The quality report records policy decisions, rejected occurrences,
de-duplication, request coverage, and requests without usable links.

For the current 2025/26 collection, the stage produces 24,509 canonical
articles and 49,286 request-bound links. It excludes 10,907 occurrences by
policy and rejects another 338 included-source occurrences whose publication
timestamp lies outside the associated request window. All 612 requests and all
306 matches retain at least one valid link.

### Build Stable News Content IDs

Group publications with identical normalized article bodies without changing
their source or request provenance:

```console
uv run python -m buli_news.main build-news-contents --season 2025
```

Input:

```text
data/interim/2025/news/articles/articles.jsonl
```

Outputs:

```text
data/interim/2025/news/contents/contents.jsonl
data/interim/2025/news/contents/article_links.jsonl
data/interim/2025/news/contents/quality.json
```

The stage normalizes each article body with Unicode NFKC normalization, Unicode
case folding, collapsed whitespace, and stripped outer whitespace. It hashes
that normalized UTF-8 text with SHA-256 to form a stable `content_id`. It does
not group similar or semantically related text.

`contents.jsonl` stores one representative original body for every distinct
normalized text. `contents/article_links.jsonl` maps every `article_id` to
exactly one `content_id`. The original article rows, titles, URLs, sources, and
request-bound match and side links remain unchanged. Content grouping therefore
cannot assign an article to another request, match, side, or team.

For the current collection, 24,509 canonical publications map to 23,738
contents. The 771 collapsed publication rows belong to 543 duplicate groups;
443 of those groups contain publications from more than one source host. The
largest group contains 30 publications. These groups are descriptive inputs
for the local-LLM stage. The request-bound annotation-task stage decides where
an identical content needs to be interpreted.

### Build Team-Specific LLM Annotation Tasks

Create one deterministic task per `request_id` and `content_id`:

```console
uv run python -m buli_news.main build-news-annotation-tasks --season 2025
```

Inputs:

```text
data/interim/2025/matches/normalized.jsonl
data/interim/2025/news/articles/articles.jsonl
data/interim/2025/news/articles/request_links.jsonl
data/interim/2025/news/contents/contents.jsonl
data/interim/2025/news/contents/article_links.jsonl
config/news_annotation_schema_v3.json
```

Outputs:

```text
data/interim/2025/news/annotations/tasks.jsonl
data/interim/2025/news/annotations/tasks_quality.json
```

A task keeps the exact request, fixture, home/away side, and target team through
which its article was collected. Content de-duplication therefore never moves
an article to the other team or another fixture. If the same normalized content
appears through several sites inside the same request, those article links form
one LLM task. If it appears in a different home- or away-team request, it forms
a separate task with that request's target-team context.

Every stored task contains the full request and fixture provenance needed for
later feature aggregation. Ollama receives only the target team, its configured
aliases, the article title, and the article body. When several publication rows
share the same content inside one request, the title and body are selected together
from the article with the lowest response position and then the lowest
`article_id`. The quality report records this deterministic selection and the
number of collapsed within-request links.

For the current collection, 49,286 request-bound article links produce 48,466
team-specific annotation tasks. The difference consists of 820 additional
publication links whose normalized content already occurs in the same request.
All 612 home/away requests and all 306 fixtures remain represented.

### Annotate News Through Local Ollama

Start Ollama and install the selected model once:

```console
ollama serve
ollama pull gemma4:12b-it-qat
```

Run a small pilot before processing the complete task file:

```console
uv run python -m buli_news.main annotate-news \
  --season 2025 \
  --model gemma4:12b-it-qat \
  --pilot-size 200 \
  --pilot-seed 42 \
  --workers 2
```

The default server is `http://localhost:11434`, and the default Ollama context
window is `40960` tokens. `--base-url`, `--model`, `--num-ctx`,
`--timeout-seconds`, `--workers`, `--output`, and `--failure-output` can be set
explicitly. Use `--task-id` for one exact task.
`--pilot-size` selects a deterministic pilot balanced across matchdays, target
teams, home/away sides, source hosts, unique contents, and article-length
quartiles. The selected task manifest is stored beside the annotation outputs
below
`data/interim/2025/news/annotations/pilots/v{schema_version}/` as
`tasks_selection_v{selection_version}_{size}_seed_{seed}.jsonl`,
`results_selection_v{selection_version}_{size}_seed_{seed}.jsonl`, and
`failures_selection_v{selection_version}_{size}_seed_{seed}.jsonl`. Explicit
`--output` and `--failure-output` paths still take precedence. `--limit`
remains available for a simple prefix of untouched tasks but is not a
stratified pilot.

A full run without `--pilot-size`, `--output`, or `--failure-output` keeps the
canonical append-only destinations
`data/interim/2025/news/annotations/results.jsonl` and
`data/interim/2025/news/annotations/failures.jsonl`. A rerun skips only
annotations with the same task, annotation configuration, and local Ollama
model digest. Concurrent requests are processed by worker threads, but all
successful and failed JSONL rows are appended by the main thread so writes
cannot interleave. An OS-level lock on the selected result path also rejects a
second annotation process before it can append duplicate rows; resume a stopped
run only after its previous process has actually exited. Use one worker unless
the local Ollama hardware has been benchmarked; two workers improved throughput
on the development machine.

To inspect the complete article set for one fixture without touching the full
run outputs, select its exact match ID:

```console
uv run python -m buli_news.main annotate-news \
  --season 2025 \
  --model gemma4:12b-it-qat \
  --match-id 77393 \
  --workers 2
```

This writes `tasks.jsonl`, `results.jsonl`, and, only when needed,
`failures.jsonl` below
`annotations/pilots/v{schema_version}/matches/{match_id}/`. `--match-id`
cannot be combined with `--pilot-size`, `--task-id`, `--limit`, or
`--retry-failures-only`.
The default context size is 40,960 tokens because the current collection also
contains a small number of unusually long article bodies. Each response may use
up to 8,192 generated tokens so Thinking can finish before Ollama emits the four
ratings and their evidence arrays. The larger context leaves additional input
headroom beyond the longest prompts observed in the 100-task pilot. The output
limit is only an upper bound; Ollama stops normally as soon as the complete
structured JSON response is finished.

Each selected task receives at most two semantic response attempts per command
execution. If both responses fail JSON, schema, rating, or evidence-presence
validation, one failure event and the last model response are appended to
`data/interim/2025/news/annotations/failures.jsonl`. The command then continues
with the next untouched task. Normal later runs defer these known failures so
they cannot repeatedly block progress. Retry only unresolved failures for the
same annotation configuration and model digest with:

```console
uv run python -m buli_news.main annotate-news \
  --season 2025 \
  --model gemma4:12b-it-qat \
  --retry-failures-only
```

Each explicit `--retry-failures-only` execution may make a new pair of semantic
attempts. If that pair also fails, it appends another failure event rather than
overwriting the earlier event. A later valid result is appended to
`annotations/results.jsonl`; all previous failure rows remain as an auditable
history. HTTP and Ollama-server failures still stop the command because they
indicate an infrastructure problem rather than one bad article response.

The default `config/news_annotation_schema_v3.json` is the complete annotation
configuration: system prompt, response JSON Schema, target-team aliases, and
numeric rating mapping. It does not inherit from or append text to another
configuration. The v1 and v2 files remain only so earlier pilot rows stay
interpretable; the active annotation code accepts the complete v3 structure.
Each request sends two chat messages:

1. a fixed system prompt defining the extraction task, all indicator meanings,
   the evidence rules, and the prohibition on external knowledge
2. a user message containing only the target team, its aliases, article title,
   and complete article body as untrusted JSON data

Ollama receives the full response JSON Schema through its structured-output
`format` field, with thinking enabled, temperature `0`, and a fixed seed. Each
indicator directly contains its rating and one or two supporting quotes. Python
does not rewrite, discard, or reclassify the model's semantic decisions. It
only checks the exact JSON fields and allowed ratings. Evidence is always an
array: it must be empty for `not_mentioned` and contain one or two non-empty
strings for every assessed rating. Python does not compare those strings
against the article text. Within one execution, an invalid response is retried
once and is then recorded as one failure event.
Article scope, target-team attribution, opponent separation, and indicator
meaning are owned by the system prompt and remain visible in the model output.

The four extracted indicators are:

```text
sporting_form
personnel_situation
physical_readiness
confidence_and_motivation
```

Ratings map to `-2`, `-1`, `0`, `1`, and `2`. `not_mentioned` remains a missing
value rather than being treated as neutral. Relevance is implicit: an article
contains an extracted signal when at least one indicator is not `not_mentioned`.
This stage persists article-level, team-contextual annotations only; it does not
yet aggregate them or train the combined prediction model.

## Planned Next Stages

1. aggregate the validated article annotations into leakage-safe home- and
   away-team news features per `match_id`
2. join those news features to the unchanged numerical feature rows
3. train the frozen selected model once with numerical features only and once
   with the identical numerical features plus news features
4. evaluate and compare both variants on the same 63 fixed test matches
