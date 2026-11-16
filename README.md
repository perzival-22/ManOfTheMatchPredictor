# Man of the Match team predictor

**Author:** Kelvin Wilch

A machine learning project that answers one question:

> For each finished 2018 or 2022 World Cup match, look at both teams' match statistics and pick **which of the two teams** the Man of the Match came from.

It does not name the player, it does not forecast before kick-off, and it does not ask "did the winning team get the award?". Who won is never part of the label.

This file explains the whole project: what is in it, how to run it, and what every part of the code does and why. The code itself lives in one notebook, `MOTM_predictor.ipynb`, and follows the same five phases as this guide.

## Contents

1. [Project files](#project-files)
2. [How to run it](#how-to-run-it)
3. [The data](#the-data)
4. [Phase 1: Data audit](#phase-1-data-audit)
5. [Phase 2: Features and split](#phase-2-features-and-split)
6. [Phase 3: Baseline](#phase-3-baseline)
7. [Phase 4: Training and tuning](#phase-4-training-and-tuning)
8. [Phase 5: Evaluation](#phase-5-evaluation)
9. [Results](#results)
10. [Limitations](#limitations)
11. [Function reference](#function-reference)
12. [Glossary](#glossary)

## Project files

| File | What it is |
|---|---|
| `MOTM_predictor.ipynb` | The whole project: data checks, features, baseline, models, evaluation and conclusions. Every section starts with a note on what it does and why, and every code cell has inline comments. |
| `world_cup_team_statistics.csv` | The dataset: 256 team rows, 128 matches, 45 columns. |
| `README.md` | This guide. |

## How to run it

You need Python 3.9 or newer.

```bash
# 1. Create and activate a virtual environment (once)
python3 -m venv env
source env/bin/activate

# 2. Install the libraries the notebook imports
pip install pandas numpy scikit-learn jupyter

# 3. Open the notebook, then choose "Run All"
jupyter notebook MOTM_predictor.ipynb
```

The notebook loads the CSV by a relative path, so keep the two files in the same folder. A full run takes a few seconds. Every random step uses the seed `42`, so the numbers are the same on every run.

To run it without opening a browser:

```bash
jupyter nbconvert --to notebook --execute --inplace MOTM_predictor.ipynb
```

If any `assert` or "Self-check" cell fails, the run stops with an error at that cell. That is deliberate: see [Self-checks](#self-checks).

## The data

Each match appears as **two rows**, one per team. The two rows mirror each other: if one says `Team = Russia, Opponent = Saudi Arabia`, the other says `Team = Saudi Arabia, Opponent = Russia`.

The label column is `Man of the Match`:

- `1` on the row of the team that supplied the award winner;
- `0` on the other row.

So each match has exactly one `1` and one `0`, and the label is balanced (128 of each) by construction.

**The key idea of the project:** the real sample size is **128 matches**, not 256 independent rows. The two rows of a match are two views of one decision. Because of that, the data is *split* by match, *cross-validated* by match and *scored* by match.

| Group of columns | Examples | Used as model input? |
|---|---|---|
| Shared match statistics | `Goal Scored`, `Ball Possession %`, `On-Target`, `Passes` | Yes (19 columns) |
| The label | `Man of the Match` | Never. It is the answer key. |
| Award details | `award_winner_team`, `award_winner_player` | Never. They contain the answer. |
| Identifiers | `Team`, `Opponent`, `Date`, `match_id`, `year` | No. They say who played, not how. |
| Source notes | `statistics_source`, `award_source`, `statistics_method` | No. Kept for auditing only. |
| 2022-only statistics | `passes_completed`, `possession_in_contest_pct` | No. Empty for all of 2018. |

## Phase 1: Data audit

**Goal:** prove the file has the shape the rest of the project assumes, before any modelling.

### 1.1 Load and look

`pd.read_csv` reads the file into a table. The cell prints the shape (`(256, 45)`), the number of matches (128), the rows per tournament (128 each, which is 64 matches each) and the label counts (128 ones, 128 zeros).

### 1.2 Structure checks

A series of `assert` statements. Each one states a rule and stops the notebook if the rule is broken:

```python
match_check = df.groupby("match_id").agg(
    rows=(TARGET, "size"),            # how many rows the match has
    positive_labels=(TARGET, "sum"),  # how many of them are labelled 1
    years=("year", "nunique"),        # how many different years appear
)
assert (match_check["rows"] == 2).all()             # exactly two teams per match
assert (match_check["positive_labels"] == 1).all()  # exactly one team supplied the winner
```

It also checks that the key columns have no gaps, that the label is only ever 0 or 1, that no team appears twice in a match, and that the two rows of every match name each other as opponent.

**Why:** if one match had a missing row or two winners, the match-level scoring later would be wrong without any visible error. An assert turns a silent problem into a loud one.

### 1.3 Missing values and source differences

`df.isna().sum()` counts the gaps per column. `pd.crosstab(df["year"], df[column])` shows, for each source column, which values appear in which year.

**What it found:**

- Several columns are empty for all 128 rows of 2018 (the 2022-only statistics).
- Every source column has one value for 2018 and a different one for 2022. The two tournaments came from different sources.
- Possession was measured differently: whole numbers from 25 to 75 in 2018, decimals from 15.82 to 84.18 in 2022, because the 2022 method removes an "in contest" share.

**What follows from it:** only statistics recorded in both years are used, and the final results are reported per year as well as overall.

## Phase 2: Features and split

### 2.1 Feature lists

A "feature" is an input column. The notebook defines two lists:

```python
FEATURES_WITH_GOALS = [...19 shared statistics...]
SCORE_LEAK_COLUMNS = ["Goal Scored", "Goals in PSO", "Own goals"]
FEATURES_NO_GOALS = [c for c in FEATURES_WITH_GOALS if c not in SCORE_LEAK_COLUMNS]
```

**Why two lists:** the award usually goes to the side that scored more, so goals nearly reveal the answer. A model that sees goals mostly learns "who scored more".

| Feature set | Columns | What it tests |
|---|---|---|
| with goals | all 19 | How well the award can be predicted with everything available |
| no goals | 16 (the 19 minus the three score columns) | Whether **how a team played** predicts the award |

The **no-goals** model is the main one. It is expected to score *lower*, because goals make the task easier. It is the main model because it asks the more interesting question, not because it is more accurate.

`Goals in PSO` is dropped because shootout goals show who went through. `Own goals` is dropped because it is part of the scoreline.

The second list is built from the first, so the two can never drift apart. Asserts confirm that the label is in neither list, no column is repeated, and every column exists in the file.

`pd.to_numeric(..., errors="coerce")` then makes sure every feature is a number. Anything unreadable becomes `NaN` (a marked gap) instead of crashing. The data turns out to have no gaps.

### 2.2 The split by match

```python
def split_by_match(frame, test_size=0.20, random_state=RANDOM_STATE):
    match_meta = frame[["match_id", "year"]].drop_duplicates("match_id")
    train_ids, test_ids = train_test_split(
        match_meta["match_id"], test_size=test_size,
        random_state=random_state, stratify=match_meta["year"],
    )
    train = frame[frame["match_id"].isin(train_ids)].copy()
    test = frame[frame["match_id"].isin(test_ids)].copy()
    return train, test
```

How it works, step by step:

1. Make a table with one line per match (its id and its year).
2. Split the list of **match ids** 80/20, not the rows.
3. Pull back the team rows for each side. Both rows of a match share its id, so they land on the same side.

| | Matches | Team rows | Per year |
|---|---|---|---|
| Training | 102 | 204 | 51 + 51 |
| Test | 26 | 52 | 13 + 13 |

**Why split by match:** the two rows of a match mirror each other. If one went to training and the other to testing, the model would have seen the opponent's side of a test match, which is peeking at the answer.

**Why `stratify`:** it keeps the 2018/2022 balance equal on both sides, so the test set cannot end up mostly one tournament by chance.

A country can appear on both sides (Brazil in a training match and a different Brazil match in the test set). That is fine. It is the *match* that must not be shared.

**What 26 test matches means:** one match is worth 1 / 26 ≈ **3.85 percentage points** of accuracy. A gap of one or two matches between methods is noise.

## Phase 3: Baseline

### 3.1 Scoring one pick per match

The question is "which of the two teams?", so there must be exactly one pick per match. `match_predictions` enforces that:

1. Attach one score to every team row.
2. Sort inside each match by score, highest first (an exact tie goes to the alphabetically first team, so the result never changes between runs).
3. Keep the top row of each match as the pick.
4. Find the true answer (the row labelled 1) and compare.

It returns one row per match with `predicted_team`, `selected_score`, `actual_team`, `year` and `correct`. `match_accuracy` is the share of `correct` matches.

**Why not ordinary row accuracy:** scoring rows one at a time allows impossible answers, such as predicting `1` for both teams of a match. Comparing the two scores inside a match forces exactly one winner.

Every method, the baseline and all the models, is graded by these same two functions.

### 3.2 The baseline rule

A hand-written rule with **no training**:

1. the team with more **goals**;
2. if level, the team with more **shots on target**;
3. if still level, the team with more **possession**;
4. if still level, a **coin flip with a fixed seed**.

```python
ranked = ranked.sort_values(["match_id", *BASELINE_ORDER, "_coin"])
picked_rows = ranked.groupby("match_id")["_row"].last().to_numpy()
```

Sorting by several columns compares them one after another, the way a dictionary sorts words letter by letter. After the sort, the last row of each match is the team that ranks highest, so that row is the pick. The function returns `1` for the picked row and `0` for the other, which `match_predictions` can grade like any model's scores.

**Why a baseline:** an accuracy number means nothing on its own. The baseline is the yardstick. It is not a training stage and it learns nothing. A model earns its place only if it beats the rule. Random picking would get about 50%.

The tie-break chain belongs to the baseline only. The models learn for themselves how much each statistic matters.

`drawn_match_ids` returns the matches where both teams scored the same number of goals. Those are the matches goals cannot decide.

### 3.3 Baseline on the training matches

The rule is right on **90.2%** of the 102 training matches, and on **12 of the 20** drawn ones. Its test score is computed in Phase 5, on the same 26 matches as the models and with draws included. Leaving draws out would let the rule skip its hardest cases.

## Phase 4: Training and tuning

### 4.1 The three models

They run from simple to more flexible, to test whether extra flexibility helps on a small dataset.

| Model | How it decides | Preparation steps |
|---|---|---|
| Logistic regression | A weighted sum of the statistics, squashed into a probability between 0 and 1 | imputer, scaler |
| Decision tree | A chain of yes/no questions about the statistics | imputer |
| KNN (k-nearest neighbours) | The `k` most similar training rows vote | imputer, scaler |

Each model is a scikit-learn `Pipeline`: a chain of preparation steps followed by the model.

- **`SimpleImputer(strategy="median")`** fills any missing value with the column's median. There are no gaps today, but the step keeps the model from crashing if one appears.
- **`StandardScaler`** subtracts each column's mean and divides by its spread, so all statistics are on a similar scale. It works per column.
  - KNN **needs** it: it measures distances, and passes (hundreds) would drown out cards (0 to 3).
  - Logistic regression **wants** it: its penalty then treats every statistic fairly, and the weights can be compared.
  - The tree **does not need** it: rescaling a column moves a threshold but does not change the split.

**Why the steps sit inside the pipeline:** they are then fitted only on the rows the model is training on. Fitting a scaler on all the data would leak a little information from the rows being scored.

`positive_scores` returns each row's score for label `1`. It looks up which output column belongs to label `1` instead of assuming it is the second one.

### 4.2 The settings to try

A setting ("hyperparameter") controls how flexible a model is. Too rigid and the model misses real patterns. Too flexible and it memorises the training matches (overfits).

| Model | Setting | Values tried | Meaning |
|---|---|---|---|
| Logistic regression | `C` | 0.01, 0.1, 1, 10, 100 | Small `C` = simpler. `C` is any positive number; it is the model's *output* that lies between 0 and 1. |
| Decision tree | `max_depth` | 2, 3, 4, 5, 6 | How many questions in a row |
| Decision tree | `min_samples_leaf` | 2, 5, 10 | Fewest rows allowed in a final group |
| KNN | `n_neighbors` | 3, 5, 7, 9, 11, 13, 15 | How many neighbours vote (odd, to avoid tied votes) |

`simplicity_key` returns a sort key where lower means simpler: smaller `C`, shallower tree with larger leaves, larger `k`. It is used only to break an exact tie.

### 4.3 Grouped 5-fold cross-validation

Cross-validation (CV) chooses settings **without touching the test set**:

1. Divide the 102 training matches into 5 folds of about 20 matches. Both rows of a match stay in the same fold (`GroupKFold` with `groups=match_id`).
2. Hold out one fold and train on the other four.
3. Rotate until every fold has been held out once.
4. Average the 5 scores. That is the setting's CV score.
5. Repeat for every setting of every model.

```python
folds = list(GroupKFold(n_splits=n_splits).split(X, y, groups=train["match_id"]))
for model_name, base_model in build_models().items():
    for params in ParameterGrid(GRIDS[model_name]):
        for fit_idx, valid_idx in folds:
            candidate = clone(base_model).set_params(**params)
            candidate.fit(X.iloc[fit_idx], y.iloc[fit_idx])
            scores = positive_scores(candidate, X.iloc[valid_idx])
            fold_scores.append(match_accuracy(train.iloc[valid_idx], scores))
```

Details that matter:

- The folds are built **once**, so every model and every setting is checked on exactly the same folds.
- `clone` makes a fresh, untrained copy for each fold, so nothing learned in one fold carries into the next.
- Each fold is scored with `match_accuracy`, the same measure as the final result.

The function returns one row per (model, setting) with `cv_mean`, `cv_std` and the five `fold_scores`. That is 5 + 15 + 7 = 27 settings, each trained 5 times.

**Why not tune on the test set:** choosing settings by test score would be tuning to the test matches, and they would stop being a fair exam.

### 4.4 Pick the best and refit

`best_settings` keeps, for each model, the setting with the highest `cv_mean`. Exact ties (compared with `np.isclose`, which handles decimals safely) go to the simpler setting.

`fit_best` then trains each model with its winning setting on **all 102** training matches. During CV each model only ever saw four fifths of the training data, so the final model is refitted on all of it.

### 4.5 and 4.6 Running it

The search runs twice, once per feature set. The model with the best CV score in each set is recorded in `selected` **before** the test matches are looked at.

| Feature set | Model | Best setting | CV mean | CV std |
|---|---|---|---|---|
| with goals | Logistic regression | `C = 0.01` | 0.874 | 0.115 |
| with goals | Decision tree | `max_depth = 2`, `min_samples_leaf = 10` | 0.833 | 0.066 |
| with goals | KNN | `k = 11` | 0.774 | 0.050 |
| no goals | Logistic regression | `C = 10` | 0.688 | 0.111 |
| no goals | KNN | `k = 3` | 0.675 | 0.147 |
| no goals | Decision tree | `max_depth = 3`, `min_samples_leaf = 10` | 0.598 | 0.159 |

Logistic regression wins in both sets, so **logistic regression (no goals)** is the main model. (`C = 10` and `C = 100` tied exactly on the no-goals set, and the tie rule kept the simpler `C = 10`.)

`cv_std` is the spread between the 5 folds. It is not a confidence interval. Each fold has about 20 matches, so one match moves a fold's score by about 5 points.

## Phase 5: Evaluation

This is the first and only time the 26 test matches are used. Whatever the numbers are, nothing is adjusted afterwards.

### 5.1 The results table

The baseline and all six tuned models are graded on the same 26 matches by `match_predictions`. `summarise` turns each comparison into one table line (correct matches, total, accuracy). Each model is given exactly the feature columns it was trained on.

### 5.2 By tournament and on draws

`breakdown` counts each method's correct picks in 2018, in 2022 (13 test matches each) and on the 8 drawn test matches.

### 5.3 The main model's mistakes

The cell lists the test matches the main model picked wrongly. `selected_score` ranks the two teams within a match. It is not a calibrated probability: each row is scored separately, so the two scores in a match do not have to add up to 1.

### 5.4 Which statistics mattered

Two views, both measured on **training data only**:

- **Permutation importance:** shuffle one statistic's column and measure how much the model's score drops. A bigger drop means the model relied on it more. Each statistic is shuffled 20 times and the drops are averaged.
- **Logistic regression weights:** the number each scaled statistic is multiplied by. Positive pushes towards "the award came from this team", negative pushes away.

This part is exploratory. Statistics that move together (attempts, shots on target, corners) share importance between them, and a weight shows an association, not a cause.

## Results

### Test set (26 matches)

| Method | With goals | No goals |
|---|---|---|
| Baseline rule | 21/26 (80.8%) | n/a |
| Logistic regression | 19/26 (73.1%) | **18/26 (69.2%)** (main model) |
| Decision tree | 18/26 (69.2%) | 13/26 (50.0%) |
| KNN | 17/26 (65.4%) | 11/26 (42.3%) |

### Breakdown

| Method | 2018 (/13) | 2022 (/13) | Draws (/8) |
|---|---|---|---|
| Baseline rule | 11 | 10 | 4 |
| Logistic regression (with goals) | 11 | 8 | 3 |
| Decision tree (with goals) | 10 | 8 | 2 |
| KNN (with goals) | 8 | 9 | 3 |
| Logistic regression (no goals) | 11 | 7 | 3 |
| Decision tree (no goals) | 6 | 7 | 3 |
| KNN (no goals) | 6 | 5 | 2 |

### What the results say

1. **No model beat the baseline.** The rule got 21 of 26 and the best model got 19. Two matches is within the noise of a 26-match test, so the rule cannot be called truly better, but the models added nothing measurable on top of it.
2. **Goals carry most of the answer.** With goals, every model reached 17 to 19. The extra statistics did not improve on "pick the team that scored more".
3. **How a team played still says something.** The main model never saw a goal and still picked 18 of 26, where a coin flip would expect 13. Its strongest signal was shots on target. With 26 matches this is a hint, not proof.
4. **More flexibility did not help.** Without goals, the tree (13/26) and KNN (11/26) did no better than a coin flip. KNN scored 0.67 in CV and 0.42 on the test matches, which shows how unstable results are on data this small.
5. **Draws stayed hard.** On the 8 drawn test matches the baseline got 4 and the main model got 3.
6. **2018 went better than 2022** for the main model (11/13 against 7/13). The tournaments were measured differently and 13 matches is very few, so no cause is read into the gap.

### What the main model leaned on

| Statistic | Permutation importance | Weight |
|---|---|---|
| On-Target | 0.126 (highest) | +0.990 (largest positive) |
| Ball Possession % | 0.054 | −0.797 (largest negative) |
| Corners | 0.013 | +0.374 |
| Free Kicks | 0.015 | +0.342 |
| Blocked | 0.022 | −0.447 |

The negative possession weight does not mean having the ball hurts a team. It means that, among teams with the same shots, passes and corners, the one with more possession was less often the award side.

## Limitations

- **Small test set.** 26 matches, so one match is 3.85 points. Gaps of one or two matches are noise.
- **Two different sources.** Shared column names do not guarantee that 2018 and 2022 measured the same thing in the same way. Possession is the clear example.
- **Label sources.** The 2018 labels come from a supplied statistics file and the 2022 labels from an online list. They should be checked against official records before any use beyond coursework.
- **No per-90 scaling.** Matches with extra time have larger raw counts, and there is no minutes column to correct for it.
- **After the match, team only.** The model needs full-time statistics and picks the team, not the player.

## Function reference

All of these are defined in the notebook, in the phase shown.

| Function | Phase | What it does |
|---|---|---|
| `split_by_match(frame)` | 2 | Splits whole matches 80/20, stratified by year. Returns `(train, test)`. |
| `match_predictions(frame, scores)` | 3 | Picks the higher-scoring team per match and grades it. One row per match. |
| `match_accuracy(frame, scores)` | 3 | Share of matches picked correctly. |
| `baseline_scores(frame, seed)` | 3 | The hand-written rule: 1 for the team it picks, 0 for the other. |
| `drawn_match_ids(frame)` | 3 | Ids of matches where the goals were level. |
| `build_models()` | 4 | The three untrained pipelines. |
| `positive_scores(fitted, features)` | 4 | A fitted model's score for label 1 on each row. |
| `simplicity_key(model_name, params)` | 4 | Sort key where lower means simpler. Breaks CV ties. |
| `grouped_cv_search(train, features)` | 4 | Scores every setting of every model with grouped 5-fold CV. |
| `best_settings(search_results)` | 4 | The winning setting per model. |
| `fit_best(train, features, best)` | 4 | Refits each winner on all training matches. |
| `summarise(name, comparison)` | 5 | One line of the results table. |
| `breakdown(comparison)` | 5 | Correct picks per year and on draws. |

Constants: `DATASET_PATH`, `TARGET`, `RANDOM_STATE`, `FEATURES_WITH_GOALS`, `SCORE_LEAK_COLUMNS`, `FEATURES_NO_GOALS`, `FEATURE_SETS`, `BASELINE_ORDER`, `GRIDS`.

### Self-checks

Four cells titled "Self-check" run the functions on tiny hand-made matches where the right answer is known in advance. The helpers `tiny_match` and `tiny_frame` build those matches.

| Self-check | What it proves |
|---|---|
| Match-level scoring | One pick per match, graded correctly; ties go alphabetically; a wrong number of scores raises an error |
| Baseline rule | Each tie-break step works in order; the coin flip is repeatable and picks exactly one team; row order does not matter |
| Grids and pipelines | Every model has a grid; the scaler is in logistic regression and KNN but not the tree; the simplicity order is right |
| Tie-breaking | An exact CV tie goes to the simpler setting |

If a check fails, the notebook stops there, so a mistake in a function cannot quietly reach the results.

## Glossary

| Term | Meaning |
|---|---|
| Feature | An input column the model learns from |
| Label / target | The answer column (`Man of the Match`) |
| Training set | The matches the model learns from (102) |
| Test set | The matches held back for the final, one-time exam (26) |
| Data leakage | Information about the answer reaching the model when it should not |
| Baseline | A simple rule with no training, used as the bar to beat |
| Hyperparameter | A setting chosen before training, such as `C`, `max_depth` or `k` |
| Overfitting | Memorising the training data instead of learning a pattern that carries over |
| Cross-validation | Scoring a setting by training and checking on rotating parts of the training set |
| Fold | One of the parts the training set is divided into for cross-validation |
| Pipeline | Preparation steps and a model chained into one object |
| Imputer | A step that fills in missing values |
| Scaling | Putting every column on a similar scale (subtract the mean, divide by the spread) |
| Permutation importance | How much a model's score drops when one column is shuffled |
| Match accuracy | The share of matches where the picked team supplied the Man of the Match |
