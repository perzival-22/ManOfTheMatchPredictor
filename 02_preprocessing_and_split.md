# Phase 2 — Prepare features and split matches

## Notebook Markdown cell

> **Feature policy.** I use numerical team statistics available for both 2018 and 2022. These are measured during or after the match, matching the post-match question. I exclude target-derived award fields, identifiers, source metadata, and 2022-only measurements. The 2018 and 2022 sources may define some shared statistics differently; I will check performance separately by year.

The first 17 columns below are the common core match stats. `Goals in PSO` and `Own goals` are also numeric and present in both tournaments. Include them because this prediction is made after the match and shootout. `1st Goal` and `Own goal Time` are event times with many structurally blank values (and values such as `90+5`), so leave them for an optional later experiment with explicit parsing and missingness indicators.

## Notebook code cell — select and check features

```python
FEATURES = [
    "Goal Scored", "Ball Possession %", "Attempts", "On-Target",
    "Off-Target", "Blocked", "Corners", "Offsides", "Free Kicks",
    "Saves", "Pass Accuracy %", "Passes", "Distance Covered (Kms)",
    "Fouls Committed", "Yellow Card", "Yellow & Red", "Red",
    "Goals in PSO", "Own goals",
]

assert TARGET not in FEATURES
assert len(FEATURES) == len(set(FEATURES))
assert set(FEATURES).issubset(df.columns)

# Coerce unexpected text to NaN. Imputation is fitted later using training data only.
df = df.copy().reset_index(drop=True)
df[FEATURES] = df[FEATURES].apply(pd.to_numeric, errors="coerce")
display(df[FEATURES].isna().sum().to_frame("missing_after_conversion"))

# If a whole column has no observed training values, reconsider that feature.
assert not df[FEATURES].isna().all().any()
```

The current workbook has no missing values in these 19 features. The imputer in each model pipeline is still useful if new rows have missing values later.

## Notebook code cell — group-aware 80/20 split

```python
from sklearn.model_selection import train_test_split

# One record per match is used for splitting. Year stratification puts both
# tournaments in each partition while keeping the two team rows together.
match_meta = df[["match_id", "year"]].drop_duplicates("match_id")
train_ids, test_ids = train_test_split(
    match_meta["match_id"],
    test_size=0.20,
    random_state=42,
    stratify=match_meta["year"],
)

train_df = df[df["match_id"].isin(train_ids)].copy()
test_df = df[df["match_id"].isin(test_ids)].copy()

X_train, y_train = train_df[FEATURES], train_df[TARGET]
X_test, y_test = test_df[FEATURES], test_df[TARGET]

assert set(train_df["match_id"]).isdisjoint(test_df["match_id"])
assert train_df["match_id"].nunique() == 102
assert test_df["match_id"].nunique() == 26
assert len(train_df) == 204 and len(test_df) == 52
assert y_train.sum() == 102 and y_test.sum() == 26

print("Training matches:", train_df["match_id"].nunique())
print("Test matches:", test_df["match_id"].nunique())
print("Train year counts:\n", train_df.groupby("year")["match_id"].nunique())
print("Test year counts:\n", test_df.groupby("year")["match_id"].nunique())
```

## Notebook Markdown cell

> The test set represents about 20% of **matches**, not a random 20% of team rows. Randomly separating the two teams from one match would leak information across the boundary because their statistics and outcome are tightly linked. I will fit imputation and logistic-regression scaling inside training pipelines only. Hyperparameter choices will use grouped cross-validation inside the 102 training matches. The 26 test matches will be examined once, after model selection.

**Adequacy:** 102 training matches support a small, regularized comparison, but 26 test matches are not enough for a precise claim that one algorithm is universally best. One corrected match changes test match accuracy by about 3.85 percentage points. More tournaments or comparable matches would improve confidence, provided measurement definitions and award labels remain consistent.
