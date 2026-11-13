# Phase 1 — Inspect and validate the data

## Notebook Markdown cell

> **Question and unit of analysis.** I predict whether the Man of the Match winner came from a given team after a match has ended. A match contributes two team rows, one labeled `1` and the other `0`. The two rows are linked observations, so later splitting and evaluation happen at the match level.

> **Data provenance.** The file contains 2018 and 2022 men's World Cup matches. The workbook's `statistics_source`, `award_source`, and `statistics_method` columns document where the data came from. The 2022 award labels are listed as `reported_online`, whereas 2018 labels are `reported`. I will retain these fields for auditing but never use them as predictors.

## Notebook code cell — load and summarize

Install packages in your own environment if needed: `python -m pip install pandas openpyxl numpy scikit-learn`.

```python
import numpy as np
import pandas as pd

DATA_PATH = "world_cup_team_statistics.xlsx"
TARGET = "Man of the Match"

df = pd.read_excel(DATA_PATH, sheet_name="world_cup_team_statistics")
print("Rows, columns:", df.shape)
print("Matches:", df["match_id"].nunique())
print("Tournament rows:\n", df["year"].value_counts().sort_index())
print("Target counts:\n", df[TARGET].value_counts().sort_index())
display(df.head())
display(df.dtypes.to_frame("dtype"))
```

Expected shape: `(256, 45)`. The target counts should be 128 zeros and 128 ones. The notebook is currently empty, so these cells start the analysis.

## Notebook code cell — check match integrity

```python
assert df["match_id"].notna().all()
assert df["Team"].notna().all()
assert df["Opponent"].notna().all()
assert df[TARGET].notna().all()
assert set(df[TARGET].unique()) == {0, 1}

match_check = df.groupby("match_id").agg(
    rows=(TARGET, "size"),
    positive_labels=(TARGET, "sum"),
    years=("year", "nunique"),
)
assert (match_check["rows"] == 2).all()
assert (match_check["positive_labels"] == 1).all()
assert (match_check["years"] == 1).all()
assert not df.duplicated(["match_id", "Team"]).any()

for _, pair in df.groupby("match_id"):
    a, b = pair.iloc[0], pair.iloc[1]
    assert a["Team"] == b["Opponent"]
    assert b["Team"] == a["Opponent"]

print("All matches have two reciprocal teams and one positive label.")
```

These assertions should stop the notebook if the file changes in a way that invalidates the setup. Inspect any failure before fitting a model.

## Notebook code cell — inspect missingness and source differences

```python
missing = df.isna().sum().sort_values(ascending=False)
display(missing[missing > 0].to_frame("missing_rows"))

source_columns = [
    "award_status", "statistics_source", "award_source", "statistics_method"
]
for column in source_columns:
    print(f"\n{column}")
    display(pd.crosstab(df["year"], df[column]))

display(
    df.groupby("year")[[
        "Ball Possession %", "Pass Accuracy %", "Passes",
        "Distance Covered (Kms)", "Attempts"
    ]].agg(["mean", "min", "max"]).round(2)
)
```

## Notebook Markdown cell — findings to record

> The target is balanced by construction, but the effective sample size is 128 matches rather than 256 independent decisions. `passes_completed`, `possession_in_contest_pct`, `incomplete_attempts`, and other supplemental columns are available only for 2022. The 2022 `statistics_method` says possession was normalized to exclude an in-contest share and pass accuracy was computed. I will use shared match-stat columns for the main comparison and report year-specific results later.

> The `award_winner_team` column exactly identifies the positive row. `award_winner_player` and `award_winner_player_id` also disclose the winner. They are label metadata, not inputs. `match_id`, team names, year, dates, source fields, URLs, and award status are also excluded from the first numerical-stat model.
