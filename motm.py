"""Man of the Match predictor: the logic behind phases 2 to 5.

The notebook imports these functions so it can stay a readable story. Each
function is small and pure (it returns new data and never edits its inputs),
which is what lets `test_motm.py` check them on tiny hand-made matches.

The question every function serves: for a finished match, which of the two
teams did the Man of the Match come from?
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, ParameterGrid, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

TARGET = "Man of the Match"
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Phase 2: features
# ---------------------------------------------------------------------------

# The 19 numeric stats that both 2018 and 2022 record. Award, ID, source and
# 2022-only columns are left out on purpose (see the notebook's feature policy).
FEATURES_WITH_GOALS: Tuple[str, ...] = (
    "Goal Scored", "Ball Possession %", "Attempts", "On-Target",
    "Off-Target", "Blocked", "Corners", "Offsides", "Free Kicks",
    "Saves", "Pass Accuracy %", "Passes", "Distance Covered (Kms)",
    "Fouls Committed", "Yellow Card", "Yellow & Red", "Red",
    "Goals in PSO", "Own goals",
)

# Columns that give away the scoreline (and so, nearly, the answer).
# Shootout goals reveal who advanced; own goals are part of the score.
SCORE_LEAK_COLUMNS: Tuple[str, ...] = ("Goal Scored", "Goals in PSO", "Own goals")

FEATURES_NO_GOALS: Tuple[str, ...] = tuple(
    column for column in FEATURES_WITH_GOALS if column not in SCORE_LEAK_COLUMNS
)

FEATURE_SETS: Dict[str, Tuple[str, ...]] = {
    "with goals": FEATURES_WITH_GOALS,
    "no goals": FEATURES_NO_GOALS,
}


def coerce_features(frame: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    """Return a copy with feature columns forced to numbers (bad text -> NaN)."""
    result = frame.copy().reset_index(drop=True)
    result[list(features)] = result[list(features)].apply(pd.to_numeric, errors="coerce")
    return result


# ---------------------------------------------------------------------------
# Phase 2: the grouped split
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Split:
    """The training and test team rows. Both rows of a match share a side."""

    train: pd.DataFrame
    test: pd.DataFrame


def split_by_match(
    frame: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = RANDOM_STATE,
) -> Split:
    """Split whole matches (not rows) 80/20, stratified by tournament year.

    Splitting one row per match_id is what keeps a match's two mirrored rows
    together; a row-level split would let the model peek at the opponent.
    """
    match_meta = frame[["match_id", "year"]].drop_duplicates("match_id")
    train_ids, test_ids = train_test_split(
        match_meta["match_id"],
        test_size=test_size,
        random_state=random_state,
        stratify=match_meta["year"],
    )
    train = frame[frame["match_id"].isin(train_ids)].copy()
    test = frame[frame["match_id"].isin(test_ids)].copy()
    return Split(train=train, test=test)


# ---------------------------------------------------------------------------
# Phase 3: match-level scoring and the baseline rule
# ---------------------------------------------------------------------------


def match_predictions(frame: pd.DataFrame, scores: Sequence[float]) -> pd.DataFrame:
    """Pick the higher-scoring team in each match and grade it.

    Returns one row per match: predicted_team, actual_team, year, correct.
    An exact score tie goes to the alphabetically first team, so the result
    is the same on every run.
    """
    scores_array = np.asarray(scores, dtype=float)
    if len(scores_array) != len(frame):
        raise ValueError("Need exactly one score per team row.")

    result = frame[["match_id", "Team", "year", TARGET]].copy()
    result["score"] = scores_array

    predicted = (
        result.sort_values(["match_id", "score", "Team"], ascending=[True, False, True])
        .drop_duplicates("match_id")[["match_id", "Team", "score"]]
        .rename(columns={"Team": "predicted_team", "score": "selected_score"})
    )
    actual = result[result[TARGET] == 1][["match_id", "Team", "year"]].rename(
        columns={"Team": "actual_team"}
    )
    comparison = predicted.merge(actual, on="match_id", validate="one_to_one")
    comparison["correct"] = comparison["predicted_team"] == comparison["actual_team"]
    return comparison.reset_index(drop=True)


def match_accuracy(frame: pd.DataFrame, scores: Sequence[float]) -> float:
    """Share of matches where the picked team supplied the Man of the Match."""
    return float(match_predictions(frame, scores)["correct"].mean())


# The baseline compares these columns in order; the first difference decides.
BASELINE_ORDER: Tuple[str, ...] = ("Goal Scored", "On-Target", "Ball Possession %")


def baseline_scores(frame: pd.DataFrame, seed: int = RANDOM_STATE) -> np.ndarray:
    """Hand-written rule, no training: 1 for the team the rule picks, 0 otherwise.

    More goals wins; if tied, more shots on target; if tied, more possession;
    if even that is tied, a coin flip with a fixed seed (same answer each run).
    """
    coin = np.random.default_rng(seed).random(len(frame))
    ranked = frame[["match_id", *BASELINE_ORDER]].copy()
    ranked["_coin"] = coin
    ranked["_row"] = np.arange(len(frame))

    # Sorting by every tiebreak at once compares them lexicographically, like
    # sorting words letter by letter. The last row of each match is the pick.
    ranked = ranked.sort_values(["match_id", *BASELINE_ORDER, "_coin"])
    picked_rows = ranked.groupby("match_id")["_row"].last().to_numpy()

    scores = np.zeros(len(frame))
    scores[picked_rows] = 1.0
    return scores


def drawn_match_ids(frame: pd.DataFrame) -> List[Any]:
    """Matches where both teams scored the same number of goals."""
    goals = frame.groupby("match_id")["Goal Scored"].agg(["min", "max"])
    return goals.index[goals["min"] == goals["max"]].tolist()


# ---------------------------------------------------------------------------
# Phase 4: models, grids and grouped cross-validation
# ---------------------------------------------------------------------------


def build_models() -> Dict[str, Pipeline]:
    """Three models from simple to more flexible, each inside a Pipeline.

    The imputer and scaler are pipeline steps so they are fit on the training
    folds only, never on the rows being scored.
    """
    return {
        "Logistic regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ]),
        # A tree only asks "is this stat above a threshold?", so scaling
        # would not change its splits.
        "Decision tree": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", DecisionTreeClassifier(random_state=RANDOM_STATE)),
        ]),
        # KNN compares distances, so unscaled passes (hundreds) would drown
        # out cards (0 to 3). The scaler is required here.
        "KNN": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier()),
        ]),
    }


GRIDS: Dict[str, Dict[str, List[Any]]] = {
    "Logistic regression": {"model__C": [0.01, 0.1, 1.0, 10.0, 100.0]},
    "Decision tree": {
        "model__max_depth": [2, 3, 4, 5, 6],
        "model__min_samples_leaf": [2, 5, 10],
    },
    "KNN": {"model__n_neighbors": [3, 5, 7, 9, 11, 13, 15]},
}


def simplicity_key(model_name: str, params: Mapping[str, Any]) -> Tuple[float, ...]:
    """Lower means simpler. Used only to break an exact cross-validation tie.

    Simpler = smaller C, shallower tree with bigger leaves, larger k.
    """
    if model_name == "Logistic regression":
        return (params["model__C"],)
    if model_name == "Decision tree":
        return (params["model__max_depth"], -params["model__min_samples_leaf"])
    if model_name == "KNN":
        return (-params["model__n_neighbors"],)
    raise ValueError(f"Unknown model: {model_name}")


def positive_scores(fitted: Pipeline, features: pd.DataFrame) -> np.ndarray:
    """The model's score for label 1 (MOTM came from this team) on each row."""
    positive_column = list(fitted.classes_).index(1)
    return fitted.predict_proba(features)[:, positive_column]


def grouped_cv_search(
    train: pd.DataFrame,
    features: Sequence[str],
    n_splits: int = 5,
) -> pd.DataFrame:
    """Try every setting of every model with 5-fold grouped CV on training rows.

    Each fold holds out about 20 whole matches, trains on the other ~80,
    and is scored by match accuracy. The test set is never touched here.
    """
    X = train[list(features)]
    y = train[TARGET]
    folds = list(GroupKFold(n_splits=n_splits).split(X, y, groups=train["match_id"]))

    rows = []
    for model_name, base in build_models().items():
        for params in ParameterGrid(GRIDS[model_name]):
            fold_scores = []
            for fit_idx, valid_idx in folds:
                candidate = clone(base).set_params(**params)
                candidate.fit(X.iloc[fit_idx], y.iloc[fit_idx])
                scores = positive_scores(candidate, X.iloc[valid_idx])
                fold_scores.append(match_accuracy(train.iloc[valid_idx], scores))
            rows.append({
                "model": model_name,
                "params": params,
                "cv_mean": float(np.mean(fold_scores)),
                "cv_std": float(np.std(fold_scores)),
                "fold_scores": [round(score, 3) for score in fold_scores],
            })
    return pd.DataFrame(rows)


def best_settings(search_results: pd.DataFrame) -> pd.DataFrame:
    """Best setting per model: highest CV mean, exact ties go to the simpler one."""
    ranked = search_results.assign(
        _simplicity=[
            simplicity_key(row.model, row.params) for row in search_results.itertuples()
        ]
    ).sort_values(["model", "cv_mean"], ascending=[True, False], kind="stable")

    def pick(group: pd.DataFrame) -> pd.Series:
        top = group[np.isclose(group["cv_mean"], group["cv_mean"].max())]
        return top.sort_values("_simplicity").iloc[0]

    best = [pick(group) for _, group in ranked.groupby("model")]
    return (
        pd.DataFrame(best)
        .drop(columns="_simplicity")
        .sort_values("cv_mean", ascending=False)
        .reset_index(drop=True)
    )


def fit_best(
    train: pd.DataFrame,
    features: Sequence[str],
    best: pd.DataFrame,
) -> Dict[str, Pipeline]:
    """Refit each model's winning setting on all training matches."""
    models = build_models()
    fitted = {}
    for row in best.itertuples():
        pipeline = clone(models[row.model]).set_params(**row.params)
        fitted[row.model] = pipeline.fit(train[list(features)], train[TARGET])
    return fitted


# ---------------------------------------------------------------------------
# Phase 5: summarising results
# ---------------------------------------------------------------------------


def summarise(name: str, comparison: pd.DataFrame) -> Dict[str, Any]:
    """One results-table row: correct matches out of total."""
    return {
        "method": name,
        "correct_matches": int(comparison["correct"].sum()),
        "total_matches": len(comparison),
        "match_accuracy": round(float(comparison["correct"].mean()), 3),
    }
