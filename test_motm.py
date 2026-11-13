"""Tests for motm.py. Run with: python -m pytest -q"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
import pytest

import motm
from motm import TARGET


def make_match(
    match_id: str,
    teams: Tuple[str, str],
    goals: Tuple[int, int],
    on_target: Tuple[int, int],
    possession: Tuple[float, float],
    motm_index: int,
    year: int = 2018,
) -> List[dict]:
    """Two mirrored team rows for one match; motm_index marks the MOTM side."""
    return [
        {
            "match_id": match_id,
            "year": year,
            "Team": teams[i],
            "Goal Scored": goals[i],
            "On-Target": on_target[i],
            "Ball Possession %": possession[i],
            TARGET: int(i == motm_index),
        }
        for i in (0, 1)
    ]


def frame(*matches: List[dict]) -> pd.DataFrame:
    return pd.DataFrame([row for match in matches for row in match])


def picked_teams(data: pd.DataFrame, seed: int = 42) -> List[str]:
    predictions = motm.match_predictions(data, motm.baseline_scores(data, seed))
    return predictions.sort_values("match_id")["predicted_team"].tolist()


# --- features ---------------------------------------------------------------


def test_no_goals_set_drops_exactly_the_score_columns() -> None:
    assert len(motm.FEATURES_WITH_GOALS) == 19
    assert len(motm.FEATURES_NO_GOALS) == 16
    assert set(motm.FEATURES_WITH_GOALS) - set(motm.FEATURES_NO_GOALS) == set(
        motm.SCORE_LEAK_COLUMNS
    )
    for features in motm.FEATURE_SETS.values():
        assert TARGET not in features


# --- baseline tiebreak chain ------------------------------------------------


def test_baseline_picks_more_goals_first() -> None:
    data = frame(make_match("m1", ("A", "B"), (0, 2), (9, 1), (70, 30), 1))
    assert picked_teams(data) == ["B"]


def test_baseline_uses_shots_on_target_when_goals_tie() -> None:
    data = frame(make_match("m1", ("A", "B"), (1, 1), (2, 6), (70, 30), 1))
    assert picked_teams(data) == ["B"]


def test_baseline_uses_possession_when_goals_and_shots_tie() -> None:
    data = frame(make_match("m1", ("A", "B"), (1, 1), (4, 4), (45, 55), 1))
    assert picked_teams(data) == ["B"]


def test_baseline_coin_flip_is_reproducible_and_always_picks_one() -> None:
    tied = [
        make_match(f"m{i}", (f"A{i}", f"B{i}"), (0, 0), (3, 3), (50, 50), 0)
        for i in range(20)
    ]
    data = frame(*tied)
    scores = motm.baseline_scores(data, seed=7)
    # Exactly one pick per match, and the same picks on a rerun.
    assert scores.reshape(-1, 2).sum(axis=1).tolist() == [1.0] * 20
    assert picked_teams(data, seed=7) == picked_teams(data, seed=7)
    # A fair coin should not always land on the same side.
    assert len({team[0] for team in picked_teams(data, seed=7)}) == 2


def test_baseline_does_not_depend_on_row_order() -> None:
    data = frame(
        make_match("m1", ("A", "B"), (1, 1), (2, 6), (70, 30), 1),
        make_match("m2", ("C", "D"), (3, 0), (5, 5), (40, 60), 0),
    )
    shuffled = data.iloc[[3, 0, 2, 1]].reset_index(drop=True)
    assert sorted(picked_teams(data)) == sorted(picked_teams(shuffled)) == ["B", "C"]


# --- match-level scoring ----------------------------------------------------


def test_match_predictions_grades_one_pick_per_match() -> None:
    data = frame(
        make_match("m1", ("A", "B"), (1, 0), (1, 1), (50, 50), 0),
        make_match("m2", ("C", "D"), (1, 0), (1, 1), (50, 50), 1),
    )
    comparison = motm.match_predictions(data, [0.9, 0.1, 0.8, 0.3])
    assert comparison["predicted_team"].tolist() == ["A", "C"]
    assert comparison["correct"].tolist() == [True, False]
    assert motm.match_accuracy(data, [0.9, 0.1, 0.8, 0.3]) == 0.5


def test_match_predictions_breaks_exact_ties_alphabetically() -> None:
    data = frame(make_match("m1", ("Zambia", "Angola"), (0, 0), (1, 1), (50, 50), 1))
    comparison = motm.match_predictions(data, [0.5, 0.5])
    assert comparison["predicted_team"].tolist() == ["Angola"]


def test_match_predictions_rejects_wrong_number_of_scores() -> None:
    data = frame(make_match("m1", ("A", "B"), (0, 0), (1, 1), (50, 50), 1))
    with pytest.raises(ValueError):
        motm.match_predictions(data, [0.5])


def test_drawn_match_ids_finds_level_scores() -> None:
    data = frame(
        make_match("m1", ("A", "B"), (1, 1), (1, 1), (50, 50), 1),
        make_match("m2", ("C", "D"), (2, 0), (1, 1), (50, 50), 0),
    )
    assert motm.drawn_match_ids(data) == ["m1"]


# --- split ------------------------------------------------------------------


def test_split_keeps_both_rows_of_a_match_together() -> None:
    matches = [
        make_match(f"m{i}", (f"A{i}", f"B{i}"), (1, 0), (1, 1), (50, 50), 0,
                   year=2018 if i % 2 else 2022)
        for i in range(20)
    ]
    split = motm.split_by_match(frame(*matches))
    assert set(split.train["match_id"]).isdisjoint(split.test["match_id"])
    assert (split.train.groupby("match_id").size() == 2).all()
    assert (split.test.groupby("match_id").size() == 2).all()
    assert split.test["match_id"].nunique() == 4
    # Stratified: each year gets the same share of test matches.
    assert split.test.groupby("year")["match_id"].nunique().tolist() == [2, 2]


# --- tuning helpers ---------------------------------------------------------


def test_simplicity_prefers_small_c_shallow_trees_and_large_k() -> None:
    key = motm.simplicity_key
    assert key("Logistic regression", {"model__C": 0.1}) < key(
        "Logistic regression", {"model__C": 10.0})
    assert key("Decision tree", {"model__max_depth": 2, "model__min_samples_leaf": 5}) < key(
        "Decision tree", {"model__max_depth": 4, "model__min_samples_leaf": 5})
    assert key("KNN", {"model__n_neighbors": 15}) < key("KNN", {"model__n_neighbors": 3})


def test_best_settings_breaks_exact_cv_ties_toward_simpler() -> None:
    results = pd.DataFrame([
        {"model": "KNN", "params": {"model__n_neighbors": 3}, "cv_mean": 0.7},
        {"model": "KNN", "params": {"model__n_neighbors": 11}, "cv_mean": 0.7},
        {"model": "KNN", "params": {"model__n_neighbors": 5}, "cv_mean": 0.6},
    ])
    best = motm.best_settings(results)
    assert best.loc[0, "params"] == {"model__n_neighbors": 11}


def test_grids_match_the_agreed_ranges() -> None:
    assert motm.GRIDS["KNN"]["model__n_neighbors"] == [3, 5, 7, 9, 11, 13, 15]
    assert motm.GRIDS["Decision tree"]["model__max_depth"] == [2, 3, 4, 5, 6]
    assert motm.GRIDS["Logistic regression"]["model__C"] == [0.01, 0.1, 1.0, 10.0, 100.0]
    assert set(motm.GRIDS) == set(motm.build_models())


def test_scaler_sits_inside_lr_and_knn_pipelines_only() -> None:
    models = motm.build_models()
    assert "scaler" in models["Logistic regression"].named_steps
    assert "scaler" in models["KNN"].named_steps
    assert "scaler" not in models["Decision tree"].named_steps


def test_grouped_cv_search_runs_on_a_tiny_dataset() -> None:
    rng = np.random.default_rng(0)
    rows = []
    for i in range(30):
        motm_side = int(rng.integers(0, 2))
        for side in (0, 1):
            row = {name: float(rng.integers(0, 10)) for name in motm.FEATURES_NO_GOALS}
            row.update({"match_id": f"m{i}", "year": 2018, "Team": f"T{i}{side}",
                        TARGET: int(side == motm_side)})
            # Give the MOTM side more shots so there is something to learn.
            row["On-Target"] += 5 * row[TARGET]
            rows.append(row)
    data = pd.DataFrame(rows)
    results = motm.grouped_cv_search(data, motm.FEATURES_NO_GOALS)
    expected_runs = sum(len(list(motm.ParameterGrid(g))) for g in motm.GRIDS.values())
    assert len(results) == expected_runs
    best = motm.best_settings(results)
    assert set(best["model"]) == set(motm.GRIDS)
    fitted = motm.fit_best(data, motm.FEATURES_NO_GOALS, best)
    scores = motm.positive_scores(fitted["KNN"], data[list(motm.FEATURES_NO_GOALS)])
    assert scores.shape == (60,)
