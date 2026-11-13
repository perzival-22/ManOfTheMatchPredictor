# Phase 3 — Establish comparison baselines

## Notebook Markdown cell

> **Primary metric.** The product makes one choice per match, so the main metric is the proportion of matches for which the selected team has target `1`. A row-by-row accuracy score can hide invalid outcomes such as predicting `1` for both teams. I choose the team with the higher model score within each match, breaking an exact tie alphabetically so the result is reproducible.

> **Reference points.** Randomly choosing one of two teams is expected to get about 50% of matches right. A simple post-match rule chooses the team with more goals, then more shots on target if goals are tied. This is a meaningful comparator because a learned model should add value beyond the most obvious match outcome stats.

## Notebook code cell — match-level scoring helper

```python
def match_predictions(frame, scores):
    """Return one predicted and one actual winner per match."""
    result = frame[["match_id", "Team", "year", TARGET]].copy()
    result["score"] = np.asarray(scores)
    assert len(result) == len(scores)

    predicted = (
        result.sort_values(
            ["match_id", "score", "Team"],
            ascending=[True, False, True],
        )
        .drop_duplicates("match_id")
        [["match_id", "Team", "score"]]
        .rename(columns={"Team": "predicted_team", "score": "selected_score"})
    )
    actual = (
        result[result[TARGET] == 1][["match_id", "Team", "year"]]
        .rename(columns={"Team": "actual_team"})
    )
    comparison = predicted.merge(actual, on="match_id", validate="one_to_one")
    comparison["correct"] = comparison["predicted_team"] == comparison["actual_team"]
    return comparison


def match_accuracy(frame, scores):
    return match_predictions(frame, scores)["correct"].mean()
```

## Notebook code cell — simple training-set rule

```python
# Score = goals first, then shots on target to break a goal tie.
# The tiny multiplier cannot outweigh a one-goal difference in this dataset.
goal_rule_scores = train_df["Goal Scored"] + 0.001 * train_df["On-Target"]
goal_rule_train_accuracy = match_accuracy(train_df, goal_rule_scores)
print(f"Goal/shots rule on training matches: {goal_rule_train_accuracy:.3f}")
print("Random-choice expectation: 0.500")
```

The goal rule uses neither fitting nor the test set. Its test score will be computed in Phase 5 alongside the learned models. Avoid selecting or changing the rule after looking at test results.

## Suggested notebook comment

```python
# Accuracy is computed from one selected team per match. Since each match
# contributes a positive and negative team row, row-level accuracy is only
# supplementary and is not the main success criterion.
```
