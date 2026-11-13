# Phase 5 — Evaluate, explain, and conclude

## Notebook Markdown cell

> I now score the held-out 26 matches. This is the first time test outcomes are used for model evaluation. I will report the simple goal/shots rule, each tuned model, the cross-validation-selected model, and results by tournament year. A single test-match error changes accuracy by roughly 3.85 percentage points, so close results should be described cautiously.

## Notebook code cell — final match-level results

```python
test_results = []
predictions_by_model = {}

rule_scores = test_df["Goal Scored"] + 0.001 * test_df["On-Target"]
predictions_by_model["Goal/shots rule"] = match_predictions(test_df, rule_scores)

for name, fitted in fitted_models.items():
    positive_class = list(fitted.classes_).index(1)
    scores = fitted.predict_proba(X_test)[:, positive_class]
    predictions_by_model[name] = match_predictions(test_df, scores)

for name, comparison in predictions_by_model.items():
    test_results.append({
        "method": name,
        "correct_matches": int(comparison["correct"].sum()),
        "total_matches": len(comparison),
        "match_accuracy": comparison["correct"].mean(),
    })

results_table = pd.DataFrame(test_results).sort_values(
    "match_accuracy", ascending=False
)
display(results_table)
print("Cross-validation-selected model:", selected_name)
display(predictions_by_model[selected_name].sort_values("match_id"))
```

`selected_score` is the model's positive-class score for the chosen team. The two team scores from a row classifier need not add to one, so do not describe it as a calibrated probability that the selected team wins the award. The match selection uses their ranking.

## Notebook code cell — inspect year results and mistakes

```python
selected_comparison = predictions_by_model[selected_name]
year_results = selected_comparison.groupby("year").agg(
    correct_matches=("correct", "sum"),
    total_matches=("correct", "size"),
    match_accuracy=("correct", "mean"),
)
display(year_results)
display(selected_comparison.loc[~selected_comparison["correct"]])
```

The proposed split has only 13 test matches per year. If 2018 and 2022 differ, consider the different source methods, styles of play, and sample variation. Do not infer a tournament-specific causal effect from this small comparison.

## Notebook code cell — optional row metrics

```python
from sklearn.metrics import classification_report, confusion_matrix

row_predictions = selected_model.predict(X_test)
print(classification_report(y_test, row_predictions, zero_division=0))
print("Row confusion matrix:\n", confusion_matrix(y_test, row_predictions))
```

These diagnostics answer a different question: how often each team row was labeled correctly at a fixed threshold. The match-level choice above remains the main result.

## Notebook code cell — inspect model associations

```python
from sklearn.inspection import permutation_importance

# This standard scorer is row-level and gives an exploratory feature ranking.
# It is evaluated only on training data to avoid learning from the test set.
importance = permutation_importance(
    selected_model, X_train, y_train,
    scoring="balanced_accuracy", n_repeats=20, random_state=42,
)
importance_table = pd.DataFrame({
    "feature": FEATURES,
    "mean_importance": importance.importances_mean,
    "std_importance": importance.importances_std,
}).sort_values("mean_importance", ascending=False)
display(importance_table.head(10))

if selected_name == "Logistic regression":
    coefficients = pd.Series(
        selected_model.named_steps["model"].coef_[0],
        index=FEATURES,
    ).sort_values()
    display(coefficients.to_frame("standardized_coefficient"))
```

Correlated stats can divide importance between themselves. Logistic coefficients indicate associations after scaling, not causal effects. These explanations are exploratory because the dataset is small.

## Notebook Markdown cell — final write-up template

> **Method.** I used [19 common numerical match statistics] to predict which team supplied the Man of the Match winner after the match. I split by `match_id` into 102 training and 26 test matches, keeping both team rows together and balancing the two tournament years. I selected hyperparameters using five-fold grouped cross-validation on training matches.

> **Results.** The goal/shots rule correctly identified [x]/26 test winners. Logistic regression identified [x]/26, the decision tree [x]/26, and KNN [x]/26. [Selected model] had the highest grouped cross-validation score and achieved [x]/26 on the held-out test matches. Its 2018 and 2022 results were [x]/13 and [x]/13.

> **Interpretation and limits.** The test set contains only 26 independent matches, so accuracy is sensitive to a few results. Shared column names do not guarantee identical measurement methods across 2018 and 2022. The workbook records 2022 award winners from an online source and 2018 winners from a supplied statistics file; labels should be verified against authoritative records if this is used beyond a course exercise. The model uses completed-match statistics and cannot predict before kickoff. It predicts the winner's team, not the player.

> **Next improvement.** Add more consistently sourced tournaments or matches, recheck feature definitions across years, and repeat a grouped evaluation. An additional train-on-2018/test-on-2022 experiment can expose year transfer issues, but it uses only 64 matches for training and should be labeled as a separate diagnostic rather than replacing the untouched 80/20 result.
