# Phase 4 — Train and tune three models

## Notebook Markdown cell

> **Logistic regression** is the simple, fast baseline. It gives team-level scores and coefficients that show directional associations. Because its inputs have different units (goals, percentages, passes, kilometres), I standardize features inside its pipeline. Regularization parameter `C` controls model flexibility.

> **Decision tree** creates readable if/then rules and can capture interactions without scaling. A deep tree can memorize this small training set, so I limit depth and minimum leaf size.

> **K-nearest neighbours (KNN)** scores a team row by letting the `k` most similar training rows vote. It measures similarity by distance, so every stat must be on a similar scale; without scaling, passes (hundreds) would drown out cards (0 to 3). I standardize features inside its pipeline and tune `k` over odd values to avoid tied votes. The three models run from simple to more flexible, and I will choose one from cross-validation results, not assume it wins in advance.

## Notebook code cell — define pipelines and candidate settings

```python
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

models = {
    "Logistic regression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000, random_state=42)),
    ]),
    "Decision tree": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", DecisionTreeClassifier(random_state=42)),
    ]),
    # KNN uses distances, so the scaler is required. It sits inside the
    # pipeline so it is fit only on the training folds each time.
    "KNN": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier()),
    ]),
}

grids = {
    "Logistic regression": {
        "model__C": [0.01, 0.1, 1.0, 10.0, 100.0],
    },
    "Decision tree": {
        "model__max_depth": [2, 3, 4, 5, 6],
        "model__min_samples_leaf": [2, 5, 10],
    },
    "KNN": {
        # Odd k avoids tied votes. Larger k = smoother, simpler model.
        "model__n_neighbors": [3, 5, 7, 9, 11, 13, 15],
    },
}
```

## Notebook code cell — grouped cross-validation

```python
# GroupKFold keeps both rows of every match in the same validation fold.
# Every fold is scored by one predicted winner per match.
cv = GroupKFold(n_splits=5)
cv_splits = list(cv.split(X_train, y_train, groups=train_df["match_id"]))
search_rows = []

for model_name, base_pipeline in models.items():
    for params in ParameterGrid(grids[model_name]):
        fold_scores = []
        for fit_idx, valid_idx in cv_splits:
            candidate = clone(base_pipeline).set_params(**params)
            candidate.fit(X_train.iloc[fit_idx], y_train.iloc[fit_idx])
            positive_class = list(candidate.classes_).index(1)
            scores = candidate.predict_proba(X_train.iloc[valid_idx])[:, positive_class]
            fold_scores.append(match_accuracy(train_df.iloc[valid_idx], scores))

        search_rows.append({
            "model": model_name,
            "params": params,
            "cv_mean": float(np.mean(fold_scores)),
            "cv_std": float(np.std(fold_scores)),
            "fold_scores": fold_scores,
        })

search_results = pd.DataFrame(search_rows)
best_by_model = (
    search_results.sort_values(
        ["cv_mean", "cv_std"], ascending=[False, True]
    )
    .drop_duplicates("model")
    .sort_values("cv_mean", ascending=False)
)
display(best_by_model[["model", "params", "cv_mean", "cv_std", "fold_scores"]])

# Choose the overall winner now, before inspecting the held-out test matches.
selected_name = best_by_model.iloc[0]["model"]
print("Selected by grouped CV:", selected_name)
```

The standard deviation shows variation across the five folds, not a confidence interval. Small differences between models may be noise. Each fold validates on only about 20–21 matches.

## Notebook code cell — fit the selected settings on all training matches

```python
fitted_models = {}
for _, row in best_by_model.iterrows():
    name = row["model"]
    fitted_models[name] = clone(models[name]).set_params(**row["params"])
    fitted_models[name].fit(X_train, y_train)

selected_model = fitted_models[selected_name]
print("Final model:", selected_model)
```

This fits each model for a course comparison. The model designated for use is the one selected by cross-validation. Do not tune settings again after reading the test scores.
