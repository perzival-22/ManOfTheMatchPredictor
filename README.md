# Man of the Match team predictor

This project predicts **which of the two teams in a completed World Cup match supplied the Man of the Match winner**. Each team is a row: `1` means the award winner came from that team, and `0` means the award winner came from its opponent. It does not identify the player or make a pre-match forecast.

Copy the notebook cells from these guides, in order, into `MOTM_predictor.ipynb`:

1. [01 — Inspect and validate the data](01_data_audit.md)
2. [02 — Prepare features and split matches](02_preprocessing_and_split.md)
3. [03 — Establish comparison baselines](03_baselines.md)
4. [04 — Train and tune three models](04_training_and_tuning.md)
5. [05 — Evaluate, explain, and conclude](05_evaluation_and_reporting.md)

The Excel file has **256 team rows, 128 matches, and 45 columns**. There are 64 matches from 2018 and 64 from 2022. Every match has exactly two rows and exactly one positive label. A grouped 80/20 split gives **102 training matches (204 rows)** and **26 test matches (52 rows)**; with the proposed year stratification, the test set has 13 matches from each tournament. This is enough to train and compare small baseline models for a course project, but only 26 independent test decisions make performance estimates noisy. A five-fold grouped cross-validation on the training set helps choose model settings; the test matches remain untouched until final evaluation.

## Working assumptions

- The match and any penalty shootout have finished, so their statistics are available.
- The desired output is one winning **team** per match.
- The first experiment uses common numeric match statistics and compares logistic regression, a decision tree, and k-nearest neighbours (KNN), a simple-to-flexible progression.
- The models output a score for each team. The higher-scoring team is the prediction for that match.

The workbook combines two tournaments with different statistics collection methods. The 2022 rows have extra fields absent from 2018, and the source notes describe a different possession calculation. The guides keep the first feature set consistent across years and ask you to discuss this limitation in the notebook.
