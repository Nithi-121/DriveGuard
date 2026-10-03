# DriveGuard demo walkthrough (about 3 minutes)

## 1. Frame the question (20 seconds)

“For each operational drive-day, can the available SMART telemetry rank a drive that will have a recorded failure within the next 30 days? The unit is a drive-day, and the result is a retrospective risk ranking—not a live prediction service.”

## 2. Show the evidence base (35 seconds)

Open `reports/eda/data_profile.md` and point to the 2013 date range, 29,072 serial numbers, low natural failure prevalence, and daily population discontinuity. Explain that 18,021 positive rows are only part of 4,261,871 eligible labeled rows; 828,906 records were censored or otherwise excluded. The test period leaves out the known population-drop interval, whose cause is unknown.

## 3. Explain the evaluation design (35 seconds)

Show the temporal split in `config.yaml`: train, then validation, then a later test interval. Describe the 30-day label purge, censor-aware negatives, and why validation/test remain unsampled. Emphasize that the alert threshold is selected on validation; do not tune to the test set.

## 4. Compare ranking and alert burden (50 seconds)

Open the Streamlit app’s **Model lab** or `reports/portfolio/portfolio_report.md`. Average precision is compared across the four methods on the complete test set. XGBoost reaches 0.120 average precision, while the transparent SMART sector-count rule reaches 0.125. Isolation Forest finds more failure events but emits about 82 alerts per 1,000 drive-days because its validation threshold does not transfer to test. Use the trade-off plot and table to explain why ranking quality, precision, recall, and workload should be considered together.

## 5. Inspect model behavior (30 seconds)

Open **Why this score** and show the global SHAP summary and a detected, missed, or false-alarm example. State that the values describe fitted-model associations, not causal failure mechanisms. The SHAP case sample is intentionally enriched and must not be used to estimate prevalence.

## 6. Close with limits and next work (20 seconds)

Mention the data’s age, SMART model dependence, source population shift, repeated daily alerts, lack of current-fleet external validation, and the need to calibrate and deduplicate alerts before any real operational use. Close with **Data quality**, showing calibration, drive-cluster uncertainty, test slices, and the data limits. The current clean rebuild audit passes, and the seven contract checks cover labels, as-of features, chronological splits, and threshold budgets.

## Dashboard tour

1. **Fleet overview** gives complete-test sample counts, prevalence, average precision, event coverage, and daily population context.
2. **Risk explorer** filters dates and drive models, visualizes the complete-test score distribution, and recalculates descriptive test metrics at an adjustable cutoff. Explain that the control is for exploration, not holdout tuning.
3. **Drive detail** follows one sampled drive's historical score trace and recorded outcome.
4. **Model lab** compares average precision and the false-positive/recall workload trade-off.
5. **Why this score** presents global and local SHAP explanations with a clear non-causal interpretation.
6. **Data quality** exposes calibration, cluster bootstrap intervals, group slices, the archive discontinuity, and deployment limits.

## Interview questions to prepare for

- **Why not accuracy?** With less than 1% prevalence, a model can achieve high accuracy by predicting healthy for every row; PR metrics and alert burden are more useful.
- **Why censor observations?** A drive not seen long enough after a snapshot cannot safely be labeled healthy for the full horizon.
- **Why time split?** Random row splits leak near-duplicate histories across evaluation and hide temporal drift.
- **Why keep the simple rule?** It is a transparent benchmark; a more complex model should justify its additional operational cost.
- **What would be required before deployment?** Recent external data, calibration, alert-episode grouping, alert capacity/cost review, and ongoing drift checks.
