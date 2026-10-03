# DriveGuard explanations

The global chart uses a deterministic row sample from the held-out test interval at approximately natural prevalence. Local cases come from the dashboard drill-down sample, which contains a 1% sample of healthy-drive histories plus all positive-labelled drive-days.

Positive SHAP contributions raise this fitted model's score relative to its baseline; negative contributions lower it. These explanations describe model behavior and are not causal findings.

## Highest mean absolute SHAP values

| Feature | Mean absolute SHAP |
|---|---:|
| numeric__smart_5_raw_num | 0.03087 |
| numeric__smart_5_std_30d | 0.02975 |
| numeric__smart_5_max_30d | 0.02650 |
| model__model_ST3000DM001 | 0.01904 |
| model__model_ST1500DL003 | 0.01757 |
| numeric__smart_9_mean_7d | 0.01331 |
| numeric__smart_197_mean_7d | 0.01279 |
| numeric__smart_9_mean_30d | 0.01144 |
| numeric__capacity_bytes | 0.00964 |
| numeric__smart_197_raw_num | 0.00877 |

## Local cases

See the three waterfall plots and `case_explanations.csv` for one detected failure, one missed failure, and one false alarm when available. These examples are selected from the stated deterministic sample, not from the entire fleet.
