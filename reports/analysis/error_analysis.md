# Calibration and error analysis

The complete test set contains 1,235,875 eligible drive-days with 6,164 positives (0.499%).

## Probability quality

| Model | Brier score | 10-bin ECE |
|---|---:|---:|
| Logistic Regression | 0.00490 | 0.00204 |
| Xgboost | 0.00471 | 0.00489 |

Brier score and expected calibration error are evaluated at the observed test prevalence. The calibration plot uses quantile bins, so bin widths in probability space can differ. A calibration method would need to be fit on validation data and then frozen before test evaluation.

## Drive-cluster 95% bootstrap intervals

| Model | Metric | Lower | Upper |
|---|---|---:|---:|
| Logistic Regression | precision | 0.1029 | 0.1593 |
| Logistic Regression | recall | 0.1859 | 0.2725 |
| Logistic Regression | false positive rate | 0.0068 | 0.0087 |
| Logistic Regression | alerts per 1000 drive days | 7.8985 | 9.8335 |
| Logistic Regression | failure event recall | 0.3284 | 0.4346 |
| Xgboost | precision | 0.1283 | 0.1805 |
| Xgboost | recall | 0.3103 | 0.4107 |
| Xgboost | false positive rate | 0.0091 | 0.0111 |
| Xgboost | alerts per 1000 drive days | 10.7183 | 12.9944 |
| Xgboost | failure event recall | 0.3944 | 0.5025 |

Intervals resample serial-number clusters, preserving within-drive repeated observations. They quantify sampling variation across drives in this test period, not uncertainty from time or source shift.

## Slice diagnostics

See `error_slices.csv` for month- and drive-model-specific prevalence, false negatives, false positives, precision, and recall at thresholds chosen on validation. Small or shifted slices should be interpreted cautiously; these are diagnostics, not new tuning targets.
