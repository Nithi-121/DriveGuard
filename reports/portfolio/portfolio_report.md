# DriveGuard: predicting hard-drive failure from SMART telemetry

## Project brief

DriveGuard is an end-to-end data science project that estimates whether an observed hard drive will have a recorded failure within the next 30 days. It covers data ingestion, censor-aware labels, point-in-time features, temporal validation, model comparisons, explainability, uncertainty, and a retrospective dashboard.

**It is a retrospective portfolio demonstration, not a live monitoring product or a recommendation to replace equipment.**

## Dataset and target

- Source: Backblaze Drive Stats 2013 (1,235,875 retained test drive-days; source manifest records 266 daily CSVs).
- Test prevalence: 0.499% (6,164/1,235,875 eligible drive-days).
- Unit: one drive-day. Positive means a recorded failure 1–30 days after the observation. Failure-day rows are excluded. Negative means complete follow-up without failure or a known failure after the horizon; insufficiently observed rows are censored.
- Validation and test preserve natural prevalence. The test period is 2013-10-15 through 2013-12-01; the sharp 2013 source population drop is excluded, and its cause is unknown.

## Full test comparison

| Method | Average precision | Recall | Precision | Test FPR | Alerts / 1,000 drive-days | Event coverage | Median lead |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic regression | 0.096 | 22.8% | 12.8% | 0.78% | 8.9 | 38.4% | 19.5 d |
| XGBoost | 0.120 | 36.0% | 15.4% | 0.99% | 11.7 | 45.0% | 24.0 d |
| SMART sector-count rule | 0.125 | 34.9% | 13.4% | 1.13% | 13.0 | 40.7% | 28.0 d |
| Isolation Forest | 0.125 | 43.0% | 2.6% | 8.06% | 82.3 | 66.4% | 24.0 d |

All thresholds were selected on validation to target no more than 1% row-level FPR. Actual test FPR differs: Isolation Forest rises to 8.1% (82.3 alerts per 1,000 drive-days), showing poor threshold transfer across the pronounced validation/test prevalence and population shift. Its score is an anomaly ranking, not a failure probability. The SMART sector-count rule has the strongest average precision (0.125) among the listed methods.

![Full-test precision–recall curves](precision_recall_all_models.png)

![Operating threshold trade-off](operating_tradeoff.png)

## Interpretation and uncertainty

XGBoost has 0.120 average precision, 36.0% row recall, 15.4% precision, and 45.0% event coverage at its validation-selected threshold. Its 95% drive-cluster bootstrap intervals are 31.0%–41.1% recall and 12.8%–18.0% precision. The intervals measure drive-sampling variability in this test period and do not account for source shift or time dependence. The supervised probability models have Brier scores 0.00490 (logistic regression) and 0.00471 (XGBoost); see the calibration plot and report. Calibration is not corrected.

## Why the project is useful

- Censor-aware target construction avoids treating unobserved outcomes as healthy.
- Features use only current or trailing telemetry; no future rows enter feature windows.
- Train, validation, and test are chronological; thresholds use validation, and the test remains unsampled.
- Reports include rare-event PR metrics, event coverage, lead time, calibration, cluster uncertainty, explainability, and model/data limitations.

## Key artifacts

- [Full model card](../model_card.md)
- [Calibration and error analysis](../analysis/error_analysis.md)
- [Dataset profile](../eda/data_profile.md)
- [SMART/global explanations](../explainability/explanations.md)
- [Data dictionary](../../docs/data_dictionary.md)
- [Run manifest](../run_manifest.json)
- [Setup and end-to-end rebuild commands](../../README.md)

## Limitations

The archive is from 2013, SMART semantics vary by drive model, the source population changes sharply during the year, and validation/test prevalence differs substantially. There is no external current-fleet validation. Daily alerts are repeated observations and have not been deduplicated into operational alert episodes. No metric here should be interpreted as a guarantee of warning for any specific drive.

Source: Backblaze, [Hard Drive Reliability and Test Data](https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data). DriveGuard is independent and not endorsed by Backblaze.
