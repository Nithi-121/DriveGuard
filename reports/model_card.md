# DriveGuard model card

## Summary

DriveGuard is a retrospective research demo that ranks drive-day observations by their estimated risk of a recorded hard-drive failure within the next 30 days. It is not a live service, a certified prognostic system, or a recommendation to replace a drive.

## Intended use

- Demonstrate a reproducible predictive-maintenance workflow using public Backblaze Drive Stats data.
- Compare weighted logistic regression, XGBoost, a documented SMART sector-count rule, and an Isolation Forest anomaly ranking.
- Study rare-event ranking, alert burden, event coverage, calibration, uncertainty, and lead time.

Do not use these artifacts to make operational decisions about real equipment. The project has no external validation on a current fleet.

## Data and labels

- Source: Backblaze 2013 daily drive snapshots (266 CSVs, 5,091,501 source rows, 29,072 serial numbers).
- Prediction unit: one eligible drive-day.
- Target: a recorded failure 1–30 days after the observation date.
- Failure-day rows are excluded. A negative requires a later failure beyond 30 days or a full 30-day observation window without failure. Other rows are censored.
- Eligible population: 4,261,871 drive-days; 18,021 positives (0.4228%).
- Held-out test: 1,235,875 drive-days; 6,164 positives (0.499%).

## Method

The supervised learners use every positive training row and a deterministic 10% sample of healthy training drive-days, weighted to account for sampling. Validation and test are unsampled. The chronological periods are training before 2013-07-01 after label purge, validation 2013-07-31 through 2013-08-17, and test 2013-10-15 through 2013-12-01. A 30-day label embargo is applied, and the sharp source-population discontinuity is left outside the evaluation windows; its cause is unknown. Validation prevalence is 0.104% while test prevalence is 0.499%.

Features include current numeric SMART values, missingness flags, trailing 7/30-day summaries, short-term changes, and one-hot encoded drive model. SMART 187 and 198 are fully missing in this archive. Feature windows use only data available on or before the prediction date. XGBoost runs on CPU with one worker thread. Threshold selection searches tied validation score groups efficiently and does not exceed the requested validation false-positive budget.

## Held-out test results

The XGBoost reference threshold is 0.012663, selected on validation to target at most 1% row-level false-positive rate. Every other listed threshold is also selected on validation. Test behavior can differ. Confidence intervals are 300 percentile bootstrap replicates that resample serial-number clusters; intervals are shown for supervised probability models only.

| Model | Average precision | Recall (95% drive-bootstrap CI) | Precision (95% drive-bootstrap CI) | Test FPR | Alerts / 1,000 drive-days | Event recall | Median detected-event lead |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic regression | 0.0959 | 22.8% (18.6%–27.3%) | 12.8% (10.3%–15.9%) | 0.8% | 8.9 | 38.4% | 19.5 days |
| XGBoost | 0.1198 | 36.0% (31.0%–41.1%) | 15.4% (12.8%–18.0%) | 1.0% | 11.7 | 45.0% | 24.0 days |
| SMART sector-count rule | 0.1253 | 34.9% | 13.4% | 1.1% | 13.0 | 40.7% | 28.0 days |
| Isolation Forest (anomaly rank) | 0.1250 | 43.0% | 2.6% | 8.1% | 82.3 | 66.4% | 24.0 days |

The SMART sector-count rule has the highest average precision among the supervised/rule methods. XGBoost has higher row recall and precision than logistic regression at the selected thresholds in this test period. Isolation Forest's validation-selected threshold produces an 8.1% test false-positive rate and high alert burden; its scores are anomaly ranks, not failure probabilities. No method is ready for operational use. These confidence intervals quantify drive-sampling variation within this retained test period, not uncertainty from temporal/source shift.

## Calibration

On the natural-prevalence test set, logistic regression Brier 0.00490, ECE 0.00204; xgboost Brier 0.00471, ECE 0.00489. Calibration has not been corrected. Any calibration transform must be fit on validation and frozen before a final test evaluation. See `reports/analysis/calibration_test.png` and `reports/analysis/calibration_bins.csv`.

## Limitations and risks

- Data is from 2013 and one public source; hardware, firmware, fleet composition, and monitoring practice change over time.
- A large population discontinuity occurs from 2013-08-20 to 2013-10-14. Its cause is unknown; it is excluded from the main evaluation windows.
- SMART raw values are vendor/model-specific encodings and are not universal physical measurements. Associations are not causal effects.
- Validation and test prevalence differ substantially; operating thresholds may not transfer across time or fleets.
- Daily alerts repeat for drives over time and are not deduplicated into operational alert episodes.
- Bootstrap intervals do not capture temporal dependence or dataset-source shift.

## Monitoring and update requirements

Before any real-world use, require current-fleet external validation, schema checks, calibration on representative recent data, operating-cost analysis, alert episode deduplication, subgroup/model checks, and an approved human review process. This repository's Streamlit dashboard is a retrospective visualization only.

## Reproducibility

See `README.md` for installation, run phases, and audit commands. `requirements-lock.txt` records the validated Windows/Python environment. Raw data and binary model artifacts are local generated files. `reports/run_manifest.json` records package versions, source provenance, split settings, and SHA-256 hashes for key artifacts.

## Source

Backblaze, [Hard Drive Reliability and Test Data](https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data). DriveGuard is independent and not endorsed by Backblaze.
