# DriveGuard data dictionary

The canonical and feature tables are Parquet datasets produced from Backblaze 2013 daily CSVs. One row represents one drive observed on one calendar day. Raw fields retain source naming when possible; derived column names are listed below. `smart_*_raw_num` values are numeric casts of vendor-reported raw attributes, not normalized physical measurements.

## Identity, provenance, and target fields

| Field | Type | Meaning |
|---|---|---|
| `date` | source date/string | Original source observation date retained from the CSV. |
| `obs_date` | date/timestamp | Parsed calendar observation date used for ordering and time windows. |
| `serial_number` | string | Drive identifier; used to group a drive's history. |
| `filename` | string | Source daily CSV filename for provenance. |
| `model` | string | Drive model label as reported in the source. |
| `capacity_bytes` | numeric | Nominal drive capacity in bytes, when supplied. |
| `failure_date` | date | Earliest observed failure date for that serial in the ingested source history; null if none is observed. |
| `will_fail_30d` | integer/null | 1 when failure is observed 1–30 days after the row; 0 when a full negative follow-up is known; null for censored/otherwise ineligible source rows. Modeling features contain eligible rows only. |
| `label_status` | category | `failure_day`, `positive`, `negative_before_later_failure`, `negative`, or `insufficient_followup`; records why the label was assigned or withheld. Failure-day rows are excluded from the labeled output. |

## SMART feature fields

The selected raw SMART fields with nonzero coverage in this archive are 5, 9, 194, and 197. Attributes 187 and 198 were fully missing and excluded. Consult the source's model-specific SMART documentation before assigning physical meaning; raw encodings and vendor interpretations vary.

| Pattern | Meaning |
|---|---|
| `smart_N_raw_num` | Numeric-cast current raw value for SMART attribute N. |
| `smart_N_missing` | 1 if that row's raw value for N is absent, else 0. |
| `smart_N_mean_7d`, `smart_N_mean_30d` | Mean of available values on this date and preceding 6/29 calendar days for that drive. |
| `smart_N_max_7d`, `smart_N_max_30d` | Maximum of available values in the same trailing calendar windows. |
| `smart_N_std_7d`, `smart_N_std_30d` | Population standard deviation over available values in the trailing calendar windows. |
| `smart_N_count_7d`, `smart_N_count_30d` | Number of non-null daily values available in the trailing calendar windows. |
| `smart_N_change` | Current raw numeric value minus the immediately prior observed value, only when the prior reading is 1–7 days earlier. Generated for attributes 5 and 197 in this archive. |

All trailing windows include the current date and exclude future observations. Missing coverage can make rolling summaries null; the model's fitted numeric imputer handles nulls, while row-level missing indicators are retained.

## Source columns

The ingested table also preserves the source CSV columns, including vendor SMART fields and the source `failure` flag. Schema varies over time; absent columns from a daily file become null after schema alignment. Do not assume every source column has consistent units or interpretation across drive models.

## Label counts and exclusions

For this 2013 run, the source contains 5,091,501 drive-day rows. After target eligibility and failure-day exclusion, the supervised feature set contains 4,261,871 rows: 18,021 positive and 4,243,850 negative. Another 828,906 rows are censored or failure-day records and do not enter supervised training/evaluation. The test dataset is further restricted to the documented chronological test interval and label horizon.
