# DriveGuard data profile

## Coverage

- Source drive-day rows: 5,091,501
- Distinct serial numbers: 29,072
- Observed date range: 2013-04-10 to 2013-12-31
- Eligible labeled rows: 4,261,871
- Positive 30-day labels: 18,021 (0.4228%)
- Negative labels: 4,243,850

## Monthly label counts

December prevalence is based only on rows with observable 30-day outcomes near the dataset boundary; it is not directly comparable with earlier months.

| Month | Eligible rows | Positives | Positive prevalence |
|---|---:|---:|---:|
| 2013-04 | 452,091 | 2,003.0 | 0.443% |
| 2013-05 | 686,934 | 2,807.0 | 0.409% |
| 2013-06 | 676,883 | 2,519.0 | 0.372% |
| 2013-07 | 715,785 | 2,411.0 | 0.337% |
| 2013-08 | 454,968 | 433.0 | 0.095% |
| 2013-09 | 25,191 | 112.0 | 0.445% |
| 2013-10 | 440,131 | 1,980.0 | 0.450% |
| 2013-11 | 782,063 | 4,078.0 | 0.521% |
| 2013-12 | 27,825 | 1,678.0 | 6.031% |

## Temporal split summary

The split uses a 30-day label embargo and keeps the low-population interval outside the main evaluation windows.

| Split | Eligible rows | Positives | Prevalence |
|---|---:|---:|---:|
| purged_or_outside | 787,903 | 4,091 | 0.519% |
| test | 1,235,875 | 6,164 | 0.499% |
| train | 1,815,908 | 7,329 | 0.404% |
| validation | 422,185 | 437 | 0.104% |

## SMART missingness

| Attribute | Missing (%) |
|---|---:|
| SMART 5 raw | 0.00% |
| SMART 9 raw | 0.00% |
| SMART 187 raw | 100.00% |
| SMART 194 raw | 0.00% |
| SMART 197 raw | 0.00% |
| SMART 198 raw | 100.00% |

## Frequent drive models

| Model | Eligible rows | Positive labels |
|---|---:|---:|
| Hitachi HDS722020ALA330 | 851,494 | 712.0 |
| Hitachi HDS5C3030ALA630 | 824,628 | 607.0 |
| ST3000DM001 | 806,319 | 6,682.0 |
| Hitachi HDS5C4040ALE630 | 428,271 | 610.0 |
| ST31500541AS | 364,897 | 2,951.0 |
| ST4000DM000 | 271,983 | 947.0 |
| Hitachi HDS723030ALA640 | 185,187 | 177.0 |
| ST31500341AS | 122,381 | 2,399.0 |
| WDC WD10EADS | 86,864 | 261.0 |
| WDC WD30EZRX | 84,010 | 420.0 |

## Data quality interpretation

Daily source population is below 5,000 rows from 2013-08-20 through 2013-10-14. Counts are roughly 720–900 rows per day in this interval, compared with thousands to tens of thousands on most surrounding dates. This report records the discontinuity but does not infer its cause. The main temporal evaluation leaves this interval out; model results still describe only the retained date ranges.
SMART 187 and SMART 198 raw values are fully missing in this archive and are excluded from the feature table. Other listed SMART values have usable coverage; their missingness indicators remain available to the model.

See `data_profile.png` for daily row counts and monthly target prevalence.
