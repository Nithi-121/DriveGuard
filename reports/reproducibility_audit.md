# Clean rebuild audit

Overall result: **PASS**.

The candidate was rebuilt from the original daily CSVs in a fresh Python virtual environment. Numeric test scores were compared by `(obs_date, serial_number)` with absolute/relative tolerance `1e-8`.

| Check | Reference | Rebuilt | Result |
|---|---:|---:|---|
| drive_days.parquet_rows | 5,091,501 | 5,091,501 | PASS |
| labeled.parquet_rows | 5,090,777 | 5,090,777 | PASS |
| features.parquet_rows | 4,261,871 | 4,261,871 | PASS |
| Label status counts | 4 categories | 4 categories | PASS |
| Feature schema | 51 columns | 51 columns | PASS |
| Holdout metrics | 96 values | 0 mismatches | PASS |
| Full test predictions | 1,235,875 rows | max score diff 1.04e-17 | PASS |

## Details

- Label status counts match: `True`.
- Feature names and DuckDB types match: `True`.
- Model-input/target values match on 42,524 deterministic sampled drive-days (excluding source filename provenance): `True`.
- Missing prediction keys: 0; extra keys: 0; label mismatches: 0.0.
- Scored-row mismatches beyond tolerance: 0.0.
- Full field-level comparison is in `reproducibility_audit.json`.
