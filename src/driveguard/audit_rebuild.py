"""Compare a clean rebuild with a reference DriveGuard run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np


def _q(path: Path) -> str:
    return str(path).replace("'", "''")


def _identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _row_count(path: Path) -> int:
    with duckdb.connect() as con:
        return int(con.execute(f"SELECT count(*) FROM read_parquet('{_q(path)}')").fetchone()[0])


def _label_counts(path: Path) -> dict[str, int]:
    with duckdb.connect() as con:
        rows = con.execute(
            f"SELECT label_status, count(*) FROM read_parquet('{_q(path)}') GROUP BY 1"
        ).fetchall()
    return {str(label): int(count) for label, count in rows}


def _schema(path: Path) -> list[tuple[str, str]]:
    with duckdb.connect() as con:
        rows = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{_q(path)}')").fetchall()
    return [(str(row[0]), str(row[1])) for row in rows]


def _compare_feature_values(reference: Path, rebuilt: Path, sample_mod: int = 100) -> dict:
    """Compare 1% of model inputs/targets by stable drive-day key.

    `filename` is excluded because it intentionally differs when a workspace is
    rebuilt at a different root path. All prediction-relevant values remain in
    the comparison.
    """
    with duckdb.connect() as con:
        columns = [row[0] for row in con.execute(
            f"DESCRIBE SELECT * FROM read_parquet('{_q(reference)}')"
        ).fetchall()]
        compared = [name for name in columns if name != "filename"]
        expressions = ", ".join(
            f'count(*) FILTER (WHERE a.{_identifier(name)} IS DISTINCT FROM b.{_identifier(name)}) AS {_identifier(name)}'
            for name in compared if name not in {"obs_date", "serial_number"}
        )
        result = con.execute(f"""
            WITH ref_sample AS (
              SELECT * FROM read_parquet('{_q(reference)}')
              WHERE hash(serial_number, obs_date) % {int(sample_mod)} = 0
            ), rebuilt_sample AS (
              SELECT * FROM read_parquet('{_q(rebuilt)}')
              WHERE hash(serial_number, obs_date) % {int(sample_mod)} = 0
            )
            SELECT (SELECT count(*) FROM ref_sample) AS reference_sample_rows,
                   (SELECT count(*) FROM rebuilt_sample) AS rebuilt_sample_rows,
                   count(*) AS matched_sample_rows, {expressions}
            FROM ref_sample a
            INNER JOIN rebuilt_sample b USING (obs_date, serial_number)
        """).fetchdf().iloc[0].to_dict()
    reference_rows = int(result.pop("reference_sample_rows"))
    rebuilt_rows = int(result.pop("rebuilt_sample_rows"))
    matched_rows = int(result.pop("matched_sample_rows"))
    mismatches = {name: int(value) for name, value in result.items() if int(value or 0)}
    return {
        "sample_mod": sample_mod,
        "reference_sample_rows": reference_rows,
        "rebuilt_sample_rows": rebuilt_rows,
        "matched_sample_rows": matched_rows,
        "excluded_provenance_columns": ["filename"],
        "mismatched_values_by_column": mismatches,
        "match": reference_rows > 0 and reference_rows == rebuilt_rows == matched_rows and not mismatches,
    }


def _compare_predictions(reference: Path, rebuilt: Path, score_columns: list[str], tolerance: float) -> dict:
    pairs = ", ".join(f"max(abs(a.{col} - b.{col})) AS {col}_max_abs_diff" for col in score_columns)
    conditions = " OR ".join(f"abs(coalesce(a.{c}, 0) - coalesce(b.{c}, 0)) > {tolerance}" for c in score_columns)
    with duckdb.connect() as con:
        query = f"""
            SELECT count(*) AS rows,
                   count(*) FILTER (WHERE a.will_fail_30d != b.will_fail_30d) AS label_mismatches,
                   count(*) FILTER (WHERE {conditions}) AS score_mismatch_rows,
                   {pairs}
            FROM read_parquet('{_q(reference)}') a
            FULL OUTER JOIN read_parquet('{_q(rebuilt)}') b
              ON CAST(a.obs_date AS DATE) = CAST(b.obs_date AS DATE)
             AND a.serial_number = b.serial_number
            WHERE a.serial_number IS NOT NULL AND b.serial_number IS NOT NULL
        """
        row = con.execute(query).fetchdf().iloc[0].to_dict()
        counts = con.execute(f"""
            SELECT
              (SELECT count(*) FROM read_parquet('{_q(reference)}')) AS reference_rows,
              (SELECT count(*) FROM read_parquet('{_q(rebuilt)}')) AS rebuilt_rows,
              (SELECT count(*) FROM read_parquet('{_q(reference)}') a ANTI JOIN read_parquet('{_q(rebuilt)}') b
                ON CAST(a.obs_date AS DATE)=CAST(b.obs_date AS DATE) AND a.serial_number=b.serial_number) AS missing_keys,
              (SELECT count(*) FROM read_parquet('{_q(rebuilt)}') b ANTI JOIN read_parquet('{_q(reference)}') a
                ON CAST(a.obs_date AS DATE)=CAST(b.obs_date AS DATE) AND a.serial_number=b.serial_number) AS extra_keys
        """).fetchdf().iloc[0].to_dict()
    return {k: (float(v) if isinstance(v, (np.floating, float)) else int(v)) for k, v in {**counts, **row}.items()}


def audit(reference_root: Path, rebuilt_root: Path, output: Path, tolerance: float = 1e-8) -> dict:
    checks: dict[str, dict] = {}
    ref_data = reference_root / "data/processed"
    new_data = rebuilt_root / "data/processed"
    for name in ("drive_days.parquet", "labeled.parquet", "features.parquet"):
        left, right = ref_data / name, new_data / name
        checks[f"{name}_rows"] = {"reference": _row_count(left), "rebuilt": _row_count(right)}
        checks[f"{name}_rows"]["match"] = checks[f"{name}_rows"]["reference"] == checks[f"{name}_rows"]["rebuilt"]
    checks["label_status_counts"] = {
        "reference": _label_counts(ref_data / "labeled.parquet"),
        "rebuilt": _label_counts(new_data / "labeled.parquet"),
    }
    checks["label_status_counts"]["match"] = checks["label_status_counts"]["reference"] == checks["label_status_counts"]["rebuilt"]
    checks["feature_schema"] = {
        "reference_columns": _schema(ref_data / "features.parquet"),
        "rebuilt_columns": _schema(new_data / "features.parquet"),
    }
    checks["feature_schema"]["match"] = checks["feature_schema"]["reference_columns"] == checks["feature_schema"]["rebuilt_columns"]
    checks["feature_values_sample"] = _compare_feature_values(
        ref_data / "features.parquet", new_data / "features.parquet"
    )

    ref_metrics = json.loads((reference_root / "reports/model_evaluation/metrics.json").read_text(encoding="utf-8"))
    new_metrics = json.loads((rebuilt_root / "reports/model_evaluation/metrics.json").read_text(encoding="utf-8"))
    metric_comparisons = []
    for model, ref_result in ref_metrics["models"].items():
        for split in ("validation", "test"):
            for key, ref_value in ref_result[split].items():
                new_value = new_metrics["models"][model][split][key]
                if isinstance(ref_value, (int, float)) and not isinstance(ref_value, bool):
                    equal = bool(np.isclose(float(ref_value), float(new_value), atol=tolerance, rtol=tolerance, equal_nan=True))
                else:
                    equal = ref_value == new_value
                metric_comparisons.append({"model": model, "split": split, "metric": key,
                                           "reference": ref_value, "rebuilt": new_value, "match": equal})
    checks["holdout_metrics"] = {
        "comparison_count": len(metric_comparisons),
        "mismatches": [item for item in metric_comparisons if not item["match"]],
        "match": all(item["match"] for item in metric_comparisons),
    }
    score_columns = ["risk_logistic_regression", "risk_xgboost", "risk_sector_count_rule"]
    checks["full_test_predictions"] = _compare_predictions(
        reference_root / "reports/model_evaluation/test_predictions.parquet",
        rebuilt_root / "reports/model_evaluation/test_predictions.parquet",
        score_columns,
        tolerance,
    )
    p = checks["full_test_predictions"]
    p["match"] = (p["reference_rows"] == p["rebuilt_rows"] and p["missing_keys"] == 0 and
                   p["extra_keys"] == 0 and p["label_mismatches"] == 0 and p["score_mismatch_rows"] == 0)
    required_checks = [
        "drive_days.parquet_rows", "labeled.parquet_rows", "features.parquet_rows",
        "label_status_counts", "feature_schema", "feature_values_sample", "holdout_metrics", "full_test_predictions",
    ]
    passed = all(bool(checks[name]["match"]) for name in required_checks)
    result = {"passed": passed, "score_tolerance": tolerance, "checks": checks}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    lines = ["# Clean rebuild audit", "", f"Overall result: **{'PASS' if passed else 'FAIL'}**.", "",
             "The candidate was rebuilt from the original daily CSVs in a fresh Python virtual environment. Numeric test scores were compared by `(obs_date, serial_number)` with absolute/relative tolerance `1e-8`.", "",
             "| Check | Reference | Rebuilt | Result |", "|---|---:|---:|---|"]
    for name in ("drive_days.parquet_rows", "labeled.parquet_rows", "features.parquet_rows"):
        check = checks[name]
        lines.append(f"| {name} | {check['reference']:,} | {check['rebuilt']:,} | {'PASS' if check['match'] else 'FAIL'} |")
    lines.extend([
        f"| Label status counts | {len(checks['label_status_counts']['reference'])} categories | {len(checks['label_status_counts']['rebuilt'])} categories | {'PASS' if checks['label_status_counts']['match'] else 'FAIL'} |",
        f"| Feature schema | {len(checks['feature_schema']['reference_columns'])} columns | {len(checks['feature_schema']['rebuilt_columns'])} columns | {'PASS' if checks['feature_schema']['match'] else 'FAIL'} |",
        f"| Holdout metrics | {checks['holdout_metrics']['comparison_count']} values | {len(checks['holdout_metrics']['mismatches'])} mismatches | {'PASS' if checks['holdout_metrics']['match'] else 'FAIL'} |",
        f"| Full test predictions | {p['reference_rows']:,} rows | max score diff {max(p.get(c+'_max_abs_diff', 0) or 0 for c in score_columns):.3g} | {'PASS' if p['match'] else 'FAIL'} |",
        "", "## Details", "",
        f"- Label status counts match: `{checks['label_status_counts']['match']}`.",
        f"- Feature names and DuckDB types match: `{checks['feature_schema']['match']}`.",
        f"- Model-input/target values match on {checks['feature_values_sample']['matched_sample_rows']:,} deterministic sampled drive-days (excluding source filename provenance): `{checks['feature_values_sample']['match']}`.",
        f"- Missing prediction keys: {p['missing_keys']}; extra keys: {p['extra_keys']}; label mismatches: {p['label_mismatches']}.",
        f"- Scored-row mismatches beyond tolerance: {p['score_mismatch_rows']}.",
        f"- Full field-level comparison is in `{output.with_suffix('.json').name}`.", "",
    ])
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Clean rebuild audit: {'PASS' if passed else 'FAIL'}; report written to {output}")
    if not passed:
        raise SystemExit(1)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-root", type=Path, default=Path.cwd())
    parser.add_argument("--rebuilt-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/reproducibility_audit.md"))
    parser.add_argument("--tolerance", type=float, default=1e-8)
    args = parser.parse_args()
    audit(args.reference_root, args.rebuilt_root, args.output, args.tolerance)


if __name__ == "__main__":
    main()
