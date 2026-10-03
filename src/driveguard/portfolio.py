"""Build a shareable comparison report, figures, and reproducibility manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve
import yaml


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build(root: Path, report_dir: Path, storage_cap_gb: float = 5.0) -> None:
    metrics_path = root / "reports/model_evaluation/metrics.json"
    anomaly_path = root / "reports/model_evaluation/isolation_forest_metrics.json"
    predictions_path = root / "reports/model_evaluation/test_predictions.parquet"
    anomaly_predictions_path = root / "reports/model_evaluation/isolation_forest_test_predictions.parquet"
    analysis_path = root / "reports/analysis/analysis_summary.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    anomaly = json.loads(anomaly_path.read_text(encoding="utf-8"))
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    frame = pd.read_parquet(predictions_path)
    anomaly_frame = pd.read_parquet(anomaly_predictions_path, columns=["obs_date", "serial_number", "anomaly_score"])
    frame = frame.merge(anomaly_frame, on=["obs_date", "serial_number"], how="left", validate="one_to_one")
    if frame["anomaly_score"].isna().any() or len(frame) != metrics["test"]["rows"]:
        raise ValueError("Anomaly scores did not align with all full-test predictions")

    y = frame["will_fail_30d"].to_numpy(dtype=np.int8)
    series = [
        ("Logistic regression", "risk_logistic_regression", metrics["models"]["logistic_regression"]["test"]),
        ("XGBoost", "risk_xgboost", metrics["models"]["xgboost"]["test"]),
        ("SMART sector-count rule", "risk_sector_count_rule", metrics["models"]["sector_count_rule"]["test"]),
        ("Isolation Forest", "anomaly_score", anomaly["test"]),
    ]
    report_dir.mkdir(parents=True, exist_ok=True)
    colors = ["#3478a8", "#de7a22", "#388c66", "#9467bd"]
    fig, ax = plt.subplots(figsize=(9, 7))
    prevalence = float(y.mean())
    ax.axhline(prevalence, linestyle="--", color="#555555", label=f"No-skill prevalence ({prevalence:.3%})")
    table_rows = []
    for (label, score_column, result), color in zip(series, colors):
        score = frame[score_column].to_numpy(dtype=float)
        precision_curve, recall_curve, _ = precision_recall_curve(y, score, drop_intermediate=True)
        ap = float(average_precision_score(y, score))
        ax.plot(recall_curve, precision_curve, color=color, linewidth=1.8, label=f"{label} (AP {ap:.3f})")
        ci = analysis["models"].get(score_column.removeprefix("risk_"), {}).get("drive_cluster_bootstrap_95pct", {}).get("intervals", {})
        rec_ci = ci.get("recall", {})
        pre_ci = ci.get("precision", {})
        table_rows.append({"label": label, "average_precision": ap, "recall": float(result["recall"]),
            "precision": float(result["precision"]), "false_positive_rate": float(result["false_positive_rate"]),
            "alerts_per_1000_drive_days": float(result["alerts_per_1000_drive_days"]),
            "failure_event_coverage": float(result["failure_events_with_any_alert_recall"]),
            "median_lead_days": result["median_lead_days_detected_events"],
            "recall_ci": rec_ci, "precision_ci": pre_ci})
    ax.set(xlabel="Recall", ylabel="Precision", title="DriveGuard: full held-out test precision–recall")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    fig.savefig(report_dir / "precision_recall_all_models.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6))
    for row, color in zip(table_rows, colors):
        ax.scatter(row["false_positive_rate"] * 100, row["recall"] * 100,
                   s=50 + row["alerts_per_1000_drive_days"] * 2, color=color, alpha=0.82, label=row["label"])
        ax.annotate(row["label"], (row["false_positive_rate"] * 100, row["recall"] * 100),
                    xytext=(7, 5), textcoords="offset points", fontsize=8)
    ax.axvline(1.0, linestyle="--", color="#555555", label="1% validation FPR target")
    ax.set(xlabel="Test false-positive rate (%)", ylabel="Test row recall (%)",
           title="Operational trade-off at validation-selected thresholds")
    ax.text(0.01, 0.02, "Bubble size scales with alerts per 1,000 drive-days.", transform=ax.transAxes, fontsize=8)
    ax.grid(alpha=0.2)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(report_dir / "operating_tradeoff.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    test = metrics["models"]["xgboost"]["test"]
    xgb_ci = analysis["models"]["xgboost"]["drive_cluster_bootstrap_95pct"]["intervals"]
    rule_row = next(row for row in table_rows if row["label"] == "SMART sector-count rule")
    anomaly_row = next(row for row in table_rows if row["label"] == "Isolation Forest")
    logistic_cal = analysis["models"]["logistic_regression"]
    xgb_cal = analysis["models"]["xgboost"]
    lines = [
        "# DriveGuard: predicting hard-drive failure from SMART telemetry", "",
        "## Project brief", "",
        "DriveGuard is an end-to-end data science project that estimates whether an observed hard drive will have a recorded failure within the next 30 days. It covers data ingestion, censor-aware labels, point-in-time features, temporal validation, model comparisons, explainability, uncertainty, and a retrospective dashboard.", "",
        "**It is a retrospective portfolio demonstration, not a live monitoring product or a recommendation to replace equipment.**", "",
        "## Dataset and target", "",
        f"- Source: Backblaze Drive Stats 2013 ({metrics['test']['rows']:,} retained test drive-days; source manifest records 266 daily CSVs).",
        f"- Test prevalence: {test['prevalence']:.3%} ({test['positive_rows']:,}/{test['rows']:,} eligible drive-days).",
        "- Unit: one drive-day. Positive means a recorded failure 1–30 days after the observation. Failure-day rows are excluded. Negative means complete follow-up without failure or a known failure after the horizon; insufficiently observed rows are censored.",
        "- Validation and test preserve natural prevalence. The test period is 2013-10-15 through 2013-12-01; the sharp 2013 source population drop is excluded, and its cause is unknown.", "",
        "## Full test comparison", "",
        "| Method | Average precision | Recall | Precision | Test FPR | Alerts / 1,000 drive-days | Event coverage | Median lead |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in table_rows:
        lines.append(f"| {row['label']} | {row['average_precision']:.3f} | {row['recall']:.1%} | {row['precision']:.1%} | {row['false_positive_rate']:.2%} | {row['alerts_per_1000_drive_days']:.1f} | {row['failure_event_coverage']:.1%} | {row['median_lead_days']:.1f} d |")
    lines.extend([
        "", (f"All thresholds were selected on validation to target no more than 1% row-level FPR. Actual test FPR differs: Isolation Forest rises to {anomaly_row['false_positive_rate']:.1%} "
        f"({anomaly_row['alerts_per_1000_drive_days']:.1f} alerts per 1,000 drive-days), showing poor threshold transfer across the pronounced validation/test prevalence and population shift. "
        f"Its score is an anomaly ranking, not a failure probability. The SMART sector-count rule has the strongest average precision ({rule_row['average_precision']:.3f}) among the listed methods."), "",
        "![Full-test precision–recall curves](precision_recall_all_models.png)", "",
        "![Operating threshold trade-off](operating_tradeoff.png)", "",
        "## Interpretation and uncertainty", "",
        (f"XGBoost has {test['average_precision']:.3f} average precision, {test['recall']:.1%} row recall, {test['precision']:.1%} precision, and "
        f"{test['failure_events_with_any_alert_recall']:.1%} event coverage at its validation-selected threshold. Its 95% drive-cluster bootstrap intervals are "
        f"{xgb_ci['recall']['lower_95']:.1%}–{xgb_ci['recall']['upper_95']:.1%} recall and {xgb_ci['precision']['lower_95']:.1%}–{xgb_ci['precision']['upper_95']:.1%} precision. "
        f"The intervals measure drive-sampling variability in this test period and do not account for source shift or time dependence. The supervised probability models have "
        f"Brier scores {logistic_cal['brier_score']:.5f} (logistic regression) and {xgb_cal['brier_score']:.5f} (XGBoost); see the calibration plot and report. Calibration is not corrected."), "",
        "## Why the project is useful", "",
        "- Censor-aware target construction avoids treating unobserved outcomes as healthy.",
        "- Features use only current or trailing telemetry; no future rows enter feature windows.",
        "- Train, validation, and test are chronological; thresholds use validation, and the test remains unsampled.",
        "- Reports include rare-event PR metrics, event coverage, lead time, calibration, cluster uncertainty, explainability, and model/data limitations.", "",
        "## Key artifacts", "",
        "- [Full model card](../model_card.md)",
        "- [Calibration and error analysis](../analysis/error_analysis.md)",
        "- [Dataset profile](../eda/data_profile.md)",
        "- [SMART/global explanations](../explainability/explanations.md)",
        "- [Data dictionary](../../docs/data_dictionary.md)",
        "- [Run manifest](../run_manifest.json)",
        "- [Setup and end-to-end rebuild commands](../../README.md)", "",
        "## Limitations", "",
        "The archive is from 2013, SMART semantics vary by drive model, the source population changes sharply during the year, and validation/test prevalence differs substantially. There is no external current-fleet validation. Daily alerts are repeated observations and have not been deduplicated into operational alert episodes. No metric here should be interpreted as a guarantee of warning for any specific drive.", "",
        "Source: Backblaze, [Hard Drive Reliability and Test Data](https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data). DriveGuard is independent and not endorsed by Backblaze.", "",
    ])
    (report_dir / "portfolio_report.md").write_text("\n".join(lines), encoding="utf-8")

    manifest_source = root / "data/manifest.csv"
    with manifest_source.open("r", encoding="utf-8", newline="") as handle:
        source_manifest = list(csv.DictReader(handle))
    packages = {}
    for package in ("duckdb", "pandas", "pyarrow", "scikit-learn", "xgboost", "shap", "streamlit", "plotly", "matplotlib", "PyYAML"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    candidates = [
        root / "config.yaml", root / "requirements-lock.txt", manifest_source, metrics_path, anomaly_path, analysis_path,
        root / "models/logistic_regression.joblib", root / "models/xgboost.joblib", root / "models/isolation_forest.joblib",
        predictions_path, anomaly_predictions_path, root / "reports/model_evaluation/test_predictions_sample.parquet",
    ]
    hashes = {str(p.relative_to(root).as_posix()): {"bytes": p.stat().st_size, "sha256": _sha256(p)} for p in candidates if p.exists()}
    all_files = [p for p in root.rglob("*") if p.is_file()]
    total_bytes = sum(p.stat().st_size for p in all_files)
    compact_bytes = sum(p.stat().st_size for p in all_files if ".venv" not in p.parts and "__pycache__" not in p.parts)
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "random_seed": 42,
        "label_horizon_days": 30,
        "temporal_split": metrics["split"],
        "source_data_manifest": source_manifest,
        "artifacts": hashes,
        "storage": {"project_budget_gb_decimal": storage_cap_gb,
                    "full_project_bytes_including_venv_and_cache": int(total_bytes),
                    "data_and_outputs_bytes_excluding_venv_and_cache": int(compact_bytes)},
        "rebuild_order": [
            "ingest", "labels", "features", "eda", "train", "anomaly", "analyze_errors", "score_sample", "explain", "streamlit app"
        ],
    }
    (root / "reports/run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote portfolio report and full-test comparison figures to {report_dir}")
    print(f"Wrote reproducibility manifest to {root / 'reports/run_manifest.json'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--outdir", type=Path, default=Path("reports/portfolio"))
    parser.add_argument("--storage-cap-gb", type=float, default=5.0)
    args = parser.parse_args()
    outdir = args.outdir if args.outdir.is_absolute() else args.root / args.outdir
    build(args.root, outdir, args.storage_cap_gb)


if __name__ == "__main__":
    main()
