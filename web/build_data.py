"""Build the compact, static data package used by the Vercel dashboard."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "public"
REPORTS = ROOT / "reports"
EVALUATION = REPORTS / "model_evaluation"


def records(frame: pd.DataFrame) -> list[dict]:
    clean = frame.copy()
    for column in clean.select_dtypes(include=["datetime", "datetimetz"]).columns:
        clean[column] = clean[column].dt.strftime("%Y-%m-%d")
    return json.loads(clean.to_json(orient="records", date_format="iso"))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "assets").mkdir(exist_ok=True)
    (OUT / "data").mkdir(exist_ok=True)

    metrics = json.loads((EVALUATION / "metrics.json").read_text(encoding="utf-8"))
    anomaly_path = EVALUATION / "isolation_forest_metrics.json"
    anomaly = json.loads(anomaly_path.read_text(encoding="utf-8")) if anomaly_path.exists() else None
    analysis = json.loads((REPORTS / "analysis" / "analysis_summary.json").read_text(encoding="utf-8"))
    full = pd.read_parquet(EVALUATION / "test_predictions.parquet")
    full["obs_date"] = pd.to_datetime(full["obs_date"])
    full["failure_date"] = pd.to_datetime(full["failure_date"], errors="coerce")
    threshold = metrics["models"]["xgboost"]["test"]["threshold_selected_on_validation_at_max_1pct_fpr"]

    daily = full.assign(
        xgb_alert=full["risk_xgboost"] >= threshold,
        xgb_tp=(full["risk_xgboost"] >= threshold) & (full["will_fail_30d"] == 1),
    ).groupby("obs_date", as_index=False).agg(
        drive_days=("will_fail_30d", "size"),
        positive_days=("will_fail_30d", "sum"),
        xgb_alerts=("xgb_alert", "sum"),
        xgb_detected_days=("xgb_tp", "sum"),
    )

    positive = full[full["will_fail_30d"] == 1]
    negative_sample = full[full["will_fail_30d"] == 0].sample(n=3200, random_state=42)
    top_alerts = full.nlargest(1800, "risk_xgboost")
    risk = pd.concat([positive, negative_sample, top_alerts]).drop_duplicates(
        ["obs_date", "serial_number"]
    )
    risk = risk.sort_values("risk_xgboost", ascending=False)

    sample = pd.read_parquet(EVALUATION / "test_predictions_sample.parquet")
    sample["obs_date"] = pd.to_datetime(sample["obs_date"])
    sample["failure_date"] = pd.to_datetime(sample["failure_date"], errors="coerce")

    score_histograms = {}
    for name, column in (
        ("XGBoost", "risk_xgboost"),
        ("Logistic regression", "risk_logistic_regression"),
        ("SMART sector-count rule", "risk_sector_count_rule"),
    ):
        values = full[column].to_numpy(dtype=float)
        counts, edges = np.histogram(values, bins=36)
        score_histograms[name] = {"counts": counts.tolist(), "edges": edges.tolist()}
    if anomaly is not None and "anomaly_isolation_forest" in sample:
        values = sample["anomaly_isolation_forest"].to_numpy(dtype=float)
        counts, edges = np.histogram(values, bins=36)
        score_histograms["Isolation Forest (review sample)"] = {
            "counts": counts.tolist(), "edges": edges.tolist(),
        }

    def read_csv(path: Path) -> list[dict]:
        if not path.exists():
            return []
        return records(pd.read_csv(path))

    payload = {
        "metrics": metrics,
        "anomaly": anomaly,
        "analysis": analysis,
        "threshold": threshold,
        "daily": records(daily),
        "risk_examples": records(risk),
        "drive_history_sample": records(sample),
        "score_histograms": score_histograms,
        "calibration": read_csv(REPORTS / "analysis" / "calibration_bins.csv"),
        "error_slices": read_csv(REPORTS / "analysis" / "error_slices.csv"),
        "explanations": read_csv(REPORTS / "explainability" / "case_explanations.csv"),
        "data_profile": (REPORTS / "eda" / "data_profile.md").read_text(encoding="utf-8"),
        "model_card": (REPORTS / "model_card.md").read_text(encoding="utf-8"),
    }
    (OUT / "data" / "driveguard.json").write_text(
        json.dumps(payload, separators=(",", ":"), allow_nan=False), encoding="utf-8"
    )

    for source, destination in (
        (REPORTS / "explainability" / "shap_global_importance.png", OUT / "assets" / "shap_global_importance.png"),
        (REPORTS / "portfolio" / "precision_recall_all_models.png", OUT / "assets" / "precision_recall_all_models.png"),
        (REPORTS / "portfolio" / "operating_tradeoff.png", OUT / "assets" / "operating_tradeoff.png"),
    ):
        if source.exists():
            shutil.copy2(source, destination)

    target = OUT / "data" / "driveguard.json"
    print(f"Wrote {target} ({target.stat().st_size / 1024 / 1024:.2f} MiB)")
    print(f"Full test rows: {len(full):,}; drive-history sample rows: {len(sample):,}; risk review rows: {len(risk):,}")


if __name__ == "__main__":
    main()
