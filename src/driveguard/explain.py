"""Create global and representative local SHAP explanations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from .train import _quoted


def explain(features: Path, sample_path: Path, model_dir: Path, metrics_path: Path, output_dir: Path) -> None:
    config = json.loads(metrics_path.read_text(encoding="utf-8"))
    threshold = float(config["models"]["xgboost"]["test"]["threshold_selected_on_validation_at_max_1pct_fpr"])
    bundle = joblib.load(model_dir / "xgboost.joblib")
    model = bundle["model"]
    preprocessor = bundle["preprocessor"]

    # Global importance uses a deterministic row sample at natural prevalence.
    p = str(features).replace("'", "''")
    con = duckdb.connect()
    try:
        schema = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{p}')").fetchall()
        columns = [r[0] for r in schema]
        numeric = [c for c in columns if c == "capacity_bytes" or c.startswith("smart_")]
        categorical = ["model"] if "model" in columns else []
        selection = list(dict.fromkeys(numeric + categorical))
        select = ", ".join(_quoted(c) for c in selection)
        global_frame = con.execute(
            f"SELECT {select} FROM read_parquet('{p}') "
            "WHERE obs_date >= DATE '2013-10-15' AND obs_date <= DATE '2013-12-01' "
            "AND hash(serial_number, obs_date) % 500 = 0"
        ).df()
    finally:
        con.close()

    sample_frame = pd.read_parquet(sample_path)
    positives = sample_frame[sample_frame["will_fail_30d"].eq(1)].copy()
    negatives = sample_frame[sample_frame["will_fail_30d"].eq(0)].copy()
    candidates: list[pd.Series] = []
    detected = positives[positives["risk_xgboost"] >= threshold]
    if not detected.empty:
        row = detected.sort_values("risk_xgboost", ascending=False).iloc[0].copy()
        row["case_type"] = "detected_failure"
        candidates.append(row)
    missed = positives[positives["risk_xgboost"] < threshold]
    if not missed.empty:
        row = missed.sort_values("risk_xgboost").iloc[0].copy()
        row["case_type"] = "missed_failure"
        candidates.append(row)
    false_alarms = negatives[negatives["risk_xgboost"] >= threshold]
    if not false_alarms.empty:
        row = false_alarms.sort_values("risk_xgboost", ascending=False).iloc[0].copy()
        row["case_type"] = "false_alarm"
        candidates.append(row)
    if not candidates or global_frame.empty:
        raise ValueError("Could not select explanation rows from the supplied data")
    cases = pd.DataFrame(candidates).reset_index(drop=True)

    names = preprocessor.get_feature_names_out()
    background_x = preprocessor.transform(global_frame[selection])
    if hasattr(background_x, "toarray"):
        background_x = background_x.toarray()
    explainer = shap.TreeExplainer(model)
    background_values = explainer.shap_values(background_x)
    if isinstance(background_values, list):
        background_values = background_values[1]
    background_values = np.asarray(background_values)
    mean_abs = np.abs(background_values).mean(axis=0)

    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 7))
    shap.summary_plot(
        background_values,
        background_x,
        feature_names=names,
        plot_type="bar",
        max_display=18,
        show=False,
    )
    plt.title("Mean absolute SHAP value on deterministic test sample")
    plt.tight_layout()
    plt.savefig(output_dir / "shap_global_importance.png", dpi=160, bbox_inches="tight")
    plt.close()

    cases = cases.reset_index(drop=True)
    cases["case_id"] = np.arange(len(cases))
    con = duckdb.connect()
    try:
        con.register("candidate_rows", cases[["case_id", "serial_number", "obs_date"]])
        case_select = ", ".join(f"f.{_quoted(c)}" for c in selection)
        case_frame = con.execute(
            f"SELECT {case_select}, c.case_id FROM read_parquet('{p}') f "
            "INNER JOIN candidate_rows c ON f.serial_number=c.serial_number AND f.obs_date=c.obs_date "
            "ORDER BY c.case_id"
        ).df()
    finally:
        con.close()
    if len(case_frame) != len(cases):
        raise ValueError("Could not restore all selected case features from the feature dataset")
    case_x = preprocessor.transform(case_frame[selection])
    if hasattr(case_x, "toarray"):
        case_x = case_x.toarray()
    case_values = explainer.shap_values(case_x)
    if isinstance(case_values, list):
        case_values = case_values[1]
    case_values = np.asarray(case_values)
    expected = explainer.expected_value
    if isinstance(expected, (list, np.ndarray)):
        expected = np.asarray(expected).reshape(-1)[-1]

    summary_rows = []
    for i, row in cases.iterrows():
        explanation = shap.Explanation(
            values=case_values[i],
            base_values=float(expected),
            data=case_x[i],
            feature_names=names,
        )
        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(explanation, max_display=12, show=False)
        plt.title(f"{row['case_type']} | {row['serial_number']} | {row['obs_date']}")
        plt.tight_layout()
        plt.savefig(output_dir / f"{row['case_type']}.png", dpi=160, bbox_inches="tight")
        plt.close()
        top = np.argsort(np.abs(case_values[i]))[::-1][:5]
        summary_rows.append(
            {
                "case_type": row["case_type"],
                "serial_number": row["serial_number"],
                "observation_date": str(row["obs_date"]),
                "failure_date": str(row["failure_date"]),
                "label": int(row["will_fail_30d"]),
                "xgboost_score": float(row["risk_xgboost"]),
                "threshold": threshold,
                "top_feature_1": names[top[0]],
                "top_feature_2": names[top[1]],
                "top_feature_3": names[top[2]],
                "top_feature_4": names[top[3]],
                "top_feature_5": names[top[4]],
            }
        )
    pd.DataFrame(summary_rows).to_csv(output_dir / "case_explanations.csv", index=False)

    feature_importance = sorted(zip(names, mean_abs), key=lambda x: x[1], reverse=True)[:10]
    lines = [
        "# DriveGuard explanations",
        "",
        "The global chart uses a deterministic row sample from the held-out test interval at approximately natural prevalence. Local cases come from the dashboard drill-down sample, which contains a 1% sample of healthy-drive histories plus all positive-labelled drive-days.",
        "",
        "Positive SHAP contributions raise this fitted model's score relative to its baseline; negative contributions lower it. These explanations describe model behavior and are not causal findings.",
        "",
        "## Highest mean absolute SHAP values",
        "",
        "| Feature | Mean absolute SHAP |",
        "|---|---:|",
    ]
    lines.extend(f"| {name} | {value:.5f} |" for name, value in feature_importance)
    lines.extend(["", "## Local cases", "", "See the three waterfall plots and `case_explanations.csv` for one detected failure, one missed failure, and one false alarm when available. These examples are selected from the stated deterministic sample, not from the entire fleet."])
    (output_dir / "explanations.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote SHAP report and local explanations to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=Path("data/processed/features.parquet"))
    parser.add_argument("--sample", type=Path, default=Path("reports/model_evaluation/test_predictions_sample.parquet"))
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--metrics", type=Path, default=Path("reports/model_evaluation/metrics.json"))
    parser.add_argument("--outdir", type=Path, default=Path("reports/explainability"))
    args = parser.parse_args()
    explain(args.features, args.sample, args.model_dir, args.metrics, args.outdir)


if __name__ == "__main__":
    main()
