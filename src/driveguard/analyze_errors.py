"""Calibration, drive-cluster uncertainty, and slice-based error analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss


PROBABILITY_MODELS = ("logistic_regression", "xgboost")


def _calibration(frame: pd.DataFrame, model: str, bins: int = 10) -> tuple[pd.DataFrame, dict]:
    y = frame["will_fail_30d"].to_numpy(dtype=np.int8)
    p = frame[f"risk_{model}"].to_numpy(dtype=float)
    p = np.clip(p, 0.0, 1.0)
    edges = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
    group = np.minimum(np.searchsorted(edges[1:-1], p, side="right"), max(len(edges) - 2, 0))
    curve = pd.DataFrame({"bin": group, "predicted": p, "observed": y}).groupby("bin", as_index=False).agg(
        rows=("observed", "size"), mean_predicted_probability=("predicted", "mean"),
        observed_prevalence=("observed", "mean"), positives=("observed", "sum"),
    )
    ece = float(np.sum(curve["rows"] / len(frame) * np.abs(
        curve["mean_predicted_probability"] - curve["observed_prevalence"]
    )))
    return curve, {"brier_score": float(brier_score_loss(y, p)), "expected_calibration_error_10_quantile_bins": ece}


def _cluster_intervals(frame: pd.DataFrame, model: str, threshold: float, seed: int, replicates: int) -> dict:
    y = frame["will_fail_30d"].to_numpy(dtype=np.int8)
    score = frame[f"risk_{model}"].to_numpy(dtype=float)
    alert = score >= threshold
    work = frame[["serial_number", "failure_date", "will_fail_30d"]].copy()
    work["tp"] = (alert & (y == 1)).astype(np.int64)
    work["fp"] = (alert & (y == 0)).astype(np.int64)
    work["fn"] = ((~alert) & (y == 1)).astype(np.int64)
    work["tn"] = ((~alert) & (y == 0)).astype(np.int64)
    work["event_key"] = work["failure_date"].where(y == 1)
    event_rows = work[y == 1].groupby(["serial_number", "event_key"], dropna=False).agg(
        detected=("tp", lambda z: bool(z.sum())),
    ).reset_index()
    event_counts = event_rows.groupby("serial_number").agg(
        events=("event_key", "size"), detected_events=("detected", "sum")
    )
    by_drive = work.groupby("serial_number")[ ["tp", "fp", "fn", "tn"] ].sum().join(event_counts, how="left").fillna(0)
    values = by_drive.to_numpy(dtype=float)
    n = len(values)
    if n < 2:
        raise ValueError("Need at least two drives for cluster bootstrap")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n, size=(replicates, n))
    totals = values[draws].sum(axis=1)
    tp, fp, fn, tn = (totals[:, i] for i in range(4))
    events, detected = totals[:, 4], totals[:, 5]
    metrics = {
        "precision": tp / np.maximum(tp + fp, 1),
        "recall": tp / np.maximum(tp + fn, 1),
        "false_positive_rate": fp / np.maximum(fp + tn, 1),
        "alerts_per_1000_drive_days": (tp + fp) / np.maximum(tp + fp + fn + tn, 1) * 1000,
        "failure_event_recall": detected / np.maximum(events, 1),
    }
    intervals = {}
    for name, sample in metrics.items():
        intervals[name] = {
            "lower_95": float(np.quantile(sample, 0.025)),
            "upper_95": float(np.quantile(sample, 0.975)),
        }
    return {"resampling_unit": "serial_number (drive cluster)", "replicates": replicates, "seed": seed,
            "unique_drives": int(n), "intervals": intervals}


def analyze(predictions: Path, metrics_path: Path, outdir: Path, replicates: int = 300, seed: int = 42) -> None:
    frame = pd.read_parquet(predictions)
    frame["obs_date"] = pd.to_datetime(frame["obs_date"])
    frame["month"] = frame["obs_date"].dt.to_period("M").astype(str)
    baseline = json.loads(metrics_path.read_text(encoding="utf-8"))
    outdir.mkdir(parents=True, exist_ok=True)
    calibration_rows = []
    summary = {"test_rows": int(len(frame)), "test_prevalence": float(frame["will_fail_30d"].mean()), "models": {}}
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    for model in PROBABILITY_MODELS:
        curve, stats = _calibration(frame, model)
        curve.insert(0, "model", model)
        calibration_rows.append(curve)
        ax.plot(curve["mean_predicted_probability"], curve["observed_prevalence"], marker="o", label=model.replace("_", " ").title())
        threshold = float(baseline["models"][model]["test"]["threshold_selected_on_validation_at_max_1pct_fpr"])
        summary["models"][model] = {
            **stats,
            "threshold_selected_on_validation": threshold,
            "drive_cluster_bootstrap_95pct": _cluster_intervals(frame, model, threshold, seed, replicates),
        }
    ax.set(xlabel="Mean predicted probability (quantile bin)", ylabel="Observed failure prevalence", title="Test-period probability calibration")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(outdir / "calibration_test.png", dpi=160)
    plt.close(fig)
    pd.concat(calibration_rows, ignore_index=True).to_csv(outdir / "calibration_bins.csv", index=False)

    slice_rows = []
    model_thresholds = {m: float(baseline["models"][m]["test"]["threshold_selected_on_validation_at_max_1pct_fpr"]) for m in PROBABILITY_MODELS}
    for group_name in ("month", "model"):
        for group_value, group in frame.groupby(group_name, dropna=False):
            if len(group) < 100:
                continue
            for model in PROBABILITY_MODELS:
                y = group["will_fail_30d"].to_numpy(dtype=np.int8)
                pred = group[f"risk_{model}"].to_numpy(dtype=float) >= model_thresholds[model]
                pos = int(y.sum())
                tp = int(np.sum(pred & (y == 1)))
                fp = int(np.sum(pred & (y == 0)))
                fn = pos - tp
                slice_rows.append({"slice": group_name, "value": str(group_value), "model": model,
                    "rows": len(group), "positives": pos, "prevalence": float(y.mean()),
                    "recall_at_validation_threshold": tp / pos if pos else None,
                    "precision_at_validation_threshold": tp / (tp + fp) if tp + fp else 0.0,
                    "false_negatives": fn, "false_positives": fp})
    slices = pd.DataFrame(slice_rows)
    slices.to_csv(outdir / "error_slices.csv", index=False)
    summary["thresholds_note"] = "All thresholds were selected on validation; test slices are diagnostic only."
    summary["calibration_note"] = "Calibration is measured on the natural-prevalence test set. Models were fit with sampled healthy training rows and inverse sampling weights; probability calibration may still require a separate validation-fitted calibrator before deployment."
    summary["uncertainty_note"] = "Percentile intervals resample whole serial-number clusters and quantify sampling variation across drives in this test period; they do not account for temporal or dataset-source shift."
    (outdir / "analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = ["# Calibration and error analysis", "",
        f"The complete test set contains {len(frame):,} eligible drive-days with {int(frame['will_fail_30d'].sum()):,} positives ({frame['will_fail_30d'].mean():.3%}).", "",
        "## Probability quality", "", "| Model | Brier score | 10-bin ECE |", "|---|---:|---:|"]
    for model, stats in summary["models"].items():
        lines.append(f"| {model.replace('_', ' ').title()} | {stats['brier_score']:.5f} | {stats['expected_calibration_error_10_quantile_bins']:.5f} |")
    lines.extend(["", "Brier score and expected calibration error are evaluated at the observed test prevalence. The calibration plot uses quantile bins, so bin widths in probability space can differ. A calibration method would need to be fit on validation data and then frozen before test evaluation.", "",
        "## Drive-cluster 95% bootstrap intervals", "", "| Model | Metric | Lower | Upper |", "|---|---|---:|---:|"])
    for model, stats in summary["models"].items():
        for metric, bounds in stats["drive_cluster_bootstrap_95pct"]["intervals"].items():
            lines.append(f"| {model.replace('_', ' ').title()} | {metric.replace('_', ' ')} | {bounds['lower_95']:.4f} | {bounds['upper_95']:.4f} |")
    lines.extend(["", "Intervals resample serial-number clusters, preserving within-drive repeated observations. They quantify sampling variation across drives in this test period, not uncertainty from time or source shift.", "",
        "## Slice diagnostics", "", "See `error_slices.csv` for month- and drive-model-specific prevalence, false negatives, false positives, precision, and recall at thresholds chosen on validation. Small or shifted slices should be interpreted cautiously; these are diagnostics, not new tuning targets.", ""])
    (outdir / "error_analysis.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote calibration and drive-cluster error analysis to {outdir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, default=Path("reports/model_evaluation/test_predictions.parquet"))
    parser.add_argument("--metrics", type=Path, default=Path("reports/model_evaluation/metrics.json"))
    parser.add_argument("--outdir", type=Path, default=Path("reports/analysis"))
    parser.add_argument("--replicates", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    analyze(args.predictions, args.metrics, args.outdir, args.replicates, args.seed)


if __name__ == "__main__":
    main()
