"""Score untouched chronological holdouts using previously saved model artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, precision_recall_curve

from .train import _metrics, _threshold_at_fpr, _quoted


def evaluate(features: Path, config_path: Path, model_dir: Path, output_dir: Path) -> None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    split = config["split"]
    horizon = int(config["label"]["horizon_days"])
    val_start = pd.Timestamp(split["validation_start"]).date().isoformat()
    val_cut = (pd.Timestamp(split["validation_end"]) - pd.Timedelta(days=horizon)).date().isoformat()
    test_start = pd.Timestamp(split["test_start"]).date().isoformat()
    test_cut = (pd.Timestamp(split["test_end"]) - pd.Timedelta(days=horizon)).date().isoformat()
    p = str(features).replace("'", "''")
    con = duckdb.connect()
    try:
        schema = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{p}')").fetchall()
        all_columns = [row[0] for row in schema]
        numeric = [c for c in all_columns if c == "capacity_bytes" or c.startswith("smart_")]
        categorical = ["model"] if "model" in all_columns else []
        feature_columns = numeric + categorical
        select = ", ".join(_quoted(c) for c in feature_columns + ["obs_date", "serial_number", "failure_date", "will_fail_30d"])
        val = con.execute(
            f"SELECT {select} FROM read_parquet('{p}') WHERE obs_date >= DATE '{val_start}' AND obs_date < DATE '{val_cut}'"
        ).df()
        test = con.execute(
            f"SELECT {select} FROM read_parquet('{p}') WHERE obs_date >= DATE '{test_start}' AND obs_date <= DATE '{test_cut}'"
        ).df()
    finally:
        con.close()

    if not len(val) or not len(test):
        raise ValueError("Validation or test split is empty")
    print(f"Loaded complete holdouts: validation {len(val):,}; test {len(test):,}.", flush=True)
    lr_bundle = joblib.load(model_dir / "logistic_regression.joblib")
    xgb_bundle = joblib.load(model_dir / "xgboost.joblib")
    previous_metrics_path = output_dir / "metrics.json"
    previous_metrics = json.loads(previous_metrics_path.read_text(encoding="utf-8")) if previous_metrics_path.exists() else {}
    training_metadata = dict(lr_bundle.get("training", {}))
    if "healthy_sample_fraction" not in training_metadata:
        training_metadata["healthy_sample_fraction"] = previous_metrics.get("healthy_sample_fraction")
        fraction = training_metadata["healthy_sample_fraction"]
        training_metadata["healthy_sample_weight"] = 1.0 / fraction if fraction else None
    previous_training = previous_metrics.get("training", {})
    for key in ("train_rows_sampled", "train_positive_rows"):
        if key not in training_metadata:
            training_metadata[key] = previous_metrics.get(key, previous_training.get(key))
    preprocessor = lr_bundle["preprocessor"]
    if numeric != lr_bundle["numeric_features"]:
        raise ValueError("Saved model feature schema does not match the current feature dataset")
    x_val = preprocessor.transform(val[feature_columns])
    x_test = preprocessor.transform(test[feature_columns])
    print("Transformed validation and test features.", flush=True)
    models = {"logistic_regression": lr_bundle["model"], "xgboost": xgb_bundle["model"]}
    scores = {
        name: (model.predict_proba(x_val)[:, 1], model.predict_proba(x_test)[:, 1])
        for name, model in models.items()
    }
    if {"smart_5_raw_num", "smart_197_raw_num"}.issubset(val.columns):
        val_rule = np.log1p(np.maximum(val["smart_5_raw_num"].fillna(0), val["smart_197_raw_num"].fillna(0))).to_numpy()
        test_rule = np.log1p(np.maximum(test["smart_5_raw_num"].fillna(0), test["smart_197_raw_num"].fillna(0))).to_numpy()
        scores["sector_count_rule"] = (val_rule, test_rule)

    output_dir.mkdir(parents=True, exist_ok=True)
    scored = test[["obs_date", "serial_number", "model", "failure_date", "will_fail_30d"]].copy()
    for name, (_, test_score) in scores.items():
        scored[f"risk_{name}"] = test_score
    scored.to_parquet(output_dir / "test_predictions.parquet", index=False, compression="zstd")
    metrics = {}
    curves = {}
    y_val = val["will_fail_30d"].to_numpy(dtype=np.int8)
    y_test = test["will_fail_30d"].to_numpy(dtype=np.int8)
    for name, (val_score, test_score) in scores.items():
        threshold = _threshold_at_fpr(y_val, val_score)
        metrics[name] = {
            "validation": _metrics(val, val_score, threshold),
            "test": _metrics(test, test_score, threshold),
        }
        curves[name] = precision_recall_curve(y_test, test_score)[:2]
    result = {
        "horizon_days": horizon,
        "split": split,
        "train_rows_sampled": training_metadata.get("train_rows_sampled"),
        "train_positive_rows": training_metadata.get("train_positive_rows"),
        "healthy_sample_fraction": training_metadata.get("healthy_sample_fraction"),
        "training": {
            **training_metadata,
            "logistic_solver": lr_bundle["model"].solver,
            "logistic_max_iter": int(lr_bundle["model"].max_iter),
            "xgboost_best_iteration": int(xgb_bundle["model"].best_iteration),
            "xgboost_n_estimators": int(xgb_bundle["model"].n_estimators),
        },
        "validation": {"start": val_start, "last_day_exclusive": val_cut, "rows": len(val)},
        "test": {"start": test_start, "last_day_inclusive": test_cut, "rows": len(test)},
        "models": metrics,
    }
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    pd.DataFrame(
        [
            {"model": name, **{f"validation_{k}": v for k, v in values["validation"].items()},
             **{f"test_{k}": v for k, v in values["test"].items()}}
            for name, values in metrics.items()
        ]
    ).to_csv(output_dir / "metrics.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    for name, (precision, recall) in curves.items():
        ap = average_precision_score(y_test, scores[name][1])
        ax.plot(recall, precision, label=f"{name} (AP={ap:.3f})")
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision–recall on future test period")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(output_dir / "precision_recall_test.png", dpi=160)
    plt.close(fig)
    print(f"Wrote validation/test metrics to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=Path("data/processed/features.parquet"))
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--outdir", type=Path, default=Path("reports/model_evaluation"))
    args = parser.parse_args()
    evaluate(args.features, args.config, args.model_dir, args.outdir)


if __name__ == "__main__":
    main()
