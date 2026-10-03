"""Fit and evaluate an Isolation Forest using healthy training-period rows only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score

from .train import _metrics, _quoted, _threshold_at_fpr


def run_anomaly(features: Path, config_path: Path, model_dir: Path, outdir: Path,
                healthy_fraction: float = 0.1, estimators: int = 200, seed: int = 42) -> None:
    if not 0 < healthy_fraction <= 1:
        raise ValueError("healthy_fraction must be in (0, 1]")
    if estimators < 1:
        raise ValueError("estimators must be positive")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    split, horizon = config["split"], int(config["label"]["horizon_days"])
    train_cut = (pd.Timestamp(split["train_end"]) - pd.Timedelta(days=horizon)).date().isoformat()
    val_start = pd.Timestamp(split["validation_start"]).date().isoformat()
    val_cut = (pd.Timestamp(split["validation_end"]) - pd.Timedelta(days=horizon)).date().isoformat()
    test_start = pd.Timestamp(split["test_start"]).date().isoformat()
    test_cut = (pd.Timestamp(split["test_end"]) - pd.Timedelta(days=horizon)).date().isoformat()
    model_dir.mkdir(parents=True, exist_ok=True)
    outdir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    try:
        p = str(features).replace("'", "''")
        all_columns = [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM read_parquet('{p}')").fetchall()]
        numeric = [c for c in all_columns if c == "capacity_bytes" or c.startswith("smart_")]
        categorical = ["model"] if "model" in all_columns else []
        metadata = ["obs_date", "serial_number", "failure_date", "will_fail_30d"]
        cols = list(dict.fromkeys(numeric + categorical + metadata))
        select = ", ".join(_quoted(c) for c in cols)
        mod = max(1, round(1 / healthy_fraction))
        filter_sample = f"hash(serial_number) % {mod} = 0" if mod > 1 else "TRUE"
        train = con.execute(
            f"SELECT {select} FROM read_parquet('{p}') WHERE obs_date < DATE '{train_cut}' "
            f"AND will_fail_30d=0 AND ({filter_sample})"
        ).df()
        validation = con.execute(
            f"SELECT {select} FROM read_parquet('{p}') WHERE obs_date >= DATE '{val_start}' "
            f"AND obs_date < DATE '{val_cut}'"
        ).df()
    finally:
        con.close()
    if train.empty or validation.empty or train["will_fail_30d"].nunique() != 1:
        raise ValueError("Training must include healthy rows; validation must be non-empty")
    if not validation["will_fail_30d"].isin([0, 1]).all():
        raise ValueError("Validation contains non-binary labels")

    # Reuse the preprocessor fitted on the supervised training period so this
    # baseline sees exactly the same as-of feature representation.
    xgb_bundle = joblib.load(model_dir / "xgboost.joblib")
    preprocessor = xgb_bundle["preprocessor"]
    if numeric != xgb_bundle["numeric_features"]:
        raise ValueError("Feature columns differ from the fitted preprocessing schema")
    x_train = preprocessor.transform(train[numeric + categorical])
    x_val = preprocessor.transform(validation[numeric + categorical])
    forest = IsolationForest(
        n_estimators=estimators, max_samples=min(512, len(train)), contamination="auto",
        # Single-worker execution avoids Windows process-pool restrictions in
        # managed environments; max_samples bounds work per tree.
        n_jobs=1, random_state=seed,
    )
    forest.fit(x_train)
    val_scores = -forest.decision_function(x_val)
    y_val = validation["will_fail_30d"].to_numpy(dtype=np.int8)
    threshold = _threshold_at_fpr(y_val, val_scores)

    # Read the full test features in Arrow batches to bound working memory.
    con = duckdb.connect()
    try:
        p = str(features).replace("'", "''")
        query = (f"SELECT {select} FROM read_parquet('{p}') WHERE obs_date >= DATE '{test_start}' "
                 f"AND obs_date <= DATE '{test_cut}' ORDER BY obs_date, serial_number")
        reader = con.execute(query).to_arrow_reader(batch_size=100_000)
        predictions = []
        scores_parts = []
        labels_parts = []
        for batch in reader:
            chunk = batch.to_pandas()
            x = preprocessor.transform(chunk[numeric + categorical])
            score = -forest.decision_function(x)
            scores_parts.append(score)
            labels_parts.append(chunk["will_fail_30d"].to_numpy(dtype=np.int8))
            predictions.append(chunk[metadata].assign(anomaly_score=score))
    finally:
        con.close()
    test = pd.concat(predictions, ignore_index=True)
    test_scores = np.concatenate(scores_parts)
    test_y = np.concatenate(labels_parts)
    test["obs_date"] = pd.to_datetime(test["obs_date"])
    test.to_parquet(outdir / "isolation_forest_test_predictions.parquet", index=False, compression="zstd")

    validation_frame = validation[["obs_date", "serial_number", "failure_date", "will_fail_30d"]]
    result = {
        "model": "IsolationForest",
        "score_definition": "negative sklearn decision_function; higher values indicate greater anomaly",
        "training_scope": "healthy training-period drive-days only",
        "healthy_training_rows": int(len(train)),
        "healthy_training_drives": int(train["serial_number"].nunique()),
        "healthy_sample_fraction": 1 / mod,
        "n_estimators": estimators,
        "seed": seed,
        "validation": {
            "rows": int(len(validation)), "positives": int(y_val.sum()),
            "prevalence": float(y_val.mean()), "average_precision": float(average_precision_score(y_val, val_scores)),
            "threshold_selected_on_validation_at_max_1pct_fpr": threshold,
            **_metrics(validation_frame, val_scores, threshold),
        },
        "test": {
            "rows": int(len(test)), "positives": int(test_y.sum()),
            "prevalence": float(test_y.mean()), "average_precision": float(average_precision_score(test_y, test_scores)),
            "threshold_selected_on_validation_at_max_1pct_fpr": threshold,
            **_metrics(test, test_scores, threshold),
        },
        "interpretation": "Unsupervised anomaly ranking, not a calibrated failure probability. Threshold is selected on validation to target at most 1% row-level FPR.",
    }
    (outdir / "isolation_forest_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    joblib.dump({"preprocessor": preprocessor, "model": forest, "numeric_features": numeric,
                 "categorical_features": categorical, "training": {"healthy_sample_fraction": 1 / mod, "seed": seed}},
                model_dir / "isolation_forest.joblib", compress=3)
    print(f"Isolation Forest test AP={result['test']['average_precision']:.4f}; ")
    print(f"recall={result['test']['recall']:.3%}, precision={result['test']['precision']:.3%}; "
          f"wrote results to {outdir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=Path("data/processed/features.parquet"))
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--outdir", type=Path, default=Path("reports/model_evaluation"))
    parser.add_argument("--healthy-fraction", type=float, default=0.1)
    parser.add_argument("--estimators", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_anomaly(args.features, args.config, args.model_dir, args.outdir, args.healthy_fraction, args.estimators, args.seed)


if __name__ == "__main__":
    main()
