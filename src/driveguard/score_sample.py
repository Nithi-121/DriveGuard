"""Write a small deterministic test-period sample for dashboard drill-down."""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd

from .train import _quoted


def score_sample(features: Path, model_dir: Path, config_path: Path, output: Path, healthy_drive_mod: int = 100) -> None:
    if healthy_drive_mod < 1:
        raise ValueError("healthy_drive_mod must be at least 1")
    config = __import__("yaml").safe_load(config_path.read_text(encoding="utf-8"))
    split = config["split"]
    horizon = int(config["label"]["horizon_days"])
    start = pd.Timestamp(split["test_start"]).date().isoformat()
    end = (pd.Timestamp(split["test_end"]) - pd.Timedelta(days=horizon)).date().isoformat()
    p = str(features).replace("'", "''")
    con = duckdb.connect()
    try:
        schema = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{p}')").fetchall()
        all_columns = [row[0] for row in schema]
        numeric = [c for c in all_columns if c == "capacity_bytes" or c.startswith("smart_")]
        categorical = ["model"] if "model" in all_columns else []
        metadata = ["obs_date", "serial_number", "model", "failure_date", "will_fail_30d"]
        select_columns = list(dict.fromkeys(numeric + categorical + metadata))
        select = ", ".join(_quoted(c) for c in select_columns)
        frame = con.execute(
            f"SELECT {select} FROM read_parquet('{p}') "
            f"WHERE obs_date >= DATE '{start}' AND obs_date <= DATE '{end}' "
            f"AND (will_fail_30d=1 OR hash(serial_number) % {healthy_drive_mod}=0)"
        ).df()
    finally:
        con.close()
    if frame.empty:
        raise ValueError("Dashboard sample query returned no rows")

    lr_bundle = joblib.load(model_dir / "logistic_regression.joblib")
    xgb_bundle = joblib.load(model_dir / "xgboost.joblib")
    if_bundle = joblib.load(model_dir / "isolation_forest.joblib")
    preprocessor = xgb_bundle["preprocessor"]
    if numeric != xgb_bundle["numeric_features"]:
        raise ValueError("Saved model feature schema does not match the current features")
    x = preprocessor.transform(frame[numeric + categorical])
    frame["risk_xgboost"] = xgb_bundle["model"].predict_proba(x)[:, 1]
    frame["risk_logistic_regression"] = lr_bundle["model"].predict_proba(x)[:, 1]
    frame["anomaly_isolation_forest"] = -if_bundle["model"].decision_function(x)
    frame["risk_sector_count_rule"] = np.log1p(
        np.maximum(frame["smart_5_raw_num"].fillna(0), frame["smart_197_raw_num"].fillna(0))
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    keep = metadata + ["risk_xgboost", "risk_logistic_regression", "risk_sector_count_rule", "anomaly_isolation_forest"]
    frame[keep].to_parquet(output, index=False, compression="zstd")
    print(
        f"Wrote {len(frame):,} sampled test-period rows ({100/healthy_drive_mod:.1f}% of healthy-drive histories plus all positive rows) to {output}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=Path("data/processed/features.parquet"))
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--output", type=Path, default=Path("reports/model_evaluation/test_predictions_sample.parquet"))
    parser.add_argument("--healthy-drive-mod", type=int, default=100)
    args = parser.parse_args()
    score_sample(args.features, args.model_dir, args.config, args.output, args.healthy_drive_mod)


if __name__ == "__main__":
    main()
