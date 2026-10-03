"""Train baselines and an XGBoost ranker on the purged chronological split."""

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
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


METADATA = {"date", "obs_date", "serial_number", "filename", "failure_date", "will_fail_30d", "label_status"}


def _quoted(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _read_rows(con, path: str, columns: list[str], where: str) -> pd.DataFrame:
    selection = ", ".join(_quoted(c) for c in columns)
    return con.execute(
        f"SELECT {selection} FROM read_parquet('{path}') WHERE {where} "
        "ORDER BY obs_date, serial_number"
    ).df()


def _threshold_at_fpr(y: np.ndarray, scores: np.ndarray, max_fpr: float = 0.01) -> float:
    negatives = scores[y == 0]
    if not len(negatives):
        raise ValueError("Validation split has no negative observations")
    if not 0 <= max_fpr <= 1:
        raise ValueError("max_fpr must be between 0 and 1")
    ordered = np.sort(negatives)
    allowed_alerts = int(np.floor(max_fpr * len(ordered) + 1e-12))
    unique, counts = np.unique(ordered, return_counts=True)
    alerts_at_or_above = np.cumsum(counts[::-1])[::-1]
    feasible = np.flatnonzero(alerts_at_or_above <= allowed_alerts)
    if len(feasible):
        return float(unique[feasible[0]])
    # This occurs when the maximum-score tie alone exceeds the budget, or when
    # the budget is zero. A threshold just above the maximum yields no alerts.
    return float(np.nextafter(ordered[-1], np.inf))


def _metrics(frame: pd.DataFrame, scores: np.ndarray, threshold: float) -> dict:
    y = frame["will_fail_30d"].to_numpy(dtype=np.int8)
    alert = scores >= threshold
    tp = int(np.sum(alert & (y == 1)))
    fp = int(np.sum(alert & (y == 0)))
    fn = int(np.sum(~alert & (y == 1)))
    tn = int(np.sum(~alert & (y == 0)))
    fpr = fp / (fp + tn) if fp + tn else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    positive = frame.loc[y == 1, ["serial_number", "failure_date", "obs_date"]].copy()
    positive["alert"] = alert[y == 1]
    positive["lead_days"] = (pd.to_datetime(positive["failure_date"]) - pd.to_datetime(positive["obs_date"])).dt.days
    if positive.empty:
        event_recall, median_lead = 0.0, None
    else:
        positive["alert_lead_days"] = positive["lead_days"].where(positive["alert"])
        events = positive.groupby(["serial_number", "failure_date"], dropna=False).agg(
            detected=("alert", "any"), lead_days=("alert_lead_days", "max")
        )
        detected = events.loc[events["detected"]]
        event_recall = float(events["detected"].mean())
        median_lead = float(detected["lead_days"].median()) if not detected.empty else None

    return {
        "rows": int(len(frame)),
        "positive_rows": int(y.sum()),
        "prevalence": float(y.mean()) if len(y) else 0.0,
        "average_precision": float(average_precision_score(y, scores)),
        "threshold_selected_on_validation_at_max_1pct_fpr": threshold,
        "recall": recall,
        "precision": precision,
        "false_positive_rate": fpr,
        "alerts_per_1000_drive_days": float(alert.mean() * 1000),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "failure_events_with_any_alert_recall": event_recall,
        "eligible_failure_events": int(len(events)) if not positive.empty else 0,
        "median_lead_days_detected_events": median_lead,
    }


def train(features: Path, config_path: Path, output_dir: Path, model_dir: Path, sample_fraction: float = 0.2) -> None:
    if not features.exists():
        raise FileNotFoundError(features)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    split = config["split"]
    horizon = int(config["label"]["horizon_days"])
    train_cut = (pd.Timestamp(split["train_end"]) - pd.Timedelta(days=horizon)).date().isoformat()
    val_start = pd.Timestamp(split["validation_start"]).date().isoformat()
    val_cut = (pd.Timestamp(split["validation_end"]) - pd.Timedelta(days=horizon)).date().isoformat()
    test_start = pd.Timestamp(split["test_start"]).date().isoformat()
    test_cut = (pd.Timestamp(split["test_end"]) - pd.Timedelta(days=horizon)).date().isoformat()
    if not (train_cut <= val_start < val_cut <= test_start < test_cut):
        raise ValueError("Configured temporal periods are empty or out of order")
    if not 0 < sample_fraction <= 1:
        raise ValueError("sample_fraction must be in (0, 1]")

    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    p = str(features).replace("'", "''")
    con = duckdb.connect()
    try:
        schema = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{p}')").fetchall()
        all_columns = [row[0] for row in schema]
        numeric = [c for c in all_columns if c == "capacity_bytes" or c.startswith("smart_")]
        categorical = ["model"] if "model" in all_columns else []
        feature_columns = numeric + categorical
        required = ["obs_date", "serial_number", "failure_date", "will_fail_30d"]
        if any(c not in all_columns for c in required) or not numeric:
            raise ValueError("Feature Parquet does not contain the required target and feature columns")
        columns = feature_columns + required
        sample_mod = max(1, round(1 / sample_fraction))
        effective_fraction = 1 / sample_mod
        frac = f"hash(serial_number) % {sample_mod} = 0" if sample_mod > 1 else "TRUE"
        train_frame = _read_rows(
            con, p, columns,
            f"obs_date < DATE '{train_cut}' AND (will_fail_30d = 1 OR ({frac}))",
        )
        val_frame = _read_rows(
            con, p, columns,
            f"obs_date >= DATE '{val_start}' AND obs_date < DATE '{val_cut}'",
        )
        test_frame = _read_rows(
            con, p, columns,
            f"obs_date >= DATE '{test_start}' AND obs_date <= DATE '{test_cut}'",
        )
    finally:
        con.close()

    if not len(train_frame) or not train_frame["will_fail_30d"].nunique() == 2:
        raise ValueError("Training split must contain both classes")
    if not (val_frame["will_fail_30d"].eq(0).any() and val_frame["will_fail_30d"].eq(1).any()):
        raise ValueError("Validation split must contain both classes")
    print(
        f"Loaded train sample {len(train_frame):,} rows, validation {len(val_frame):,}, "
        f"test {len(test_frame):,}; natural holdout prevalence preserved.",
        flush=True,
    )

    y_train = train_frame["will_fail_30d"].to_numpy(dtype=np.int8)
    y_val = val_frame["will_fail_30d"].to_numpy(dtype=np.int8)
    y_test = test_frame["will_fail_30d"].to_numpy(dtype=np.int8)
    # Correct the training loss for the deterministic healthy-row sample.
    weights = np.where(y_train == 1, 1.0, 1.0 / effective_fraction)

    preprocessor = ColumnTransformer(
        [
            ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
            ("model", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True))]), categorical),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )
    x_train = preprocessor.fit_transform(train_frame[feature_columns])
    x_val = preprocessor.transform(val_frame[feature_columns])
    x_test = preprocessor.transform(test_frame[feature_columns])
    print("Finished preprocessing; fitting logistic baseline.", flush=True)

    predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    lr = LogisticRegression(max_iter=80, solver="liblinear", random_state=int(config.get("seed", 42)))
    lr.fit(x_train, y_train, sample_weight=weights)
    predictions["logistic_regression"] = (lr.predict_proba(x_val)[:, 1], lr.predict_proba(x_test)[:, 1])
    joblib.dump(
        {"preprocessor": preprocessor, "model": lr, "numeric_features": numeric,
         "training": {"healthy_sample_fraction": effective_fraction, "healthy_sample_weight": 1.0 / effective_fraction,
                      "train_rows_sampled": int(len(train_frame)), "train_positive_rows": int(y_train.sum()), "split": split}},
        model_dir / "logistic_regression.joblib",
    )
    print("Logistic baseline fitted; fitting XGBoost.", flush=True)

    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise RuntimeError("Install the optional model package with: pip install -r requirements-model.txt") from exc
    xgb = XGBClassifier(
        n_estimators=180,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=10,
        reg_lambda=2.0,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        early_stopping_rounds=30,
        # Single-threaded CPU histogram building avoids run-to-run floating
        # point reduction differences in early stopping on rare-event data.
        device="cpu",
        n_jobs=1,
        random_state=int(config.get("seed", 42)),
    )
    xgb.fit(x_train, y_train, sample_weight=weights, eval_set=[(x_val, y_val)], verbose=False)
    predictions["xgboost"] = (xgb.predict_proba(x_val)[:, 1], xgb.predict_proba(x_test)[:, 1])
    joblib.dump(
        {"preprocessor": preprocessor, "model": xgb, "numeric_features": numeric,
         "training": {"healthy_sample_fraction": effective_fraction, "healthy_sample_weight": 1.0 / effective_fraction,
                      "train_rows_sampled": int(len(train_frame)), "train_positive_rows": int(y_train.sum()), "split": split}},
        model_dir / "xgboost.joblib",
    )

    # Simple telemetry rule, ranked by the maximum of two observable sector counters.
    if "smart_5_raw_num" in test_frame and "smart_197_raw_num" in test_frame:
        val_rule = np.log1p(
            np.maximum(val_frame["smart_5_raw_num"].fillna(0), val_frame["smart_197_raw_num"].fillna(0))
        ).to_numpy()
        test_rule = np.log1p(
            np.maximum(test_frame["smart_5_raw_num"].fillna(0), test_frame["smart_197_raw_num"].fillna(0))
        ).to_numpy()
        predictions["sector_count_rule"] = (val_rule, test_rule)

    metrics: dict[str, dict] = {}
    curves = {}
    for name, (val_score, test_score) in predictions.items():
        threshold = _threshold_at_fpr(y_val, val_score)
        metrics[name] = {
            "validation": _metrics(val_frame, val_score, threshold),
            "test": _metrics(test_frame, test_score, threshold),
        }
        curves[name] = precision_recall_curve(y_test, test_score)[:2]

    # Persist scores produced in this same full-test evaluation pass. This lets
    # downstream analysis reuse exact holdout predictions without rescoring.
    scored_test = test_frame[["obs_date", "serial_number", "model", "failure_date", "will_fail_30d"]].copy()
    for name, (_, test_score) in predictions.items():
        scored_test[f"risk_{name}"] = test_score
    scored_test.to_parquet(output_dir / "test_predictions.parquet", index=False, compression="zstd")

    result = {
        "dataset": str(features),
        "horizon_days": horizon,
        "split": split,
        "train_rows_sampled": int(len(train_frame)),
        "train_positive_rows": int(y_train.sum()),
        "healthy_sample_fraction": effective_fraction,
        "training": {
            "healthy_drive_sample_fraction": effective_fraction,
            "healthy_sample_weight": 1.0 / effective_fraction,
            "logistic_solver": lr.solver,
            "logistic_max_iter": int(lr.max_iter),
            "logistic_iterations_used": int(lr.n_iter_[0]),
            "xgboost_n_estimators": int(xgb.n_estimators),
            "xgboost_best_iteration": int(xgb.best_iteration),
            "xgboost_max_depth": int(xgb.max_depth),
            "xgboost_n_jobs": int(xgb.n_jobs),
        },
        "validation": {"start": val_start, "last_day_exclusive": val_cut, "rows": int(len(val_frame))},
        "test": {"start": test_start, "last_day_inclusive": test_cut, "rows": int(len(test_frame))},
        "models": metrics,
    }
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    rows = []
    for name, values in metrics.items():
        row = {"model": name}
        row.update({f"validation_{k}": v for k, v in values["validation"].items()})
        row.update({f"test_{k}": v for k, v in values["test"].items()})
        rows.append(row)
    pd.DataFrame(rows).to_csv(output_dir / "metrics.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    for name, (precision, recall) in curves.items():
        ax.plot(recall, precision, label=f"{name} (AP={metrics[name]['test']['average_precision']:.3f})")
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
    parser.add_argument("--outdir", type=Path, default=Path("reports/model_evaluation"))
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--sample-fraction", type=float, default=0.1)
    args = parser.parse_args()
    train(args.features, args.config, args.outdir, args.model_dir, args.sample_fraction)


if __name__ == "__main__":
    main()
