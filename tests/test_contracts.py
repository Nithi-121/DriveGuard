from __future__ import annotations

import numpy as np
import pandas as pd

from driveguard.features import build_features
from driveguard.labels import build_labels
from driveguard.train import _threshold_at_fpr
from driveguard.evaluate import temporal_masks


def _source_rows() -> pd.DataFrame:
    rows = [
        ("2020-01-01", "A", 0),
        ("2020-01-05", "A", 0),
        ("2020-01-10", "A", 1),
        ("2020-01-01", "B", 0),
        ("2020-02-15", "B", 0),
        ("2020-01-01", "C", 0),
        ("2020-02-15", "C", 1),
    ]
    return pd.DataFrame(rows, columns=["date", "serial_number", "failure"]).assign(
        filename="fixture.csv", model="fixture-model", capacity_bytes="1000",
        smart_5_raw="1", smart_9_raw="2", smart_194_raw="3", smart_197_raw="4",
    )


def test_labels_distinguish_positive_negative_censored_and_failure_day(tmp_path):
    source = tmp_path / "source.parquet"
    output = tmp_path / "labels.parquet"
    _source_rows().to_parquet(source, index=False)

    build_labels(source, output, horizon_days=30)
    labels = pd.read_parquet(output)

    assert len(labels) == 5
    assert (labels["label_status"] == "positive").sum() == 2
    assert (labels["label_status"] == "negative").sum() == 1
    assert (labels["label_status"] == "negative_before_later_failure").sum() == 1
    assert (labels["label_status"] == "insufficient_followup").sum() == 1
    assert not labels["label_status"].eq("failure_day").any()
    assert set(labels.loc[labels["label_status"] == "positive", "serial_number"]) == {"A"}


def test_feature_windows_do_not_use_future_values(tmp_path):
    dates = pd.date_range("2020-01-01", periods=4).strftime("%Y-%m-%d")
    source_frame = pd.DataFrame({
        "date": dates,
        "serial_number": ["A"] * 4,
        "filename": ["fixture.csv"] * 4,
        "model": ["fixture-model"] * 4,
        "capacity_bytes": [1000] * 4,
        "smart_5_raw": [10, 20, 30, 40],
        "smart_9_raw": [1, 2, 3, 4],
        "smart_194_raw": [5, 5, 5, 5],
        "smart_197_raw": [1, 3, 5, 7],
    })
    label_frame = pd.DataFrame({
        "date": dates,
        "serial_number": ["A"] * 4,
        "failure_date": [None] * 4,
        "will_fail_30d": [0] * 4,
        "label_status": ["negative"] * 4,
    })
    labels = tmp_path / "labels.parquet"
    label_frame.to_parquet(labels, index=False)

    first_source = tmp_path / "source_first.parquet"
    source_frame.to_parquet(first_source, index=False)
    first_features = tmp_path / "features_first.parquet"
    build_features(first_source, labels, first_features)

    changed = source_frame.copy()
    changed.loc[3, "smart_5_raw"] = 999_999
    second_source = tmp_path / "source_changed_future.parquet"
    changed.to_parquet(second_source, index=False)
    second_features = tmp_path / "features_second.parquet"
    build_features(second_source, labels, second_features)

    first = pd.read_parquet(first_features)
    second = pd.read_parquet(second_features)
    first["obs_date"] = pd.to_datetime(first["obs_date"])
    second["obs_date"] = pd.to_datetime(second["obs_date"])
    first = first.set_index("obs_date")
    second = second.set_index("obs_date")
    target_date = pd.Timestamp("2020-01-03")
    assert first.loc[target_date, "smart_5_mean_7d"] == 20
    assert first.loc[target_date, "smart_5_mean_7d"] == second.loc[target_date, "smart_5_mean_7d"]
    assert first.loc[target_date, "smart_5_raw_num"] == second.loc[target_date, "smart_5_raw_num"]


def test_temporal_split_is_purged_disjoint_and_chronological():
    dates = pd.Series(pd.to_datetime([
        "2013-06-30", "2013-07-01", "2013-07-30", "2013-07-31",
        "2013-08-17", "2013-08-18", "2013-10-14", "2013-10-15",
        "2013-12-01", "2013-12-02",
    ]))
    masks = temporal_masks(
        dates,
        train_end="2013-07-31",
        validation_start="2013-07-31",
        validation_end="2013-09-17",
        test_start="2013-10-15",
        test_end="2013-12-31",
        horizon_days=30,
    )
    assert masks["train"].tolist() == [True, False, False, False, False, False, False, False, False, False]
    assert masks["validation"].tolist() == [False, False, False, True, True, False, False, False, False, False]
    assert masks["test"].tolist() == [False, False, False, False, False, False, False, True, True, False]
    assert not (masks["train"] & masks["validation"]).any()
    assert not (masks["validation"] & masks["test"]).any()


def test_validation_threshold_respects_requested_fpr():
    y = np.array([0, 0, 0, 0, 1], dtype=np.int8)
    scores = np.array([0.10, 0.20, 0.20, 0.40, 0.90])
    threshold = _threshold_at_fpr(y, scores, max_fpr=0.25)
    observed_fpr = np.mean(scores[y == 0] >= threshold)
    assert observed_fpr <= 0.25
    lower_score = np.max(scores[(y == 0) & (scores < threshold)])
    assert np.mean(scores[y == 0] >= lower_score) > 0.25


def test_validation_threshold_ties_never_exceed_alert_budget():
    y = np.array([0] * 10 + [1], dtype=np.int8)
    scores = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9, 0.9, 1.0])
    threshold = _threshold_at_fpr(y, scores, max_fpr=0.1)
    assert np.mean(scores[y == 0] >= threshold) <= 0.1


def test_validation_threshold_zero_budget_is_above_maximum_negative():
    y = np.array([0, 0, 0, 1], dtype=np.int8)
    scores = np.array([0.2, 0.4, 0.8, 0.9])
    threshold = _threshold_at_fpr(y, scores, max_fpr=0.01)
    assert threshold > np.max(scores[y == 0])
    assert not np.any(scores[y == 0] >= threshold)


def test_validation_threshold_scans_tied_score_groups_efficiently():
    y = np.array([0] * 100 + [1], dtype=np.int8)
    scores = np.array([0.1] * 95 + [0.5] * 4 + [0.9] + [1.0])
    threshold = _threshold_at_fpr(y, scores, max_fpr=0.01)
    assert threshold == 0.9
    assert np.mean(scores[y == 0] >= threshold) <= 0.01
