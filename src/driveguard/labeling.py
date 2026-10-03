"""Censor-aware 30-day drive failure labels."""

from __future__ import annotations

import pandas as pd


def add_failure_horizon_labels(
    frame: pd.DataFrame,
    horizon_days: int = 30,
    observed_through: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Label eligible drive-days from failure events and observed follow-up.

    Required columns: date, serial_number, failure. A positive must fail on a
    later day within the horizon. A negative requires a full horizon of
    follow-up or a known failure after the horizon. Other rows are censored.
    """
    required = {"date", "serial_number", "failure"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if horizon_days < 1:
        raise ValueError("horizon_days must be positive")

    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    out["failure"] = pd.to_numeric(out["failure"], errors="coerce").fillna(0).astype("int8")
    if out["date"].isna().any() or out["serial_number"].isna().any():
        raise ValueError("date and serial_number must be present and parseable")
    if not out["failure"].isin([0, 1]).all():
        raise ValueError("failure must contain only 0/1 values")

    failure_dates = (
        out.loc[out["failure"].eq(1)]
        .groupby("serial_number")["date"]
        .agg(list)
        .to_dict()
    )
    if any(len(dates) > 1 for dates in failure_dates.values()):
        raise ValueError("A serial_number has multiple observed failure events; investigate identity/history")

    end = pd.Timestamp(observed_through).normalize() if observed_through is not None else out["date"].max()
    horizon = pd.Timedelta(days=horizon_days)
    fail_date = out["serial_number"].map(
        {serial: dates[0] for serial, dates in failure_dates.items()}
    )
    delta = fail_date - out["date"]
    after_today = delta.gt(pd.Timedelta(0))
    positive = after_today & delta.le(horizon)
    negative = (after_today & delta.gt(horizon)) | (fail_date.isna() & ((end - out["date"]) >= horizon))
    eligible = positive | negative

    out["failure_date"] = fail_date
    out["label_eligible"] = eligible.astype("int8")
    out["will_fail_30d"] = positive.where(eligible, pd.NA).astype("Int8")
    out["censor_reason"] = "eligible"
    out.loc[~eligible & out["date"].ge(end - horizon), "censor_reason"] = "insufficient_followup"
    out.loc[~eligible & out["failure"].eq(1), "censor_reason"] = "failure_day"
    return out
