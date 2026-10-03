"""Small, explicit temporal split helper with a label-horizon embargo."""

from __future__ import annotations

import pandas as pd


def temporal_masks(
    dates: pd.Series,
    train_end: str,
    validation_end: str,
    test_end: str,
    horizon_days: int = 30,
    validation_start: str | None = None,
    test_start: str | None = None,
) -> dict[str, pd.Series]:
    """Return chronological masks with label-horizon embargoes and optional extra gaps."""
    d = pd.to_datetime(dates).dt.normalize()
    tr = pd.Timestamp(train_end).normalize()
    va = pd.Timestamp(validation_end).normalize()
    te = pd.Timestamp(test_end).normalize()
    vstart = pd.Timestamp(validation_start).normalize() if validation_start else tr
    tstart = pd.Timestamp(test_start).normalize() if test_start else va
    embargo = pd.Timedelta(days=horizon_days)
    if not tr <= vstart < va <= tstart < te:
        raise ValueError("Require train_end <= validation_start < validation_end <= test_start < test_end")
    return {
        "train": d.lt(tr - embargo),
        "validation": d.ge(vstart) & d.lt(va - embargo),
        "test": d.ge(tstart) & d.le(te - embargo),
    }
