"""Date window helpers for risk period filtering."""
from __future__ import annotations

import pandas as pd


def frost_window_mask(times: pd.DatetimeIndex, year: int,
                      start: tuple[int, int] = (3, 15),
                      end: tuple[int, int] = (4, 30)) -> pd.Series:
    """Return boolean mask for frost risk window within the given year."""
    t = pd.Series(times)
    window_start = pd.Timestamp(year=year, month=start[0], day=start[1])
    window_end = pd.Timestamp(year=year, month=end[0], day=end[1], hour=23)
    return (t >= window_start) & (t <= window_end)


def is_march(dt: pd.Timestamp) -> bool:
    return dt.month == 3


def is_april(dt: pd.Timestamp) -> bool:
    return dt.month == 4
