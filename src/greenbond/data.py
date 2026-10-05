"""Validated local inputs and backward-only alignment on the index calendar."""
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd

TARGET = "green_bond_index"


def normalize(frame: pd.DataFrame, *, target: bool = False) -> pd.DataFrame:
    """Remove exact duplicates; reject ambiguous dates and nonnumeric observations."""
    frame = frame.copy().drop_duplicates()
    if "date" not in frame:
        raise ValueError("Input must contain a date column.")
    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    if frame["date"].isna().any():
        raise ValueError("Dates must not be missing.")
    if frame["date"].duplicated().any():
        raise ValueError("Conflicting observations for the same date; resolve upstream.")
    for column in frame.columns.drop("date"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    frame = frame.replace([np.inf, -np.inf], np.nan).sort_values("date")
    if target:
        if TARGET not in frame:
            raise ValueError(f"Input must contain {TARGET}.")
        frame = frame.dropna(subset=[TARGET])
        if (frame[TARGET] <= 0).any():
            raise ValueError("Index levels must be positive.")
    return frame.set_index("date")


def read_csv(path: Path) -> pd.DataFrame:
    return normalize(pd.read_csv(path), target=True)


def align_predictors(target: pd.DataFrame, daily: pd.DataFrame,
                     monthly: pd.DataFrame | None = None,
                     macro_lag_days: int = 45) -> pd.DataFrame:
    """Carry the last available value forward; never interpolate from the future.

    Monthly dates are observation dates, shifted by an explicit calendar-day lag.
    A lag is an assumption, not a substitute for release timestamps or vintages.
    """
    if macro_lag_days < 0:
        raise ValueError("Macro availability lag must be nonnegative.")
    result = target.copy()
    sources = [daily]
    if monthly is not None:
        monthly = monthly.copy()
        monthly.index = monthly.index + pd.Timedelta(days=macro_lag_days)
        sources.append(monthly)
    for source in sources:
        overlap = set(result.columns).intersection(source.columns)
        if overlap:
            raise ValueError(f"Overlapping predictor columns: {sorted(overlap)}")
        calendar = source.index.union(target.index).sort_values()
        available = source.reindex(calendar).ffill().reindex(target.index)
        result = result.join(available)
    return result


def read_legacy(directory: Path, macro_lag_days: int = 45) -> pd.DataFrame:
    raw = pd.read_excel(directory / "SPgreenbondindex.xls", skiprows=6)
    raw = raw.iloc[:, :2]
    raw.columns = ["date", TARGET]
    # The vendor export contains a textual footer after the observations.
    dates = pd.to_datetime(raw["date"], errors="coerce")
    values = pd.to_numeric(raw[TARGET], errors="coerce")
    raw = raw.loc[dates.notna() & values.notna()].copy()
    target = normalize(raw, target=True)
    daily = normalize(pd.read_excel(directory / "predictors.xlsx").rename(columns={"Date": "date"}))
    monthly = normalize(pd.read_excel(directory / "predictors_mon.xlsx").rename(columns={"Date": "date"}))
    return align_predictors(target, daily, monthly, macro_lag_days)


def quality_audit(frame: pd.DataFrame) -> dict:
    """Use SQLite for transparent row, duplicate, and missing-value checks."""
    with sqlite3.connect(":memory:") as connection:
        frame.reset_index().to_sql("observations", connection, index=False)
        missing = {}
        for column in frame.columns:
            quoted = '"' + column.replace('"', '""') + '"'
            missing[column] = connection.execute(
                f"SELECT COUNT(*) FROM observations WHERE {quoted} IS NULL"
            ).fetchone()[0]
        duplicates = connection.execute(
            "SELECT COUNT(*) FROM (SELECT date FROM observations GROUP BY date HAVING COUNT(*) > 1)"
        ).fetchone()[0]
    return {"rows": len(frame), "start": str(frame.index.min().date()),
            "end": str(frame.index.max().date()), "duplicate_dates": duplicates,
            "missing_by_column": missing}


def synthetic_data(rows: int = 900, seed: int = 42) -> pd.DataFrame:
    """Illustrative artificial index and predictors, unrelated to vendor observations."""
    if rows < 150:
        raise ValueError("Demo requires at least 150 observations.")
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-01", periods=rows, name="date")
    signal = np.sin(np.arange(rows) / 24) + rng.normal(0, 0.2, rows)
    changes = np.zeros(rows)
    for t in range(1, rows):
        changes[t] = 0.10 * signal[t - 1] + 0.2 * changes[t - 1] + rng.normal(0, 0.20)
    frame = pd.DataFrame({TARGET: 130 + np.cumsum(changes),
                          "market_signal": signal,
                          "treasury_yield": 2 + np.cumsum(rng.normal(0, .015, rows)),
                          "volatility_index": 18 + 3 * np.cos(np.arange(rows) / 45)
                          + rng.normal(0, 1, rows)}, index=dates)
    frame.loc[dates[::41], "treasury_yield"] = np.nan
    return frame
