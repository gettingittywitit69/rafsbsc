"""Ken French data fetchers and a helper to cache them locally."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from pandas_datareader import data as web
import numpy as np


DEFAULT_START = "1926-07-01"
_DEFAULT_START_TS = pd.Timestamp(DEFAULT_START)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_DIR = _PROJECT_ROOT / "data" / "famafrench_cache"
__all__ = ["KenFrenchLoader", "fetch_french25_excess", "fetch_french49_excess", "ensure_french_datasets"]


def _normalize_index(idx) -> pd.DatetimeIndex:
    """Convert Ken French indices to month-end timestamps."""
    if isinstance(idx, pd.PeriodIndex):
        return idx.to_timestamp("M")
    if isinstance(idx, pd.DatetimeIndex):
        return idx
    try:
        return pd.to_datetime(idx, format="%Y%m")
    except (TypeError, ValueError):
        return pd.to_datetime(idx)


class KenFrenchLoader:
    """
    Single point of contact for Ken French tables with memoized disk cache.

    The loader caches CSVs under ``cache_dir`` and reuses them across calls,
    so repeated requests for the same dataset/table/date range avoid extra
    downloads during a notebook session.
    """

    def __init__(self, cache_dir: Path | str = DEFAULT_CACHE_DIR):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._memo = {}

    def _cache_path(self, dataset: str, table: int, start_date: Optional[str], end_date: Optional[str]) -> Path:
        start_tag = pd.to_datetime(start_date).strftime("%Y%m") if start_date else "start"
        end_tag = pd.to_datetime(end_date).strftime("%Y%m") if end_date else "latest"
        safe_ds = dataset.replace("/", "_")
        return self.cache_dir / f"{safe_ds}_tbl{table}_{start_tag}_{end_tag}.csv"

    def _candidate_cache_paths(
        self,
        dataset: str,
        table: int,
        start_date: Optional[str],
        end_date: Optional[str],
    ) -> list[Path]:
        exact_path = self._cache_path(dataset, table, start_date, end_date)
        if exact_path.exists():
            return [exact_path]
        safe_ds = dataset.replace("/", "_")
        return sorted(self.cache_dir.glob(f"{safe_ds}_tbl{table}_*.csv"))

    def load_table(
        self,
        dataset: str,
        table: int = 0,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        key = (dataset, table, start_date, end_date)
        if key in self._memo:
            return self._memo[key].copy()

        cache_path = self._cache_path(dataset, table, start_date, end_date)
        cached_paths = self._candidate_cache_paths(dataset, table, start_date, end_date)
        fetched_from_cache = bool(cached_paths)
        if fetched_from_cache:
            df = pd.read_csv(cached_paths[0], index_col=0, parse_dates=True)
        else:
            pdr_start = pd.to_datetime(start_date) if start_date is not None else None
            pdr_end = pd.to_datetime(end_date) if end_date is not None else None
            fetched = web.DataReader(dataset, "famafrench", start=pdr_start, end=pdr_end)
            df = fetched[table].copy()

        df.index = _normalize_index(df.index)
        df = df.replace([-99.99, -99.9, -999], np.nan)

        if not fetched_from_cache:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(cache_path)

        if start_date is not None:
            df = df.loc[pd.to_datetime(start_date) :]
        if end_date is not None:
            df = df.loc[: pd.to_datetime(end_date)]

        self._memo[key] = df
        return df.copy()


_DEFAULT_LOADER = KenFrenchLoader()


def fetch_french25_excess(
    start: Optional[str] = None,
    end: Optional[str] = None,
    value_weighted: bool = True,
    loader: Optional[KenFrenchLoader] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load 25 Portfolios (monthly) and risk-free rate from Ken French's data library.

    Returns a tuple of (excess_returns_df, rf_series) with returns in decimal form.
    """
    loader = loader or _DEFAULT_LOADER
    pdr_start = pd.to_datetime(start) if start is not None else _DEFAULT_START_TS
    pdr_end = pd.to_datetime(end) if end is not None else None

    ff = loader.load_table("F-F_Research_Data_Factors", table=0, start_date=pdr_start, end_date=pdr_end)
    rf = ff["RF"] / 100.0

    ret25 = loader.load_table("25_Portfolios_5x5", table=0 if value_weighted else 1, start_date=pdr_start, end_date=pdr_end)
    ret25 = ret25 / 100.0
    ret25.columns = ret25.columns.str.strip()
    ret25 = ret25.mask(ret25 <= -0.99)

    df = ret25.join(rf, how="inner")
    df.index = _normalize_index(df.index)

    if start is not None:
        df = df.loc[pd.to_datetime(start) :]
    if end is not None:
        df = df.loc[: pd.to_datetime(end)]

    rf_series = df["RF"]
    excess = df.drop(columns="RF").sub(rf_series, axis=0)
    return excess, rf_series


def fetch_french49_excess(
    start: Optional[str] = None,
    end: Optional[str] = None,
    value_weighted: bool = True,
    loader: Optional[KenFrenchLoader] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load 49 Industry Portfolios (monthly) and risk-free rate from Ken French's data library.

    Returns a tuple of (excess_returns_df, rf_series) with returns in decimal form.
    """
    loader = loader or _DEFAULT_LOADER
    pdr_start = pd.to_datetime(start) if start is not None else _DEFAULT_START_TS
    pdr_end = pd.to_datetime(end) if end is not None else None

    ff = loader.load_table("F-F_Research_Data_Factors", table=0, start_date=pdr_start, end_date=pdr_end)
    rf = ff["RF"] / 100.0

    ret49 = loader.load_table("49_Industry_Portfolios", table=0 if value_weighted else 1, start_date=pdr_start, end_date=pdr_end)
    ret49 = ret49 / 100.0
    ret49.columns = ret49.columns.str.strip()
    ret49 = ret49.mask(ret49 <= -0.99)

    df = ret49.join(rf, how="inner")
    df.index = _normalize_index(df.index)

    if start is not None:
        df = df.loc[pd.to_datetime(start) :]
    if end is not None:
        df = df.loc[: pd.to_datetime(end)]

    rf_series = df["RF"]
    excess = df.drop(columns="RF").sub(rf_series, axis=0)
    return excess, rf_series


def ensure_french_datasets(
    output_dir: Path | str = "data",
    start: Optional[str] = DEFAULT_START,
    end: Optional[str] = None,
    value_weighted: bool = True,
    refresh: bool = False,
):
    """
    Ensure CSVs for French 25/49 excess returns and risk-free rate exist locally.

    If the files already exist and `refresh` is False, they are loaded from disk.
    Otherwise, data is fetched from the Ken French data library and written to disk.

    Returns
    -------
    dict
        {
            "excess_25": DataFrame,
            "excess_49": DataFrame,
            "risk_free": Series,
            "paths": {
                "excess_25": Path,
                "excess_49": Path,
                "risk_free": Path,
            }
        }
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    p25_path = output_dir / "french25_excess_monthly.csv"
    p49_path = output_dir / "french49_excess_monthly.csv"
    rf_path = output_dir / "risk_free_monthly.csv"

    need_fetch = refresh or not (p25_path.exists() and p49_path.exists() and rf_path.exists())

    if need_fetch:
        ex25, rf = fetch_french25_excess(start=start, end=end, value_weighted=value_weighted)
        ex49, _ = fetch_french49_excess(start=start, end=end, value_weighted=value_weighted)

        ex25.to_csv(p25_path, float_format="%.10f")
        rf.to_frame(name="RF").to_csv(rf_path, float_format="%.10f")
        ex49.to_csv(p49_path, float_format="%.10f")
    else:
        ex25 = pd.read_csv(p25_path, index_col=0, parse_dates=True)
        rf_df = pd.read_csv(rf_path, index_col=0, parse_dates=True)
        rf = rf_df.iloc[:, 0] if rf_df.shape[1] else pd.Series(dtype=float)
        ex49 = pd.read_csv(p49_path, index_col=0, parse_dates=True)

    return {
        "excess_25": ex25,
        "excess_49": ex49,
        "risk_free": rf,
        "paths": {"excess_25": p25_path, "excess_49": p49_path, "risk_free": rf_path},
    }


if __name__ == "__main__":
    result = ensure_french_datasets()
    ex25, ex49 = result["excess_25"], result["excess_49"]
    print("25-portfolios excess shape:", ex25.shape)
    print("49-industry excess shape:", ex49.shape)
    rf = result["risk_free"]
    print("RF shape:", rf.shape)
    print("Start:", ex49.index.min().date(), "End:", ex49.index.max().date())
