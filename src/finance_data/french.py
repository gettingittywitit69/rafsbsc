"""
Extended Fama-French data loaders that return tidy long-format DataFrames.

Each loader targets a specific family of strategies and returns a DataFrame with
columns:
    - date (Timestamp, month-end)
    - group (str)
    - strategy_id (str)
    - return_excess (float, decimal monthly excess return)

Implementation note:
- No pandas_datareader. Data are downloaded directly from Ken French's FTP as ZIP+CSV.
- Portfolio returns are converted to excess returns using the U.S. RF series from
  the corresponding Fama-French factors file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple, List, Dict
import io
import zipfile
import urllib.request

import numpy as np
import pandas as pd

__all__ = [
    "load_all_strategies_long",
    "load_us_ff5_factors",
    "load_us_industries_30",
    "load_us_industries_49",
    "load_us_momentum_factor",
    "load_us_research_factors_wide",
    "load_us_size_bm_25",
    "load_us_size_deciles",
    "pivot_family",
]

DEFAULT_START = "1926-07-01"
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CSV_CACHE_DIR = _PROJECT_ROOT / "data" / "famafrench_cache"

_FTP_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"

# Map the old pandas_datareader dataset keys to Ken French ZIP filenames.
_DATASET_TO_ZIP: Dict[str, str] = {
    "F-F_Research_Data_Factors": "F-F_Research_Data_Factors_CSV.zip",
    "F-F_Research_Data_5_Factors_2x3": "F-F_Research_Data_5_Factors_2x3_CSV.zip",
    "F-F_Momentum_Factor": "F-F_Momentum_Factor_CSV.zip",
    "Portfolios_Formed_on_ME": "Portfolios_Formed_on_ME_CSV.zip",
    "25_Portfolios_5x5": "25_Portfolios_5x5_CSV.zip",
    "30_Industry_Portfolios": "30_Industry_Portfolios_CSV.zip",
    "49_Industry_Portfolios": "49_Industry_Portfolios_CSV.zip",
}

_SENTINELS = {-99.99, -999.0, -999.99, -9999.0}


@dataclass(frozen=True)
class _KenFrenchZipClient:
    cache_dir: Path = Path.home() / ".cache" / "kenfrench"

    def __post_init__(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch_zip_bytes(self, zip_name: str, *, force: bool = False) -> bytes:
        zip_path = self.cache_dir / zip_name
        if zip_path.exists() and not force:
            return zip_path.read_bytes()

        url = _FTP_BASE + zip_name
        with urllib.request.urlopen(url) as resp:  # nosec (user-controlled URL not used)
            data = resp.read()

        zip_path.write_bytes(data)
        return data


_KF = _KenFrenchZipClient()


def _resolve_zip_name(dataset: str) -> str:
    if dataset.lower().endswith(".zip"):
        return dataset
    if dataset in _DATASET_TO_ZIP:
        return _DATASET_TO_ZIP[dataset]
    # Allow callers to pass the basename without suffix
    maybe = dataset + "_CSV.zip"
    return maybe


def _extract_first_csv_text(zip_bytes: bytes) -> str:
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        members = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not members:
            raise ValueError("ZIP contained no CSV file.")
        # Typically exactly one; take the first deterministically.
        name = sorted(members)[0]
        raw = zf.read(name)
    return raw.decode("latin1")


def _split_csv_line(line: str) -> List[str]:
    # Ken French CSVs are simple: no embedded commas inside quoted fields.
    return [c.strip() for c in line.split(",")]


def _is_monthly_or_daily_date_token(tok: str) -> bool:
    return tok.isdigit() and (len(tok) == 6 or len(tok) == 8)


def _parse_date_index(date_tokens: pd.Index) -> pd.DatetimeIndex:
    # Convert YYYYMM -> month-end timestamps; YYYYMMDD -> timestamps.
    s = pd.Index(date_tokens.astype(str))
    if len(s) == 0:
        return pd.DatetimeIndex([])
    L = len(s[0])
    if L == 6:
        dt = pd.to_datetime(s, format="%Y%m", errors="coerce")
        return (dt + pd.offsets.MonthEnd(0)).to_numpy()
    if L == 8:
        return pd.to_datetime(s, format="%Y%m%d", errors="coerce").to_numpy()
    raise ValueError("Unsupported date token length.")


def _parse_all_tables(csv_text: str) -> List[pd.DataFrame]:
    """
    Parse all "numeric-date tables" in a Ken French CSV.

    We detect a table when:
      - A header line is followed by a data line whose first field is YYYYMM or YYYYMMDD.
    We then read until the first field stops being YYYYMM(/DD).

    Returns a list of DataFrames with a month-end DatetimeIndex (for monthly tables).
    """
    lines = csv_text.splitlines()
    tables: List[pd.DataFrame] = []
    i = 0
    n = len(lines)

    while i < n - 1:
        hdr = _split_csv_line(lines[i])
        nxt = _split_csv_line(lines[i + 1])

        if not nxt or not _is_monthly_or_daily_date_token(nxt[0]):
            i += 1
            continue

        # Decide whether header includes DATE column.
        # If next row has one more field than header, header is missing DATE.
        if len(nxt) == len(hdr) + 1:
            columns = ["DATE"] + hdr
        else:
            columns = hdr[:]
        if not columns:
            i += 1
            continue
        columns[0] = "DATE"

        # Read data rows
        j = i + 1
        rows: List[List[str]] = []
        while j < n:
            row = _split_csv_line(lines[j])
            if not row or not _is_monthly_or_daily_date_token(row[0]):
                break

            # Normalize row length to columns length
            if len(row) < len(columns):
                row = row + [""] * (len(columns) - len(row))
            elif len(row) > len(columns):
                row = row[: len(columns)]
            rows.append(row)
            j += 1

        if rows:
            df = pd.DataFrame(rows, columns=columns)
            df["DATE"] = _parse_date_index(df["DATE"])
            df = df.set_index("DATE")
            df.index.name = "date"

            # Clean column names + numeric conversion
            df.columns = pd.Index([str(c).strip() for c in df.columns])
            for c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

            # Sentinels -> NaN
            df = df.replace(list(_SENTINELS), np.nan)

            # Drop rows that are entirely missing
            df = df.dropna(how="all")

            # Only keep monthly tables as monthly (date is month-end). Daily stays daily.
            tables.append(df)

        i = j + 1

    return tables


def _load_cached_table_csv(
    dataset: str,
    table: int,
    start_date: Optional[str],
    end_date: Optional[str],
) -> Optional[pd.DataFrame]:
    safe_ds = dataset.replace("/", "_")
    start_tag = pd.to_datetime(start_date).strftime("%Y%m") if start_date else "start"
    end_tag = pd.to_datetime(end_date).strftime("%Y%m") if end_date else "latest"
    exact = _CSV_CACHE_DIR / f"{safe_ds}_tbl{table}_{start_tag}_{end_tag}.csv"
    candidates = [exact] if exact.exists() else sorted(_CSV_CACHE_DIR.glob(f"{safe_ds}_tbl{table}_*.csv"))
    if not candidates:
        return None

    df = pd.read_csv(candidates[0], index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    if start_date is not None:
        df = df[df.index >= pd.to_datetime(start_date)]
    if end_date is not None:
        df = df[df.index <= pd.to_datetime(end_date)]
    return df


def _fetch_ff_table(
    dataset: str,
    table: int = 0,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    *,
    force_download: bool = False,
) -> pd.DataFrame:
    """
    Download a single Ken French dataset and return the selected table as a DataFrame.

    - dataset: old pandas_datareader key (e.g. "25_Portfolios_5x5") or a ZIP filename.
    - table: zero-based table index among detected monthly/daily tables.
    """
    cached_df = _load_cached_table_csv(dataset, table, start_date, end_date)
    if cached_df is not None:
        return cached_df

    zip_name = _resolve_zip_name(dataset)
    zip_bytes = _KF.fetch_zip_bytes(zip_name, force=force_download)
    text = _extract_first_csv_text(zip_bytes)
    tables = _parse_all_tables(text)
    if table < 0 or table >= len(tables):
        raise IndexError(f"Requested table={table}, but only {len(tables)} table(s) found in {zip_name}.")
    df = tables[table].copy()

    if start_date is not None:
        df = df[df.index >= pd.to_datetime(start_date)]
    if end_date is not None:
        df = df[df.index <= pd.to_datetime(end_date)]

    return df


def _to_long(df: pd.DataFrame, group: str, rename_map: Optional[dict] = None) -> pd.DataFrame:
    """Convert a wide DataFrame to long with standardized column names."""
    rename_map = rename_map or {}
    tidy_df = df.copy()
    tidy_df.index.name = "date"
    tidy_df.columns = pd.Index([str(c).strip() for c in tidy_df.columns])
    tidy = tidy_df.rename(columns=rename_map).reset_index()
    tidy = tidy.rename(columns={"Date": "date", "index": "date"})
    tidy = tidy.melt(id_vars="date", var_name="strategy_id", value_name="return_excess")
    tidy["group"] = group
    tidy = tidy.dropna(subset=["return_excess"]).sort_values(["date", "group", "strategy_id"])
    tidy = tidy[["date", "group", "strategy_id", "return_excess"]].reset_index(drop=True)
    return tidy


def load_us_ff5_factors(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load U.S. Fama-French 5 factors (monthly) and return tidy factors + RF.

    Factors are already excess returns; values are converted to decimals.
    """
    df = _fetch_ff_table(
        "F-F_Research_Data_5_Factors_2x3",
        table=0,
        start_date=start_date,
        end_date=end_date,
    )

    if "RF" not in df.columns:
        raise KeyError("Expected column 'RF' not found in FF5 factors table.")

    rf = (df["RF"] / 100.0).rename("RF")
    factors = df.drop(columns="RF") / 100.0

    rename_map = {c: str(c).replace("-", "_").replace(" ", "_") for c in factors.columns}
    long = _to_long(factors, group="US_factors_5", rename_map=rename_map)
    return long, rf


def load_us_research_factors_wide(
    start_date: Optional[str] = DEFAULT_START,
    end_date: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load U.S. research factors (Mkt-RF, SMB, HML) plus RF in wide monthly format.

    Values are returned in decimal form with a month-end DatetimeIndex.
    """
    df = _fetch_ff_table(
        "F-F_Research_Data_Factors",
        table=0,
        start_date=start_date,
        end_date=end_date,
    )
    df = df / 100.0
    if "RF" not in df.columns:
        raise KeyError("Expected column 'RF' not found in FF3 factors table.")
    rf = df["RF"].rename("RF")
    factors = df.drop(columns="RF")
    return factors, rf


def load_us_momentum_factor(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load the U.S. momentum factor (UMD) and return a tidy excess series.

    Series is already an excess return; values are converted to decimals.
    """
    df = _fetch_ff_table(
        "F-F_Momentum_Factor",
        table=0,
        start_date=start_date,
        end_date=end_date,
    )
    df = df / 100.0
    if df.shape[1] != 1:
        # Some files may carry multiple columns; pick the first as "UMD".
        df = df.iloc[:, [0]]
    rename_map = {df.columns[0]: "UMD"}
    return _to_long(df, group="US_momentum_factor", rename_map=rename_map)


def load_us_size_deciles(
    rf: pd.Series,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load value-weighted 10 size portfolios (monthly) and convert to excess returns.
    """
    df = _fetch_ff_table(
        "Portfolios_Formed_on_ME",
        table=0,  # first monthly table is typically value-weighted returns
        start_date=start_date,
        end_date=end_date,
    )

    dec_cols = ["Lo 10", "2-Dec", "3-Dec", "4-Dec", "5-Dec", "6-Dec", "7-Dec", "8-Dec", "9-Dec", "Hi 10"]
    missing = [c for c in dec_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing expected size-decile column(s): {missing}")

    dec_df = df[dec_cols] / 100.0
    rf_aligned = rf.reindex(dec_df.index)
    excess = dec_df.sub(rf_aligned, axis=0)

    rename_map = {col: f"SIZE_{i+1}" for i, col in enumerate(dec_cols)}
    return _to_long(excess, group="US_size_10", rename_map=rename_map)


def load_us_size_bm_25(
    rf: pd.Series,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load value-weighted 25 portfolios formed on Size and Book-to-Market and convert to excess.
    """
    df = _fetch_ff_table(
        "25_Portfolios_5x5",
        table=0,  # first monthly table is typically value-weighted returns
        start_date=start_date,
        end_date=end_date,
    )
    ret_df = df / 100.0
    rf_aligned = rf.reindex(ret_df.index)
    excess = ret_df.sub(rf_aligned, axis=0)
    rename_map = {col: str(col).replace(" ", "_").replace("-", "_") for col in excess.columns}
    return _to_long(excess, group="US_size_BM_25", rename_map=rename_map)


def load_us_industries_30(
    rf: pd.Series,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Load 30 industry portfolios (value-weighted) and convert to excess returns."""
    df = _fetch_ff_table(
        "30_Industry_Portfolios",
        table=0,  # first monthly table is typically value-weighted returns
        start_date=start_date,
        end_date=end_date,
    )
    df = df / 100.0
    rf_aligned = rf.reindex(df.index)
    excess = df.sub(rf_aligned, axis=0)
    return _to_long(excess, group="US_industries_30")


def load_us_industries_49(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load 49 industry portfolios (value-weighted, monthly) and convert to excess returns
    using the U.S. RF series from F-F Research Factors.
    """
    _, rf = load_us_research_factors_wide(
        start_date=start_date or DEFAULT_START,
        end_date=end_date,
    )
    df = _fetch_ff_table(
        "49_Industry_Portfolios",
        table=0,  # first monthly table is typically value-weighted returns
        start_date=start_date,
        end_date=end_date,
    )
    df = df / 100.0
    rf_aligned = rf.reindex(df.index)
    excess = df.sub(rf_aligned, axis=0)
    return _to_long(excess, group="US_industries_49")


def load_all_strategies_long(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load and combine all supported strategy families into a single long DataFrame.

    Families included by default:
      - US 49 Industry portfolios
      - US 30 Industry portfolios
      - US FF5 factors + UMD momentum factor
      - US size deciles (10)
      - US size-BM 25 portfolios
    """
    us_ff5_long, rf_us = load_us_ff5_factors(start_date=start_date, end_date=end_date)
    us_mom_long = load_us_momentum_factor(start_date=start_date, end_date=end_date)
    us_size10_long = load_us_size_deciles(rf=rf_us, start_date=start_date, end_date=end_date)
    us_sizebm_long = load_us_size_bm_25(rf=rf_us, start_date=start_date, end_date=end_date)
    us_ind30_long = load_us_industries_30(rf=rf_us, start_date=start_date, end_date=end_date)
    us_ind49_long = load_us_industries_49(start_date=start_date, end_date=end_date)

    frames = [us_ind49_long, us_ind30_long, us_ff5_long, us_mom_long, us_size10_long, us_sizebm_long]
    frames = [f for f in frames if f is not None and not f.empty]
    combined = pd.concat(frames, ignore_index=True, sort=False)

    if start_date is not None:
        combined = combined[combined["date"] >= pd.to_datetime(start_date)]
    if end_date is not None:
        combined = combined[combined["date"] <= pd.to_datetime(end_date)]

    combined = combined.sort_values(["date", "group", "strategy_id"]).reset_index(drop=True)
    return combined


def pivot_family(long_df: pd.DataFrame, groups: Iterable[str]) -> pd.DataFrame:
    """
    Convenience helper to pivot one or more groups into a wide date x strategy table.
    """
    fam = long_df[long_df["group"].isin(groups)].copy()
    wide = fam.pivot(index="date", columns="strategy_id", values="return_excess")
    wide = wide.sort_index()
    return wide.dropna(how="all")
