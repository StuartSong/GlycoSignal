"""
glycosignal.io
==============

Loading and parsing CGM data from files and folders.

All loaders return DataFrames with canonical column names:
  - ``Timestamp`` -- datetime-parseable
  - ``Glucose`` -- numeric, mg/dL

Optional metadata columns may also be present: ``subject``, ``filename``,
``sensor``, ``date``.

Usage
-----
    >>> from glycosignal import io
    >>> df = io.load_csv("my_data.csv")
    >>> df = io.load_cgm_folder("data/subjects/")
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .preprocessing import standardize_columns
from .schemas import COL_GLUCOSE, COL_TIMESTAMP
from .utils import require_dataframe


# ─────────────────────────────────────────────────────────────────────────────
# Auto-detection helpers
# ─────────────────────────────────────────────────────────────────────────────

_TIMESTAMP_CANDIDATES = ["timestamp", "time", "datetime", "date_time"]
_GLUCOSE_CANDIDATES = ["glucose value (mg/dl)", "glucose", "gl", "sgv", "glucose_mg_dl", "bg"]
_SUBJECT_CANDIDATES = ["id", "ptid", "subject", "patient_id", "subjectid"]


def _detect_col(df: pd.DataFrame, candidates: list[str], label: str) -> str | None:
    """Return the first column whose lowercase name matches any candidate."""
    lower_map = {c.lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Unified CSV loader
# ─────────────────────────────────────────────────────────────────────────────

def load_csv(
    filepath: str | Path,
    timestamp_col: str | None = None,
    glucose_col: str | None = None,
    subject_col: str | None = None,
    column_map: dict[str, str] | None = None,
    **read_csv_kwargs,
) -> pd.DataFrame:
    """Load a single CGM CSV file and return a standardized DataFrame.

    Column names are auto-detected when not explicitly specified.  The
    returned DataFrame always contains ``Timestamp`` and ``Glucose`` columns.

    Parameters
    ----------
    filepath : str or Path
        Path to the CSV file.
    timestamp_col : str | None
        Column name for timestamps.  Auto-detected if None.
    glucose_col : str | None
        Column name for glucose values.  Auto-detected if None.
    subject_col : str | None
        Column name for subject/patient ID.  Auto-detected if None.  When
        found, the column is included in the output as ``"subject"``.
    column_map : dict[str, str] | None
        Additional column name mappings (merged with auto-detection).
    **read_csv_kwargs
        Passed directly to ``pd.read_csv``.

    Returns
    -------
    pd.DataFrame
        DataFrame with at least ``Timestamp`` and ``Glucose`` columns.

    Raises
    ------
    FileNotFoundError
        If *filepath* does not exist.
    ValueError
        If required columns cannot be found (auto-detection and explicit name
        both fail).
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    df = pd.read_csv(filepath, **read_csv_kwargs)

    # Resolve column names
    ts_col = timestamp_col or _detect_col(df, _TIMESTAMP_CANDIDATES, "Timestamp")
    gl_col = glucose_col or _detect_col(df, _GLUCOSE_CANDIDATES, "Glucose")
    su_col = subject_col or _detect_col(df, _SUBJECT_CANDIDATES, "subject")

    if ts_col is None:
        raise ValueError(
            f"Cannot find a Timestamp column in {filepath.name}. "
            f"Available columns: {list(df.columns)}. "
            f"Pass timestamp_col=... explicitly."
        )
    if gl_col is None:
        raise ValueError(
            f"Cannot find a Glucose column in {filepath.name}. "
            f"Available columns: {list(df.columns)}. "
            f"Pass glucose_col=... explicitly."
        )

    # Build rename map
    rename: dict[str, str] = {}
    if ts_col != COL_TIMESTAMP:
        rename[ts_col] = COL_TIMESTAMP
    if gl_col != COL_GLUCOSE:
        rename[gl_col] = COL_GLUCOSE
    if su_col and su_col != "subject":
        rename[su_col] = "subject"
    if column_map:
        rename.update(column_map)

    out = df.rename(columns=rename)

    # Apply any remaining standardization
    out = standardize_columns(out)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Folder loader (per-subject CSV files)
# ─────────────────────────────────────────────────────────────────────────────

def load_cgm_folder(
    folder_path: str | Path,
    timestamp_col: str | None = None,
    glucose_col: str | None = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Load CGM data from a folder of per-subject CSV files.

    Each CSV is expected to represent one subject or recording session.
    The loader auto-detects timestamp and glucose columns.

    Adds two metadata columns to the output:
    - ``filename`` -- stem of the source CSV (no extension).
    - ``subject`` -- derived from the filename (stem without trailing date suffix).

    Parameters
    ----------
    folder_path : str or Path
        Directory containing ``.csv`` files.
    timestamp_col : str | None
        Override the auto-detected timestamp column name.
    glucose_col : str | None
        Override the auto-detected glucose column name.
    show_progress : bool
        Show a tqdm progress bar while loading.

    Returns
    -------
    pd.DataFrame
        Concatenated DataFrame with ``Timestamp``, ``Glucose``, ``filename``,
        and ``subject`` columns.

    Raises
    ------
    FileNotFoundError
        If *folder_path* does not exist or contains no CSV files.
    """
    folder_path = Path(folder_path)
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    csv_files = sorted(f for f in folder_path.iterdir() if f.suffix.lower() == ".csv")
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {folder_path}")

    iterator = tqdm(csv_files, desc="Loading CSVs") if show_progress else csv_files
    dfs: list[pd.DataFrame] = []

    for filepath in iterator:
        df = load_csv(
            filepath,
            timestamp_col=timestamp_col,
            glucose_col=glucose_col,
        )

        # Keep only Timestamp, Glucose, and any existing subject column
        keep_cols = [COL_TIMESTAMP, COL_GLUCOSE]
        if "subject" in df.columns:
            keep_cols.append("subject")
        df = df[keep_cols].copy()

        # Derive filename and subject from path
        stem = filepath.stem
        parts = stem.rsplit("_", 1)
        subject = parts[0] if (len(parts) == 2 and len(parts[1]) >= 8) else stem

        df["filename"] = stem
        if "subject" not in df.columns:
            df["subject"] = subject

        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# Single multi-subject file loader
# ─────────────────────────────────────────────────────────────────────────────

def load_cgm_file(
    filepath: str | Path,
    timestamp_col: str | None = None,
    glucose_col: str | None = None,
    subject_col: str | None = None,
) -> pd.DataFrame:
    """Load CGM data from a single file that contains multiple subjects.

    Expects a ``subject`` (or ``id`` / ``ptid``) column to identify records.

    Parameters
    ----------
    filepath : str or Path
        Path to the CSV file.
    timestamp_col : str | None
        Override the auto-detected timestamp column.
    glucose_col : str | None
        Override the auto-detected glucose column.
    subject_col : str | None
        Override the auto-detected subject/ID column.

    Returns
    -------
    pd.DataFrame
        DataFrame with ``Timestamp``, ``Glucose``, and ``subject`` columns.

    Raises
    ------
    FileNotFoundError
        If *filepath* does not exist.
    ValueError
        If required columns cannot be resolved.
    """
    df = load_csv(
        filepath,
        timestamp_col=timestamp_col,
        glucose_col=glucose_col,
        subject_col=subject_col,
    )

    if "subject" not in df.columns:
        raise ValueError(
            f"No subject/ID column found in {Path(filepath).name}. "
            f"Pass subject_col=... explicitly. "
            f"Available columns: {list(df.columns)}."
        )

    df["subject"] = df["subject"].astype(str)

    keep = [COL_TIMESTAMP, COL_GLUCOSE, "subject"]
    extra = [c for c in df.columns if c not in keep]
    return df[keep + extra]


# ─────────────────────────────────────────────────────────────────────────────
# Device format presets
# ─────────────────────────────────────────────────────────────────────────────

def load_dexcom(filepath: str | Path) -> pd.DataFrame:
    """Load a Dexcom CGM export CSV.

    Dexcom exports use the column ``"Timestamp (YYYY-MM-DDThh:mm:ss)"`` and
    ``"Glucose Value (mg/dL)"``.  Only CGM reading rows are kept (device events
    and calibrations are dropped).

    Parameters
    ----------
    filepath : str or Path
        Path to the Dexcom CSV export.

    Returns
    -------
    pd.DataFrame
        DataFrame with ``Timestamp`` and ``Glucose`` columns.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    df = pd.read_csv(filepath, skiprows=1)  # Dexcom has a header row to skip

    # Dexcom column name patterns
    ts_candidates = [c for c in df.columns if "timestamp" in c.lower()]
    gl_candidates = [c for c in df.columns if "glucose value" in c.lower()]

    ts_col = ts_candidates[0] if ts_candidates else None
    gl_col = gl_candidates[0] if gl_candidates else None

    return load_csv(
        filepath,
        timestamp_col=ts_col,
        glucose_col=gl_col,
    )


def load_libre(filepath: str | Path) -> pd.DataFrame:
    """Load a FreeStyle Libre CGM export CSV.

    Libre exports include a 2-row header; the ``"Device Timestamp"`` and
    ``"Historic Glucose mg/dL"`` columns are used.

    Parameters
    ----------
    filepath : str or Path
        Path to the Libre CSV export.

    Returns
    -------
    pd.DataFrame
        DataFrame with ``Timestamp`` and ``Glucose`` columns.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    df = pd.read_csv(filepath, skiprows=2)

    ts_candidates = [c for c in df.columns if "timestamp" in c.lower() or "time" in c.lower()]
    gl_candidates = [c for c in df.columns if "glucose" in c.lower() and "historic" in c.lower()]
    if not gl_candidates:
        gl_candidates = [c for c in df.columns if "glucose" in c.lower()]

    ts_col = ts_candidates[0] if ts_candidates else None
    gl_col = gl_candidates[0] if gl_candidates else None

    return load_csv(
        filepath,
        timestamp_col=ts_col,
        glucose_col=gl_col,
    )
