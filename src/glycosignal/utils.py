"""
glycosignal.utils
=================

Internal utility helpers shared across GlycoSignal modules.

These functions are not part of the public API and may change without notice.
"""

from __future__ import annotations

import re

import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Wide-format window helpers (used by features.py for backward-compat paths)
# ─────────────────────────────────────────────────────────────────────────────

_TIME_COL_RE = re.compile(r"^\d+:\d{2}$")


def is_time_column(col_name: str) -> bool:
    """Return True if *col_name* looks like a wide-format time column (``HH:MM``).

    Parameters
    ----------
    col_name : str
        Column name to test.

    Returns
    -------
    bool
        ``True`` for patterns like ``"00:00"``, ``"09:05"``, ``"23:55"``.
    """
    return bool(_TIME_COL_RE.fullmatch(str(col_name)))


def infer_time_columns(df: pd.DataFrame) -> list[str]:
    """Return all wide-format time columns from *df* in their original order.

    Parameters
    ----------
    df : pd.DataFrame
        Wide-format sliding-window DataFrame.

    Returns
    -------
    list[str]
        Columns matching the ``HH:MM`` pattern, preserving column order.
    """
    return [c for c in df.columns if is_time_column(c)]


def infer_id_columns(df: pd.DataFrame, time_cols: list[str]) -> list[str]:
    """Return columns that are *not* wide-format time columns.

    Parameters
    ----------
    df : pd.DataFrame
        Wide-format sliding-window DataFrame.
    time_cols : list[str]
        Already-identified time columns (from :func:`infer_time_columns`).

    Returns
    -------
    list[str]
        Non-time columns in their original order.
    """
    time_set = set(time_cols)
    return [c for c in df.columns if c not in time_set]


# ─────────────────────────────────────────────────────────────────────────────
# General helpers
# ─────────────────────────────────────────────────────────────────────────────

def require_dataframe(obj: object, param_name: str = "data") -> pd.DataFrame:
    """Raise a clear TypeError if *obj* is not a pandas DataFrame.

    Parameters
    ----------
    obj : object
        The object to validate.
    param_name : str
        Name used in the error message.

    Returns
    -------
    pd.DataFrame
        *obj* unchanged.

    Raises
    ------
    TypeError
        If *obj* is not a ``pd.DataFrame``.
    """
    if not isinstance(obj, pd.DataFrame):
        raise TypeError(
            f"'{param_name}' must be a pd.DataFrame, got {type(obj).__name__}."
        )
    return obj


def require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    """Raise ValueError if *df* is missing any of the required *columns*.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to check.
    columns : list[str]
        Column names that must be present.

    Raises
    ------
    ValueError
        If one or more columns are absent from *df*.
    """
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"DataFrame is missing required column(s): {missing}. "
            f"Available columns: {list(df.columns)}."
        )
