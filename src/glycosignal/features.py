"""
glycosignal.features
====================

Feature map builders: compute glycemic features across windows or records and
return ML-ready DataFrames.

Three entry points:

1. :func:`build_feature_vector` -- compute features for a *single* window/record.
2. :func:`build_feature_map` -- compute features across all windows in a
   long-format windowed DataFrame (output of :func:`~glycosignal.windows.create_sliding_windows`).
3. :func:`build_feature_table` -- compute features across a list of DataFrames
   (one per subject / recording session).

Backward compatibility:
    :func:`build_feature_map_wide` accepts the legacy wide-format windowed
    DataFrame (``HH:MM`` columns) produced by the original ``cgm_feature_map.py``
    script.

Usage
-----
    >>> from glycosignal import windows, features
    >>> result = windows.create_sliding_windows(df, window_hours=24)
    >>> X = features.build_feature_map(result.windows)

    >>> X = features.build_feature_map(
    ...     result.windows,
    ...     feature_names=["mean_glucose", "cv", "lbgi"],
    ... )
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from tqdm import tqdm

from .registry import DEFAULT_FEATURE_NAMES, DEFAULT_REGISTRY
from .schemas import COL_GLUCOSE, COL_TIMESTAMP, CGMInput, _ensure_prepared
from .utils import infer_id_columns, infer_time_columns, require_columns, require_dataframe


# ─────────────────────────────────────────────────────────────────────────────
# Single-window feature computation
# ─────────────────────────────────────────────────────────────────────────────

def build_feature_vector(
    data: CGMInput,
    feature_names: list[str] | None = None,
    registry=None,
) -> dict[str, float | int]:
    """Compute glycemic features for a single window or record.

    Parameters
    ----------
    data : DataFrame or PreparedCGMData
        A single window or recording session (must have ``Timestamp`` and
        ``Glucose`` columns if a DataFrame).
    feature_names : list[str] | None
        Feature names to compute.  ``None`` uses :data:`~glycosignal.registry.DEFAULT_FEATURE_NAMES`.
    registry : FeatureRegistry | None
        Feature registry to use.  ``None`` uses :data:`~glycosignal.registry.DEFAULT_REGISTRY`.

    Returns
    -------
    dict[str, float | int]
        Mapping from feature name to scalar value.

    Raises
    ------
    KeyError
        If any requested feature name is not registered.
    """
    reg = registry or DEFAULT_REGISTRY
    names = feature_names if feature_names is not None else DEFAULT_FEATURE_NAMES

    # Validate all names before computing
    for name in names:
        if name not in reg:
            raise KeyError(
                f"Feature {name!r} is not registered. "
                f"Call glycosignal.registry.list_features() to see available features."
            )

    d = _ensure_prepared(data)
    return {name: reg.compute(name, d) for name in names}


# ─────────────────────────────────────────────────────────────────────────────
# Feature map from long-format windowed DataFrame
# ─────────────────────────────────────────────────────────────────────────────

def build_feature_map(
    windowed_df: pd.DataFrame,
    feature_names: list[str] | None = None,
    include_metadata: bool = True,
    registry=None,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Compute glycemic features for every window in a long-format windowed DataFrame.

    This is the primary feature-map builder.  It accepts the output of
    :func:`~glycosignal.windows.create_sliding_windows` and returns an
    ML-ready feature matrix with one row per window.

    Parameters
    ----------
    windowed_df : pd.DataFrame
        Long-format DataFrame with at minimum a ``window_id``, ``Timestamp``,
        and ``Glucose`` column.  Typically the ``.windows`` attribute of a
        :class:`~glycosignal.windows.WindowResult`.
    feature_names : list[str] | None
        Feature names to compute.  ``None`` uses the default core feature set.
    include_metadata : bool
        When True, non-time-series columns (``window_id``, ``subject``,
        ``date``, etc.) are included at the front of the output DataFrame.
    registry : FeatureRegistry | None
        Feature registry.  ``None`` uses the global default.
    show_progress : bool
        Show a tqdm progress bar.

    Returns
    -------
    pd.DataFrame
        One row per window.  Columns: metadata (if requested) followed by one
        column per feature.

    Raises
    ------
    ValueError
        If required columns are missing.
    KeyError
        If any requested feature name is not registered.
    """
    require_dataframe(windowed_df, "windowed_df")
    require_columns(windowed_df, ["window_id", COL_TIMESTAMP, COL_GLUCOSE])

    reg = registry or DEFAULT_REGISTRY
    names = feature_names if feature_names is not None else DEFAULT_FEATURE_NAMES

    # Validate all feature names up front
    for name in names:
        if name not in reg:
            raise KeyError(
                f"Feature {name!r} is not registered. "
                f"Call glycosignal.registry.list_features() to see available features."
            )

    # Identify metadata columns (everything except Timestamp and Glucose)
    meta_cols = [c for c in windowed_df.columns if c not in (COL_TIMESTAMP, COL_GLUCOSE)]

    rows: list[dict] = []
    groups = windowed_df.groupby("window_id", sort=False)
    iterator = tqdm(groups, desc="Extracting features") if show_progress else groups

    for window_id, grp in iterator:
        row: dict = {}

        if include_metadata:
            for col in meta_cols:
                row[col] = grp[col].iloc[0]

        # Compute features on this window's sub-DataFrame
        sub = grp[[COL_TIMESTAMP, COL_GLUCOSE]].reset_index(drop=True)
        d = _ensure_prepared(sub)
        for name in names:
            try:
                row[name] = reg.compute(name, d)
            except Exception:
                row[name] = np.nan

        rows.append(row)

    if not rows:
        cols = (meta_cols if include_metadata else []) + names
        return pd.DataFrame(columns=cols)

    out = pd.DataFrame(rows)

    # Ensure column order: metadata first, then features
    if include_metadata:
        ordered = meta_cols + [n for n in names if n not in meta_cols]
    else:
        ordered = [n for n in names]
    out = out.reindex(columns=ordered)

    return out.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Feature table from list of DataFrames
# ─────────────────────────────────────────────────────────────────────────────

def build_feature_table(
    records: Sequence[pd.DataFrame],
    feature_names: list[str] | None = None,
    record_ids: Sequence[str] | None = None,
    registry=None,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Compute features for a list of DataFrames (one per record/subject).

    Parameters
    ----------
    records : sequence of DataFrames
        Each DataFrame must have ``Timestamp`` and ``Glucose`` columns.
    feature_names : list[str] | None
        Feature names to compute.  ``None`` uses the default core feature set.
    record_ids : sequence of str | None
        Optional identifiers for each record.  If provided, a ``record_id``
        column is added to the output.  Length must match ``records``.
    registry : FeatureRegistry | None
        Feature registry.  ``None`` uses the global default.
    show_progress : bool
        Show a tqdm progress bar.

    Returns
    -------
    pd.DataFrame
        One row per record.  Optional ``record_id`` column followed by features.

    Raises
    ------
    ValueError
        If *record_ids* length does not match *records*.
    KeyError
        If any requested feature name is not registered.
    """
    records = list(records)
    if record_ids is not None:
        record_ids = list(record_ids)
        if len(record_ids) != len(records):
            raise ValueError(
                f"record_ids length ({len(record_ids)}) must match "
                f"records length ({len(records)})."
            )

    reg = registry or DEFAULT_REGISTRY
    names = feature_names if feature_names is not None else DEFAULT_FEATURE_NAMES

    rows: list[dict] = []
    iterator = enumerate(records)
    if show_progress:
        iterator = tqdm(list(iterator), desc="Computing feature table")

    for i, df in iterator:
        row: dict = {}
        if record_ids is not None:
            row["record_id"] = record_ids[i]
        row.update(build_feature_vector(df, feature_names=names, registry=reg))
        rows.append(row)

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Backward-compatible wide-format feature map
# ─────────────────────────────────────────────────────────────────────────────

def build_feature_map_wide(
    df: pd.DataFrame,
    id_cols: list[str] | None = None,
    feature_names: list[str] | None = None,
    registry=None,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Compute features from a **wide-format** sliding-window DataFrame.

    This function accepts the legacy ``HH:MM`` column format produced by the
    original ``cgm_feature_map.py`` script (and by
    :func:`~glycosignal.windows.pivot_windows_wide`).

    Parameters
    ----------
    df : pd.DataFrame
        Wide-format DataFrame with time-grid columns like ``"00:00"``,
        ``"00:05"``, ..., ``"23:55"``.
    id_cols : list[str] | None
        Identifier columns to carry through to the output.  Defaults to
        ``["subject", "date"]`` when present.
    feature_names : list[str] | None
        Feature names to compute.  ``None`` uses the default core feature set.
    registry : FeatureRegistry | None
        Feature registry.  ``None`` uses the global default.
    show_progress : bool
        Show a tqdm progress bar.

    Returns
    -------
    pd.DataFrame
        One row per window with id columns followed by feature columns.

    Raises
    ------
    ValueError
        If no time-grid columns are found.
    """
    require_dataframe(df, "df")

    time_cols = infer_time_columns(df)
    if not time_cols:
        raise ValueError(
            "No time-grid columns found (expected patterns like '00:00', '00:05', ...). "
            "Use build_feature_map() for long-format windowed DataFrames."
        )

    if id_cols is None:
        id_cols = [c for c in ("subject", "date") if c in df.columns]

    reg = registry or DEFAULT_REGISTRY
    names = feature_names if feature_names is not None else DEFAULT_FEATURE_NAMES

    has_date = "date" in df.columns
    rows: list[dict] = []

    iterator = df.iterrows()
    if show_progress:
        iterator = tqdm(list(iterator), total=len(df), desc="Extracting features (wide)")

    for row_idx, row in iterator:
        date_val = row["date"] if has_date else None
        d = _wide_row_to_prepared(row, time_cols, date_value=date_val, reg=reg)

        feat_row: dict = {}
        for col in id_cols:
            feat_row[col] = row.get(col)

        for name in names:
            try:
                feat_row[name] = reg.compute(name, d)
            except Exception:
                feat_row[name] = np.nan

        rows.append(feat_row)

    if not rows:
        return pd.DataFrame(columns=id_cols + names)

    out = pd.DataFrame(rows)
    ordered = id_cols + [n for n in names if n not in id_cols]
    return out.reindex(columns=ordered).reset_index(drop=True)


def _wide_row_to_prepared(row: pd.Series, time_cols: list[str], date_value=None, reg=None):
    """Convert one wide-format window row into a PreparedCGMData object."""
    from .schemas import PreparedCGMData, _time_weights

    glucose = pd.to_numeric(
        pd.Series(row[time_cols].values, dtype=float), errors="coerce"
    )

    base_date = (
        pd.to_datetime(date_value)
        if date_value is not None
        else pd.Timestamp("2000-01-01")
    )

    timestamps = pd.Series([
        base_date + pd.Timedelta(
            hours=int(t.split(":")[0]),
            minutes=int(t.split(":")[1]),
        )
        for t in time_cols
    ])

    valid = glucose.notna()
    timestamps = timestamps[valid.values].reset_index(drop=True)
    glucose = glucose[valid.values].reset_index(drop=True)

    if glucose.empty:
        return PreparedCGMData(
            glucose=np.array([], dtype=np.float64),
            timestamps=np.array([], dtype="datetime64[ns]"),
            weights=np.array([], dtype=np.float64),
            total_minutes=0.0,
            n_readings=0,
        )

    df_long = pd.DataFrame({COL_TIMESTAMP: timestamps, COL_GLUCOSE: glucose})
    from .schemas import prepare
    return prepare(df_long)
