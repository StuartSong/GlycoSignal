"""
cgm_feature_map_from_windows — Extract glycemic features from sliding-window output
====================================================================================

Reads a wide-format CSV produced by ``cgm_sliding_window.py`` (columns like
``00:00``, ``00:05``, …, ``23:55``) and computes glycemic features per window.

Feature selection
-----------------
Edit the ``FEATURES`` list below to add, remove, or reorder features.
Each entry is ``(output_column_name, function)``.
The function receives a ``cgmquantify_stuart.PreparedCGMData`` object.
"""

import os
import re

import numpy as np
import pandas as pd
from tqdm import tqdm

import glycosignal as gs


# ============================================================
#  Feature registry — edit this list to add / remove features
# ============================================================
# Each tuple is (output_column_name, callable).
# The callable receives a PreparedCGMData object.
# To remove a feature: delete or comment out the line.
# To add a feature:    append a new (name, callable) tuple.
# ============================================================
FEATURES = [
    # --- Summary statistics --------------------------------
    ("mean",           gs.mean_glucose),
    ("median",         gs.median_glucose),
    ("min",            gs.min_glucose),
    ("max",            gs.max_glucose),
    ("fq",             gs.q1_glucose),
    ("tq",             gs.q3_glucose),

    # --- Variability ---------------------------------------
    ("sd",             gs.sd),
    ("cv",             gs.cv),

    # --- Time-in-range (minutes) ---------------------------
    ("TOR",            gs.TOR),
    ("TIR",            gs.TIR),
    ("TIR_70_180",     lambda d: gs.TIR_lo_hi(d, upper=180, lower=70)),
    ("TITR",           lambda d: gs.TIR_lo_hi(d, upper=140, lower=70)),

    # --- Time above / below threshold (minutes) -----------
    ("TA140",          lambda d: gs.TAT(d, threshold=140)),
    ("TA180",          lambda d: gs.TAT(d, threshold=180)),
    ("TA200",          lambda d: gs.TAT(d, threshold=200)),
    ("TA250",          lambda d: gs.TAT(d, threshold=250)),
    ("TB70",           lambda d: gs.TBT(d, threshold=70)),
    ("TB54",           lambda d: gs.TBT(d, threshold=54)),

    # --- Risk indices & variability ------------------------
    ("J_index",        gs.J_index),
    ("LBGI",           gs.LBGI),
    ("HBGI",           gs.HBGI),
    ("ADRR",           gs.ADRR),

    # --- Composite scores ----------------------------------
    ("GRI",            gs.GRI),

    # --- Excursion metrics ---------------------------------
    ("MGE",            gs.MGE),
    ("MGN",            gs.MGN),
    ("MAGE",           gs.MAGE),

    # --- Peak counts ---------------------------------------
    ("PA140",          lambda d: gs.count_peaks(d, threshold=140)),
    ("PA180",          lambda d: gs.count_peaks(d, threshold=180)),
    ("PA200",          lambda d: gs.count_peaks(d, threshold=200)),
    ("PB_139",         lambda d: gs.count_peaks_in_range(d, lower=0, upper=139)),
    ("PA_140_179",     lambda d: gs.count_peaks_in_range(d, lower=140, upper=179)),
    ("PA_180_199",     lambda d: gs.count_peaks_in_range(d, lower=180, upper=199)),
]


# ============================================================
#  Helpers
# ============================================================

_TIME_COL_RE = re.compile(r"\d+:\d{2}")


def _is_time_column(col_name):
    return bool(_TIME_COL_RE.fullmatch(str(col_name)))


def _infer_time_columns(df):
    return [c for c in df.columns if _is_time_column(c)]


def _infer_id_columns(df, time_cols):
    return [c for c in df.columns if c not in time_cols]


def _window_row_to_prepared(row, time_cols, date_value=None):
    """Convert one wide-format window row into a PreparedCGMData object.

    Reconstructs proper timestamps from time-grid column names and the
    row's date so that all downstream metrics use actual intervals.
    """
    glucose = pd.to_numeric(
        pd.Series(row[time_cols].values, dtype=float), errors="coerce"
    )

    base_date = (
        pd.to_datetime(date_value) if date_value is not None
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
        return gs.PreparedCGMData(
            glucose=np.array([], dtype=np.float64),
            timestamps=np.array([], dtype="datetime64[ns]"),
            weights=np.array([], dtype=np.float64),
            total_minutes=0.0,
            n_readings=0,
        )

    df_long = pd.DataFrame({"Timestamp": timestamps, "Glucose": glucose})
    return gs.prepare(df_long)


def _build_window_id(row, row_index, preferred_col="filename"):
    # Prefer the original window/file id when available.
    if preferred_col in row.index and pd.notna(row[preferred_col]):
        return str(row[preferred_col])

    # Fallback: build a stable id from subject/date if present.
    fallback_parts = [c for c in ("subject", "date") if c in row.index]
    if fallback_parts:
        parts = [str(row[c]) for c in fallback_parts if pd.notna(row[c])]
        if parts:
            return "_".join(parts)

    return f"window_{row_index}"


# ============================================================
#  Main extraction
# ============================================================

def create_feature_map(df, id_cols=None, features=None, show_progress=True):
    """Extract glycemic features from a wide-format sliding-window DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Output of ``cgm_sliding_window.create_sliding_windows``.
    id_cols : list[str] | None
        Identifier columns to include in output (in order). If None, defaults
        to ``["subject", "date"]`` when present.
    features : list[tuple[str, callable]] | None
        Feature registry.  Defaults to module-level ``FEATURES``.
    show_progress : bool
        Show a tqdm progress bar.

    Returns
    -------
    pd.DataFrame
        One row per window with id columns followed by feature columns.
    """
    if features is None:
        features = FEATURES

    time_cols = _infer_time_columns(df)
    if not time_cols:
        raise ValueError(
            "No time-grid columns found (expected '00:00', '00:05', …)."
        )

    if id_cols is None:
        id_cols = [c for c in ("subject", "date") if c in df.columns]

    has_date = "date" in df.columns

    iterator = df.iterrows()
    if show_progress:
        iterator = tqdm(iterator, total=len(df), desc="Extracting features")

    rows = []
    for row_idx, row in iterator:
        date_val = row["date"] if has_date else None
        data = _window_row_to_prepared(row, time_cols, date_value=date_val)

        feat_dict = {"id": _build_window_id(row, row_idx)}
        for col in id_cols:
            feat_dict[col] = row[col]
        for feat_name, feat_func in features:
            feat_dict[feat_name] = feat_func(data)

        rows.append(feat_dict)

    out = pd.DataFrame(rows)
    desired_order = (
        ["id"]
        + list(id_cols)
        + [name for name, _ in features]
    )
    return out.reindex(columns=desired_order)


# ============================================================
#  Run
# ============================================================

if __name__ == "__main__":
    file_dirs = [
        "Data/processed_cf_windows_24h_0h.csv",
        "Data/processed_shah_windows_24h_0h.csv",
        "Data/processed_aleppo_windows_24h_0h.csv",
    ]
    for input_csv in file_dirs:
        if not os.path.exists(input_csv):
            print(f"Skipping missing file: {input_csv}")
            continue

        input_root, input_ext = os.path.splitext(input_csv)
        output_csv = f"{input_root}_feature_map{input_ext or '.csv'}"

        # Keep only these identifier columns in output (id is always included).
        id_columns = ["subject", "date"]

        windows_df = pd.read_csv(input_csv)
        feature_map_df = create_feature_map(
            windows_df,
            id_cols=id_columns,
        )
        feature_map_df.to_csv(output_csv, index=False)

        print(f"Input windows:  {windows_df.shape}")
        print(f"Feature map:    {feature_map_df.shape}")
        print(f"Saved to:       {output_csv}")
