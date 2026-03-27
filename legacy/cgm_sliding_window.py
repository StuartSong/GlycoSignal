import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.interpolate import PchipInterpolator


def load_cgm_folder(folder_path="Data/Processed"):
    """
    Load CGM data from a folder of per-subject CSV files (e.g. CFBR data).
    Each CSV has columns: Timestamp, Glucose Value (mg/dL).

        Returns standardised DataFrame with columns:
            Timestamp, Glucose Value (mg/dL), filename, subject
    """
    dfs = []
    csv_files = [f for f in sorted(os.listdir(folder_path)) if f.endswith(".csv")]
    for f in tqdm(csv_files, desc="Loading CSVs"):
        filepath = os.path.join(folder_path, f)
        df = pd.read_csv(filepath)

        # Auto-detect column names
        possible_ts = [c for c in df.columns if c.lower() in ["timestamp", "time"]]
        ts_col = possible_ts[0] if possible_ts else df.columns[0]

        possible_gl = [c for c in df.columns
                       if c.lower() in ["glucose value (mg/dl)", "gl", "glucose"]]
        gl_col = possible_gl[0] if possible_gl else df.columns[-1]

        df = df[[ts_col, gl_col]].copy()
        df.columns = ["Timestamp", "Glucose Value (mg/dL)"]

        # Filename = original stem; subject = stem without date suffix
        name = os.path.splitext(f)[0]
        parts = name.rsplit("_", 1)
        subject_wo_date = parts[0] if (len(parts) == 2 and len(parts[1]) >= 8) else name
        df["filename"] = name
        df["subject"] = subject_wo_date
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def load_cgm_file(filepath):
    """
    Load CGM data from a single CSV with an id column (e.g. Aleppo, Shah).
    Expects columns: id, time, gl.

        Returns standardised DataFrame with columns:
            Timestamp, Glucose Value (mg/dL), subject
    """
    df = pd.read_csv(filepath)

    # Auto-detect column names
    possible_id = [c for c in df.columns if c.lower() in ["id", "ptid", "subject"]]
    id_col = possible_id[0] if possible_id else df.columns[0]

    possible_ts = [c for c in df.columns if c.lower() in ["timestamp", "time"]]
    ts_col = possible_ts[0] if possible_ts else df.columns[1]

    possible_gl = [c for c in df.columns
                   if c.lower() in ["glucose value (mg/dl)", "gl", "glucose"]]
    gl_col = possible_gl[0] if possible_gl else df.columns[-1]

    out = df[[ts_col, gl_col, id_col]].copy()
    out.columns = ["Timestamp", "Glucose Value (mg/dL)", "subject"]
    out["subject"] = out["subject"].astype(str)
    return out


def _fill_row_nans_with_pchip(values, max_gap_points=24, left_context=None, right_context=None):
    """
    Fill NaNs using PCHIP interpolation only for short consecutive gaps.

    A gap is interpolated only when its length is strictly less than
    ``max_gap_points`` (e.g., < 12 points for < 1 hour on a 5-min grid).
    Optional context rows (previous/next day from same subject) can be passed
    to support edge-gap interpolation.
    Observed values remain unchanged.
    """
    y = np.asarray(values, dtype=float)
    n_points = y.size
    x = np.arange(n_points, dtype=float)
    missing_mask = np.isnan(y)

    if not missing_mask.any():
        return y.copy()

    x_parts = [x]
    y_parts = [y]

    if left_context is not None:
        left_arr = np.asarray(left_context, dtype=float)
        if left_arr.size == n_points:
            x_parts.insert(0, np.arange(-n_points, 0, dtype=float))
            y_parts.insert(0, left_arr)

    if right_context is not None:
        right_arr = np.asarray(right_context, dtype=float)
        if right_arr.size == n_points:
            x_parts.append(np.arange(n_points, 2 * n_points, dtype=float))
            y_parts.append(right_arr)

    x_all = np.concatenate(x_parts)
    y_all = np.concatenate(y_parts)
    mask_all = ~np.isnan(y_all)
    n_valid_all = int(mask_all.sum())

    if n_valid_all < 2:
        return y.copy()

    filled = y.copy()

    # Find consecutive NaN runs and keep only short runs for interpolation.
    runs = []
    run_start = None
    for idx, is_missing in enumerate(missing_mask):
        if is_missing and run_start is None:
            run_start = idx
        elif (not is_missing) and run_start is not None:
            runs.append((run_start, idx - 1))
            run_start = None
    if run_start is not None:
        runs.append((run_start, y.size - 1))

    try:
        interp = PchipInterpolator(x_all[mask_all], y_all[mask_all], extrapolate=False)
        for start, end in runs:
            run_len = end - start + 1
            if run_len < max_gap_points:
                xi = x[start:end + 1]
                filled[start:end + 1] = np.round(interp(xi), 1)
    except Exception:
        mask_current = ~np.isnan(y)
        if int(mask_current.sum()) < 2:
            return filled
        for start, end in runs:
            run_len = end - start + 1
            if run_len < max_gap_points:
                xi = x[start:end + 1]
                filled[start:end + 1] = np.round(np.interp(xi, x[mask_current], y[mask_current]), 1)

    return filled


def _generate_missingness_report(window_stats, min_data_percent, points_per_window, output_path):
    """
    Generate a PDF report showing per-window data coverage for each group/year,
    indicating which windows were kept and which were dropped and why.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from collections import defaultdict

    threshold_points = (min_data_percent / 100) * points_per_window

    # Group stats by (group_value, year), preserving insertion order
    grouped = defaultdict(list)
    for stat in window_stats:
        year = stat["date"][:4]
        key = (stat["group"], year)
        grouped[key].append(stat)

    # Build text lines for each group block
    all_blocks = []
    for (group_value, year), stats in grouped.items():
        title = (
            f"{group_value} {year}   "
            f"(threshold = {threshold_points:.1f} points for {min_data_percent:.0f}% coverage)"
        )
        header = f"{'Date':<14}  {'Points':>6}  {'Coverage':>9}  Kept"
        separator = "-" * 50
        rows = []
        for stat in stats:
            if stat["kept"]:
                kept_str = "YES"
            elif stat["drop_reason"] == "remaining NaN after interpolation":
                kept_str = "NO   \u2190 dropped (unfillable gap)"
            else:
                kept_str = "NO   \u2190 dropped"
            rows.append(
                f"{stat['date']:<14}  {stat['points']:>6}  "
                f"{stat['coverage_pct']:>8.1f}%  {kept_str}"
            )
        block = [title, header, separator] + rows + [""]
        all_blocks.append(block)

    # Paginate: pack blocks until ~60 lines, then flush a page
    lines_per_page = 60
    font_size = 8

    def _flush_page(pdf, lines):
        fig, ax = plt.subplots(figsize=(10, 14))
        ax.axis("off")
        ax.text(
            0.02, 0.98,
            "\n".join(lines),
            transform=ax.transAxes,
            fontsize=font_size,
            fontfamily="monospace",
            verticalalignment="top",
        )
        fig.tight_layout(pad=1.0)
        pdf.savefig(fig)
        plt.close(fig)

    with PdfPages(output_path) as pdf:
        current_lines = []
        for block in all_blocks:
            if current_lines and len(current_lines) + len(block) > lines_per_page:
                _flush_page(pdf, current_lines)
                current_lines = []
            current_lines.extend(block)
        if current_lines:
            _flush_page(pdf, current_lines)


def create_sliding_windows(
    df,
    window_hours=24,
    overlap_hours=0,
    tolerance_minutes=2.5,
    group_col="subject",
    id_cols=None,
    min_data_percent_per_window=70,
    dropped_after_interp_output_path=None,
    report_output_path=None,
):
    """
    Transform a concatenated CGM dataframe into a wide sliding-window
    representation with a standardised 5-minute time grid.

    Parameters
    ----------
    df : pd.DataFrame
        Output of ``load_cgm_folder`` or ``load_cgm_file``. Must contain
        ``Timestamp`` and ``Glucose Value (mg/dL)`` and the grouping column
        specified by ``group_col``.
    window_hours : float
        Length of each window in hours (default 24).
    overlap_hours : float
        Overlap between consecutive windows in hours (default 0).
    min_data_percent_per_window : float | str
        Minimum percentage of non-missing 5-minute grid values required to
        keep a window row. Accepts values from 0 to 100. You can pass either
        a number (e.g., ``70``) or a string with ``%`` (e.g., ``"70%"``).
        Default is 0 (i.e., no minimum percentage filter).
    dropped_after_interp_output_path : str | None
        Optional CSV path. If provided, rows that are dropped after
        interpolation (because NaNs still remain) are saved to this path,
        using their pre-interpolation values so missing patterns can be
        inspected.
    report_output_path : str | None
        Optional path ending in ``.pdf``. If provided, a PDF report is written
        showing per-window coverage for every group/year, marking which windows
        were kept and which were dropped (insufficient data or unfillable gap).
    tolerance_minutes : float
            group_col : str
                Column to group by when creating windows. Use ``filename`` for CFBR
                (per-file recordings) or ``subject`` for datasets where one subject may
                span multiple files/rows.
            id_cols : tuple[str, ...] | list[str] | None
                Columns to include at the front of the output for each window. If None,
                defaults to ``(group_col,)``.
        Maximum distance (in minutes) between an actual CGM reading and a
        target 5-minute grid point. Readings outside this tolerance are
        treated as missing.  Default 2.5 (half the grid spacing).

    Returns
    -------
    result_df : pd.DataFrame
        One row per window. The first columns are ``id_cols`` (or ``group_col``)
        followed by ``date`` (window start date). The remaining columns
        correspond to the time grid (e.g. 288 for 24h), named ``00:00``,
        ``00:05``, …, ``23:55``.
    metadata : dict
        ``n_subjects``              – unique subjects
        ``n_valid_windows``         – total rows in result_df
        ``n_discarded_partial_days``– first-day partials dropped per group
        ``n_dropped_min_data_pct``  – windows dropped before interpolation due to
                                      insufficient observed data percentage
        ``n_dropped_after_interp``  – rows dropped due to remaining NaNs
    """
    if overlap_hours >= window_hours:
        raise ValueError("overlap_hours must be less than window_hours")

    if isinstance(min_data_percent_per_window, str):
        pct_str = min_data_percent_per_window.strip()
        if pct_str.endswith("%"):
            pct_str = pct_str[:-1].strip()
        min_data_percent_per_window = float(pct_str)

    min_data_percent_per_window = float(min_data_percent_per_window)
    if not (0 <= min_data_percent_per_window <= 100):
        raise ValueError("min_data_percent_per_window must be between 0 and 100")

    if id_cols is None:
        id_cols = (group_col,)

    missing_cols = [c for c in ["Timestamp", "Glucose Value (mg/dL)", group_col] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df = df.copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df["Glucose Value (mg/dL)"] = pd.to_numeric(
        df["Glucose Value (mg/dL)"], errors="coerce"
    )
    df = df.dropna(subset=["Timestamp", "Glucose Value (mg/dL)", group_col])

    # ----- grid parameters ------------------------------------------------
    grid_freq = "5min"
    points_per_window = int(window_hours * 60 / 5)
    time_offsets = pd.timedelta_range(start="0h", periods=points_per_window, freq=grid_freq)

    # Column names: "00:00" … "23:55" (or beyond 24 h if needed)
    col_names = []
    for td in time_offsets:
        total_min = int(td.total_seconds() // 60)
        h, m = divmod(total_min, 60)
        col_names.append(f"{h:02d}:{m:02d}")

    tolerance = pd.Timedelta(minutes=tolerance_minutes)
    step = pd.Timedelta(hours=window_hours - overlap_hours)
    window_delta = pd.Timedelta(hours=window_hours)

    # ----- iterate over each subject -------------------------------------
    rows = []
    window_stats = []  # collect stats for all windows (kept + dropped) for the report
    n_discarded = 0
    n_dropped_min_data_pct = 0

    for group_value, grp in tqdm(df.groupby(group_col), desc=f"Creating windows ({group_col})"):
        grp = grp.sort_values("Timestamp").reset_index(drop=True)

        first_ts = grp["Timestamp"].iloc[0]
        first_midnight = first_ts.normalize()

        # Discard the first incomplete calendar day
        if first_ts != first_midnight:
            first_midnight += pd.Timedelta(days=1)
            n_discarded += 1

        last_ts = grp["Timestamp"].iloc[-1]
        if first_midnight > last_ts:
            continue  # no complete day available

        # Pre-round all timestamps to nearest 5 min and compute distances
        grp["rounded"] = grp["Timestamp"].dt.round(grid_freq)
        grp["dist"] = (grp["Timestamp"] - grp["rounded"]).abs()

        # Discard readings outside tolerance once (avoids re-filtering per window)
        grp = grp.loc[grp["dist"] <= tolerance]

        if grp.empty:
            continue

        # ----- slide windows ----------------------------------------------
        window_start = first_midnight
        while window_start <= last_ts:
            window_end = window_start + window_delta

            # Select readings whose rounded time falls inside this window
            mask = (grp["rounded"] >= window_start) & (grp["rounded"] < window_end)
            w = grp.loc[mask]

            # Average duplicate rounded timestamps (handles equidistant case)
            resampled = w.groupby("rounded")["Glucose Value (mg/dL)"].mean()

            # Reindex to the full 5-minute target grid
            targets = pd.date_range(start=window_start, periods=points_per_window, freq=grid_freq)
            resampled = resampled.reindex(targets)

            # Skip windows with too little data
            non_na = int(resampled.notna().sum())
            data_pct = (non_na / points_per_window) * 100
            if data_pct < min_data_percent_per_window:
                n_dropped_min_data_pct += 1
                if report_output_path is not None:
                    window_stats.append({
                        "group": group_value,
                        "date": window_start.strftime("%Y-%m-%d"),
                        "points": non_na,
                        "coverage_pct": data_pct,
                        "kept": False,
                        "drop_reason": "insufficient data",
                        "rows_idx": None,
                    })
                window_start += step
                continue

            if report_output_path is not None:
                window_stats.append({
                    "group": group_value,
                    "date": window_start.strftime("%Y-%m-%d"),
                    "points": non_na,
                    "coverage_pct": data_pct,
                    "kept": True,
                    "drop_reason": None,
                    "rows_idx": len(rows),  # position this row will occupy in `rows`
                })

            row = {
                "date": window_start.strftime("%Y-%m-%d"),
            }
            for col in id_cols:
                if col == group_col:
                    row[col] = group_value
                else:
                    row[col] = grp[col].iloc[0] if col in grp.columns else None
            row.update(dict(zip(col_names, resampled.values)))
            rows.append(row)

            window_start += step

    # ----- assemble output ------------------------------------------------
    result_df = pd.DataFrame(rows)

    # Fill each row's NaNs via short-gap PCHIP interpolation; keep observed values unchanged.
    # For edge gaps, use previous/next day rows from the same subject/group as context.
    n_dropped_after_interp = 0
    if not result_df.empty:
        if "subject" in result_df.columns:
            context_key_col = "subject"
        elif group_col in result_df.columns:
            context_key_col = group_col
        else:
            context_key_col = id_cols[0]

        pre_interp_result_df = result_df.copy()
        pre_interp_result_df["_date_dt"] = pd.to_datetime(pre_interp_result_df["date"], errors="coerce")
        filled_values = np.empty((len(pre_interp_result_df), len(col_names)), dtype=float)

        for _, grp_idx in pre_interp_result_df.groupby(context_key_col).groups.items():
            group_rows = pre_interp_result_df.loc[list(grp_idx)].sort_values("_date_dt")
            row_indices = group_rows.index.to_list()
            group_matrix = group_rows[col_names].to_numpy(dtype=float)

            for pos_in_group, row_idx in enumerate(row_indices):
                row_values = group_matrix[pos_in_group]
                left_context = None
                right_context = None

                if np.isnan(row_values[0]) and pos_in_group > 0:
                    left_context = group_matrix[pos_in_group - 1]
                if np.isnan(row_values[-1]) and pos_in_group < (len(row_indices) - 1):
                    right_context = group_matrix[pos_in_group + 1]

                filled_values[row_idx] = _fill_row_nans_with_pchip(
                    row_values,
                    max_gap_points=12,
                    left_context=left_context,
                    right_context=right_context,
                )

        pre_interp_result_df = pre_interp_result_df.drop(columns=["_date_dt"])
        filled_result_df = result_df.copy()
        filled_result_df.loc[:, col_names] = filled_values

        # Drop rows that still contain missing values after interpolation.
        remaining_missing_mask = filled_result_df[col_names].isna().any(axis=1)
        n_dropped_after_interp = int(remaining_missing_mask.sum())

        # Update report stats for windows dropped after interpolation
        if report_output_path is not None and n_dropped_after_interp > 0:
            dropped_set = set(remaining_missing_mask.index[remaining_missing_mask].tolist())
            for stat in window_stats:
                if stat["rows_idx"] is not None and stat["rows_idx"] in dropped_set:
                    stat["kept"] = False
                    stat["drop_reason"] = "remaining NaN after interpolation"

        if dropped_after_interp_output_path is not None:
            dropped_pre_interp = pre_interp_result_df.loc[remaining_missing_mask].reset_index(drop=True)
            dropped_pre_interp.to_csv(dropped_after_interp_output_path, index=False)

        if n_dropped_after_interp > 0:
            result_df = filled_result_df.loc[~remaining_missing_mask].reset_index(drop=True)
        else:
            result_df = filled_result_df

    metadata = {
        "n_subjects": df["subject"].nunique() if "subject" in df.columns else df[group_col].nunique(),
        "n_valid_windows": len(result_df),
        "n_discarded_partial_days": n_discarded,
        "n_dropped_min_data_pct": n_dropped_min_data_pct,
        "n_dropped_after_interp": n_dropped_after_interp,
    }

    if report_output_path is not None and window_stats:
        _generate_missingness_report(
            window_stats,
            min_data_percent_per_window,
            points_per_window,
            report_output_path,
        )

    return result_df, metadata


def format_window_str(window_hours, overlap_hours):
    """Format window parameters for filename."""
    if window_hours == int(window_hours):
        w_str = f"{int(window_hours)}h"
    else:
        w_str = f"{window_hours}h"
    
    if overlap_hours == int(overlap_hours):
        o_str = f"{int(overlap_hours)}h"
    else:
        o_str = f"{overlap_hours}h"
    
    return f"{w_str}_{o_str}"


if __name__ == "__main__":
    window_str = format_window_str(24, 0)
    
    # ---- Process CFBR data -------------------------------------------
    print("=" * 60)
    print("Processing CFBR Data (Data/Processed/)")
    print("=" * 60)
    df_cfbr = load_cgm_folder(folder_path="Data/Processed")
    print(f"Combined shape: {df_cfbr.shape}")

    windows_cfbr, meta_cfbr = create_sliding_windows(
        df_cfbr,
        group_col="filename",
        id_cols=("filename", "subject"),
        report_output_path=f"Data/missingness_report_cfbr_windows_{window_str}.pdf",
        # dropped_after_interp_output_path=f"Data/dropped_after_interp_cf_windows_{window_str}.csv",
    )
    print(f"Sliding-window shape: {windows_cfbr.shape}")
    print(f"Metadata: {meta_cfbr}\n")

    cfbr_output = f"Data/processed_cf_windows_{window_str}.csv"
    windows_cfbr.to_csv(cfbr_output, index=False)
    print(f"Saved: {cfbr_output}\n")

    # # ---- Process Aleppo & Shah -----------------------------------------------
    # datasets_to_process = [
    #     ("aleppo", "Public Datasets/Aleppo_T1D_226/HDeviceCGM_processed.csv"),
    #     ("shah", "Public Datasets/Shah_Healthy_169/NonDiabDeviceCGM_processed.csv"),
    # ]

    # for dataset_key, filepath in datasets_to_process:
    #     print("=" * 60)
    #     print(f"Processing {dataset_key} ({filepath})")
    #     print("=" * 60)

    #     df_pub = load_cgm_file(filepath)
    #     print(f"Combined shape: {df_pub.shape}")

    #     windows_pub, meta_pub = create_sliding_windows(
    #         df_pub,
    #         group_col="subject",
    #         id_cols=("subject",),
    #         report_output_path=f"Data/missingness_report_{dataset_key}_windows_{window_str}.pdf",
    #         # dropped_after_interp_output_path=f"Data/dropped_after_interp_{dataset_key}_windows_{window_str}.csv",
    #     )
    #     print(f"Sliding-window shape: {windows_pub.shape}")
    #     print(f"Metadata: {meta_pub}\n")

    #     pub_output = f"Data/processed_{dataset_key}_windows_{window_str}.csv"
    #     windows_pub.to_csv(pub_output, index=False)
    #     print(f"Saved: {pub_output}\n")
