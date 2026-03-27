"""
glycosignal.cli
===============

Command-line interface for GlycoSignal.

Entry point: ``glycosignal`` (configured in pyproject.toml).

Commands
--------
- ``glycosignal summary input.csv``
- ``glycosignal windows input.csv [--window-hours N] [--overlap-hours N] [--output FILE]``
- ``glycosignal features windows.csv [--output FILE] [--features F1,F2,...]``
- ``glycosignal list-features [--category CAT]``
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_df(filepath: str, subject_col: str | None = None):
    """Load a CSV with column auto-detection."""
    from .io import load_csv
    return load_csv(filepath, subject_col=subject_col)


def _print_table(d: dict, title: str = "") -> None:
    """Print a dict as a formatted two-column table."""
    if title:
        print(f"\n{title}")
        print("-" * 50)
    for key, val in d.items():
        if isinstance(val, float):
            print(f"  {key:<35} {val:>10.3f}")
        else:
            print(f"  {key:<35} {val!s:>10}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Subcommand: summary
# ─────────────────────────────────────────────────────────────────────────────

def cmd_summary(args: argparse.Namespace) -> None:
    """Print a comprehensive glycemic summary to stdout."""
    from . import metrics
    from .io import load_csv
    from .preprocessing import clean_cgm, standardize_columns

    df = load_csv(args.input)
    df = standardize_columns(df)
    df = clean_cgm(df)

    if df.empty:
        print("Error: No valid CGM data found after cleaning.", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  GlycoSignal Summary: {Path(args.input).name}")
    print(f"{'='*60}")
    print(f"  Readings : {len(df)}")

    from .schemas import prepare
    prep = prepare(df)

    _print_table(metrics.basic_stats(prep), "Basic Statistics")
    _print_table(metrics.variability_metrics(prep), "Variability")
    _print_table(metrics.time_in_ranges(prep), "Time-in-Range")
    _print_table(metrics.risk_indices(prep), "Risk Indices")


# ─────────────────────────────────────────────────────────────────────────────
# Subcommand: windows
# ─────────────────────────────────────────────────────────────────────────────

def cmd_windows(args: argparse.Namespace) -> None:
    """Create sliding windows and save to CSV."""
    from .io import load_csv
    from .preprocessing import clean_cgm, standardize_columns
    from .windows import create_sliding_windows

    df = load_csv(args.input, subject_col=args.subject_col)
    df = standardize_columns(df)
    df = clean_cgm(df)

    if df.empty:
        print("Error: No valid CGM data found after cleaning.", file=sys.stderr)
        sys.exit(1)

    if "subject" not in df.columns:
        df["subject"] = Path(args.input).stem

    group_col = args.group_col or "subject"
    result = create_sliding_windows(
        df,
        window_hours=args.window_hours,
        overlap_hours=args.overlap_hours,
        min_fraction=args.min_fraction,
        group_col=group_col,
        show_progress=True,
    )

    output = args.output or (
        Path(args.input).stem + f"_windows_{args.window_hours}h_{args.overlap_hours}h.csv"
    )
    result.windows.to_csv(output, index=False)

    print(f"\nWindowing complete:")
    for k, v in result.metadata.items():
        print(f"  {k:<40} {v}")
    print(f"\nSaved: {output}")


# ─────────────────────────────────────────────────────────────────────────────
# Subcommand: features
# ─────────────────────────────────────────────────────────────────────────────

def cmd_features(args: argparse.Namespace) -> None:
    """Compute feature map from a windowed CSV and save to file."""
    import pandas as pd
    from .features import build_feature_map, build_feature_map_wide
    from .utils import infer_time_columns

    df = pd.read_csv(args.input)

    feature_names = None
    if args.features:
        feature_names = [f.strip() for f in args.features.split(",")]

    # Detect format: long (has window_id) or wide (has HH:MM columns)
    time_cols = infer_time_columns(df)
    if time_cols:
        print("Detected wide-format windowed input.")
        result = build_feature_map_wide(df, feature_names=feature_names, show_progress=True)
    elif "window_id" in df.columns:
        print("Detected long-format windowed input.")
        result = build_feature_map(df, feature_names=feature_names, show_progress=True)
    else:
        print(
            "Error: Input does not appear to be a windowed DataFrame. "
            "Run 'glycosignal windows' first.",
            file=sys.stderr,
        )
        sys.exit(1)

    output = args.output or (Path(args.input).stem + "_features.csv")
    result.to_csv(output, index=False)
    print(f"\nFeature map shape: {result.shape}")
    print(f"Saved: {output}")


# ─────────────────────────────────────────────────────────────────────────────
# Subcommand: list-features
# ─────────────────────────────────────────────────────────────────────────────

def cmd_list_features(args: argparse.Namespace) -> None:
    """Print all registered feature names and descriptions."""
    from .registry import get_feature_metadata

    meta = get_feature_metadata()
    if args.category:
        meta = meta[meta["category"] == args.category]
        if meta.empty:
            print(f"No features found in category {args.category!r}.")
            return

    # Group by category
    for cat, grp in meta.groupby("category"):
        print(f"\n{cat.upper()}")
        print("-" * 60)
        for _, row in grp.iterrows():
            print(f"  {row['name']:<35}  {row['description']}")

    print(f"\nTotal: {len(meta)} features")


# ─────────────────────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="glycosignal",
        description=(
            "GlycoSignal: CGM data analysis and ML feature extraction.\n\n"
            "Examples:\n"
            "  glycosignal summary data.csv\n"
            "  glycosignal windows data.csv --window-hours 24\n"
            "  glycosignal features windows.csv --output features.csv\n"
            "  glycosignal list-features\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s " + _get_version(),
    )

    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    subparsers.required = True

    # ── summary ──────────────────────────────────────────────────────────
    p_summary = subparsers.add_parser("summary", help="Print glycemic summary metrics.")
    p_summary.add_argument("input", help="Path to CGM CSV file.")
    p_summary.set_defaults(func=cmd_summary)

    # ── windows ──────────────────────────────────────────────────────────
    p_windows = subparsers.add_parser("windows", help="Create sliding windows from CGM data.")
    p_windows.add_argument("input", help="Path to CGM CSV file.")
    p_windows.add_argument(
        "--window-hours", type=float, default=24.0, dest="window_hours",
        help="Window duration in hours (default: 24).",
    )
    p_windows.add_argument(
        "--overlap-hours", type=float, default=0.0, dest="overlap_hours",
        help="Window overlap in hours (default: 0).",
    )
    p_windows.add_argument(
        "--min-fraction", type=float, default=0.7, dest="min_fraction",
        help="Minimum completeness fraction 0–1 (default: 0.7).",
    )
    p_windows.add_argument(
        "--group-col", default=None, dest="group_col",
        help="Column to group by (default: 'subject').",
    )
    p_windows.add_argument(
        "--subject-col", default=None, dest="subject_col",
        help="Override subject column name.",
    )
    p_windows.add_argument("--output", "-o", default=None, help="Output CSV path.")
    p_windows.set_defaults(func=cmd_windows)

    # ── features ─────────────────────────────────────────────────────────
    p_features = subparsers.add_parser("features", help="Build feature map from windowed data.")
    p_features.add_argument("input", help="Path to windowed CSV file.")
    p_features.add_argument(
        "--features", default=None,
        help="Comma-separated feature names (default: all core features).",
    )
    p_features.add_argument("--output", "-o", default=None, help="Output CSV path.")
    p_features.set_defaults(func=cmd_features)

    # ── list-features ─────────────────────────────────────────────────────
    p_list = subparsers.add_parser("list-features", help="List all registered feature names.")
    p_list.add_argument(
        "--category", default=None,
        help="Filter by category (e.g. basic_stats, variability, time_in_range, risk, excursion, peak).",
    )
    p_list.set_defaults(func=cmd_list_features)

    return parser


def _get_version() -> str:
    try:
        from . import __version__
        return __version__
    except ImportError:
        return "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
