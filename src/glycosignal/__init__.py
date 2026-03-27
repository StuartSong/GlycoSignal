"""
GlycoSignal
===========

A production-quality Python package for continuous glucose monitor (CGM) data
analysis, glycemic feature computation, and ML-ready feature extraction.

Quick start
-----------
    >>> import glycosignal
    >>> print(glycosignal.__version__)

    >>> from glycosignal import metrics
    >>> mean = metrics.mean_glucose(df)
    >>> summary = metrics.summary_dict(df)

    >>> from glycosignal import windows, features
    >>> result = windows.create_sliding_windows(df, window_hours=24)
    >>> X = features.build_feature_map(result.windows)

    >>> from glycosignal import registry
    >>> registry.list_features()

Submodules
----------
- :mod:`glycosignal.schemas` -- canonical data contract (``PreparedCGMData``, ``prepare``)
- :mod:`glycosignal.io` -- data loading from CSV files and folders
- :mod:`glycosignal.preprocessing` -- cleaning, validation, interpolation, resampling
- :mod:`glycosignal.metrics` -- individual glycemic metrics and grouped summaries
- :mod:`glycosignal.registry` -- feature registry with metadata
- :mod:`glycosignal.features` -- ML-ready feature map builders
- :mod:`glycosignal.windows` -- sliding window creation
- :mod:`glycosignal.detect` -- event detection (hypo, hyper, nocturnal, postprandial)
- :mod:`glycosignal.plotting` -- matplotlib visualization utilities
- :mod:`glycosignal.report` -- HTML summary report generation
- :mod:`glycosignal.cli` -- command-line interface
"""

__version__ = "0.1.0"

# ─────────────────────────────────────────────────────────────────────────────
# Submodule imports (lazy-style: import the modules, not everything from them)
# ─────────────────────────────────────────────────────────────────────────────

from . import (  # noqa: F401
    detect,
    features,
    io,
    metrics,
    plotting,
    preprocessing,
    registry,
    report,
    schemas,
    windows,
)

# ─────────────────────────────────────────────────────────────────────────────
# Convenience top-level re-exports
# ─────────────────────────────────────────────────────────────────────────────

# Core data preparation
from .schemas import PreparedCGMData, prepare  # noqa: F401

# Most commonly used metric functions
from .metrics import (  # noqa: F401
    adrr,
    basic_stats,
    conga24,
    count_peaks,
    count_peaks_in_range,
    cv,
    gri,
    hbgi,
    j_index,
    lbgi,
    mage,
    max_glucose,
    mean_glucose,
    mean_glucose_excursion,
    mean_glucose_normal,
    median_glucose,
    min_glucose,
    q1_glucose,
    q3_glucose,
    risk_indices,
    sd,
    summary_dict,
    time_above_range_minutes,
    time_above_range_percent,
    time_below_range_minutes,
    time_below_range_percent,
    time_in_range_minutes,
    time_in_range_percent,
    time_in_ranges,
    time_outside_range_minutes,
    time_outside_range_percent,
    variability_metrics,
)

# Feature map builders
from .features import (  # noqa: F401
    build_feature_map,
    build_feature_map_wide,
    build_feature_table,
    build_feature_vector,
)

# Window creation
from .windows import create_sliding_windows, pivot_windows_wide  # noqa: F401

# Registry convenience
from .registry import (  # noqa: F401
    get_feature,
    get_feature_metadata,
    get_feature_names,
    list_features,
)

# IO
from .io import load_cgm_file, load_cgm_folder, load_csv  # noqa: F401

# Preprocessing
from .preprocessing import (  # noqa: F401
    clean_cgm,
    convert_units,
    detect_gaps,
    interpolate_cgm,
    resample_cgm,
    standardize_columns,
    validate_cgm,
)

# Event detection
from .detect import (  # noqa: F401
    detect_hyperglycemia,
    detect_hypoglycemia,
    detect_nocturnal_events,
    detect_postprandial_excursions,
)

# Reporting
from .report import generate_summary_report  # noqa: F401

__all__ = [
    "__version__",
    # Submodules
    "schemas", "io", "preprocessing", "metrics", "registry",
    "features", "windows", "detect", "plotting", "report",
    # Data structures
    "PreparedCGMData", "prepare",
    # Metrics
    "mean_glucose", "median_glucose", "min_glucose", "max_glucose",
    "q1_glucose", "q3_glucose", "sd", "cv", "j_index", "mage", "conga24",
    "time_in_range_minutes", "time_in_range_percent",
    "time_below_range_minutes", "time_below_range_percent",
    "time_above_range_minutes", "time_above_range_percent",
    "time_outside_range_minutes", "time_outside_range_percent",
    "lbgi", "hbgi", "adrr", "gri",
    "mean_glucose_excursion", "mean_glucose_normal",
    "count_peaks", "count_peaks_in_range",
    "basic_stats", "variability_metrics", "time_in_ranges",
    "risk_indices", "summary_dict",
    # Feature maps
    "build_feature_vector", "build_feature_map",
    "build_feature_map_wide", "build_feature_table",
    # Windows
    "create_sliding_windows", "pivot_windows_wide",
    # Registry
    "list_features", "get_feature", "get_feature_names", "get_feature_metadata",
    # IO
    "load_csv", "load_cgm_folder", "load_cgm_file",
    # Preprocessing
    "standardize_columns", "clean_cgm", "validate_cgm", "detect_gaps",
    "resample_cgm", "interpolate_cgm", "convert_units",
    # Detection
    "detect_hypoglycemia", "detect_hyperglycemia",
    "detect_nocturnal_events", "detect_postprandial_excursions",
    # Report
    "generate_summary_report",
]
