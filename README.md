# GlycoSignal

A production-quality Python package for continuous glucose monitor (CGM) data analysis,
glycemic feature computation, and ML-ready feature extraction.

GlycoSignal has two core promises:

1. **Every glycemic feature is directly callable** -- compute mean glucose, CV, MAGE, LBGI,
   time-in-range, and 30+ other metrics one at a time on any DataFrame.
2. **ML-ready feature maps in one call** -- generate a feature matrix from any number of
   CGM windows or subjects, ready for scikit-learn or any other ML pipeline.

---

## Installation

```bash
pip install GlycoSignal
```

For development (includes pytest, black, ruff, mypy):

```bash
pip install "GlycoSignal[dev]"
```

For HTML report generation (adds Jinja2):

```bash
pip install "GlycoSignal[report]"
```

---

## Quickstart

```python
import pandas as pd
import glycosignal

# Load your CGM data
df = glycosignal.load_csv("my_cgm_data.csv")
df = glycosignal.clean_cgm(df)

# Compute a single metric
mean = glycosignal.mean_glucose(df)
print(f"Mean glucose: {mean:.1f} mg/dL")

# Compute a full summary
summary = glycosignal.summary_dict(df)
print(summary)

# Generate a 24h windowed feature matrix for ML
result = glycosignal.create_sliding_windows(df, window_hours=24)
X = glycosignal.build_feature_map(result.windows)
print(X.shape)
```

---

## Individual Glycemic Features

Every metric is a standalone function. Import and call them directly:

```python
from glycosignal import metrics

# Basic statistics
mean   = metrics.mean_glucose(df)
median = metrics.median_glucose(df)
mn     = metrics.min_glucose(df)
mx     = metrics.max_glucose(df)
q1     = metrics.q1_glucose(df)
q3     = metrics.q3_glucose(df)

# Variability
std    = metrics.sd(df)
coef_v = metrics.cv(df)          # percent
j      = metrics.j_index(df)
m      = metrics.mage(df)
c24    = metrics.conga24(df)

# Time-in-range (minutes)
tir_min  = metrics.time_in_range_minutes(df, low=70, high=180)
tbr_min  = metrics.time_below_range_minutes(df, threshold=70)
tar_min  = metrics.time_above_range_minutes(df, threshold=180)

# Time-in-range (percent)
tir_pct  = metrics.time_in_range_percent(df, low=70, high=180)
tbr_pct  = metrics.time_below_range_percent(df, threshold=70)
tar_pct  = metrics.time_above_range_percent(df, threshold=180)

# Risk indices
low_risk  = metrics.lbgi(df)
high_risk = metrics.hbgi(df)
daily_risk= metrics.adrr(df)
gri_score = metrics.gri(df)
```

**Performance tip:** When computing many features on the same dataset, call
`prepare()` once:

```python
from glycosignal.schemas import prepare

prepared = prepare(df)          # validates once, builds time-weights
mean     = metrics.mean_glucose(prepared)
cv       = metrics.cv(prepared)
lbgi     = metrics.lbgi(prepared)
```

---

## Grouped Summaries

```python
from glycosignal import metrics

# Six-number summary
stats = metrics.basic_stats(df)
# {'mean': 128.4, 'median': 126.0, 'min': 62.0, 'max': 248.0, 'q1': 102.0, 'q3': 155.0}

# Variability metrics
var = metrics.variability_metrics(df)
# {'sd': 34.2, 'cv': 26.6, 'j_index': 26.7, 'mage': 45.8}

# Time-in-range for multiple ranges
tir = metrics.time_in_ranges(df, ranges=[(70, 180), (70, 140)])
# {'tir_70_180_min': 1152.0, 'tir_70_180_pct': 80.0, ...}

# Risk indices
risk = metrics.risk_indices(df)
# {'lbgi': 0.8, 'hbgi': 3.2, 'adrr': 14.1, 'gri': 12.4}

# Everything combined
everything = metrics.summary_dict(df)
```

---

## Feature Map Generation (for ML)

### Style A: Windowed feature map (recommended for ML)

```python
from glycosignal import windows, features

# Step 1: load and clean
df = glycosignal.load_csv("cgm.csv")
df = glycosignal.clean_cgm(df)

# Step 2: create 24-hour windows (no overlap)
result = windows.create_sliding_windows(df, window_hours=24, overlap_hours=0)
print(result.metadata)
# {'n_groups': 1, 'n_valid_windows': 7, 'n_discarded_partial_days': 1, ...}

# Step 3: compute all default features
X = features.build_feature_map(result.windows)
# DataFrame with 7 rows × (metadata + 20 features)

# Step 4: fit your ML model
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X.drop(columns=["window_id", "subject", "date"]), y)
```

### Style B: Select specific features

```python
X = features.build_feature_map(
    result.windows,
    feature_names=["mean_glucose", "cv", "tir_70_180_pct", "mage", "lbgi"],
)
```

### Style C: Feature vector for a single window

```python
from glycosignal import features

vec = features.build_feature_vector(window_df, feature_names=["mean_glucose", "cv"])
# {'mean_glucose': 128.4, 'cv': 26.6}
```

### Style D: Feature table from a list of DataFrames

```python
from glycosignal import features

records = [df_subject1, df_subject2, df_subject3]
X = features.build_feature_table(records, record_ids=["S01", "S02", "S03"])
```

---

## Feature Registry

Inspect the built-in feature catalogue:

```python
import glycosignal

# List all feature names
glycosignal.list_features()
# ['adrr', 'conga24', 'cv', 'gri', 'hbgi', 'j_index', 'lbgi', ...]

# Filter by category
glycosignal.list_features(category="time_in_range")
# ['tar_180_min', 'tar_180_pct', 'tbr_54_min', ...]

# Get full metadata
meta = glycosignal.get_feature_metadata()
# DataFrame with columns: name, description, category, output_type

# Inspect a specific feature
desc = glycosignal.get_feature("gri")
print(desc.description)
# 'Glucose Risk Index (Klonoff et al. 2023)'
```

---

## Data Loading

```python
from glycosignal import io

# Auto-detect column names
df = io.load_csv("data.csv")

# Explicit column names
df = io.load_csv("data.csv", timestamp_col="time_utc", glucose_col="bg_mg_dl")

# Load a folder of per-subject CSVs
df = io.load_cgm_folder("data/subjects/")
# Returns df with: Timestamp, Glucose, filename, subject

# Load a single multi-subject file
df = io.load_cgm_file("all_subjects.csv", subject_col="ptid")

# Dexcom and Libre presets
df = io.load_dexcom("dexcom_export.csv")
df = io.load_libre("libre_export.csv")
```

---

## Preprocessing

```python
from glycosignal import preprocessing

# Rename non-standard columns
df = preprocessing.standardize_columns(df)

# Clean (drop NaN, sort, enforce positive)
df = preprocessing.clean_cgm(df, drop_duplicates=True, sort=True)

# Validate and get a structured report
report = preprocessing.validate_cgm(df)
print(report.summary())

# Find gaps
gaps = preprocessing.detect_gaps(df, expected_interval_minutes=5)

# Resample to regular 5-min grid
df_resampled = preprocessing.resample_cgm(df, freq="5min", method="nearest")

# Interpolate short gaps (PCHIP)
df_interp = preprocessing.interpolate_cgm(df, method="pchip", max_gap_points=12)

# Unit conversion
df_mgdl = preprocessing.convert_units(df_mmol, from_unit="mmol/L", to_unit="mg/dL")
```

---

## Event Detection

```python
from glycosignal import detect

# Hypoglycemic episodes
hypo = detect.detect_hypoglycemia(df, threshold=70, min_duration_minutes=15)
# DataFrame: start_time, end_time, duration_minutes, nadir_glucose, event_type

# Hyperglycemic episodes
hyper = detect.detect_hyperglycemia(df, threshold=180, min_duration_minutes=15)

# Nocturnal events (midnight–6am)
nocturnal = detect.detect_nocturnal_events(df, start_hour=0, end_hour=6)

# Postprandial excursions (heuristic)
excursions = detect.detect_postprandial_excursions(df, rise_threshold=50)
```

---

## Plotting

```python
from glycosignal import plotting

# Time series with TIR bands
fig, ax = plotting.plot_glucose_timeseries(df, subject="P001")
fig.savefig("timeseries.png", dpi=150)

# Daily overlay (all days on one 24h axis)
fig, ax = plotting.plot_daily_overlay(df)

# Ambulatory Glucose Profile (AGP)
fig, ax = plotting.plot_agp(df)

# Glucose distribution histogram
fig, ax = plotting.plot_histogram(df, bins=50)
```

All plotting functions return `(fig, ax)` and never call `plt.show()`.

---

## Reporting

```python
from glycosignal import report

# Generate a self-contained HTML report
report.generate_summary_report(df, output_path="cgm_report.html")
```

---

## CLI

After installation, the `glycosignal` command is available:

```bash
# Print glycemic summary metrics
glycosignal summary data.csv

# Create sliding windows
glycosignal windows data.csv --window-hours 24 --overlap-hours 0 --output windows.csv

# Build feature map from windowed output
glycosignal features windows.csv --output features.csv

# Select specific features
glycosignal features windows.csv --features mean_glucose,cv,lbgi,gri

# Generate HTML report
glycosignal report data.csv --output report.html

# List all registered features
glycosignal list-features

# Filter features by category
glycosignal list-features --category risk
```

---

## Canonical Data Format

All GlycoSignal functions expect DataFrames with these columns:

| Column | Type | Description |
|--------|------|-------------|
| `Timestamp` | datetime | Reading timestamp |
| `Glucose` | float | Glucose in mg/dL |

Optional metadata columns: `subject`, `filename`, `sensor`, `date`, `window_id`.

Use `preprocessing.standardize_columns(df)` to rename common alternatives
(`"time"`, `"gl"`, `"Glucose Value (mg/dL)"`, etc.) to canonical names.

---

## Migration from Script Version

The original three scripts map to these new modules:

| Old script | New module(s) | What changed |
|------------|--------------|--------------|
| `glycosignal.py` | `glycosignal.metrics`, `glycosignal.schemas` | Functions renamed to lowercase snake_case; backward-compat aliases preserved (e.g. `LBGI`, `MAGE`, `TIR`) |
| `cgm_sliding_window.py` `load_cgm_folder()` / `load_cgm_file()` | `glycosignal.io` | Canonical column `Glucose` instead of `Glucose Value (mg/dL)` |
| `cgm_sliding_window.py` `create_sliding_windows()` | `glycosignal.windows` | Output is long-format (not wide); use `pivot_windows_wide()` for the old format |
| `cgm_feature_map.py` `create_feature_map()` | `glycosignal.features.build_feature_map_wide()` | Accepts wide-format input; `build_feature_map()` accepts long-format |
| `cgm_feature_map.py` `FEATURES` list | `glycosignal.registry.DEFAULT_REGISTRY` | Structured registry with metadata; extend with `DEFAULT_REGISTRY.register()` |

### Minimal migration example

```python
# Old script usage
import glycosignal as gs
result = gs.mean_glucose(df)

# New package usage (same result)
from glycosignal import metrics
result = metrics.mean_glucose(df)

# Or at top level
import glycosignal
result = glycosignal.mean_glucose(df)
```

```python
# Old: wide-format windowed CSV → feature map
from cgm_feature_map import create_feature_map
X = create_feature_map(windows_df)

# New: same wide-format input still works
from glycosignal.features import build_feature_map_wide
X = build_feature_map_wide(windows_df)

# New preferred: long-format pipeline
from glycosignal import windows, features
result = windows.create_sliding_windows(df)
X = features.build_feature_map(result.windows)
```

```python
# Old: load folder + create windows
from cgm_sliding_window import load_cgm_folder, create_sliding_windows
df = load_cgm_folder("Data/Processed")

# New
from glycosignal import io, preprocessing, windows
df = io.load_cgm_folder("Data/Processed")
df = preprocessing.standardize_columns(df)
df = preprocessing.clean_cgm(df)
result = windows.create_sliding_windows(df, group_col="filename")
```

---

## Development

```bash
git clone https://github.com/glycosignal/glycosignal
cd glycosignal
pip install -e ".[dev]"
pytest
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
