<p align="center">
  <img src="assets/logo.png" alt="GlycoSignal" width="320"/>
</p>

<p align="center">
  <a href="https://pypi.org/project/GlycoSignal/"><img src="https://img.shields.io/pypi/v/GlycoSignal?color=blue" alt="PyPI version"/></a>
  <a href="https://pypi.org/project/GlycoSignal/"><img src="https://img.shields.io/pypi/pyversions/GlycoSignal" alt="Python versions"/></a>
  <a href="https://github.com/glycosignal/glycosignal/actions/workflows/tests.yml"><img src="https://github.com/glycosignal/glycosignal/actions/workflows/tests.yml/badge.svg" alt="Tests"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License: MIT"/></a>
</p>

# GlycoSignal

CGM analysis for Python. Compute any glycemic metric individually, or generate ML-ready feature matrices from sliding windows.

---

## Installation

```bash
pip install GlycoSignal
```

Optional extras:

```bash
pip install "GlycoSignal[dev]"     # pytest, black, ruff, mypy
pip install "GlycoSignal[report]"  # HTML reports (adds Jinja2)
```

---

## Input Data Format

GlycoSignal reads **CSV files**. Your CSV needs at minimum a timestamp column and a glucose column.

| Column | Type | Required | Description |
|--------|------|----------|-------------|
| `Timestamp` | datetime | Yes | Reading timestamp (any format `pandas` can parse) |
| `Glucose` | float | Yes | Glucose value in mg/dL |
| `subject` | string | For multi-subject files | Subject or patient identifier |

**Column names are auto-detected.** You do not need to rename your columns before loading. GlycoSignal recognizes these common names (case-insensitive):

- Timestamp: `Timestamp`, `time`, `datetime`, `date_time`, `date`
- Glucose: `Glucose`, `Glucose Value (mg/dL)`, `gl`, `sgv`, `glucose_mg_dl`, `bg`, `blood_glucose`
- Subject: `subject`, `id`, `ptid`, `patient_id`, `subjectid`

If your column names are not recognized, pass them explicitly:

```python
df = glycosignal.load_csv("data.csv", timestamp_col="time_utc", glucose_col="bg_mg_dl")
```

**Example CSV (single subject):**

```
Timestamp,Glucose
2024-01-15 08:00:00,123
2024-01-15 08:05:00,121
2024-01-15 08:10:00,125
```

**Example CSV (multiple subjects in one file):**

```
Timestamp,Glucose,subject
2024-01-15 08:00:00,123,P001
2024-01-15 08:00:00,135,P002
2024-01-15 08:05:00,121,P001
2024-01-15 08:05:00,140,P002
```

### Working with multiple subjects

If your data has multiple subjects in **one file**, use `load_cgm_file` and specify which column identifies each subject:

```python
from glycosignal import io

df = io.load_cgm_file("all_subjects.csv", subject_col="ptid")
```

If each subject is in a **separate CSV file** inside a folder, use `load_cgm_folder`. It auto-derives a `subject` column from each filename:

```python
df = io.load_cgm_folder("data/subjects/")
# Adds 'subject' and 'filename' columns automatically
```

When building sliding windows or feature maps from multi-subject data, use `group_col` to process each subject independently:

```python
from glycosignal import windows, features

result = windows.create_sliding_windows(df, window_hours=24, group_col="subject")
X = features.build_feature_map(result.windows)
# X contains rows for all subjects, with a 'subject' column preserved
```

### Unit conversion

GlycoSignal expects glucose in **mg/dL**. If your data is in mmol/L, convert first:

```python
from glycosignal import preprocessing

df = preprocessing.convert_units(df, from_unit="mmol/L", to_unit="mg/dL")
```

### Device-specific loaders

For Dexcom and FreeStyle Libre exports (which have non-standard headers), use the dedicated loaders:

```python
df = io.load_dexcom("dexcom_export.csv")     # skips Dexcom header row
df = io.load_libre("libre_export.csv")       # skips Libre 2-row header
```

---

## What you can do in 30 seconds

```python
import glycosignal

df = glycosignal.load_csv("examples/sample_cgm.csv")  # included in repo
df = glycosignal.clean_cgm(df)

# One metric
print(glycosignal.mean_glucose(df))        # 128.4
print(glycosignal.time_in_range_percent(df, low=70, high=180))  # 81.2

# Full feature matrix (20+ features, one row per 24h window)
result = glycosignal.create_sliding_windows(df, window_hours=24)
X = glycosignal.build_feature_map(result.windows)
print(X.shape)  # (7, 23)
```

---

## Core Workflows

### A. Compute glycemic metrics

Every metric is a standalone function. Call them on any cleaned DataFrame or pass a pre-prepared object for efficiency.

```python
from glycosignal import metrics

metrics.mean_glucose(df)                           # 128.4
metrics.cv(df)                                     # 26.6 (%)
metrics.time_in_range_percent(df, low=70, high=180)  # 81.2
metrics.lbgi(df)                                   # 0.8
metrics.mage(df)                                   # 45.3
metrics.gri(df)                                    # 12.4
```

Group them when you need everything at once:

```python
metrics.basic_stats(df)
# {'mean': 128.4, 'median': 126.0, 'min': 62.0, 'max': 248.0, 'q1': 102.0, 'q3': 155.0}

metrics.variability_metrics(df)
# {'sd': 34.2, 'cv': 26.6, 'j_index': 26.7, 'mage': 45.8}

metrics.risk_indices(df)
# {'lbgi': 0.8, 'hbgi': 3.2, 'adrr': 14.1, 'gri': 12.4}

metrics.summary_dict(df)   # all of the above combined
```

**Performance tip:** Call `prepare()` once when computing many features on the same data:

```python
from glycosignal.schemas import prepare

p = prepare(df)
metrics.mean_glucose(p)
metrics.cv(p)
metrics.lbgi(p)
```

#### Callable metric functions

These are called directly from the `metrics` module and accept parameters. All accept a DataFrame or `PreparedCGMData`.

| Feature | Description | Computation |
|---|---|---|
| **Basic stats** | | |
| `mean_glucose(data)` | Mean BGL | μ = (1/N) Σ Xᵢ |
| `median_glucose(data)` | Median BGL | Middle value of sorted readings |
| `min_glucose(data)` | Minimum BGL | Min(X₁, ..., Xₙ) |
| `max_glucose(data)` | Maximum BGL | Max(X₁, ..., Xₙ) |
| `q1_glucose(data)` | First quartile of BGL | Q1 = Percentile(X, 25) |
| `q3_glucose(data)` | Third quartile of BGL | Q3 = Percentile(X, 75) |
| **Variability** | | |
| `sd(data)` | Standard deviation of BGL | σ = √(Σ(Xᵢ - μ)² / N) |
| `cv(data)` | Coefficient of variation | CV = (σ / μ) × 100 |
| `j_index(data)` | J-index (glycemic variability) | J = 0.001 × (μ + σ)² |
| `mage(data)` | Mean Amplitude of Glucose Excursions | Mean of alternating peak-nadir amplitudes exceeding σ |
| `conga24(data)` | Continuous Overall Net Glycemic Action | SD of {G(t) - G(t - 24h)} for all matched pairs |
| **Time-in-range (parametric)** | | |
| `time_in_range_minutes(data, low, high)` | BGL time inside [low, high] | TIR = Δt × Σ(low ≤ BGL(t) ≤ high) |
| `time_in_range_percent(data, low, high)` | Percent time inside [low, high] | TIR% = (TIR / T) × 100 |
| `time_below_range_minutes(data, threshold)` | BGL time below threshold | TBR = Δt × Σ(BGL(t) ≤ threshold) |
| `time_below_range_percent(data, threshold)` | Percent time below threshold | TBR% = (TBR / T) × 100 |
| `time_above_range_minutes(data, threshold)` | BGL time above threshold | TAR = Δt × Σ(BGL(t) ≥ threshold) |
| `time_above_range_percent(data, threshold)` | Percent time above threshold | TAR% = (TAR / T) × 100 |
| `time_outside_range_minutes(data, low, high)` | BGL time outside [low, high] | TOR = Δt × Σ(BGL(t) < low or BGL(t) > high) |
| `time_outside_range_percent(data, low, high)` | Percent time outside [low, high] | TOR% = (TOR / T) × 100 |
| **Risk indices** | | |
| `lbgi(data)` | Low Blood Glucose Index | LBGI = (1/N) Σ rl(Xᵢ); f(X) = ln(X)^1.084 - 5.381; rl = 22.77 × f² if f ≤ 0 |
| `hbgi(data)` | High Blood Glucose Index | HBGI = (1/N) Σ rh(Xᵢ); rh = 22.77 × f² if f > 0 |
| `adrr(data)` | Average Daily Risk Range | ADRR = Max(rl) + Max(rh) |
| `gri(data)` | Glucose Risk Index | GRI = 3.0×%TBR₅₄ + 2.4×%TBR₇₀ + 1.6×%TAR₂₅₀ + 0.8×%TAR₁₈₀, capped at 100 |
| **Excursions** | | |
| `mean_glucose_excursion(data)` | Mean BGL outside mean ± SD | Mean of Xᵢ where Xᵢ < μ - σ or Xᵢ > μ + σ |
| `mean_glucose_normal(data)` | Mean BGL inside mean ± SD | Mean of Xᵢ where μ - σ ≤ Xᵢ ≤ μ + σ |
| **Peak counts** | | |
| `count_peaks(data, threshold)` | Episodes above threshold | Count of rising-edge crossings above threshold |
| `count_peaks_in_range(data, lower, upper)` | Episodes entering [lower, upper] | Count of rising-edge entries into [lower, upper] |

> **Notation:** N = readings, Xᵢ = glucose value, μ = mean, σ = SD, Δt = interval between readings, T = total monitoring time.

---

#### Registry feature names

These are the 32 fixed-parameter feature names used in `build_feature_map(feature_names=[...])`. They are pre-wired to specific clinical thresholds and require no parameters.

```python
glycosignal.list_features()           # all 32 names
glycosignal.list_features(category="time_in_range")  # filtered by category
```

| Feature name | Category | Description |
|---|---|---|
| `mean_glucose` | basic_stats | Mean glucose (mg/dL) |
| `median_glucose` | basic_stats | Median glucose (mg/dL) |
| `min_glucose` | basic_stats | Minimum glucose (mg/dL) |
| `max_glucose` | basic_stats | Maximum glucose (mg/dL) |
| `q1_glucose` | basic_stats | 25th percentile glucose (mg/dL) |
| `q3_glucose` | basic_stats | 75th percentile glucose (mg/dL) |
| `sd` | variability | Standard deviation of glucose (mg/dL) |
| `cv` | variability | Coefficient of variation (%) |
| `j_index` | variability | J-index: 0.001 × (mean + SD)² |
| `mage` | variability | Mean Amplitude of Glucose Excursions (mg/dL) |
| `conga24` | variability | SD of glucose differences 24h apart (mg/dL) |
| `tir_70_180_min` | time_in_range | Minutes in target range 70–180 mg/dL |
| `tir_70_180_pct` | time_in_range | Percent time in target range 70–180 mg/dL |
| `tir_70_140_min` | time_in_range | Minutes in tight range 70–140 mg/dL |
| `tir_70_140_pct` | time_in_range | Percent time in tight range 70–140 mg/dL |
| `tbr_70_min` | time_in_range | Minutes below 70 mg/dL (level 1 hypoglycemia) |
| `tbr_70_pct` | time_in_range | Percent time below 70 mg/dL |
| `tbr_54_min` | time_in_range | Minutes below 54 mg/dL (level 2 hypoglycemia) |
| `tbr_54_pct` | time_in_range | Percent time below 54 mg/dL |
| `tar_180_min` | time_in_range | Minutes above 180 mg/dL (level 1 hyperglycemia) |
| `tar_180_pct` | time_in_range | Percent time above 180 mg/dL |
| `tar_250_min` | time_in_range | Minutes above 250 mg/dL (level 2 hyperglycemia) |
| `tar_250_pct` | time_in_range | Percent time above 250 mg/dL |
| `lbgi` | risk | Low Blood Glucose Index |
| `hbgi` | risk | High Blood Glucose Index |
| `adrr` | risk | Average Daily Risk Range |
| `gri` | risk | Glucose Risk Index (Klonoff et al. 2023) |
| `mean_glucose_excursion` | excursion | Mean of readings outside mean ± 1SD (mg/dL) |
| `mean_glucose_normal` | excursion | Mean of readings inside mean ± 1SD (mg/dL) |
| `peaks_above_140` | peak | Excursion episodes above 140 mg/dL (count) |
| `peaks_above_180` | peak | Excursion episodes above 180 mg/dL (count) |
| `peaks_above_250` | peak | Severe hyperglycemic episodes above 250 mg/dL (count) |

---

### B. Build ML feature matrices

The recommended pipeline: clean data, create windows, extract features, train.

```python
import glycosignal
from glycosignal import windows, features
from sklearn.ensemble import RandomForestClassifier

df = glycosignal.load_csv("cgm.csv")
df = glycosignal.clean_cgm(df)

result = windows.create_sliding_windows(df, window_hours=24, overlap_hours=0)
X = features.build_feature_map(result.windows)
# X has one row per window, 20+ feature columns

feature_cols = [c for c in X.columns if c not in ("window_id", "subject", "date")]
clf = RandomForestClassifier()
clf.fit(X[feature_cols], y)
```

Select a specific subset of features:

```python
X = features.build_feature_map(
    result.windows,
    feature_names=["mean_glucose", "cv", "tir_70_180_pct", "mage", "lbgi"],
)
```

Build a feature vector for a single window:

```python
features.build_feature_vector(window_df, feature_names=["mean_glucose", "cv"])
# {'mean_glucose': 128.4, 'cv': 26.6}
```

Build a table from a list of DataFrames (one per subject):

```python
features.build_feature_table(
    [df_s01, df_s02, df_s03],
    record_ids=["S01", "S02", "S03"],
)
```

---

## Conceptual Model

```
CSV / DataFrame
    -> io.load_csv() / load_cgm_folder()
    -> preprocessing.standardize_columns() + clean_cgm()
    -> metrics.*()              # individual features, any time
    -> windows.create_sliding_windows()
    -> features.build_feature_map()
    -> ML model
```

- **Timestamps and glucose values** are the only required columns.
- **Preprocessing** is explicit: nothing is cleaned silently.
- **Metrics** accept a raw DataFrame or a pre-prepared object interchangeably.
- **Windows** output long-format rows with a `window_id` column.
- **Feature maps** are plain DataFrames, ready for scikit-learn or any other tool.
- **Registry** is inspectable: every feature has a name, description, and category.

---

## Feature System

GlycoSignal has 32 built-in features organized by category. You can call each individually, use grouped helpers, or let the registry compute them in bulk.

```python
import glycosignal

glycosignal.list_features()
# ['adrr', 'conga24', 'cv', 'gri', 'hbgi', 'j_index', 'lbgi', 'mage', ...]

glycosignal.list_features(category="risk")
# ['adrr', 'gri', 'hbgi', 'lbgi']

glycosignal.get_feature_metadata()
# DataFrame: name | description | category | output_type

glycosignal.get_feature("gri").description
# 'Glucose Risk Index (Klonoff et al. 2023)'
```

To register a custom feature:

```python
from glycosignal.registry import DEFAULT_REGISTRY

DEFAULT_REGISTRY.register(
    name="my_metric",
    func=my_function,
    description="Custom metric",
    category="variability",
)
```

---

## Additional Capabilities

### Data loading

Loads CSVs with automatic column detection. Supports per-subject folders, multi-subject files, Dexcom, and Libre exports.

```python
from glycosignal import io

df = io.load_csv("data.csv")
df = io.load_csv("data.csv", timestamp_col="time_utc", glucose_col="bg_mg_dl")
df = io.load_cgm_folder("data/subjects/")     # adds filename + subject columns
df = io.load_cgm_file("all.csv", subject_col="ptid")
df = io.load_dexcom("dexcom_export.csv")
df = io.load_libre("libre_export.csv")
```

### Preprocessing

All functions return cleaned copies. Nothing is modified in place.

```python
from glycosignal import preprocessing

df = preprocessing.standardize_columns(df)          # rename "gl", "time", etc.
df = preprocessing.clean_cgm(df)                    # drop NaN, sort, enforce positive
report = preprocessing.validate_cgm(df)             # structured quality report
gaps = preprocessing.detect_gaps(df)                # DataFrame of gap intervals
df = preprocessing.resample_cgm(df, freq="5min")    # regular grid
df = preprocessing.interpolate_cgm(df, method="pchip", max_gap_points=12)
df = preprocessing.convert_units(df, from_unit="mmol/L", to_unit="mg/dL")
```

### Event detection

Returns a DataFrame of episodes with `start_time`, `end_time`, `duration_minutes`, and `event_type`.

```python
from glycosignal import detect

detect.detect_hypoglycemia(df, threshold=70, min_duration_minutes=15)
detect.detect_hyperglycemia(df, threshold=180, min_duration_minutes=15)
detect.detect_nocturnal_events(df, start_hour=0, end_hour=6)
detect.detect_postprandial_excursions(df, rise_threshold=50)
```

### Plotting

All functions return `(fig, ax)` and never call `plt.show()`.

```python
from glycosignal import plotting

fig, ax = plotting.plot_glucose_timeseries(df, subject="P001")
fig, ax = plotting.plot_daily_overlay(df)
fig, ax = plotting.plot_agp(df)
fig, ax = plotting.plot_histogram(df)
```

### Reporting

Generates a self-contained HTML report with summary metrics, TIR, risk indices, and embedded plots.

```python
from glycosignal import report

report.generate_summary_report(df, output_path="cgm_report.html")
```

---

## CLI

```bash
glycosignal summary data.csv
glycosignal windows data.csv --window-hours 24 --overlap-hours 0 --output windows.csv
glycosignal features windows.csv --output features.csv
glycosignal features windows.csv --features mean_glucose,cv,lbgi,gri
glycosignal report data.csv --output report.html
glycosignal list-features
glycosignal list-features --category risk
```

---

## Migration from Script Version

| Old script | New location | Key change |
|---|---|---|
| `glycosignal.py` | `glycosignal.metrics` + `glycosignal.schemas` | Lowercase names; uppercase aliases (`LBGI`, `MAGE`, `TIR`) still work |
| `cgm_sliding_window.py` (loaders) | `glycosignal.io` | Column renamed to `Glucose` from `Glucose Value (mg/dL)` |
| `cgm_sliding_window.py` (windowing) | `glycosignal.windows` | Output is long-format; use `pivot_windows_wide()` for the old format |
| `cgm_feature_map.py` `create_feature_map()` | `glycosignal.features.build_feature_map_wide()` | Long-format preferred via `build_feature_map()` |
| `cgm_feature_map.py` `FEATURES` list | `glycosignal.registry.DEFAULT_REGISTRY` | Structured registry with metadata |

**Metric calls are backward-compatible:**

```python
# Old
import glycosignal as gs
gs.mean_glucose(df)

# New (identical result)
import glycosignal
glycosignal.mean_glucose(df)
```

**Feature map migration:**

```python
# Old
from cgm_feature_map import create_feature_map
X = create_feature_map(windows_df)

# New (wide-format still works)
from glycosignal.features import build_feature_map_wide
X = build_feature_map_wide(windows_df)

# New preferred (long-format pipeline)
from glycosignal import windows, features
result = windows.create_sliding_windows(df)
X = features.build_feature_map(result.windows)
```

**Folder loading migration:**

```python
# Old
from cgm_sliding_window import load_cgm_folder
df = load_cgm_folder("Data/Processed")

# New
from glycosignal import io, preprocessing
df = io.load_cgm_folder("Data/Processed")
df = preprocessing.standardize_columns(df)
df = preprocessing.clean_cgm(df)
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

MIT. See [LICENSE](LICENSE).
