<p align="center">
  <img src="assets/logo.png" alt="GlycoSignal" width="320"/>
</p>

# GlycoSignal

CGM analysis for Python. Compute any glycemic metric individually, or generate ML-ready feature matrices from sliding windows.

---

## What you can do in 30 seconds

```python
import glycosignal

df = glycosignal.load_csv("cgm.csv")
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

#### All available metrics

| Feature | Description | Computation |
|---|---|---|
| `mean_glucose` | Mean BGL | μ = (1/N) Σ Xᵢ |
| `median_glucose` | Median BGL | Middle value of sorted readings |
| `min_glucose` | Minimum BGL | Min(X₁, ..., Xₙ) |
| `max_glucose` | Maximum BGL | Max(X₁, ..., Xₙ) |
| `q1_glucose` | First quartile of BGL | Q1 = Percentile(X, 25) |
| `q3_glucose` | Third quartile of BGL | Q3 = Percentile(X, 75) |
| `sd` | Standard deviation of BGL | σ = √(Σ(Xᵢ - μ)² / N) |
| `cv` | Coefficient of variation of BGL | CV = (σ / μ) × 100 |
| `j_index` | J-index (glycemic variability) | J = 0.001 × (μ + σ)² |
| `mage` | Mean Amplitude of Glucose Excursions | Mean of alternating peak-nadir amplitudes exceeding σ |
| `conga24` | Continuous Overall Net Glycemic Action | SD of {G(t) - G(t - 24h)} for all matched pairs |
| `time_in_range_minutes` | BGL time inside [low, high] | TIR = Δt × Σ(low ≤ BGL(t) ≤ high) |
| `time_in_range_percent` | Percent time inside [low, high] | TIR% = (TIR / T) × 100 |
| `time_below_range_minutes` | BGL time below threshold | TBR = Δt × Σ(BGL(t) ≤ threshold) |
| `time_below_range_percent` | Percent time below threshold | TBR% = (TBR / T) × 100 |
| `time_above_range_minutes` | BGL time above threshold | TAR = Δt × Σ(BGL(t) ≥ threshold) |
| `time_above_range_percent` | Percent time above threshold | TAR% = (TAR / T) × 100 |
| `lbgi` | Low Blood Glucose Index | LBGI = (1/N) Σ rl(Xᵢ); f(X) = ln(X)^1.084 - 5.381; rl = 22.77 × f² if f ≤ 0 |
| `hbgi` | High Blood Glucose Index | HBGI = (1/N) Σ rh(Xᵢ); rh = 22.77 × f² if f > 0 |
| `adrr` | Average Daily Risk Range | ADRR = Max(rl(X₁)...rl(Xₙ)) + Max(rh(X₁)...rh(Xₙ)) |
| `gri` | Glucose Risk Index | GRI = 3.0 × %TBR₅₄ + 2.4 × %TBR₇₀ + 1.6 × %TAR₂₅₀ + 0.8 × %TAR₁₈₀, capped at 100 |
| `mean_glucose_excursion` | Mean BGL outside mean ± SD | MGE = (1/K) Σ Xᵢ where Xᵢ < μ - σ or Xᵢ > μ + σ |
| `mean_glucose_normal` | Mean BGL inside mean ± SD | MGN = (1/K) Σ Xᵢ where μ - σ ≤ Xᵢ ≤ μ + σ |
| `count_peaks` | Episodes above threshold | Count of rising-edge crossings above threshold |
| `count_peaks_in_range` | Episodes entering [lower, upper] | Count of rising-edge entries into [lower, upper] |

> **Notation:** N = number of readings, Xᵢ = glucose reading, μ = mean, σ = standard deviation, Δt = time interval between readings, T = total monitoring time, BGL = blood glucose level. LBGI/HBGI share the transform f(X) = ln(X)^1.084 - 5.381 with rl/rh penalizing low/high values respectively.

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

## Canonical Data Format

All functions expect a DataFrame with these two columns:

| Column | Type | Description |
|--------|------|-------------|
| `Timestamp` | datetime | Reading timestamp |
| `Glucose` | float | Glucose in mg/dL |

Optional metadata columns: `subject`, `filename`, `sensor`, `date`, `window_id`.

`preprocessing.standardize_columns(df)` maps common alternatives (`"time"`, `"gl"`, `"Glucose Value (mg/dL)"`, etc.) to canonical names automatically.

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
