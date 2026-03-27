<p align="center">
  <img src="https://raw.githubusercontent.com/StuartSong/GlycoSignal/main/assets/logo.png" alt="GlycoSignal" width="320"/>
</p>

<p align="center">
  <a href="https://pypi.org/project/GlycoSignal/"><img src="https://img.shields.io/pypi/v/glycosignal?color=blue" alt="PyPI version"/></a>
  <a href="https://pypi.org/project/GlycoSignal/"><img src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue" alt="Python versions"/></a>
  <a href="https://github.com/StuartSong/GlycoSignal/actions/workflows/tests.yml"><img src="https://github.com/StuartSong/GlycoSignal/actions/workflows/tests.yml/badge.svg" alt="Tests"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License: MIT"/></a>
</p>

# GlycoSignal

Analyze continuous glucose monitor (CGM) data in Python with individually callable glycemic metrics, sliding-window pipelines, and ML-ready feature matrices.

---

## Table of Contents

- [Installation](#installation)
- [Input Data Format](#input-data-format)
  - [Single subject](#single-subject)
  - [Multiple subjects](#multiple-subjects)
  - [Unit conversion](#unit-conversion)
  - [Device-specific loaders](#device-specific-loaders)
- [Quickstart](#quickstart)
- [Computing Glycemic Metrics](#computing-glycemic-metrics)
  - [Individual metrics](#individual-metrics)
  - [Grouped summaries](#grouped-summaries)
  - [Full metric reference](#full-metric-reference)
- [Building ML Feature Matrices](#building-ml-feature-matrices)
  - [Feature registry](#feature-registry)
- [Additional Capabilities](#additional-capabilities)
  - [Preprocessing](#preprocessing)
  - [Event detection](#event-detection)
  - [Plotting](#plotting)
  - [Reporting](#reporting)
  - [Command-line interface](#command-line-interface)
- [License](#license)

---

## Installation

```bash
pip install GlycoSignal
```

Optional extras:

```bash
pip install "GlycoSignal[report]"  # HTML reports (adds Jinja2)
```

---

## Input Data Format

GlycoSignal reads **CSV files**. The only required columns are a timestamp and a glucose value.

| Column | Type | Required | Description |
|--------|------|----------|-------------|
| `Timestamp` | datetime | Yes | Reading timestamp (any format pandas can parse) |
| `Glucose` | float | Yes | Glucose value in mg/dL |
| `subject` | string | Multi-subject only | Subject or patient identifier |

**Column names are auto-detected** (case-insensitive). Recognized alternatives:

- Timestamp: `Timestamp`, `time`, `datetime`, `date_time`, `date`
- Glucose: `Glucose`, `Glucose Value (mg/dL)`, `gl`, `sgv`, `glucose_mg_dl`, `bg`, `blood_glucose`
- Subject: `subject`, `id`, `ptid`, `patient_id`, `subjectid`

If your column names are not recognized, pass them explicitly:

```python
df = glycosignal.load_csv("data.csv", timestamp_col="time_utc", glucose_col="bg_mg_dl")
```

### Single subject

```
Timestamp,Glucose
2024-01-15 08:00:00,123
2024-01-15 08:05:00,121
2024-01-15 08:10:00,125
```

### Multiple subjects

**One file with a subject column:**

```
Timestamp,Glucose,subject
2024-01-15 08:00:00,123,P001
2024-01-15 08:00:00,135,P002
2024-01-15 08:05:00,121,P001
```

```python
from glycosignal import io

df = io.load_cgm_file("all_subjects.csv", subject_col="ptid")
```

**One CSV per subject in a folder** (subject column derived from filename):

```python
df = io.load_cgm_folder("data/subjects/")
```

When windowing or building feature maps from multi-subject data, pass `group_col`:

```python
from glycosignal import windows, features

result = windows.create_sliding_windows(df, window_hours=24, group_col="subject")
X = features.build_feature_map(result.windows)
```

### Unit conversion

GlycoSignal expects glucose in **mg/dL**. Convert mmol/L first:

```python
from glycosignal import preprocessing

df = preprocessing.convert_units(df, from_unit="mmol/L", to_unit="mg/dL")
```

### Device-specific loaders

```python
df = io.load_dexcom("dexcom_export.csv")   # skips Dexcom header row
df = io.load_libre("libre_export.csv")     # skips Libre 2-row header
```

---

## Quickstart

```python
import glycosignal

df = glycosignal.load_csv("examples/sample_cgm.csv")  # sample file included in repo
df = glycosignal.clean_cgm(df)

# One metric
print(glycosignal.mean_glucose(df))                          # 138.5
print(glycosignal.time_in_range_percent(df, low=70, high=180))  # 93.1

# Full feature matrix (32 features, one row per 24h window)
result = glycosignal.create_sliding_windows(df, window_hours=24)
X = glycosignal.build_feature_map(result.windows)
print(X.shape)
```

---

## Computing Glycemic Metrics

### Individual metrics

Every metric is a standalone function. Call directly on any cleaned DataFrame:

```python
from glycosignal import metrics

metrics.mean_glucose(df)                             # 138.5
metrics.cv(df)                                       # 17.7 (%)
metrics.time_in_range_percent(df, low=70, high=180)  # 93.1
metrics.lbgi(df)                                     # 0.01
metrics.mage(df)                                     # 27.0
metrics.gri(df)                                      # 7.2
```

### Grouped summaries

```python
metrics.basic_stats(df)
# {'mean': 138.5, 'median': 132.5, 'min': 102.0, 'max': 193.0, 'q1': 117.0, 'q3': 158.25}

metrics.variability_metrics(df)
# {'sd': 24.5, 'cv': 17.7, 'j_index': 26.6, 'mage': 27.0}

metrics.risk_indices(df)
# {'lbgi': 0.01, 'hbgi': 2.3, 'adrr': 10.5, 'gri': 7.2}

metrics.summary_dict(df)    # all of the above in one dict
```

**Performance tip:** Call `prepare()` once when computing many metrics on the same data:

```python
from glycosignal.schemas import prepare

p = prepare(df)
metrics.mean_glucose(p)
metrics.cv(p)
metrics.lbgi(p)
```

### Full metric reference

#### Callable metric functions

All functions accept a DataFrame or `PreparedCGMData` object.

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
| `j_index(data)` | J-index | J = 0.001 × (μ + σ)² |
| `mage(data)` | Mean Amplitude of Glucose Excursions | Mean of alternating peak-nadir amplitudes exceeding σ |
| `conga24(data)` | Continuous Overall Net Glycemic Action | SD of {G(t) - G(t - 24h)} for all matched pairs |
| **Time-in-range** | | |
| `time_in_range_minutes(data, low, high)` | Minutes inside [low, high] | TIR = Δt × Σ(low ≤ BGL(t) ≤ high) |
| `time_in_range_percent(data, low, high)` | Percent time inside [low, high] | TIR% = (TIR / T) × 100 |
| `time_below_range_minutes(data, threshold)` | Minutes below threshold | TBR = Δt × Σ(BGL(t) ≤ threshold) |
| `time_below_range_percent(data, threshold)` | Percent time below threshold | TBR% = (TBR / T) × 100 |
| `time_above_range_minutes(data, threshold)` | Minutes above threshold | TAR = Δt × Σ(BGL(t) ≥ threshold) |
| `time_above_range_percent(data, threshold)` | Percent time above threshold | TAR% = (TAR / T) × 100 |
| `time_outside_range_minutes(data, low, high)` | Minutes outside [low, high] | TOR = Δt × Σ(BGL < low or BGL > high) |
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

> N = readings, Xᵢ = glucose value, μ = mean, σ = SD, Δt = interval between readings, T = total monitoring time.

---

## Building ML Feature Matrices

The full pipeline: load, clean, window, extract features, train.

```python
import glycosignal
from glycosignal import windows, features
from sklearn.ensemble import RandomForestClassifier

df = glycosignal.load_csv("cgm.csv")
df = glycosignal.clean_cgm(df)

result = windows.create_sliding_windows(df, window_hours=24, overlap_hours=0)
X = features.build_feature_map(result.windows)

feature_cols = [c for c in X.columns if c not in ("window_id", "subject", "date")]
clf = RandomForestClassifier()
clf.fit(X[feature_cols], y)
```

Select specific features:

```python
X = features.build_feature_map(
    result.windows,
    feature_names=["mean_glucose", "cv", "tir_70_180_pct", "mage", "lbgi"],
)
```

Feature vector for a single window:

```python
features.build_feature_vector(window_df, feature_names=["mean_glucose", "cv"])
# {'mean_glucose': 138.5, 'cv': 17.7}
```

Feature table from a list of DataFrames (one per subject):

```python
features.build_feature_table(
    [df_s01, df_s02, df_s03],
    record_ids=["S01", "S02", "S03"],
)
```

### Feature registry

GlycoSignal has 32 built-in features organized by category, pre-wired to standard clinical thresholds.

```python
glycosignal.list_features()                         # all 32 names
glycosignal.list_features(category="risk")          # ['adrr', 'gri', 'hbgi', 'lbgi']
glycosignal.get_feature_metadata()                  # DataFrame: name | description | category
glycosignal.get_feature("gri").description          # 'Glucose Risk Index (Klonoff et al. 2023)'
```

| Feature name | Category | Description |
|---|---|---|
| `mean_glucose` | basic_stats | Mean glucose (mg/dL) |
| `median_glucose` | basic_stats | Median glucose (mg/dL) |
| `min_glucose` | basic_stats | Minimum glucose (mg/dL) |
| `max_glucose` | basic_stats | Maximum glucose (mg/dL) |
| `q1_glucose` | basic_stats | 25th percentile glucose (mg/dL) |
| `q3_glucose` | basic_stats | 75th percentile glucose (mg/dL) |
| `sd` | variability | Standard deviation (mg/dL) |
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
| `peaks_above_140` | peak | Episodes above 140 mg/dL (count) |
| `peaks_above_180` | peak | Episodes above 180 mg/dL (count) |
| `peaks_above_250` | peak | Episodes above 250 mg/dL (count) |

Add a custom feature:

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

### Preprocessing

All functions return cleaned copies. Nothing is modified in place.

```python
from glycosignal import preprocessing

df = preprocessing.clean_cgm(df)                    # drop NaN, sort, enforce positive
report = preprocessing.validate_cgm(df)             # structured quality report
gaps = preprocessing.detect_gaps(df)                # DataFrame of gap intervals
df = preprocessing.resample_cgm(df, freq="5min")    # regular grid
df = preprocessing.interpolate_cgm(df, method="pchip", max_gap_points=12)
```

### Event detection

Returns a DataFrame with `start_time`, `end_time`, `duration_minutes`, and `event_type`.

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
fig.savefig("output.png", dpi=150)
```

### Reporting

Generates a self-contained HTML report with summary metrics, TIR, risk indices, and embedded plots.

```python
from glycosignal import report

report.generate_summary_report(df, output_path="cgm_report.html")
```

### Command-line interface

After installation, the `glycosignal` command is available from any terminal:

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

## License

MIT. Copyright (c) 2024 Jiafeng Song. See [LICENSE](LICENSE).
