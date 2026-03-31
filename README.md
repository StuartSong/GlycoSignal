<p align="center">
  <img src="https://raw.githubusercontent.com/StuartSong/GlycoSignal/main/assets/logo.png" alt="GlycoSignal" width="320"/>
</p>

<p align="center">
  <a href="https://pypi.org/project/GlycoSignal/"><img src="https://img.shields.io/badge/pypi-v0.1.3-blue" alt="PyPI version"/></a>
  <a href="https://pypi.org/project/GlycoSignal/"><img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python versions"/></a>
  <a href="https://github.com/StuartSong/GlycoSignal/actions/workflows/tests.yml"><img src="https://github.com/StuartSong/GlycoSignal/actions/workflows/tests.yml/badge.svg" alt="Tests"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License: MIT"/></a>
</p>

# GlycoSignal

Python library for CGM data analysis — individually callable glycemic metrics, day/sliding-window segmentation, and ML-ready feature matrices.

```bash
pip install GlycoSignal
```

---

## Quick start

```python
import glycosignal

df = glycosignal.load_csv("cgm.csv")
df = glycosignal.clean_cgm(df)

# Individual metrics
glycosignal.mean_glucose(df)                             # 138.5
glycosignal.time_in_range_percent(df, low=70, high=180)  # 93.1

# Segment → feature matrix  (the standard pipeline)
segs = glycosignal.create_day_segments(df)               # 24 h, midnight anchor
X    = glycosignal.build_feature_map(segs.windows)       # 32 features × n_days
```

---

## Input format

Two required columns; everything else is auto-detected (case-insensitive).

| Column | Required | Recognized aliases |
|--------|----------|--------------------|
| `Timestamp` | Yes | `time`, `datetime`, `date_time`, `date` |
| `Glucose` | Yes | `Glucose Value (mg/dL)`, `gl`, `sgv`, `glucose_mg_dl`, `bg` |
| `subject` | Multi-subject only | `id`, `ptid`, `patient_id`, `subjectid` |

```python
# Override auto-detection
df = glycosignal.load_csv("data.csv", timestamp_col="time_utc", glucose_col="bg_mg_dl")

# Multiple subjects — one file with subject column
df = glycosignal.io.load_cgm_file("all.csv", subject_col="ptid")

# Multiple subjects — one CSV per subject in a folder
df = glycosignal.io.load_cgm_folder("data/subjects/")
```

---

## Data segmentation

Both functions return `WindowResult(windows, metadata)`. The `windows` DataFrame flows directly into `build_feature_map()`.

### Daily segments

```python
from glycosignal import windows

segs = windows.create_day_segments(df)                    # midnight–midnight (default)
segs = windows.create_day_segments(df, anchor_time="08:00")             # 8 AM–8 AM
segs = windows.create_day_segments(df, anchor_time="08:00",
                                   min_fraction=0.7,      # drop days < 70 % coverage
                                   group_col="subject")   # multi-subject

print(segs.metadata)
# {'n_groups': 1, 'n_valid_windows': 7, 'n_discarded_partial_days': 0, ...}
```

### Sliding windows

```python
result = windows.create_sliding_windows(df, window_hours=6, step_hours=1)
result = windows.create_sliding_windows(df, window_hours=12, step_hours=12,
                                        anchor_time="08:00")
result = windows.create_sliding_windows(df, window_hours=24, overlap_hours=12)
result = windows.create_sliding_windows(df, window_hours=24, step_hours=6,
                                        group_col="subject")
```

Key parameters: `window_hours`, `step_hours`, `anchor_time`, `min_fraction` (default `0.0` — keep all windows), `group_col`, `interpolate`, `max_gap_points`.

### Output

The `windows` DataFrame is **long-format** — one row per (window, time-point):

| `window_id` | `subject` | `date` | `Timestamp` | `Glucose` |
|-------------|-----------|--------|-------------|-----------|
| `S01_2023-01-02` | S01 | 2023-01-02 | 2023-01-02 00:00 | 112.4 |

Non-midnight anchors append `_HHMM` to the window ID (e.g. `S01_2023-01-02_0800`).

### Saving as wide-format CSV

```python
wide = windows.pivot_windows_wide(segs.windows)
wide.to_csv("segments.csv", index=False)
# → columns: date | subject | 00:00 | 00:05 | … | 23:55
# For non-midnight anchor (e.g. "08:00"): 08:00 | … | 23:55 | 00:00 | … | 07:55
```

---

## Glycemic metrics

Every metric is a standalone function. See the **[full metric reference](https://github.com/StuartSong/GlycoSignal/blob/main/docs/METRICS.md)** for all formulas.

```python
from glycosignal import metrics

# Core
metrics.mean_glucose(df)                              # 138.5
metrics.median_glucose(df)
metrics.std_glucose(df)
metrics.cv(df)                                        # 17.7 %
metrics.time_in_range_percent(df, low=70, high=180)   # 93.1 %
metrics.time_above_range(df, threshold=180)
metrics.time_below_range(df, threshold=70)

# Variability
metrics.mage(df)                                      # 27.0
metrics.j_index(df)
metrics.conga(df)
metrics.modd(df)
metrics.grade(df)

# Risk indices
metrics.lbgi(df)
metrics.hbgi(df)
metrics.adrr(df)
metrics.gri(df)                                       # 7.2

# Grouped helpers — return dicts
metrics.basic_stats(df)          # mean, median, min, max, q1, q3
metrics.variability_metrics(df)  # sd, cv, j_index, mage
metrics.risk_indices(df)         # lbgi, hbgi, adrr, gri
metrics.summary_dict(df)         # all of the above combined
```

**Performance tip** — call `prepare()` once when computing many metrics on the same data:

```python
from glycosignal.schemas import prepare
p = prepare(df)
metrics.mean_glucose(p); metrics.cv(p); metrics.lbgi(p)
```

---

## Feature matrices

```python
from glycosignal import windows, features

segs = windows.create_day_segments(df)
X = features.build_feature_map(segs.windows)           # 32 features × n_windows

# Subset
X = features.build_feature_map(segs.windows,
    feature_names=["mean_glucose", "cv", "tir_70_180_pct", "mage", "lbgi"])

# Single window → dict
features.build_feature_vector(window_df, feature_names=["mean_glucose", "cv"])

# List of DataFrames (one per subject)
features.build_feature_table([df_s01, df_s02], record_ids=["S01", "S02"])
```

### Feature registry

```python
glycosignal.list_features()                      # all 32 names
glycosignal.list_features(category="risk")       # ['adrr', 'gri', 'hbgi', 'lbgi']
glycosignal.get_feature_metadata()               # DataFrame: name | description | category
glycosignal.get_feature("gri").description       # 'Glucose Risk Index (Klonoff et al. 2023)'

# Register a custom feature
from glycosignal.registry import DEFAULT_REGISTRY
DEFAULT_REGISTRY.register(name="my_metric", func=my_fn,
                           description="...", category="variability")
```

---

## Preprocessing

```python
from glycosignal import preprocessing

df     = preprocessing.clean_cgm(df)                              # drop NaN, sort, enforce positive
report = preprocessing.validate_cgm(df)                           # structured quality report
gaps   = preprocessing.detect_gaps(df)                            # DataFrame of gap intervals
df     = preprocessing.resample_cgm(df, freq="5min")              # regular grid
df     = preprocessing.interpolate_cgm(df, method="pchip",
                                        max_gap_points=12)
df     = preprocessing.convert_units(df, from_unit="mmol/L",
                                      to_unit="mg/dL")
```

---

## Event detection

Returns a DataFrame with `start_time`, `end_time`, `duration_minutes`, `event_type`.

```python
from glycosignal import detect

detect.detect_hypoglycemia(df, threshold=70, min_duration_minutes=15)
detect.detect_hyperglycemia(df, threshold=180, min_duration_minutes=15)
detect.detect_nocturnal_events(df, start_hour=0, end_hour=6)
detect.detect_postprandial_excursions(df, rise_threshold=50)
```

---

## Plotting

All functions return `(fig, ax)` and never call `plt.show()`.

```python
from glycosignal import plotting

fig, ax = plotting.plot_glucose_timeseries(df, subject="P001")
fig, ax = plotting.plot_daily_overlay(df)
fig, ax = plotting.plot_agp(df)
fig, ax = plotting.plot_histogram(df)
fig.savefig("output.png", dpi=150)
```

---

## CLI

```bash
glycosignal summary data.csv
glycosignal windows data.csv --window-hours 24 --overlap-hours 0 --output windows.csv
glycosignal features windows.csv --output features.csv
glycosignal features windows.csv --features mean_glucose,cv,lbgi,gri
glycosignal list-features
glycosignal list-features --category risk
```

---

## License

MIT. Copyright (c) 2024 Jiafeng Song. See [LICENSE](LICENSE).
