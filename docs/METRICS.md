# GlycoSignal — Full Metric Reference

All functions accept a `DataFrame` or `PreparedCGMData` object (from `glycosignal.schemas.prepare`).

> **N** = number of readings · **Xᵢ** = glucose value · **μ** = mean · **σ** = SD · **Δt** = interval between readings · **T** = total monitoring time

---

## Basic stats

| Function | Description | Computation |
|---|---|---|
| `mean_glucose(data)` | Mean BGL | μ = (1/N) Σ Xᵢ |
| `median_glucose(data)` | Median BGL | Middle value of sorted readings |
| `min_glucose(data)` | Minimum BGL | Min(X₁, …, Xₙ) |
| `max_glucose(data)` | Maximum BGL | Max(X₁, …, Xₙ) |
| `q1_glucose(data)` | First quartile of BGL | Q1 = Percentile(X, 25) |
| `q3_glucose(data)` | Third quartile of BGL | Q3 = Percentile(X, 75) |

---

## Variability

| Function | Description | Computation |
|---|---|---|
| `sd(data)` | Standard deviation of BGL | σ = √(Σ(Xᵢ - μ)² / N) |
| `cv(data)` | Coefficient of variation | CV = (σ / μ) × 100 |
| `j_index(data)` | J-index | J = 0.001 × (μ + σ)² |
| `mage(data)` | Mean Amplitude of Glucose Excursions | Mean of alternating peak-nadir amplitudes exceeding σ |
| `conga24(data)` | Continuous Overall Net Glycemic Action | SD of {G(t) − G(t − 24h)} for all matched pairs |

---

## Time-in-range

| Function | Description | Computation |
|---|---|---|
| `time_in_range_minutes(data, low, high)` | Minutes inside [low, high] | TIR = Δt × Σ(low ≤ BGL(t) ≤ high) |
| `time_in_range_percent(data, low, high)` | Percent time inside [low, high] | TIR% = (TIR / T) × 100 |
| `time_below_range_minutes(data, threshold)` | Minutes below threshold | TBR = Δt × Σ(BGL(t) ≤ threshold) |
| `time_below_range_percent(data, threshold)` | Percent time below threshold | TBR% = (TBR / T) × 100 |
| `time_above_range_minutes(data, threshold)` | Minutes above threshold | TAR = Δt × Σ(BGL(t) ≥ threshold) |
| `time_above_range_percent(data, threshold)` | Percent time above threshold | TAR% = (TAR / T) × 100 |
| `time_outside_range_minutes(data, low, high)` | Minutes outside [low, high] | TOR = Δt × Σ(BGL < low or BGL > high) |
| `time_outside_range_percent(data, low, high)` | Percent time outside [low, high] | TOR% = (TOR / T) × 100 |

---

## Risk indices

| Function | Description | Computation |
|---|---|---|
| `lbgi(data)` | Low Blood Glucose Index | LBGI = (1/N) Σ rl(Xᵢ); f(X) = ln(X)^1.084 − 5.381; rl = 22.77 × f² if f ≤ 0 |
| `hbgi(data)` | High Blood Glucose Index | HBGI = (1/N) Σ rh(Xᵢ); rh = 22.77 × f² if f > 0 |
| `adrr(data)` | Average Daily Risk Range | ADRR = Max(rl) + Max(rh) |
| `gri(data)` | Glucose Risk Index | GRI = 3.0×%TBR₅₄ + 2.4×%TBR₇₀ + 1.6×%TAR₂₅₀ + 0.8×%TAR₁₈₀, capped at 100 |

---

## Excursions

| Function | Description | Computation |
|---|---|---|
| `mean_glucose_excursion(data)` | Mean BGL outside mean ± SD | Mean of Xᵢ where Xᵢ < μ − σ or Xᵢ > μ + σ |
| `mean_glucose_normal(data)` | Mean BGL inside mean ± SD | Mean of Xᵢ where μ − σ ≤ Xᵢ ≤ μ + σ |

---

## Peak counts

| Function | Description | Computation |
|---|---|---|
| `count_peaks(data, threshold)` | Episodes above threshold | Count of rising-edge crossings above threshold |
| `count_peaks_in_range(data, lower, upper)` | Episodes entering [lower, upper] | Count of rising-edge entries into [lower, upper] |

---

## Grouped summary helpers

These convenience functions return multiple metrics at once as a dict.

| Function | Returns |
|---|---|
| `metrics.basic_stats(df)` | `mean`, `median`, `min`, `max`, `q1`, `q3` |
| `metrics.variability_metrics(df)` | `sd`, `cv`, `j_index`, `mage` |
| `metrics.risk_indices(df)` | `lbgi`, `hbgi`, `adrr`, `gri` |
| `metrics.summary_dict(df)` | All of the above combined |

---

## Performance tip

Call `prepare()` once when computing many metrics on the same data:

```python
from glycosignal.schemas import prepare

p = prepare(df)
metrics.mean_glucose(p)
metrics.cv(p)
metrics.lbgi(p)
```
