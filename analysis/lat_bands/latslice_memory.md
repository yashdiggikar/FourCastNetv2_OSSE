# Section 5.7 Lat-Slice Analysis — Memory File
## Reference for replicating the latitude-slice RMSE analysis on other models (Aurora, Pangu, etc.)

---

## Overview

This analysis computes cosine-latitude-weighted RMSE time series for FCNv2 (4 runs)
versus ERA5 truth, evaluated at 4 specific NH latitude slices (each ±5° around centre).
Figures LS.1–LS.5 are produced. Output in `FourCastNet/lat_bands/`.

This is the **high-resolution companion** to the broad-band analysis in `FourCastNet/lats/`.
The slice analysis reveals within-band variation (e.g., 2°N vs 14°N) that is hidden by
broad-band averaging.

---

## Directory Structure

```
FourCastNet/lat_bands/
├── values/                              # CSV files
│   ├── FCN_{run}_{var}_latslice_rmse.csv    # 20 files (4 runs × 5 vars)
│   └── FCN_latslice_pct_improvement_summary.csv
├── plots/                               # PNG figures
│   ├── Figure_LS1_FCN_latslice_setup.png
│   ├── Figure_LS2_FCN_t_insertion_latslice_rmse.png
│   ├── Figure_LS3_FCN_wind_insertion_latslice_rmse.png
│   ├── Figure_LS4_FCN_z_insertion_latslice_rmse.png
│   └── Figure_LS5_FCN_latslice_bar_summary.png
├── logs/
│   └── section_5_7_latslice_debug_log.txt
├── section_5_7_latslice_results_analysis.txt
└── latslice_memory.md   ← this file
```

Main script: `FourCastNet/section_5_7_latslice_analysis.py`

---

## Latitude Slice Definitions

| Slice Key | Centre | Range | Physical Regime |
|-----------|--------|-------|----------------|
| `"2°N"`  | 2°N  | −3° to  7° | Deep tropics / ITCZ, near-zero Coriolis |
| `"14°N"` | 14°N |  9° to 19° | Subtropical jet / Hadley cell subsidence |
| `"40°N"` | 40°N | 35° to 45° | Midlatitude storm track, baroclinic instability |
| `"60°N"` | 60°N | 55° to 65° | Subpolar / polar front, Rossby waves |

Half-width = 5° (constant). NH only. SH mirror shown in setup diagram (half opacity)
but **not used for RMSE calculation**.

CSV column names for slice RMSE: `"2°N"`, `"14°N"`, `"40°N"`, `"60°N"`

---

## Variables Computed

| Key | Label | Unit | FCN Channels | ERA5 Source |
|-----|-------|------|-------------|-------------|
| t500 | T500 | K | ch 54 | pressure 't' lev 500 |
| u500 | U500 | m/s | ch 15 | pressure 'u' lev 500 |
| z500 | Z500 | m²/s² | ch 41 | pressure 'z' lev 500 |
| wind500 | Wind500 | m/s | sqrt(ch15²+ch28²) | sqrt(u²+v²) lev 500 |
| wind10 | Wind10 | m/s | sqrt(ch0²+ch1²) | surface '10u','10v' |

---

## FCN Channel Map (same for all 4 FCN runs)

```python
CH = {"u10m": 0, "v10m": 1, "u500": 15, "v500": 28, "z500": 41, "t500": 54}
```

Read with h5py: `f["prediction"][:, ch_idx, :, :]` → shape (120, 721, 1440)

---

## ERA5 File Map

```
era5_pressure_2023_{02,03,04}.grib  → pressure-level vars (t, u, v, z at 500 hPa)
era5_surface_2023_{02,03,04}.grib   → surface vars (10u, 10v)
Location: /umbc/rs/pi_halem/users/yashd1/AURORA_DATA/rmse_data/
```

### ERA5 Loading Pattern (cfgrib) — identical to broad-band

```python
# Pressure-level variable
ds = xr.open_dataset(fpath, engine='cfgrib',
                     filter_by_keys={'typeOfLevel': 'isobaricInhPa',
                                     'level': 500, 'shortName': 't'},
                     backend_kwargs={'indexpath': ''})

# Surface 10m wind
ds = xr.open_dataset(fpath, engine='cfgrib',
                     filter_by_keys={'typeOfLevel': 'surface', 'shortName': '10u'},
                     backend_kwargs={'indexpath': ''})

# ALWAYS drop scalar coords before concat
drop = [c for c in ds.coords
        if ds.coords[c].dims == () or
        c in ('step','number','isobaricInhPa','surface','valid_time','heightAboveGround')]
ds = ds.drop_vars(drop, errors='ignore')
```

### ERA5 Time Matching Pattern

```python
era5_ts = pd.to_datetime(da.time.values)
deltas  = np.abs(era5_ts - pd.Timestamp(fcn_valid_time))   # TimedeltaIndex
idx     = deltas.argmin()
if deltas[idx].total_seconds() <= 3600:     # ← use deltas[idx], NOT .iloc[idx]
    result[k] = arr[idx]
```

---

## Slice Mask and Cosine-Latitude Weighting

```python
lats_1d  = np.linspace(90, -90, 721)   # 90→-90, matches FCN grid
cos_w_2d = np.cos(np.deg2rad(lats_1d))[:, None] * np.ones((1, 1440))

HALF_WIDTH = 5
SLICES = {
    "2°N":  (-3,  7),
    "14°N": ( 9, 19),
    "40°N": (35, 45),
    "60°N": (55, 65),
}

def _slice_mask(lo, hi):
    return (lats_1d >= lo) & (lats_1d < hi)

SLICE_MASKS = {name: _slice_mask(lo, hi) for name, (lo, hi) in SLICES.items()}

def weighted_rmse(fcn_t, era5_t, mask):
    sq   = (fcn_t - era5_t) ** 2
    w    = cos_w_2d * mask[:, None]   # zero outside slice
    wsum = w.sum()
    return float(np.sqrt((sq * w).sum() / wsum)) if wsum > 1e-10 else np.nan
```

---

## Percentage Improvement Formula

```python
pct = 100 × (mean_RMSE_control − mean_RMSE_insertion) / mean_RMSE_control
```

Only mature phase (day 30–60, steps 60–119) is computed.  
Early phase = exactly 0% for all experiments — insertion activates at step 60 (day 30),
so steps 0–59 are identical between control and all insertion runs.

---

## Figures Produced

| Figure | Type | Layout | What It Shows |
|--------|------|--------|---------------|
| LS.1 | Global map | 1 panel | 4 slice fills + centre lines (NH solid, SH faded) |
| LS.2 | Time series | 2×4 | T-ins effect on U500 & Z500 per slice |
| LS.3 | Time series | 2×4 | Wind-ins effect on T500 & Z500 per slice |
| LS.4 | Time series | 2×4 | Z-ins effect on T500 & Wind500 per slice |
| LS.5 | Bar chart | 1×3 | % improvement per slice, mature phase |

### Time Series Figure Layout (LS.2–LS.4)

- Rows = variables (2), Columns = slices (4) — 2×4 grid per figure
- 4 lines per panel: black solid (control), coloured solid (focus run), thin dotted (other 2)
- Grey shading = mature phase (day 30–60); vertical dashed line at day 30
- X-axis: lead day 0–60; Y-axis: RMSE [units]
- Panel title format: `"T500  —  40°N  (35° – 45°)"`

### Bar Chart Layout (LS.5)

- 3 panels (one per insertion experiment): T-ins, Wind-ins, Z-ins
- X-axis per panel = downstream variables (excluding self-inserted variable)
- 4 bars per group = 4 lat slices, colour-coded
- Bar width = 0.18 (narrower than broad-band 0.22 to fit 4 bars instead of 3)
- Bar offset formula:
  ```python
  slice_off = {sn: (i - (n_slices-1)/2) * width for i, sn in enumerate(snames)}
  ```
- Y-axis: % RMSE improvement vs control; annotations = value on bar top

---

## Key Physical Results (FCNv2 — Actual Computed Values)

### Control RMSE Baseline (mature phase)

| Variable | 2°N | 14°N | 40°N | 60°N |
|----------|-----|------|------|------|
| Z500 (m²/s²) | 145.3 | 243.9 | 1358.5 | 2196.6 |
| T500 (K) | 1.07 | 2.04 | 5.79 | 7.94 |
| U500 (m/s) | 6.98 | 9.20 | 14.78 | 15.49 |
| Wind500 (m/s) | 4.42 | 5.79 | 12.55 | 12.49 |
| Wind10 (m/s) | 2.30 | 2.70 | 3.92 | 3.63 |

RMSE increases strongly poleward for all vars. 40°N Z500 is 9× larger than 2°N.

### T-insertion downstream improvements (% RMSE reduction, mature phase)

| Downstream Var | 2°N | 14°N | 40°N | 60°N | Key finding |
|----------------|-----|------|------|------|-------------|
| U500 | 6.0% | **34.5%** | 22.9% | 26.2% | Peak at 14°N — subtropical jet is thermally driven |
| Z500 | 18.6% | 37.9% | **39.8%** | 36.8% | Flat 37–40% from 14–60°N; low at 2°N (non-geostrophic) |
| Wind500 | **-3.0%** | 19.4% | 21.3% | 18.0% | NEGATIVE at 2°N — analysis shock (f→0) |
| Wind10 | **22.6%** | 21.8% | 7.6% | 5.5% | Peak in tropics — Hadley circulation coupling |

### Wind-insertion downstream improvements

| Downstream Var | 2°N | 14°N | 40°N | 60°N | Key finding |
|----------------|-----|------|------|------|-------------|
| T500 | 32.5% | 57.0% | **71.2%** | 61.0% | Clear poleward gradient — advection dominates |
| Z500 | 33.5% | 53.7% | **84.8%** | 72.8% | 84.8% is highest non-trivial value in entire dataset |
| Wind10 | 45.5% | 53.7% | **62.0%** | 61.0% | Strong vertical coupling learned by FCNv2 |

### Z-insertion downstream improvements

| Downstream Var | 2°N | 14°N | 40°N | 60°N | Key finding |
|----------------|-----|------|------|------|-------------|
| T500 | 16.1% | 46.8% | 80.8% | **83.5%** | Largest lat gradient of any experiment (16→84%) |
| U500 | 35.4% | 58.3% | **70.7%** | 69.5% | 35.4% at 2°N despite f→0 — ageostrophic coupling |
| Wind500 | 20.9% | 41.6% | **66.6%** | 64.4% | Monotonic poleward increase |
| Wind10 | 37.5% | 47.0% | **54.4%** | 54.5% | Modest gradient — surface-500hPa coupling |

---

## Six Key Scientific Findings

1. **Analysis shock at 2°N:** T-insertion worsens Wind500 by 3.0% — the only negative value.
   Cause: f→0 at equator → geostrophic adjustment timescale diverges → imbalance persists.

2. **Z500 = master variable in extratropics:** Z-ins T500 improvement 80–84% at 40–60°N.
   Hypsometric: T̄ = (g/R)·ΔZ/ln(p₁/p₂) holds at ≥14°N, breaks down at 2°N (16%).

3. **Dominant coupling direction is Wind→T at midlatitudes, not T→Wind:**
   Wind-ins T500 at 40°N = 71.2% vs T-ins U500 at 40°N = 22.9%.
   FCNv2 better represents temperature advection than thermal-wind balance inversion.

4. **Subtropical jet is thermally driven in FCNv2:**
   T-ins U500 peaks at 14°N (34.5%), exceeds 40°N (22.9%) — Hadley cell thermal forcing.

5. **T-insertion improves tropical surface winds:**
   Wind10 improvement: 22.6% at 2°N, 5.5% at 60°N — Hadley circulation pathway learned.

6. **Wind-insertion Z500 peaks at storm track (40°N = 84.8%):**
   Highest non-trivial downstream improvement in entire analysis.
   FCNv2 Z500 errors at 40°N are caused by incorrect winds, not mass-field bias.

---

## Comparison with Broad-Band Analysis (lats/)

| Comparison | Slice result | Broad-band result | Note |
|------------|-------------|------------------|------|
| Analysis shock | -3.0% Wind500 at 2°N | Hidden in Broad Tropics | Band average masks it |
| Z-ins T500 in tropics | 16.1% at 2°N, 46.8% at 14°N | 56.2% Broad Tropics | Subtropical mean pulls up tropical |
| Wind-ins Z500 mid | 84.8% at 40°N | ~83.2% Broad Midlatitudes | Similar; 40°N is dominant contributor |

**Recommendation:** Use slice analysis to discuss mechanisms; use broad-band bar chart for summary.

---

## How to Replicate for Aurora (or Any Other Model)

1. **Locate model forecast files** — load channels as (H, W) = (721, 1440) arrays per step.

2. **Map Aurora channels** to the same variables (T500, U500, V500, Z500, U10, V10).

3. **Reuse `SLICE_MASKS`, `weighted_rmse()`, ERA5 loading functions exactly** — they are
   model-independent.

4. **Change only:**
   - `FCN_FILES` → Aurora file paths
   - `CH` → Aurora channel-to-variable mapping
   - `N_STEPS` → Aurora forecast length
   - `FCN_IC` → Aurora initial condition datetime
   - `RUNS` → list of Aurora run names
   - CSV filenames: replace `FCN_` prefix with `Aurora_`
   - Figure titles and stems

5. **Narrower slices = lower SNR.** Each ±5° slice has fewer grid points than a broad band
   (~40 lat × 1440 lon). For polar slices this is fine. At 2°N, 10° width = ~40 grid rows,
   sufficient for stable weighted RMSE. Avoid going below ±3°.

6. **Memory budget** is the same as broad-band (one ERA5 month at a time).

---

## Common Pitfalls

| Issue | Cause | Fix |
|-------|-------|-----|
| `deltas.iloc[idx]` error | TimedeltaIndex has no `.iloc` | Use `deltas[idx]` |
| ERA5 lat upside-down | Ascending lat in cfgrib output | `isel(latitude=slice(None,None,-1))` |
| Empty ERA5 dataset for wind10 | Wrong typeOfLevel | Surface 10m uses `'surface'` not `'heightAboveGround'` |
| NaN RMSE for all steps | ERA5 file missing for Feb | Feb file must cover day 0–28 |
| OOM kill | Loading all 4 runs simultaneously | One run at a time, del + gc.collect() |
| Scalar coord concat fail | Monthly GRIB conflicting step/level | Drop all scalar coords before concat |
| 0% improvement bar chart | Plotting early phase | Insertion only activates at day 30; only use mature phase |

---

## Run Command

```bash
source /umbc/rs/pi_halem/users/yashd1/venvs/aurora310/bin/activate
python3 -u /umbc/rs/pi_halem/users/yashd1/FourCastNet/section_5_7_latslice_analysis.py
```

Approx runtime: 30–60 minutes (same as broad-band; GRIB loading dominates).
Peak RAM: ~2 GB (one variable's ERA5 + one FCN run simultaneously).
Output: 20 CSV files + 5 PNG figures + log.
