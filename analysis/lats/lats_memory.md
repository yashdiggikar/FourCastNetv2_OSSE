# Section 5.7 Lat-Band Analysis — Memory File
## Reference for replicating the latitude-band RMSE analysis on other models (Aurora, Pangu, etc.)

---

## Overview

This analysis computes cosine-latitude-weighted RMSE time series for FCNv2 (4 runs)
versus ERA5 truth, broken into three dynamical latitude bands.
Figures 5.35–5.39 are produced. Output in `FourCastNet/lats/`.

---

## Directory Structure

```
FourCastNet/lats/
├── values/                          # CSV files
│   ├── FCN_{run}_{var}_latband_rmse.csv    # 20 files (4 runs × 5 vars)
│   └── FCN_latband_pct_improvement_summary.csv
├── plots/                           # PNG figures
│   ├── Figure_5_35_FCN_latband_setup.png
│   ├── Figure_5_36_FCN_t_insertion_latband_rmse.png
│   ├── Figure_5_37_FCN_wind_insertion_latband_rmse.png
│   ├── Figure_5_38_FCN_z_insertion_latband_rmse.png
│   └── Figure_5_39_FCN_latband_bar_summary.png
├── logs/
│   └── section_5_7_debug_log.txt
├── section_5_7_theoretical_analysis.txt
├── section_5_7_methods_and_interpretation.txt
└── lats_memory.md   ← this file
```

Main script: `FourCastNet/section_5_7_latband_analysis.py`

---

## Latitude Band Definitions

| Band | Latitude Range | Physics |
|------|---------------|---------|
| Tropics | \|lat\| < 30° | Convective, non-geostrophic |
| Midlatitudes | 30° ≤ \|lat\| < 60° (both hemispheres) | Baroclinic, quasi-geostrophic |
| Polar | \|lat\| ≥ 60° (both hemispheres) | Barotropic, Rossby waves |

Lat slice centres (diagram only): 2°, 14°, 40°, 60°N

---

## Variables Computed

| Key | Label | Unit | FCN Channels | ERA5 Source |
|-----|-------|------|-------------|-------------|
| t500 | T500 | K | ch 54 | pressure 't' lev 500 |
| u500 | U500 | m/s | ch 15 | pressure 'u' lev 500 |
| z500 | Z500 | m²/s² | ch 41 | pressure 'z' lev 500 |
| wind500 | Wind500 | m/s | sqrt(ch15²+ch28²) | sqrt(u²+v²) lev 500 |
| wind10 | Wind10 | m/s | sqrt(ch0²+ch1²) | surface '10u','10v' |

**NOT computed** (incompatible): Q850/r500 — FCN has relative humidity (r500),
ERA5 uses specific humidity (q) — physically incomparable.

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

### ERA5 Loading Pattern (cfgrib)

```python
# Pressure-level variable
ds = xr.open_dataset(grib_file, engine='cfgrib',
                     filter_by_keys={'typeOfLevel': 'isobaricInhPa',
                                    'level': 500, 'shortName': 't'},
                     backend_kwargs={'indexpath': ''})

# Surface 10m wind
ds = xr.open_dataset(grib_file, engine='cfgrib',
                     filter_by_keys={'typeOfLevel': 'surface', 'shortName': '10u'},
                     backend_kwargs={'indexpath': ''})

# ALWAYS drop scalar coords before using/concatenating
drop = [c for c in ds.coords
        if ds.coords[c].dims == () or
        c in ('step','number','isobaricInhPa','surface','valid_time','heightAboveGround')]
ds = ds.drop_vars(drop, errors='ignore')
```

### ERA5 Time Matching Pattern

```python
era5_ts = pd.to_datetime(da.time.values)
tgt_pd  = pd.Timestamp(fcn_valid_time)
deltas  = np.abs(era5_ts - tgt_pd)          # → TimedeltaIndex
idx     = deltas.argmin()
if deltas[idx].total_seconds() <= 3600:     # ← use deltas[idx], NOT deltas.iloc[idx]
    arr[step_k] = da.values[idx]
```

### ERA5 Lat Direction Check

```python
lat_vals = da.latitude.values   # cfgrib uses 'latitude', not 'lat'
if lat_vals[0] < lat_vals[-1]:  # ascending → flip to match FCN's 90→-90
    da = da.isel(latitude=slice(None, None, -1))
```

---

## Cosine-Latitude Weighting

```python
lats_1d  = np.linspace(90, -90, 721)        # 90→-90
cos_w_2d = np.cos(np.deg2rad(lats_1d))[:, None] * np.ones((1, 1440))

def weighted_rmse(fcn_t, era5_t, lat_mask):
    sq   = (fcn_t - era5_t) ** 2
    w    = cos_w_2d * lat_mask[:, None]    # zero outside band
    wsum = w.sum()
    return float(np.sqrt((sq * w).sum() / wsum)) if wsum > 1e-10 else np.nan
```

---

## Percentage Improvement Formula

```python
pct = 100 × (mean_RMSE_control − mean_RMSE_insertion) / mean_RMSE_control
```

Computed per (insertion_run, variable, band, phase).
Positive = insertion improves. Negative = insertion worsens (analysis shock).

Phase means are taken over all timesteps within the phase window.

---

## Figures Produced

| Figure | Type | Layout | What It Shows |
|--------|------|--------|---------------|
| 5.35 | Global map | 1 panel | Band colour shading + slice centre lines |
| 5.36 | Time series | 2×3 | T-ins effect on U500 & Z500 per band |
| 5.37 | Time series | 2×3 | Wind-ins effect on T500 & Z500 per band |
| 5.38 | Time series | 2×3 | Z-ins effect on T500 & Wind500 per band |
| 5.39 | Bar chart | 2×5 | % improvement per band, phase, variable |

### Time Series Figure Layout (5.36–5.38)

- Rows = variables (2)
- Columns = bands (Tropics, Midlatitudes, Polar)
- 4 lines per panel: black solid (control), coloured dashed (focus run), thin dotted (other 2)
- Grey shading = mature phase (day 30–60)
- X-axis: lead day 0–60; Y-axis: RMSE [units]

### Bar Chart Layout (5.39)

- Top row = Early phase (day 0–30)
- Bottom row = Mature phase (day 30–60)
- 5 sub-panels per row (one per variable)
- 3 bars per lat band: T-ins (red), Wind-ins (blue), Z-ins (orange)
- Y-axis: % RMSE improvement vs control
- Annotations: % value on bar tops

---

## How to Replicate for Aurora (or Any Other Model)

1. **Locate model forecast files** — load as numpy arrays (H, W) = (721, 1440) per step.

2. **Map Aurora channels** to the same variables (T500, U500, V500, Z500, U10, V10).
   Aurora uses pressure levels with the same ERA5 variable names.

3. **Define time axis**: Aurora IC → datetime list, one per step.

4. **Reuse ERA5 loading functions exactly as-is** (the ERA5 files and loading code
   are model-independent).

5. **Reuse `weighted_rmse()` and `BAND_MASKS` exactly**.

6. **Change only**:
   - `FCN_FILES` → Aurora forecast file paths
   - `CH` → Aurora channel-to-variable mapping
   - `N_STEPS` → Aurora forecast length
   - `FCN_IC` → Aurora initial condition datetime
   - `RUNS` → list of Aurora run names
   - CSV filenames: replace `FCN_` prefix with `Aurora_`
   - Figure titles and stems

7. **Memory budget**: Aurora files may be larger or smaller. Adjust to load
   one channel per run at a time if OOM.

---

## Key Physical Results (FCNv2)

### Expected patterns (update with actual computed values):

| Metric | Expected Finding |
|--------|-----------------|
| Best band for Z-ins | Midlatitudes (geostrophic balance active) |
| Worst band for T-ins | Polar (analysis shock from imbalanced insertion) |
| Largest early-phase improvement | Midlatitudes (fast baroclinic instability decay) |
| Largest mature-phase improvement | All bands for Z-ins (corrects systematic geopotential bias) |
| Wind10 improvement from 500hPa insertions | Weaker than 500hPa vars (indirect surface coupling) |

---

## Common Pitfalls

| Issue | Cause | Fix |
|-------|-------|-----|
| `deltas.iloc[idx]` error | TimedeltaIndex has no .iloc | Use `deltas[idx]` |
| ERA5 lat upside-down | Ascending lat in cfgrib output | Flip with `isel(latitude=slice(None,None,-1))` |
| Empty ERA5 dataset | Wrong typeOfLevel | Surface 10m wind uses `'surface'` not `'heightAboveGround'` |
| OOM kill | Loading all 4 runs simultaneously | Process one run at a time, del + gc.collect() |
| NaN RMSE | ERA5 not found for FCN step | Check era5_files list; Feb file must be included for day 0–28 |
| Scalar coord concat fail | Monthly GRIB files have conflicting step/level coords | Drop all scalar coords before concat |

---

## Run Command

```bash
source /umbc/rs/pi_halem/users/yashd1/venvs/aurora310/bin/activate
python3 -u /umbc/rs/pi_halem/users/yashd1/FourCastNet/section_5_7_latband_analysis.py
```

Approx runtime: 30–60 minutes (3 months × 5 variables × 4 runs GRIB loading dominates).
Peak RAM: ~2 GB (one variable's ERA5 + one FCN run simultaneously).
Output: 20 CSV files + 5 PNG figures + log.
