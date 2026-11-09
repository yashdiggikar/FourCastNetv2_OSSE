# Section 5.8 Spatial Bias Analysis — Reference Memory File

**Purpose:** This file documents every calculation, every output, and every design decision
made for the FCN spatial bias analysis so the identical analysis can be reproduced for
Aurora, Pangu-Weather, or any other ML weather model with minimal re-work.

---

## 1. What This Analysis Does (One-Paragraph Summary)

Load mature-phase (lead day 30 to end) forecasts from a control run and three
insertion runs. Load ERA5 truth for the same time window. Align grids and times.
Compute `bias(lat,lon) = mean(forecast - truth)` over all mature-phase timesteps
at every grid point. Compute `bias_reduction(lat,lon) = |control_bias| - |insertion_bias|`
for every insertion × variable pair. Classify every grid point into a failure-mode
category based on which insertion helps/hurts Z500. Plot five figures (5.43–5.47).
Save all 2D fields as CSV files and all plots as PNG + PDF.

---

## 2. Experiment Setup

| Parameter | FCN Value | Notes for other models |
|---|---|---|
| Model | FourCastNetv2 | Change to Aurora, Pangu, etc. |
| IC date | 2023-02-01 00:00 UTC | Can be any date with ERA5 coverage |
| Forecast length | 60 days (120 × 12-hr steps) | Aurora is 6-hourly → 240 steps |
| Mature phase start | Lead day 30 | Keep at day 30 regardless of model |
| Mature phase end | Last forecast step | Adjust if run length differs |
| Mature phase steps | 60 (steps 60–119) | Recalculate for different cadence |
| ERA5 cadence | 6-hourly | ERA5 is always 6-hourly |
| FCN cadence | 12-hourly | Aurora = 6-hourly; Pangu = 6-hourly |
| Grid resolution | 0.25 deg (721 × 1440) | All current ML models use 0.25 deg |
| Diagnostic level | 500 hPa | Can add 250, 850 hPa |
| Diagnostic variable for failure map | Z500 | Use Z500 first; T500 as fallback |

---

## 3. Input Files Used

### FCN

| Run | File path |
|---|---|
| Control | `/umbc/rs/pi_halem/users/yashd1/FourCastNet/forecast_normal_60days_12hr.nc` |
| T-insertion | `/umbc/rs/pi_halem/users/yashd1/FourCastNet/Prediction_Files_Fcn/t_ins/t_ins/forecast_truth_injected_60days_12hr_FINAL.nc` |
| Wind-insertion | `/umbc/rs/pi_halem/users/yashd1/FourCastNet/Prediction_Files_Fcn/wind/wind_ins/forecast_uv_injected_60days_12hr_FINAL.nc` |
| Z-insertion | `/umbc/rs/pi_halem/users/yashd1/FourCastNet/Prediction_Files_Fcn/z_ins/forecast_z_injected_60days_12hr_FINAL.nc` |

FCN file format: **NetCDF4**
- Variable: `prediction`, shape `(time, channel, lat, lon)` = `(120, 73, 721, 1440)`
- Time: `hours since 2023-02-01 00:00:00`
- Channels: string array of length 73

### ERA5 (for mature phase only — February not needed)

| File | Period | Size |
|---|---|---|
| `era5_pressure_2023_03.grib` | 2023-03-01 to 2023-03-31 | 16 GB |
| `era5_pressure_2023_04.grib` | 2023-04-01 to 2023-04-15 | 7.6 GB |

ERA5 directory: `/umbc/rs/pi_halem/users/yashd1/AURORA_DATA/rmse_data/`

ERA5 file format: **GRIB2**, read with `cfgrib` + `xarray`
- Filter key: `{'shortName': 't', 'typeOfLevel': 'isobaricInhPa', 'level': 500}`
- Coordinate names after loading: `latitude`, `longitude`, `time`
- Rename needed: `latitude→lat`, `longitude→lon`

---

## 4. FCN Channel Mapping (73 channels)

```
Index  Name     Variable               Level
0      u10m     zonal wind             10 m (surface)
1      v10m     meridional wind        10 m (surface)
2      u100m    zonal wind             100 m (surface)
3      v100m    meridional wind        100 m (surface)
4      t2m      temperature            2 m (surface)
5      sp       surface pressure       surface
6      msl      mean sea level pressure surface
7      tcwv     total column water vapour surface
8–20   u50–u1000  zonal wind           50–1000 hPa (13 levels)
21–33  v50–v1000  meridional wind      50–1000 hPa (13 levels)
34–46  z50–z1000  geopotential         50–1000 hPa (13 levels)
47–59  t50–t1000  temperature          50–1000 hPa (13 levels)
60–72  r50–r1000  relative humidity    50–1000 hPa (13 levels)
```

**Channels used in this analysis:**

| Channel name | Index | ERA5 shortName | Units | Notes |
|---|---|---|---|---|
| `t500` | 54 | `t` | K | Direct comparison |
| `u500` | 15 | `u` | m/s | Direct comparison |
| `v500` | 28 | `v` | m/s | Direct comparison |
| `z500` | 41 | `z` | m²/s² | Direct comparison. Both FCN and ERA5 store geopotential (not geopotential height). No conversion needed. |
| `r500` | 67 | — | % | **NOT used.** FCN relative humidity ≠ ERA5 specific humidity. Skip. |

**Wind speed derived variable:**
```
wind500(t, lat, lon) = sqrt(u500² + v500²)
```
Computed from FCN u/v and ERA5 u/v separately, then differenced.

---

## 5. How FCN Files Are Read (memory-efficient method)

Use **h5py** to load only the channels and timesteps you need.
Do NOT use `xr.open_dataset` on the full file — it reads all 73 channels × 120 steps
into RAM and will OOM-kill on most nodes (~8.8 GB per file).

```python
import h5py
import numpy as np

MATURE_IDX = np.arange(60, 120)   # steps 60-119 = lead day 30 to 59.5

with h5py.File('forecast_normal_60days_12hr.nc', 'r') as f:
    channel_list = [c.decode('utf-8') for c in f['channel'][:]]
    cidx = channel_list.index('t500')
    arr = f['prediction'][MATURE_IDX, cidx, :, :]   # shape (60, 721, 1440)
```

Peak memory per variable per run: 60 × 721 × 1440 × 4 bytes = ~248 MB.
With 5 variables per run processed sequentially: ~1.2 GB peak per run.

---

## 6. How ERA5 Files Are Read

Use `cfgrib` + `xarray`, loading one variable at a time with `filter_by_keys`.
Load only the mature-phase months (March + April). Drop scalar coordinates
before concatenating monthly files.

```python
import xarray as xr

def load_era5_500(grib_files, short_name, level=500, target_times=None):
    parts = []
    for f in grib_files:
        ds = xr.open_dataset(str(f), engine='cfgrib',
            backend_kwargs={'filter_by_keys': {
                'shortName': short_name,
                'typeOfLevel': 'isobaricInhPa',
                'level': level,
            }})
        scalar_coords = [c for c in ds.coords if ds.coords[c].dims == ()
                         or c in ('step','number','isobaricInhPa','valid_time')]
        ds = ds.drop_vars(scalar_coords, errors='ignore')
        da = ds[list(ds.data_vars)[0]]
        da = da.rename({'latitude': 'lat', 'longitude': 'lon'})
        parts.append(da)
    combined = xr.concat(parts, dim='time').sortby('time')
    if target_times is not None:
        combined = combined.sel(time=target_times)
    return combined.values.astype(np.float32)
```

ERA5 shortName → variable mapping:

| Variable | shortName |
|---|---|
| Temperature | `t` |
| Zonal wind | `u` |
| Meridional wind | `v` |
| Geopotential | `z` |
| Specific humidity | `q` |
| Relative humidity | `r` |

---

## 7. Time Alignment Procedure

FCN is 12-hourly (00 and 12 UTC). ERA5 is 6-hourly (00, 06, 12, 18 UTC).

```
ERA5 times that match FCN = ERA5 times where hour is 0 or 12 UTC
Common times = intersection(FCN times, ERA5 filtered times)
```

For this experiment:
- FCN: 120 steps, 2023-02-01 00:00 to 2023-04-01 12:00
- ERA5 at 00/12 UTC: 120 times in the same range
- Common times: 120 (perfect match)
- Mature-phase subset: 60 times from 2023-03-03 00:00 to 2023-04-01 12:00

**For Aurora (6-hourly output):** FCN mature steps 60–119 correspond to lead
hours 720–1428 (every 12 hr). For Aurora at 6-hourly, the equivalent mature
phase would be steps 120–239 (lead hour 720–1434 at 6-hr intervals = 120 steps).
ERA5 then matches every step without subsampling.

---

## 8. Bias Calculation

```
bias(lat, lon) = (1/N) * Σ_t [forecast(t, lat, lon) − truth(t, lat, lon)]

N = 60 (number of mature-phase timesteps)
```

Positive = forecast too high. Negative = forecast too low.

In code:
```python
bias = np.nanmean(fcn_array - era5_array, axis=0)   # axis 0 = time
# fcn_array and era5_array both have shape (60, 721, 1440)
# result bias has shape (721, 1440)
```

Derived statistics:
```
mean_bias = np.mean(bias)           # scalar: global average signed bias
MAB       = np.mean(np.abs(bias))   # scalar: global mean absolute bias
bias_min  = bias.min()              # worst negative point
bias_max  = bias.max()              # worst positive point
```

---

## 9. Bias Reduction Calculation

```
bias_reduction(lat, lon) = |control_bias(lat, lon)| − |insertion_bias(lat, lon)|

Positive = insertion improved (reduced absolute error)
Negative = insertion worsened (increased absolute error)
```

In code:
```python
br = np.abs(control_bias) - np.abs(insertion_bias)   # shape (721, 1440)
mean_BR = np.mean(br)
pct_reduction = 100 * mean_BR / np.mean(np.abs(control_bias))
```

**The table of all computed bias reductions:**

| Insertion | Variable | Ctrl MAB | Ins MAB | Mean BR | % Reduction |
|---|---|---|---|---|---|
| t_ins | T500 | 2.2017 K | 0.0000 | +2.2017 | 100.0% |
| t_ins | U500 | 4.6814 m/s | 3.7111 | +0.9703 | 20.7% |
| t_ins | V500 | 3.3702 m/s | 2.3520 | +1.0182 | 30.2% |
| t_ins | Z500 | 588.41 m²/s² | 351.32 | +237.08 | 40.3% |
| t_ins | Wind500 | 3.4664 m/s | 2.9761 | +0.4903 | 14.1% |
| wind_ins | T500 | 2.2017 K | 1.2987 | +0.9030 | 41.0% |
| wind_ins | U500 | 4.6814 m/s | 0.0000 | +4.6814 | 100.0% |
| wind_ins | V500 | 3.3702 m/s | 0.0000 | +3.3702 | 100.0% |
| wind_ins | Z500 | 588.41 m²/s² | 204.82 | +383.58 | 65.2% |
| wind_ins | Wind500 | 3.4664 m/s | 0.0000 | +3.4664 | 100.0% |
| z_ins | T500 | 2.2017 K | 0.2605 | +1.9412 | 88.2% |
| z_ins | U500 | 4.6814 m/s | 1.3103 | +3.3711 | 72.0% |
| z_ins | V500 | 3.3702 m/s | 0.6662 | +2.7040 | 80.2% |
| z_ins | Z500 | 588.41 m²/s² | 0.0000 | +588.41 | 100.0% |
| z_ins | Wind500 | 3.4664 m/s | 1.0216 | +2.4447 | 70.5% |

---

## 10. Failure Mode Classification

Diagnostic variable: **Z500** (preferred; fall back to T500 if Z500 unavailable).

Threshold: `0.10 × np.percentile(np.abs(control_bias_Z500), 90)`
FCN value: `0.10 × 1460.86 = 146.09 m²/s²`

Nine classes:

| ID | Name | Condition | FCN area % |
|---|---|---|---|
| 0 | Neutral / mixed | all |impr| < threshold | 38.84% |
| 1 | All insertions improve | all three > +threshold | 32.04% |
| 2 | All insertions worsen | all three < −threshold | 0.00% |
| 3 | Temp-insertion best | best=temp, max > threshold, not class 1 | 0.00% |
| 4 | Wind-insertion best | best=wind, max > threshold, not class 1 | 0.00% |
| 5 | Geo-insertion best | best=geo, max > threshold, not class 1 | 18.11% |
| 6 | Temp-insertion worst | worst=temp, min < −threshold | 9.57% |
| 7 | Wind-insertion worst | worst=wind, min < −threshold | 1.44% |
| 8 | Geo-insertion worst | worst=geo, min < −threshold | 0.00% |

Area weighting: `weight(lat) = cos(lat × π/180)` — applied to get true spherical area %.

---

## 11. All Output Files

### Directory structure
```
/umbc/rs/pi_halem/users/yashd1/FourCastNet/bias/
├── plots/
│   ├── Figure_5_43_FCN_control_mature_phase_bias_maps.png         (1.9 MB)
│   ├── Figure_5_43_FCN_control_mature_phase_bias_maps.pdf         (90 MB)
│   ├── Figure_5_44_FCN_temperature_insertion_bias_reduction.png   (2.6 MB)
│   ├── Figure_5_44_FCN_temperature_insertion_bias_reduction.pdf   (112 MB)
│   ├── Figure_5_45_FCN_wind_insertion_bias_reduction.png          (2.4 MB)
│   ├── Figure_5_45_FCN_wind_insertion_bias_reduction.pdf          (103 MB)
│   ├── Figure_5_46_FCN_geopotential_insertion_bias_reduction.png  (2.8 MB)
│   ├── Figure_5_46_FCN_geopotential_insertion_bias_reduction.pdf  (108 MB)
│   ├── Figure_5_47_FCN_failure_mode_summary_map.png               (481 KB)
│   └── Figure_5_47_FCN_failure_mode_summary_map.pdf               (16 MB)
├── values/
│   ├── FCN_control_t500_bias.csv          (32 MB, 1M rows)
│   ├── FCN_control_u500_bias.csv          (31 MB)
│   ├── FCN_control_v500_bias.csv          (31 MB)
│   ├── FCN_control_z500_bias.csv          (31 MB)
│   ├── FCN_control_wind500_bias.csv       (31 MB)
│   ├── FCN_t_ins_t500_bias.csv            (16 MB)  ← smaller: all zeros
│   ├── FCN_t_ins_u500_bias.csv            (31 MB)
│   ├── FCN_t_ins_v500_bias.csv            (32 MB)
│   ├── FCN_t_ins_z500_bias.csv            (31 MB)
│   ├── FCN_t_ins_wind500_bias.csv         (31 MB)
│   ├── FCN_wind_ins_{var}_bias.csv        (×5 files, 31–35 MB each)
│   ├── FCN_z_ins_{var}_bias.csv           (×5 files, 16–32 MB each)
│   ├── FCN_{run}_{var}_combined.csv       (×15 files, 54–72 MB each)
│   │     columns: lat, lon, control_bias, insertion_bias, bias_reduction
│   ├── FCN_failure_mode_map.csv           (34 MB)
│   │     columns: lat, lon, class_id, class_name
│   └── FCN_failure_mode_area_summary.csv  (283 bytes, 9 rows)
│         columns: class_id, class_name, area_pct
├── logs/
│   └── section_5_8_debug_log.txt          (24 KB — full run log)
├── section_5_8_bias_analysis.py           (analysis script)
├── section_5_8_theoretical_analysis.txt  (physics + ML interpretation)
├── section_5_8_methods_and_interpretation.txt  (formulas + plot guide)
└── bias_memory.md                         (this file)
```

### CSV column reference

| File pattern | Columns | Meaning |
|---|---|---|
| `*_bias.csv` | lat, lon, bias | Raw signed bias at each grid point |
| `*_combined.csv` | lat, lon, control_bias, insertion_bias, bias_reduction | Side-by-side comparison |
| `failure_mode_map.csv` | lat, lon, class_id, class_name | One class per grid point |
| `failure_mode_area_summary.csv` | class_id, class_name, area_pct | 9-row summary table |

---

## 12. Figure Descriptions (for caption writing)

| Figure | Rows × Cols | Variables | What it shows |
|---|---|---|---|
| 5.43 | 4 × 1 | T500, U500, Z500, Wind500 | Control run mature-phase mean bias maps. Red = FCN too high, blue = too low. |
| 5.44 | 2 × 3 | U500, Z500 | Control bias / T-insertion bias / bias reduction for each variable. Tests thermal wind and hypsometric coupling. |
| 5.45 | 2 × 3 | T500, Z500 | Control bias / Wind-insertion bias / bias reduction. Tests kinematic (advection) coupling. |
| 5.46 | 2 × 3 | T500, Wind500 | Control bias / Z-insertion bias / bias reduction. Tests geostrophic and hypsometric coupling. |
| 5.47 | 1 × 1 | Z500 (diagnostic) | Failure-mode classification map. 9 discrete colors. Identifies where each insertion type helps or hurts. |

All plots use: RdBu_r diverging colormap, Plate Carrée projection, coastlines, borders, gridlines.
Colormap limits: 97th percentile of absolute bias values (symmetric ±vmax).

---

## 13. Python Environment

**Activation command:**
```bash
source /umbc/rs/pi_halem/users/yashd1/venvs/aurora310/bin/activate
```

**Required packages:**
- numpy, pandas — data and CSV
- h5py — efficient NetCDF4 channel slicing (avoids OOM)
- xarray, cfgrib — ERA5 GRIB loading
- matplotlib (Agg backend) — plotting without display
- cartopy — map projections and coastlines
- scipy (optional, for interpolation if grids differ)

**Known issue:** Loading all FCN channels for all runs simultaneously OOM-kills
at ~16 GB. Solution: use h5py to slice one channel at a time, process one run
at a time, free arrays with `del` + `gc.collect()` between runs.

---

## 14. Adaptation Guide for Aurora

When running this same analysis for Aurora, the following changes are needed:

### File format
Aurora output is likely NetCDF4 or HDF5. Check the variable and dimension names.
Aurora uses a different channel/variable naming convention — look for a metadata
file or `variables` attribute.

### Variable mapping for Aurora
Aurora standard variables at 500 hPa to look for:

| Physical variable | Likely Aurora name | ERA5 shortName |
|---|---|---|
| Temperature | `temperature_500` or `t` at level 500 | `t` |
| Zonal wind | `u_component_of_wind_500` or `u` | `u` |
| Meridional wind | `v_component_of_wind_500` or `v` | `v` |
| Geopotential | `geopotential_500` or `z` | `z` |
| Specific humidity | `specific_humidity_500` or `q` | `q` |

Note: Aurora uses **specific humidity** (not relative humidity), so Q500 bias
comparison IS valid for Aurora. This is an additional variable not available for FCN.

### Forecast cadence
Aurora produces 6-hourly output. The mature phase at lead day 30 starts at step 120
(120 × 6 hr = 720 hr = 30 days). The mature phase contains 120 steps (step 120
to step 239, if the run is 60 days).

ERA5 can be matched at every step (no subsampling needed since both are 6-hourly).

### Insertion runs for Aurora
The equivalent runs to look for:
- Control run
- Temperature insertion (t_ins or temp_ins)
- Wind insertion (uv_ins or wind_ins)
- Geopotential insertion (z_ins or geo_ins)
- Any additional: Q insertion (q_ins), if available

### Plot figure numbering
When replicating for Aurora, use figure numbers 5.48–5.52 (or as assigned in thesis
outline). The analysis structure is identical — only the model name, file paths,
channel names, and cadence change.

### Output directory
Suggested: `/umbc/rs/pi_halem/users/yashd1/AURORA_DATA/bias/` with same subfolder
structure: `plots/`, `values/`, `logs/`.

### Script changes needed
1. Update `CONTROL_FILE`, `TINS_FILE`, `WIND_FILE`, `ZINS_FILE` paths
2. Update `ERA5_MATURE_FILES` if ERA5 is in a different location
3. Update channel loading: replace h5py channel-name lookup with Aurora's variable structure
4. Update `FCN_CHANNELS_NEEDED` list
5. Update `mature_idx` for 6-hourly cadence: `np.arange(120, 240)` for 60-day run
6. Add Q500 to `BIAS_VARS` since Aurora has specific humidity
7. Change all figure titles, file name stems, and model label from "FCN" to "Aurora"

---

## 15. Key Numbers to Compare Across Models

When Aurora (or another model) is done, compare these numbers directly against FCN:

| Metric | FCN value | Aurora value | Better model |
|---|---|---|---|
| Control T500 MAB (K) | 2.2017 | — | lower is better |
| Control Z500 MAB (m²/s²) | 588.41 | — | lower is better |
| Control U500 MAB (m/s) | 4.6814 | — | lower is better |
| Control Wind500 MAB (m/s) | 3.4664 | — | lower is better |
| Z-ins → T500 % reduction | 88.2% | — | higher = stronger Z-T coupling |
| Z-ins → Wind500 % reduction | 70.5% | — | higher = stronger geostrophic coupling |
| T-ins → Z500 % reduction | 40.3% | — | higher = better hypsometric representation |
| Wind-ins → Z500 % reduction | 65.2% | — | higher = better advection representation |
| Failure class 1 area % (all improve) | 32.04% | — | higher = more consistent coupled errors |
| Failure class 6 area % (T-ins worst) | 9.57% | — | lower = less analysis shock risk |
| Failure class 5 area % (geo-ins only) | 18.11% | — | lower = more physically coupled model |
| Z500 control bias range (m²/s²) | −9389 to +4330 | — | smaller extremes = better |

---

## 16. Quick Replication Checklist

To run this analysis for a new model:

- [ ] Identify control run and insertion run file paths
- [ ] Confirm variable/channel names at 500 hPa (T, U, V, Z, optionally Q)
- [ ] Confirm forecast cadence (12-hr for FCN, 6-hr for Aurora/Pangu)
- [ ] Confirm forecast length and compute mature-phase step indices
- [ ] Check that ERA5 truth files cover the mature-phase period
- [ ] Confirm grid resolution (both model and ERA5 should be 0.25 deg)
- [ ] Check unit consistency (Z should be m²/s² or convert)
- [ ] Update `ERA5_MATURE_FILES` — only include months needed for mature phase
- [ ] Update all file paths and model-name strings in the script
- [ ] Create output directory: `{MODEL}/bias/{plots,values,logs}`
- [ ] Run the script in the `aurora310` environment (has all required packages)
- [ ] Check the debug log for any warnings or skipped variables
- [ ] Fill in the "Aurora value" column in Section 15 above
- [ ] Update `MEMORY.md` in `~/.claude/projects/` with new results

---

## 17. Runtime and Resource Notes

| Step | FCN runtime | Memory peak | Notes |
|---|---|---|---|
| FCN data loading (h5py, all 4 runs) | ~8 min | ~1.2 GB/run | Sequential, one run at a time |
| ERA5 loading (2 months, 5 variables) | ~3 min | ~2 GB | cfgrib loads all at once |
| Bias calculation | < 1 min | < 0.5 GB | Numpy in-memory |
| Figure generation | ~20 min | ~1 GB | PDF render is slow (cartopy) |
| CSV writing | ~5 min | ~0.5 GB | 37 files × 30–70 MB |
| **Total** | **~40 min** | **~4 GB peak** | On UMBC HPCF with aurora310 env |

For Aurora (6-hourly, potentially larger files): expect similar or slightly longer
loading times depending on file format. The cartopy PDF rendering time dominates
and will be the same regardless of model.

---

*Generated: 2026-06-01 | FCN IC: 2023-02-01 | Analysis script: section_5_8_bias_analysis.py*
