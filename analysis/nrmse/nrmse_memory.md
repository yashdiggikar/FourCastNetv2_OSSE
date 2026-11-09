# Global NRMSE Analysis — Memory File
## Reference for replicating the global NRMSE analysis on other models (Aurora, Pangu, etc.)

---

## Overview

Computes cosine-latitude-weighted **global RMSE** time series for FourCastNetv2
(4 runs: control, t_ins, wind_ins, z_ins) vs ERA5, then normalises by the
**ERA5 temporal standard deviation** to produce dimensionless NRMSE.

IC: 2023-02-01 00Z  |  120 steps × 0.5 day = 60 days  |  Grid: 721×1440 (0.25°)

---

## Directory Structure

```
FourCastNet/nrmse/
├── values/
│   ├── FCN_{var}_global_rmse.csv        # 5 files — step, lead_day, control, t_ins, wind_ins, z_ins
│   ├── FCN_{var}_global_nrmse.csv       # 5 files — same columns, RMSE/σ
│   └── FCN_era5_std_normalization.csv   # per-variable σ_ERA5 normalization factors
├── logs/
│   └── compute_global_nrmse_log.txt
├── compute_global_nrmse.py
├── nrmse_results_analysis.txt
└── nrmse_memory.md   ← this file
```

---

## Normalization Formula

```
NRMSE(t) = RMSE(t) / σ_ERA5(var)
```

```python
# ERA5 temporal std at each grid point
std_field = np.nanstd(era5_arr, axis=0)        # shape (721, 1440)

# Cosine-lat weighted global spatial mean
cos_w_2d  = np.cos(np.deg2rad(lats_1d))[:, None] * np.ones((1, 1440))
sigma     = (std_field * cos_w_2d).sum() / cos_w_2d.sum()   # scalar per variable
```

σ_ERA5 is a **single scalar per variable** — does NOT vary with lead time.
NRMSE < 1.0 = skillful (error < natural variability). NRMSE > 1.0 = no skill.

---

## ERA5 Normalization Factors (actual computed values)

| Variable | σ_ERA5 | Unit |
|----------|--------|------|
| T500 | 2.803 | K |
| U500 | 8.060 | m s⁻¹ |
| Z500 | 634.99 | m² s⁻² |
| Wind500 | 6.579 | m s⁻¹ |
| Wind10 | 2.336 | m s⁻¹ |

---

## Key NRMSE Values (mature phase mean, day 30–60)

| Variable | Control | T-Ins | Wind-Ins | Z-Ins |
|----------|---------|-------|----------|-------|
| T500 | 1.717 | 0.000* | **0.595** | **0.381** |
| U500 | 1.502 | 1.220 | 0.000* | **0.516** |
| Z500 | 1.962 | 1.360 | **0.396** | 0.000* |
| Wind500 | 1.528 | 1.289 | 0.000* | **0.576** |
| Wind10 | 1.542 | 1.434 | **0.594** | 0.714 |

`*` = trivially forced to ERA5 truth (NRMSE ≈ 0).
Bold = below 1.0 (skillful). T-Ins brings no variable below 1.0.

---

## Variables and FCN Channel Map

| Key | Channels | ERA5 Source |
|-----|----------|-------------|
| t500 | ch 54 | pressure 't' lev 500 |
| u500 | ch 15 | pressure 'u' lev 500 |
| z500 | ch 41 | pressure 'z' lev 500 |
| wind500 | sqrt(ch15²+ch28²) | sqrt(u²+v²) lev 500 |
| wind10 | sqrt(ch0²+ch1²) | surface '10u','10v' |

```python
CH = {"u10m": 0, "v10m": 1, "u500": 15, "v500": 28, "z500": 41, "t500": 54}
```

---

## ERA5 Files

```
era5_pressure_2023_{02,03,04}.grib  →  /umbc/rs/pi_halem/users/yashd1/AURORA_DATA/rmse_data/
era5_surface_2023_{02,03,04}.grib
```

Loading pattern (cfgrib):
```python
ds = xr.open_dataset(fpath, engine='cfgrib',
     filter_by_keys={'typeOfLevel':'isobaricInhPa','level':500,'shortName':'t'},
     backend_kwargs={'indexpath':''})
# Drop scalar coords; flip lat if ascending; match time with deltas[idx] not .iloc[idx]
```

---

## Insertion Activation Note

Steps 0–59 (day 0–29.5): all 4 runs identical — insertion not yet active.
Steps 60–119 (day 30–60): insertion runs diverge from control.

---

## How to Replicate for Aurora

1. Replace `FCN_FILES` with Aurora file paths; update `CH` to Aurora channel map.
2. Change `N_STEPS`, `FCN_IC`.
3. Rename output CSV prefix `FCN_` → `Aurora_`.
4. **Reuse `FCN_era5_std_normalization.csv` directly** — σ_ERA5 factors are model-independent.
5. All ERA5 loading, `global_rmse()`, `era5_global_std()` functions unchanged.

---

## Common Pitfalls

| Issue | Fix |
|-------|-----|
| `deltas.iloc[idx]` error | Use `deltas[idx]` — TimedeltaIndex has no `.iloc` |
| ERA5 lat upside-down | `isel(latitude=slice(None,None,-1))` |
| wind10 ERA5 empty | Use `typeOfLevel='surface'` not `'heightAboveGround'` |
| NRMSE looks wrong | Check σ_ERA5 is positive and reasonable vs table above |

---

## Run Command

```bash
source /umbc/rs/pi_halem/users/yashd1/venvs/aurora310/bin/activate
python3 -u /umbc/rs/pi_halem/users/yashd1/FourCastNet/nrmse/compute_global_nrmse.py
```

Runtime: ~45 min. Output: 11 CSVs (5 RMSE + 5 NRMSE + 1 normalization).
