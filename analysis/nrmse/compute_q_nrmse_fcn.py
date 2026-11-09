#!/usr/bin/env python3
"""
Add Q/RH NRMSE for FCN — relative humidity at 500 hPa (FCN channel r500, index 67).
ERA5 RH is derived from ERA5 q and T at 500 hPa (both already in era5_pressure GRIBs).
Appends FCN_q500_global_nrmse.csv and FCN_q500_global_rmse.csv to nrmse/values/.
"""
import warnings; warnings.filterwarnings("ignore")
import gc
from pathlib import Path
from datetime import datetime, timedelta

import h5py
import numpy as np
import pandas as pd
import xarray as xr

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE    = Path("/umbc/rs/pi_halem/users/yashd1/FourCastNet")
ERA5DIR = Path("/umbc/rs/pi_halem/users/yashd1/AURORA_DATA/rmse_data")
VALDIR  = BASE / "nrmse/values"
LOGFILE = BASE / "nrmse/logs/compute_q_nrmse_fcn.txt"
LOGFILE.parent.mkdir(parents=True, exist_ok=True)

log = open(LOGFILE, "w", buffering=1)
def L(msg): print(msg); log.write(msg + "\n")

# ── FCN run files & channel index ─────────────────────────────────────────────
FCN_FILES = {
    "control":  BASE / "forecast_normal_60days_12hr.nc",
    "t_ins":    BASE / "Prediction_Files_Fcn/t_ins/t_ins/forecast_truth_injected_60days_12hr_FINAL.nc",
    "wind_ins": BASE / "Prediction_Files_Fcn/wind/wind_ins/forecast_uv_injected_60days_12hr_FINAL.nc",
    "z_ins":    BASE / "Prediction_Files_Fcn/z_ins/forecast_z_injected_60days_12hr_FINAL.nc",
}
R500_CH = 67     # r500 = relative humidity at 500 hPa (0-indexed)
RUNS    = ["control", "t_ins", "wind_ins", "z_ins"]

ERA5_PRES = [ERA5DIR / f"era5_pressure_2023_{m:02d}.grib" for m in [2, 3, 4]]

# ── Time / grid ───────────────────────────────────────────────────────────────
FCN_IC    = datetime(2023, 2, 1, 0, 0)
N_STEPS   = 120
fcn_times = [FCN_IC + timedelta(hours=12 * k) for k in range(N_STEPS)]
lead_days = np.array([0.5 * k for k in range(N_STEPS)])

NLAT, NLON = 721, 1440
lats_1d  = np.linspace(90, -90, NLAT)
cos_w_2d = np.cos(np.deg2rad(lats_1d))[:, None] * np.ones((1, NLON))
total_w  = cos_w_2d.sum()

# ── ERA5 loader ───────────────────────────────────────────────────────────────
def load_era5_var(shortname, level, era5_files):
    """Return float32 array (N_STEPS, 721, 1440), matched to fcn_times."""
    result = np.full((N_STEPS, NLAT, NLON), np.nan, dtype=np.float32)
    for fpath in era5_files:
        if not fpath.exists():
            continue
        L(f"    Loading {fpath.name}  var={shortname} lev={level}")
        try:
            ds = xr.open_dataset(str(fpath), engine="cfgrib",
                                 filter_by_keys={"shortName": shortname,
                                                 "typeOfLevel": "isobaricInhPa",
                                                 "level": int(level)},
                                 backend_kwargs={"indexpath": ""})
            drop = [c for c in ds.coords
                    if c in ("step", "number", "isobaricInhPa", "valid_time")]
            ds = ds.drop_vars(drop, errors="ignore")
            da = ds[list(ds.data_vars)[0]]
        except Exception as e:
            L(f"      SKIP: {e}"); continue
        era5_ts = pd.to_datetime(da.time.values)
        if da.latitude.values[0] < da.latitude.values[-1]:
            da = da.isel(latitude=slice(None, None, -1))
        arr = da.values.astype(np.float32)
        da.close(); gc.collect()
        for k, tgt in enumerate(fcn_times):
            if not np.all(np.isnan(result[k])):
                continue
            deltas = np.abs(era5_ts - pd.Timestamp(tgt))
            idx    = deltas.argmin()
            if deltas[idx].total_seconds() <= 3600:
                result[k] = arr[idx]
        del arr; gc.collect()
    matched = int(np.sum(~np.all(np.isnan(result), axis=(1, 2))))
    L(f"    ERA5 {shortname} lev={level}: matched {matched}/{N_STEPS} steps")
    return result

# ── Compute ERA5 RH from q and T at 500 hPa (Bolton 1980 formula) ────────────
def era5_rh500(era5_q, era5_t, p_hpa=500.0):
    """Convert ERA5 specific humidity (kg/kg) + T (K) to relative humidity (%)."""
    # Saturation vapour pressure: Magnus approximation
    Tc      = era5_t - 273.15
    e_sat   = 6.1078 * np.exp(17.2694 * Tc / (Tc + 237.29))  # hPa
    # Actual vapour pressure from q: e = q * p / (0.622 + 0.378 * q) ≈ q * p / 0.622
    e_act   = era5_q * p_hpa / (0.622 + 0.378 * era5_q)
    rh      = np.clip(100.0 * e_act / e_sat, 0.0, 100.0)
    return rh.astype(np.float32)

# ── Metrics ───────────────────────────────────────────────────────────────────
def global_rmse(pred, truth):
    sq = (pred - truth) ** 2
    return np.sqrt((sq * cos_w_2d).sum(axis=(1, 2)) / total_w).astype(np.float64)

def era5_global_std(arr):
    std_field = np.nanstd(arr, axis=0)
    w         = cos_w_2d * (~np.all(np.isnan(arr), axis=0))
    wsum      = w.sum()
    return float((std_field * w).sum() / wsum) if wsum > 0 else np.nan

# ── Load ERA5 q and T at 500 hPa, derive RH ──────────────────────────────────
L("=" * 70)
L("FCN Q/RH500 NRMSE  |  IC=2023-02-01")
L("=" * 70)

L("Loading ERA5 q500 and T500 to derive RH500 ...")
era5_q = load_era5_var("q", 500, ERA5_PRES)
era5_t = load_era5_var("t", 500, ERA5_PRES)
era5_rh = era5_rh500(era5_q, era5_t)
del era5_q, era5_t; gc.collect()
L(f"  ERA5 RH500 sample: min={np.nanmin(era5_rh):.2f}%  max={np.nanmax(era5_rh):.2f}%")

sigma = era5_global_std(era5_rh)
L(f"  ERA5 RH500 global temporal std σ = {sigma:.6f} %")

# ── Compute NRMSE per run ─────────────────────────────────────────────────────
rmse_cols  = {"step": np.arange(N_STEPS), "lead_day": lead_days}
nrmse_cols = {"step": np.arange(N_STEPS), "lead_day": lead_days}

for run in RUNS:
    L(f"  Run: {run}")
    with h5py.File(str(FCN_FILES[run]), "r") as f:
        fcn_r500 = f["prediction"][:, R500_CH, :, :].astype(np.float32)
    rmse_ts = global_rmse(fcn_r500, era5_rh)
    rmse_cols[run]  = rmse_ts
    nrmse_cols[run] = rmse_ts / sigma if sigma > 0 else np.full_like(rmse_ts, np.nan)
    mature = rmse_ts[lead_days >= 30].mean()
    L(f"    mature RMSE={mature:.4f}  NRMSE={mature/sigma:.4f}")
    del fcn_r500; gc.collect()

del era5_rh; gc.collect()

# ── Save ─────────────────────────────────────────────────────────────────────
pd.DataFrame(rmse_cols ).to_csv(VALDIR / "FCN_q500_global_rmse.csv",  index=False)
pd.DataFrame(nrmse_cols).to_csv(VALDIR / "FCN_q500_global_nrmse.csv", index=False)
L(f"Saved: FCN_q500_global_rmse.csv + FCN_q500_global_nrmse.csv")
L("DONE")
log.close()
