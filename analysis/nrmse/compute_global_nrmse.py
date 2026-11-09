#!/usr/bin/env python3
"""
Global NRMSE for FourCastNetv2 — 4 runs vs ERA5 truth
Normalization factor: cosine-latitude weighted global mean of ERA5 temporal std
  σ_global(var) = cos-lat-weighted mean of std_t( ERA5(lat,lon,t) ) over t=0..119

Outputs (all in nrmse/values/):
  FCN_{var}_global_rmse.csv    — columns: step, lead_day, control, t_ins, wind_ins, z_ins
  FCN_{var}_global_nrmse.csv   — same, divided by σ_global(var)
  FCN_era5_std_normalization.csv — per-variable normalization factors
"""

import warnings; warnings.filterwarnings("ignore")
import gc
from pathlib import Path
from datetime import datetime, timedelta

import h5py
import numpy as np
import pandas as pd
import xarray as xr

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE    = Path("/umbc/rs/pi_halem/users/yashd1/FourCastNet")
ERA5DIR = Path("/umbc/rs/pi_halem/users/yashd1/AURORA_DATA/rmse_data")
OUTDIR  = BASE / "nrmse"
VALDIR  = OUTDIR / "values"
LOGFILE = OUTDIR / "logs" / "compute_global_nrmse_log.txt"

log = open(LOGFILE, "w", buffering=1)
def L(msg): print(msg); log.write(msg + "\n")

# ─── FCN run files ─────────────────────────────────────────────────────────────
FCN_FILES = {
    "control":  BASE / "forecast_normal_60days_12hr.nc",
    "t_ins":    BASE / "Prediction_Files_Fcn/t_ins/t_ins/forecast_truth_injected_60days_12hr_FINAL.nc",
    "wind_ins": BASE / "Prediction_Files_Fcn/wind/wind_ins/forecast_uv_injected_60days_12hr_FINAL.nc",
    "z_ins":    BASE / "Prediction_Files_Fcn/z_ins/forecast_z_injected_60days_12hr_FINAL.nc",
}
CH = {"u10m": 0, "v10m": 1, "u500": 15, "v500": 28, "z500": 41, "t500": 54}

ERA5_PRES = [ERA5DIR / f"era5_pressure_2023_{m:02d}.grib" for m in [2, 3, 4]]
ERA5_SURF = [ERA5DIR / f"era5_surface_2023_{m:02d}.grib"  for m in [2, 3, 4]]

# ─── Time axis ─────────────────────────────────────────────────────────────────
FCN_IC    = datetime(2023, 2, 1, 0, 0)
N_STEPS   = 120
fcn_times = [FCN_IC + timedelta(hours=12 * k) for k in range(N_STEPS)]
lead_days = np.array([0.5 * k for k in range(N_STEPS)])

# ─── Grid ──────────────────────────────────────────────────────────────────────
NLAT, NLON = 721, 1440
lats_1d    = np.linspace(90, -90, NLAT)
cos_w_1d   = np.cos(np.deg2rad(lats_1d))                    # (721,)
cos_w_2d   = cos_w_1d[:, None] * np.ones((1, NLON))         # (721, 1440)
total_w    = cos_w_2d.sum()

# ─── ERA5 loading ──────────────────────────────────────────────────────────────
def _load_era5_month(fpath, shortname, type_of_level, level):
    kw = {"typeOfLevel": type_of_level, "shortName": shortname}
    if level is not None:
        kw["level"] = int(level)
    ds = xr.open_dataset(str(fpath), engine="cfgrib",
                         filter_by_keys=kw,
                         backend_kwargs={"indexpath": ""})
    drop = [c for c in ds.coords if ds.coords[c].dims == () or
            c in ("step", "number", "isobaricInhPa", "surface",
                  "valid_time", "heightAboveGround")]
    ds = ds.drop_vars(drop, errors="ignore")
    varname = list(ds.data_vars)[0]
    return ds[varname]

def load_era5_array(shortname, type_of_level, level, era5_files):
    """Load ERA5 → float32 array (N_STEPS, 721, 1440), NaN where not found."""
    result = np.full((N_STEPS, NLAT, NLON), np.nan, dtype=np.float32)
    for fpath in era5_files:
        if not fpath.exists():
            continue
        L(f"    Loading {fpath.name}  var={shortname}")
        try:
            da = _load_era5_month(fpath, shortname, type_of_level, level)
        except Exception as e:
            L(f"      SKIP ({e})"); continue
        era5_ts = pd.to_datetime(da.time.values)
        lat_vals = da.latitude.values if "latitude" in da.coords else da.lat.values
        if lat_vals[0] < lat_vals[-1]:
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
    L(f"    ERA5 {shortname} (lev={level}): matched {matched}/{N_STEPS} steps")
    return result

def load_fcn_channel(run_name, ch_key):
    with h5py.File(str(FCN_FILES[run_name]), "r") as f:
        arr = f["prediction"][:, CH[ch_key], :, :]
    return arr.astype(np.float32)

# ─── Metric helpers ────────────────────────────────────────────────────────────
def global_rmse(fcn_arr, era5_arr):
    """Cosine-lat weighted global RMSE, shape (N_STEPS,)."""
    sq    = (fcn_arr - era5_arr) ** 2         # (T, 721, 1440)
    rmse  = np.sqrt((sq * cos_w_2d).sum(axis=(1, 2)) / total_w)
    return rmse.astype(np.float64)

def era5_global_std(era5_arr):
    """
    Cosine-lat weighted global mean of the temporal std field.
    Step 1: std over T dimension → (721, 1440)
    Step 2: cosine-lat weighted global spatial mean → scalar
    """
    valid_mask = ~np.all(np.isnan(era5_arr), axis=0)   # grid points with data
    std_field  = np.nanstd(era5_arr, axis=0)            # (721, 1440)
    w          = cos_w_2d * valid_mask
    wsum       = w.sum()
    sigma      = float((std_field * w).sum() / wsum) if wsum > 0 else np.nan
    return sigma

# ─── Variable definitions ──────────────────────────────────────────────────────
VARIABLES = {
    "t500":    ("T500",    "K",      "t",   500, "isobaricInhPa", ERA5_PRES),
    "u500":    ("U500",    "m s⁻¹",  "u",   500, "isobaricInhPa", ERA5_PRES),
    "z500":    ("Z500",    "m²s⁻²",  "z",   500, "isobaricInhPa", ERA5_PRES),
    "wind500": ("Wind500", "m s⁻¹",  None,  None, None,            None),
    "wind10":  ("Wind10",  "m s⁻¹",  None,  None, None,            None),
}
RUNS = ["control", "t_ins", "wind_ins", "z_ins"]

# ─── Main loop ─────────────────────────────────────────────────────────────────
L("=" * 70)
L("FCN GLOBAL RMSE + NRMSE  |  IC=2023-02-01  |  120 steps x 0.5 day")
L("=" * 70)

norm_rows = []   # for normalization CSV

for var_key, (lbl, unit, era5_short, era5_level, era5_type, era5_files) in VARIABLES.items():
    L(f"\n{'─'*60}")
    L(f"Variable: {var_key}  ({lbl}  [{unit}])")

    # ── Load ERA5 ────────────────────────────────────────────────────────────
    if var_key == "wind500":
        L("  Loading ERA5 U500 + V500 for Wind500 ...")
        e_u = load_era5_array("u", "isobaricInhPa", 500, ERA5_PRES)
        e_v = load_era5_array("v", "isobaricInhPa", 500, ERA5_PRES)
        era5_arr = np.sqrt(e_u**2 + e_v**2)
        del e_u, e_v; gc.collect()
    elif var_key == "wind10":
        L("  Loading ERA5 10U + 10V for Wind10 ...")
        e_u = load_era5_array("10u", "surface", None, ERA5_SURF)
        e_v = load_era5_array("10v", "surface", None, ERA5_SURF)
        era5_arr = np.sqrt(e_u**2 + e_v**2)
        del e_u, e_v; gc.collect()
    else:
        L(f"  Loading ERA5 {era5_short} ...")
        era5_arr = load_era5_array(era5_short, era5_type, era5_level, era5_files)

    # ── ERA5 temporal std (normalization factor) ──────────────────────────────
    sigma = era5_global_std(era5_arr)
    L(f"  ERA5 global temporal std (σ) = {sigma:.6f}  [{unit}]")
    norm_rows.append({"variable": var_key, "label": lbl, "unit": unit,
                      "era5_global_std": sigma})

    # ── Compute RMSE for all runs, collect into one DataFrame ─────────────────
    rmse_cols  = {"step": np.arange(N_STEPS), "lead_day": lead_days}
    nrmse_cols = {"step": np.arange(N_STEPS), "lead_day": lead_days}

    for run in RUNS:
        L(f"  Run: {run}")
        if var_key == "wind500":
            f_u   = load_fcn_channel(run, "u500")
            f_v   = load_fcn_channel(run, "v500")
            fcn_arr = np.sqrt(f_u**2 + f_v**2)
            del f_u, f_v; gc.collect()
        elif var_key == "wind10":
            f_u   = load_fcn_channel(run, "u10m")
            f_v   = load_fcn_channel(run, "v10m")
            fcn_arr = np.sqrt(f_u**2 + f_v**2)
            del f_u, f_v; gc.collect()
        else:
            fcn_arr = load_fcn_channel(run, var_key)

        rmse_ts = global_rmse(fcn_arr, era5_arr)
        rmse_cols[run]  = rmse_ts
        nrmse_cols[run] = rmse_ts / sigma if sigma > 0 else np.full_like(rmse_ts, np.nan)

        mature_mean = rmse_ts[lead_days >= 30].mean()
        nrmse_mature = mature_mean / sigma if sigma > 0 else np.nan
        L(f"    mature RMSE={mature_mean:.4f}  NRMSE={nrmse_mature:.4f}")

        del fcn_arr; gc.collect()

    # ── Save CSVs ─────────────────────────────────────────────────────────────
    df_rmse  = pd.DataFrame(rmse_cols)
    df_nrmse = pd.DataFrame(nrmse_cols)

    rmse_path  = VALDIR / f"FCN_{var_key}_global_rmse.csv"
    nrmse_path = VALDIR / f"FCN_{var_key}_global_nrmse.csv"
    df_rmse.to_csv(rmse_path,  index=False)
    df_nrmse.to_csv(nrmse_path, index=False)
    L(f"  Saved: {rmse_path.name}")
    L(f"  Saved: {nrmse_path.name}")

    del era5_arr; gc.collect()

# ─── Save normalization factors ───────────────────────────────────────────────
df_norm = pd.DataFrame(norm_rows)
norm_path = VALDIR / "FCN_era5_std_normalization.csv"
df_norm.to_csv(norm_path, index=False)
L(f"\nNormalization factors saved: {norm_path.name}")
L(df_norm.to_string(index=False))

L("\n" + "=" * 70)
L("DONE — all global RMSE and NRMSE CSVs written.")
L("=" * 70)
log.close()
