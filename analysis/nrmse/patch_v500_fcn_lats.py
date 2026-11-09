#!/usr/bin/env python3
"""Patch: compute FCN v500 latband + latslice RMSE CSVs (missing from original run)."""
import warnings; warnings.filterwarnings("ignore")
import gc
from pathlib import Path
from datetime import datetime, timedelta

import h5py
import numpy as np
import pandas as pd
import xarray as xr

# ── Paths ──────────────────────────────────────────────────────────────────
BASE       = Path("/umbc/rs/pi_halem/users/yashd1/FourCastNet")
ERA5DIR    = Path("/umbc/rs/pi_halem/users/yashd1/AURORA_DATA/rmse_data")
LATBAND_VALDIR  = BASE / "lats" / "values"
LATSLICE_VALDIR = BASE / "lat_bands" / "values"
ERA5_PRES  = [ERA5DIR / f"era5_pressure_2023_{m:02d}.grib" for m in [2, 3, 4]]

FCN_FILES = {
    "control":  BASE / "forecast_normal_60days_12hr.nc",
    "t_ins":    BASE / "Prediction_Files_Fcn/t_ins/t_ins/forecast_truth_injected_60days_12hr_FINAL.nc",
    "wind_ins": BASE / "Prediction_Files_Fcn/wind/wind_ins/forecast_uv_injected_60days_12hr_FINAL.nc",
    "z_ins":    BASE / "Prediction_Files_Fcn/z_ins/forecast_z_injected_60days_12hr_FINAL.nc",
}
V500_CH = 28
RUNS    = ["control", "t_ins", "wind_ins", "z_ins"]

# ── Time / grid ────────────────────────────────────────────────────────────
FCN_IC    = datetime(2023, 2, 1, 0, 0)
N_STEPS   = 120
fcn_times = [FCN_IC + timedelta(hours=12 * k) for k in range(N_STEPS)]
lead_days = np.array([0.5 * k for k in range(N_STEPS)])

NLAT, NLON = 721, 1440
lats_1d    = np.linspace(90, -90, NLAT)
cos_w_2d   = np.cos(np.deg2rad(lats_1d))[:, None] * np.ones((1, NLON))

# ── Latitude bands ─────────────────────────────────────────────────────────
BANDS = {
    "Tropics":      [(-30,  30)],
    "Midlatitudes": [(-60, -30), (30,  60)],
    "Polar":        [(-90, -60), (60,  90)],
}

def _band_mask(lat_ranges):
    m = np.zeros(NLAT, bool)
    for lo, hi in lat_ranges:
        m |= (lats_1d >= lo) & (lats_1d < hi)
    return m

BAND_MASKS = {name: _band_mask(r) for name, r in BANDS.items()}

# ── Latitude slices (±5° around centre, NH) ────────────────────────────────
HALF = 5
SLICES = {
    "2°N":  ( 2 - HALF,  2 + HALF),
    "14°N": (14 - HALF, 14 + HALF),
    "40°N": (40 - HALF, 40 + HALF),
    "60°N": (60 - HALF, 60 + HALF),
}

def _slice_mask(lo, hi):
    return (lats_1d >= lo) & (lats_1d < hi)

SLICE_MASKS = {name: _slice_mask(lo, hi) for name, (lo, hi) in SLICES.items()}

def weighted_rmse(fc_t, era5_t, mask):
    sq   = (fc_t - era5_t) ** 2
    w    = cos_w_2d * mask[:, None]
    wsum = w.sum()
    return float(np.sqrt((sq * w).sum() / wsum)) if wsum > 1e-10 else np.nan

# ── ERA5 v500 loader ───────────────────────────────────────────────────────
def load_era5_v500():
    result = np.full((N_STEPS, NLAT, NLON), np.nan, dtype=np.float32)
    for fpath in ERA5_PRES:
        if not fpath.exists():
            print(f"  MISSING: {fpath.name}"); continue
        print(f"  ERA5 {fpath.name}")
        try:
            ds = xr.open_dataset(str(fpath), engine="cfgrib",
                                 filter_by_keys={"typeOfLevel": "isobaricInhPa",
                                                 "shortName": "v", "level": 500},
                                 backend_kwargs={"indexpath": ""})
            varname = list(ds.data_vars)[0]
            da = ds[varname]
        except Exception as e:
            print(f"  SKIP {fpath.name}: {e}"); continue
        era5_ts  = pd.to_datetime(da.time.values)
        lat_name = "latitude" if "latitude" in da.dims else "lat"
        if da[lat_name].values[0] < da[lat_name].values[-1]:
            da = da.isel({lat_name: slice(None, None, -1)})
        arr = da.values.astype(np.float32)
        da.close(); gc.collect()
        for k, tgt in enumerate(fcn_times):
            if not np.all(np.isnan(result[k])): continue
            d = np.abs(era5_ts - pd.Timestamp(tgt))
            i = d.argmin()
            if d[i].total_seconds() <= 3600: result[k] = arr[i]
        del arr; gc.collect()
    matched = int(np.sum(~np.all(np.isnan(result), axis=(1, 2))))
    print(f"  ERA5 v500: matched {matched}/{N_STEPS}")
    return result

def load_fcn_v500(run_name):
    with h5py.File(str(FCN_FILES[run_name]), "r") as f:
        return f["prediction"][:, V500_CH, :, :].astype(np.float32)

# ── Main ───────────────────────────────────────────────────────────────────
print("Loading ERA5 V500 ...")
era5_v = load_era5_v500()

for run in RUNS:
    print(f"  Run: {run}")
    fcn_v = load_fcn_v500(run)

    # latband
    rows_band = []
    for k in range(N_STEPS):
        row = {"step": k, "lead_day": lead_days[k]}
        if np.all(np.isnan(era5_v[k])):
            for bn in BANDS: row[bn] = np.nan
        else:
            for bn, mask in BAND_MASKS.items():
                row[bn] = weighted_rmse(fcn_v[k], era5_v[k], mask)
        rows_band.append(row)
    df_band = pd.DataFrame(rows_band)
    df_band["run"] = run
    out_band = LATBAND_VALDIR / f"FCN_{run}_v500_latband_rmse.csv"
    df_band.to_csv(out_band, index=False)
    print(f"    → {out_band.name}")

    # latslice
    rows_slice = []
    for k in range(N_STEPS):
        row = {"step": k, "lead_day": lead_days[k]}
        if np.all(np.isnan(era5_v[k])):
            for sn in SLICES: row[sn] = np.nan
        else:
            for sn, mask in SLICE_MASKS.items():
                row[sn] = weighted_rmse(fcn_v[k], era5_v[k], mask)
        rows_slice.append(row)
    df_slice = pd.DataFrame(rows_slice)
    df_slice["run"] = run
    out_slice = LATSLICE_VALDIR / f"FCN_{run}_v500_latslice_rmse.csv"
    df_slice.to_csv(out_slice, index=False)
    print(f"    → {out_slice.name}")

    del fcn_v; gc.collect()

del era5_v; gc.collect()
print("DONE — FCN v500 latband + latslice CSVs saved.")
