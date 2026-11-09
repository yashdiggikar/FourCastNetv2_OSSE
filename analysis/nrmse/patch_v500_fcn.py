#!/usr/bin/env python3
"""Patch: compute and save FCN_v500_global_rmse.csv + nrmse.csv."""
import warnings; warnings.filterwarnings("ignore")
import gc
from pathlib import Path
from datetime import datetime, timedelta
import h5py, numpy as np, pandas as pd, xarray as xr

BASE      = Path("/umbc/rs/pi_halem/users/yashd1/FourCastNet")
ERA5DIR   = Path("/umbc/rs/pi_halem/users/yashd1/AURORA_DATA/rmse_data")
VALDIR    = BASE / "nrmse/values"
ERA5_PRES = [ERA5DIR / f"era5_pressure_2023_{m:02d}.grib" for m in [2, 3, 4]]

FCN_IC    = datetime(2023, 2, 1, 0, 0)
N_STEPS   = 120
fcn_times = [FCN_IC + timedelta(hours=12*k) for k in range(N_STEPS)]
lead_days = np.array([0.5*k for k in range(N_STEPS)])

NLAT, NLON = 721, 1440
lats_1d  = np.linspace(90, -90, NLAT)
cos_w_2d = np.cos(np.deg2rad(lats_1d))[:, None] * np.ones((1, NLON))
total_w  = cos_w_2d.sum()

FCN_FILES = {
    "control":  BASE / "forecast_normal_60days_12hr.nc",
    "t_ins":    BASE / "Prediction_Files_Fcn/t_ins/t_ins/forecast_truth_injected_60days_12hr_FINAL.nc",
    "wind_ins": BASE / "Prediction_Files_Fcn/wind/wind_ins/forecast_uv_injected_60days_12hr_FINAL.nc",
    "z_ins":    BASE / "Prediction_Files_Fcn/z_ins/forecast_z_injected_60days_12hr_FINAL.nc",
}
CH   = {"v500": 28}
RUNS = ["control", "t_ins", "wind_ins", "z_ins"]

def load_era5_array(shortname, level):
    result = np.full((N_STEPS, NLAT, NLON), np.nan, dtype=np.float32)
    for fpath in ERA5_PRES:
        if not fpath.exists(): continue
        print(f"  ERA5 {fpath.name}")
        try:
            ds = xr.open_dataset(str(fpath), engine="cfgrib",
                                 filter_by_keys={"typeOfLevel": "isobaricInhPa",
                                                 "shortName": shortname, "level": level},
                                 backend_kwargs={"indexpath": ""})
            varname = list(ds.data_vars)[0]
            da = ds[varname]
        except Exception as e:
            print(f"  SKIP {fpath.name}: {e}"); continue
        ts = pd.to_datetime(da.time.values)
        lat_name = "latitude" if "latitude" in da.dims else "lat"
        if da[lat_name].values[0] < da[lat_name].values[-1]:
            da = da.isel({lat_name: slice(None, None, -1)})
        arr = da.values.astype(np.float32)
        da.close(); gc.collect()
        for k, tgt in enumerate(fcn_times):
            if not np.all(np.isnan(result[k])): continue
            d = np.abs(ts - pd.Timestamp(tgt))
            i = d.argmin()
            if d[i].total_seconds() <= 3600: result[k] = arr[i]
        del arr; gc.collect()
    print(f"  matched {int(np.sum(~np.all(np.isnan(result),axis=(1,2))))}/{N_STEPS}")
    return result

def load_fcn_v500(run_name):
    with h5py.File(str(FCN_FILES[run_name]), "r") as f:
        return f["prediction"][:, CH["v500"], :, :].astype(np.float32)

def global_rmse(fc, truth):
    return np.sqrt(((fc - truth)**2 * cos_w_2d).sum(axis=(1,2)) / total_w).astype(np.float64)

def era5_std(arr):
    std_f = np.nanstd(arr, axis=0)
    w = cos_w_2d * (~np.all(np.isnan(arr), axis=0))
    return float((std_f * w).sum() / w.sum())

print("Loading ERA5 V500 ...")
era5_v = load_era5_array("v", 500)
sigma  = era5_std(era5_v)
print(f"σ_v500 = {sigma:.6f} m/s")

rmse_d  = {"step": np.arange(N_STEPS), "lead_day": lead_days}
nrmse_d = {"step": np.arange(N_STEPS), "lead_day": lead_days}
for run in RUNS:
    print(f"  Run: {run}")
    fc = load_fcn_v500(run)
    r  = global_rmse(fc, era5_v)
    rmse_d[run]  = r
    nrmse_d[run] = r / sigma if sigma > 0 else np.full_like(r, np.nan)
    del fc; gc.collect()

pd.DataFrame(rmse_d).to_csv(VALDIR / "FCN_v500_global_rmse.csv", index=False)
pd.DataFrame(nrmse_d).to_csv(VALDIR / "FCN_v500_global_nrmse.csv", index=False)
print("Saved FCN_v500_global_rmse.csv + nrmse.csv")

norm_f = VALDIR / "FCN_era5_std_normalization.csv"
df_n = pd.read_csv(norm_f)
if "v500" not in df_n["variable"].values:
    df_n = pd.concat([df_n, pd.DataFrame([{"variable":"v500","label":"V500",
                                            "unit":"m s⁻¹","era5_global_std":sigma}])],
                     ignore_index=True)
    df_n.to_csv(norm_f, index=False)
    print("Updated FCN_era5_std_normalization.csv")
print("DONE")
