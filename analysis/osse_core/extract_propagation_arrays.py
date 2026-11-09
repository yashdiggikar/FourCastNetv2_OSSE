#!/usr/bin/env python3
"""
Lightweight extraction of the propagation-map |error| arrays for the 4
land/ocean insertion experiments (land_temp, ocean_temp, land_uv, ocean_uv)
— NOT the full analyze_osse.py pipeline. Loads only the 3 day-snapshot
timesteps needed (Day30 / Day45 / Day60) from each experiment's raw
forecast .nc plus the matching ERA5 T500/U500 truth snapshots, computes the
per-grid |error| array exactly as analyze_osse.py does, and saves each as a
.npy so propagation maps can be replotted later without touching the raw
~43GB .nc files again.

Output: fcn_inference/{t_land,t_ocean,uv_land,uv_ocean}/propagation_error_{exp}.npy
        shape (3, 721, 1440), plus propagation_meta_{exp}.json (vmax, day labels).
"""
import warnings; warnings.filterwarnings("ignore")
import gc, json
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import xarray as xr
import netCDF4 as nc4

BASE      = Path("/umbc/rs/pi_halem/users/yashd1")
FCN_DIR   = BASE / "fcn_inference"
ERA5_DIR  = BASE / "AURORA_DATA" / "rmse_data"
ERA5_PRES = [ERA5_DIR / f"era5_pressure_2023_{m:02d}.grib" for m in [2, 3, 4]]

PLOT_LEVEL = 500
FCN_IC = datetime(2023, 2, 1, 0, 0)

EXPERIMENTS = [
    ("land_temp",  FCN_DIR / "forecast_t_landonly_day30_60.nc",          FCN_DIR / "t_land",  "temp"),
    ("ocean_temp", FCN_DIR / "t_ocean/forecast_t_ocean_day30_60.nc",     FCN_DIR / "t_ocean", "temp"),
    ("land_uv",    FCN_DIR / "uv_land/forecast_uv_landonly_day30_60.nc", FCN_DIR / "uv_land", "uv"),
    ("ocean_uv",   FCN_DIR / "uv_ocean/forecast_uv_ocean_day30_60.nc",   FCN_DIR / "uv_ocean","uv"),
]

DAY_CONFIGS = [
    ("First Injection (Day 30)", 120, FCN_IC + timedelta(hours=720)),
    ("Day 45", 179, FCN_IC + timedelta(hours=1080)),
    ("Day 60", 239, FCN_IC + timedelta(hours=1440)),
]


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def build_channel_index(nc_file):
    with nc4.Dataset(str(nc_file)) as ds:
        n_ch = ds.dimensions["channel"].size
        return {str(ds.variables["channel"][i]): i for i in range(n_ch)}


def load_era5_var(shortname, level_hpa, grib_files):
    log(f"Loading ERA5 {shortname} @ {level_hpa} hPa ...")
    dsets = []
    for f in grib_files:
        if not f.exists():
            continue
        ds = xr.open_dataset(str(f), engine="cfgrib",
                              filter_by_keys={"shortName": shortname,
                                              "typeOfLevel": "isobaricInhPa",
                                              "level": int(level_hpa)},
                              indexpath=None)
        dsets.append(ds)
    combined = xr.concat(dsets, dim="time").sortby("time")
    times = combined.time.values
    data = combined[shortname].values.astype(np.float32)
    for ds in dsets:
        ds.close()
    del combined
    log(f"  -> shape {data.shape}")
    return times, data


def era5_idx(era5_times, valid_time_dt):
    vt = np.datetime64(valid_time_dt)
    deltas = np.abs(era5_times - vt)
    idx = int(np.argmin(deltas))
    if deltas[idx] > np.timedelta64(3, "h"):
        return None
    return idx


log("Building FCN channel index ...")
CH_IDX = build_channel_index(FCN_DIR / "t_ocean/forecast_t_ocean_day30_60.nc")
T500_CH = CH_IDX.get("t500", -1)
U500_CH = CH_IDX.get("u500", -1)
log(f"  T500_CH={T500_CH} U500_CH={U500_CH}")

log("=== Loading ERA5 T500 truth snapshots ===")
era5_tm, era5_t500m = load_era5_var("t", PLOT_LEVEL, ERA5_PRES)
t500_snaps = np.stack([era5_t500m[era5_idx(era5_tm, vt), :, :] for _, _, vt in DAY_CONFIGS]).astype(np.float32)
del era5_t500m; gc.collect()

log("=== Loading ERA5 U500 truth snapshots ===")
era5_um, era5_u500m = load_era5_var("u", PLOT_LEVEL, ERA5_PRES)
u500_snaps = np.stack([era5_u500m[era5_idx(era5_um, vt), :, :] for _, _, vt in DAY_CONFIGS]).astype(np.float32)
del era5_u500m; gc.collect()

for exp_name, nc_file, out_dir, ins_type in EXPERIMENTS:
    log(f"Extracting propagation array: {exp_name} ...")
    ch_idx = T500_CH if ins_type == "temp" else U500_CH
    var_label = "T @ 500 hPa (K)" if ins_type == "temp" else "U @ 500 hPa (m/s)"
    truth_snaps = t500_snaps if ins_type == "temp" else u500_snaps

    with nc4.Dataset(str(nc_file)) as ds:
        exp_slices = np.stack([
            ds.variables["prediction"][exp_idx, ch_idx, :, :].data.astype(np.float32)
            for _, exp_idx, _ in DAY_CONFIGS
        ])

    errors = np.abs(exp_slices - truth_snaps)
    vmax = max(float(np.nanpercentile(errors, 97)), 0.1)
    vmax = round(vmax, 1)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"propagation_error_{exp_name}.npy", errors)
    meta = {
        "exp_name": exp_name, "ins_type": ins_type, "var_label": var_label,
        "vmax": vmax, "day_labels": [d for d, _, _ in DAY_CONFIGS],
        "shape": list(errors.shape),
    }
    with open(out_dir / f"propagation_meta_{exp_name}.json", "w") as f:
        json.dump(meta, f, indent=2)
    log(f"  saved -> {out_dir}/propagation_error_{exp_name}.npy  (vmax={vmax})")
    del exp_slices, errors

log("Done - all 4 FCN propagation arrays extracted.")
