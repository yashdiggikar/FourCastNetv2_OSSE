#!/usr/bin/env python3
"""
Global (cosine-lat-weighted) RMSE vs ERA5 for the FCNv2 Correlated OSSE
Perturbation experiments (t_perturbed_corr, uv_perturbed_corr).

Reuses the exact same area-weighting/time-matching methodology as
fcn_inference/analyze_osse.py (global_rmse(), era5_idx()) applied to the two
new global-insertion experiments, and compares against the SAME control
forecast (FourCastNet/forecast_normal_60days_12hr.nc) used everywhere else
in this project for FCN.

Full variable/level grid (per user request): T, U, V, Z, R (relative
humidity) at 300/500/1000 hPa, plus surface T2M/U10/V10 -- computed for
BOTH experiments regardless of which family was directly inserted, so this
also reports cross-variable coupling (e.g. does inserting T improve Z/U/V),
analogous to the main manuscript's coupling-score matrix.

Output: one CSV per experiment, columns:
    lead_hour, valid_time, control_rmse_<VAR>, experiment_rmse_<VAR>, ...
written to this directory (PERTURBATION_EXPERIMENTS/FCN/<experiment>/).

Run:
    source /umbc/rs/pi_halem/users/yashd1/venvs/aurora310/bin/activate
    python compute_global_rmse_corr.py
"""
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import xarray as xr
import netCDF4 as nc4
import cftime as cft

BASE      = Path("/umbc/rs/pi_halem/users/yashd1")
FCN_DIR   = BASE / "fcn_inference"
ERA5_DIR  = BASE / "AURORA_DATA" / "rmse_data"
CTRL_FILE = BASE / "FourCastNet" / "forecast_normal_60days_12hr.nc"
OUT_DIR   = BASE / "PERTURBATION_EXPERIMENTS" / "FCN"

EXPERIMENTS = [
    ("t_perturbed_corr",  FCN_DIR / "t_perturbed_corr"  / "forecast_t_perturbed_corr_global_day30_60.nc",  "temp"),
    ("uv_perturbed_corr", FCN_DIR / "uv_perturbed_corr" / "forecast_uv_perturbed_corr_global_day30_60.nc", "uv"),
]

LEVELS = [300, 500, 1000]  # hPa
UPPER_VARS = {"T": "t", "U": "u", "V": "v", "Z": "z", "R": "r"}  # label -> ERA5 shortName (== FCN channel prefix)
# label, ERA5 lookup key (passed to load_era5_sfc, which internally maps via
# _SFC_GRIB_SN), FCN's own saved-channel name -- these differ for wind
# (era5 key "u10"/"v10" vs FCN channel "u10m"/"v10m") though not for t2m;
# conflating them was a real bug caught before running (same class of
# 2t/t2m-vs-u10/u10m naming confusion documented elsewhere in this project).
SFC_VARS = [("T2M", "t2m", "t2m"), ("U10", "u10", "u10m"), ("V10", "v10", "v10m")]
ERA5_PRES  = [ERA5_DIR / f"era5_pressure_2023_{m:02d}.grib" for m in [2, 3, 4]]
ERA5_SURF  = [ERA5_DIR / f"era5_surface_2023_{m:02d}.grib"  for m in [2, 3, 4]]

FCN_LATS  = np.linspace(90.0, -90.0, 721)
FCN_LONS  = np.arange(0.0, 360.0, 0.25)
COS_W_FCN = np.cos(np.deg2rad(FCN_LATS))[:, np.newaxis]   # (721,1)


def global_rmse(fc, truth, w2d):
    """Cosine-lat-weighted global RMSE. fc/truth: (lat,lon); w2d: (lat,1)."""
    w_total = float(w2d.sum()) * fc.shape[1]
    return float(np.sqrt(np.sum(w2d * (fc - truth) ** 2) / w_total))


def build_channel_index(nc_file):
    with nc4.Dataset(str(nc_file)) as ds:
        n_ch = ds.dimensions["channel"].size
        return {str(ds.variables["channel"][i]): i for i in range(n_ch)}


def fcn_valid_times(nc_file):
    with nc4.Dataset(str(nc_file)) as ds:
        tv = ds.variables["time"]
        nums, units, cal = tv[:].data, tv.units, tv.calendar
    return [cft.num2date(n, units, cal, only_use_cftime_datetimes=False) for n in nums]


def ctrl_time_to_dt(hours_since, ref=datetime(2023, 2, 1)):
    return ref + timedelta(hours=float(hours_since))


def load_era5_var(shortname, level_hpa, grib_files):
    dsets = []
    for f in grib_files:
        if not Path(f).exists():
            continue
        dsets.append(xr.open_dataset(str(f), engine="cfgrib",
                      filter_by_keys={"shortName": shortname, "typeOfLevel": "isobaricInhPa",
                                       "level": int(level_hpa)}))
    if not dsets:
        return None, None
    combined = xr.concat(dsets, dim="time").sortby("time")
    times, data = combined.time.values, combined[shortname].values.astype(np.float32)
    for ds in dsets: ds.close()
    return times, data


_SFC_GRIB_SN = {"t2m": "2t", "u10": "10u", "v10": "10v"}
def load_era5_sfc(shortname, grib_files):
    grib_sn = _SFC_GRIB_SN.get(shortname, shortname)
    dsets = []
    for f in grib_files:
        if not Path(f).exists():
            continue
        dsets.append(xr.open_dataset(str(f), engine="cfgrib", filter_by_keys={"shortName": grib_sn}))
    if not dsets:
        return None, None
    combined = xr.concat(dsets, dim="time").sortby("time")
    times, data = combined.time.values, combined[shortname].values.astype(np.float32)
    for ds in dsets: ds.close()
    return times, data


def nearest_idx(times, target_dt, tol_hours=3):
    vt = np.datetime64(target_dt)
    deltas = np.abs(times - vt)
    idx = int(np.argmin(deltas))
    return idx if deltas[idx] <= np.timedelta64(tol_hours, "h") else None


def load_fcn_channel(ds, ch_idx):
    return ds.variables["prediction"][:, ch_idx, :, :].data.astype(np.float32)


print("Loading FCN control ...")
ctrl_ds = nc4.Dataset(str(CTRL_FILE))
ctrl_t  = ctrl_ds.variables["time"][:].data
ctrl_dts_np = np.array([np.datetime64(ctrl_time_to_dt(h)) for h in ctrl_t])
CTRL_CH_IDX = build_channel_index(CTRL_FILE)

print(f"Loading ERA5 truth: {list(UPPER_VARS)} x {LEVELS}hPa + surface {[l for l,_,_ in SFC_VARS]} ...")
ERA5_UPPER = {}  # (label, level) -> (times, data)
for label, sn in UPPER_VARS.items():
    for lev in LEVELS:
        try:
            ERA5_UPPER[(label, lev)] = load_era5_var(sn, lev, ERA5_PRES)
        except Exception as e:
            # Some shortName/level combinations are genuinely absent from this
            # ERA5 grib archive (empty filtered dataset) -- skip that one
            # combination rather than crashing the whole loading loop, same
            # defensive pattern as the per-level grib loop in
            # compute_correlation_length_corr.py.
            print(f"  {label}{lev}: FAILED to load ({e!r}), skipping this combination.")
            ERA5_UPPER[(label, lev)] = (None, None)
            continue
        print(f"  {label}{lev}: {'OK' if ERA5_UPPER[(label, lev)][0] is not None else 'MISSING'}")
ERA5_SFC = {label: load_era5_sfc(era5_key, ERA5_SURF) for label, era5_key, _ in SFC_VARS}

VAR_PAIRS = [(f"{label}{lev}", f"{UPPER_VARS[label]}{lev}", *ERA5_UPPER[(label, lev)])
             for label in UPPER_VARS for lev in LEVELS]
VAR_PAIRS += [(label, fcn_ch, *ERA5_SFC[label]) for label, _, fcn_ch in SFC_VARS]

for exp_name, nc_file, ins_type in EXPERIMENTS:
    out_dir = OUT_DIR / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if not nc_file.exists():
        print(f"[{exp_name}] SKIP — output not yet present at {nc_file} (job still running?).")
        continue

    print(f"\n=== {exp_name} ===")
    ch_idx = build_channel_index(nc_file)
    valid_times = fcn_valid_times(nc_file)

    # Open the (large, NFS-hosted) experiment file exactly once and reuse the
    # handle for every channel read below -- repeatedly open/closing the same
    # multi-GB file per-channel was found to segfault on this filesystem
    # (same HDF5/NFS flakiness class already seen on the Prithvi control file).
    exp_ds = nc4.Dataset(str(nc_file))

    # Every variable/level is evaluated for every experiment (not just the
    # directly-inserted family) so this also reports cross-variable coupling.
    var_pairs = VAR_PAIRS

    # Full-array load per variable (avoid re-opening the file per step)
    result = {"lead_hour": [s * 6 for s in range(len(valid_times))],
              "valid_time": [str(v) for v in valid_times]}
    for label, ch_name, era5_times, era5_data in var_pairs:
        if ch_name not in ch_idx or era5_times is None:
            print(f"  [{exp_name}] {label}: channel or ERA5 data unavailable, skipping.")
            continue
        fc_data = load_fcn_channel(exp_ds, ch_idx[ch_name])   # (n_steps, 721, 1440)
        exp_rmse = np.full(len(valid_times), np.nan, dtype=np.float64)
        ctrl_rmse = np.full(len(valid_times), np.nan, dtype=np.float64)
        ctrl_ch = CTRL_CH_IDX.get(ch_name)
        ctrl_data = load_fcn_channel(ctrl_ds, ctrl_ch) if ctrl_ch is not None else None
        for s, vt in enumerate(valid_times):
            ei = nearest_idx(era5_times, vt)
            if ei is None:
                continue
            exp_rmse[s] = global_rmse(fc_data[s], era5_data[ei], COS_W_FCN)
            if ctrl_data is not None:
                ci = nearest_idx(ctrl_dts_np, np.datetime64(vt))
                if ci is not None:
                    ctrl_rmse[s] = global_rmse(ctrl_data[ci], era5_data[ei], COS_W_FCN)
        result[f"control_rmse_{label}"] = ctrl_rmse
        result[f"experiment_rmse_{label}"] = exp_rmse
        print(f"  [{exp_name}] {label}: mean control_rmse={np.nanmean(ctrl_rmse):.4f}  "
              f"mean experiment_rmse={np.nanmean(exp_rmse):.4f}")

        # Write after every variable, not just once at the end, so results
        # already computed survive even if a later variable/step crashes.
        out_csv = out_dir / f"global_rmse_{exp_name}.csv"
        pd.DataFrame(result).to_csv(out_csv, index=False)

    print(f"  Saved: {out_dir / f'global_rmse_{exp_name}.csv'}")
    exp_ds.close()

ctrl_ds.close()
print("\nDone.")
