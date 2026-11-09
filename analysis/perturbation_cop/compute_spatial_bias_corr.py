#!/usr/bin/env python3
"""
Spatial bias (mean signed error, forecast - truth) for the FCNv2 Correlated
OSSE Perturbation experiments (t_perturbed_corr, uv_perturbed_corr).

Computed over the POST-INJECTION period only (step >= 120, day 30-60),
for both the control forecast and the experiment forecast, so the two can
be directly compared: does the correlated perturbation introduce spatial
bias patterns the control forecast doesn't have?

This is specifically designed to surface the documented limitation in
PERTURBATION_EXPERIMENTS/METHOD_OVERVIEW.md Section 2.3: cos(phi) clipping
in the streamfunction-derived wind perturbation over/under-scales noise
magnitude near the poles vs. equator — a spatial bias map is the direct
empirical check for whether that shows up as an actual bias pattern.

Output per experiment (into PERTURBATION_EXPERIMENTS/FCN/<experiment>/):
    bias_control_<VAR>.npy, bias_experiment_<VAR>.npy   (lat,lon) mean signed error
    spatial_bias_<experiment>.csv                        summary table (global + 3 lat bands)
    spatial_bias_map_<experiment>_<VAR>.png              experiment-minus-control bias difference map

Run:
    source /umbc/rs/pi_halem/users/yashd1/venvs/aurora310/bin/activate
    python compute_spatial_bias_corr.py
"""
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import xarray as xr
import netCDF4 as nc4
import cftime as cft
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

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
SFC_VARS = [("T2M", "t2m", "t2m"), ("U10", "u10", "u10m"), ("V10", "v10", "v10m")]  # label, era5 key, FCN channel
INJECT_STEP = 120   # steps >= this are post-injection (day 30-60)
ERA5_PRES   = [ERA5_DIR / f"era5_pressure_2023_{m:02d}.grib" for m in [2, 3, 4]]
ERA5_SURF   = [ERA5_DIR / f"era5_surface_2023_{m:02d}.grib"  for m in [2, 3, 4]]

FCN_LATS = np.linspace(90.0, -90.0, 721)
FCN_LONS = np.arange(0.0, 360.0, 0.25)

# Latitude-band masks (diagnoses the cos(phi)-clipping limitation directly)
POLAR_MASK  = np.abs(FCN_LATS) > 60
MIDLAT_MASK = (np.abs(FCN_LATS) <= 60) & (np.abs(FCN_LATS) >= 30)
TROP_MASK   = np.abs(FCN_LATS) < 30


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
                                      "level": int(level_hpa)}, indexpath=None))
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
        dsets.append(xr.open_dataset(str(f), engine="cfgrib", filter_by_keys={"shortName": grib_sn}, indexpath=None))
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


def region_stats(bias_map):
    return {
        "global_mean_bias": float(np.nanmean(bias_map)),
        "global_mean_abs_bias": float(np.nanmean(np.abs(bias_map))),
        "polar_mean_bias": float(np.nanmean(bias_map[POLAR_MASK, :])),
        "midlat_mean_bias": float(np.nanmean(bias_map[MIDLAT_MASK, :])),
        "tropics_mean_bias": float(np.nanmean(bias_map[TROP_MASK, :])),
    }


print("Loading FCN control ...")
ctrl_ds = nc4.Dataset(str(CTRL_FILE))
ctrl_t  = ctrl_ds.variables["time"][:].data
ctrl_dts_np = np.array([np.datetime64(ctrl_time_to_dt(h)) for h in ctrl_t])
CTRL_CH_IDX = build_channel_index(CTRL_FILE)

print(f"Loading ERA5 truth: {list(UPPER_VARS)} x {LEVELS}hPa + surface {[l for l,_,_ in SFC_VARS]} ...")
ERA5_UPPER = {}
for label, sn in UPPER_VARS.items():
    for lev in LEVELS:
        try:
            ERA5_UPPER[(label, lev)] = load_era5_var(sn, lev, ERA5_PRES)
        except Exception as e:
            print(f"  {label}{lev}: FAILED to load ({e!r}), skipping this combination.")
            ERA5_UPPER[(label, lev)] = (None, None)
ERA5_SFC = {label: load_era5_sfc(era5_key, ERA5_SURF) for label, era5_key, _ in SFC_VARS}

VAR_PAIRS = [(f"{label}{lev}", f"{UPPER_VARS[label]}{lev}", *ERA5_UPPER[(label, lev)])
             for label in UPPER_VARS for lev in LEVELS]
VAR_PAIRS += [(label, fcn_ch, *ERA5_SFC[label]) for label, _, fcn_ch in SFC_VARS]

summary_rows = []

for exp_name, nc_file, ins_type in EXPERIMENTS:
    out_dir = OUT_DIR / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if not nc_file.exists():
        print(f"[{exp_name}] SKIP — output not yet present at {nc_file} (job still running?).")
        continue

    print(f"\n=== {exp_name} ===")
    ch_idx = build_channel_index(nc_file)
    valid_times = fcn_valid_times(nc_file)
    post_steps = [s for s in range(len(valid_times)) if s >= INJECT_STEP]

    # Open the (large, NFS-hosted) experiment file exactly once and reuse the
    # handle for every channel read below -- repeatedly open/closing the same
    # multi-GB file per-channel was found to segfault on this filesystem
    # (same HDF5/NFS flakiness class already seen on the Prithvi control file).
    exp_ds = nc4.Dataset(str(nc_file))

    # Every variable/level is evaluated for every experiment (cross-variable
    # coupling included), not just the directly-inserted family.
    var_pairs = VAR_PAIRS
    last_step = len(valid_times) - 1  # day-60-only snapshot, alongside the day-30-60 mean below

    for label, ch_name, era5_times, era5_data in var_pairs:
        if ch_name not in ch_idx or era5_times is None:
            print(f"  [{exp_name}] {label}: unavailable, skipping.")
            continue
        fc_data = load_fcn_channel(exp_ds, ch_idx[ch_name])
        ctrl_ch = CTRL_CH_IDX.get(ch_name)
        ctrl_data = load_fcn_channel(ctrl_ds, ctrl_ch) if ctrl_ch is not None else None

        exp_diffs, ctrl_diffs = [], []
        for s in post_steps:
            vt = valid_times[s]
            ei = nearest_idx(era5_times, vt)
            if ei is None:
                continue
            exp_diffs.append(fc_data[s] - era5_data[ei])
            if ctrl_data is not None:
                ci = nearest_idx(ctrl_dts_np, np.datetime64(vt))
                if ci is not None:
                    ctrl_diffs.append(ctrl_data[ci] - era5_data[ei])

        if not exp_diffs:
            print(f"  [{exp_name}] {label}: no matched timesteps, skipping.")
            continue

        exp_bias = np.nanmean(np.stack(exp_diffs), axis=0)     # (721,1440), mean over day 30-60
        ctrl_bias = np.nanmean(np.stack(ctrl_diffs), axis=0) if ctrl_diffs else np.full_like(exp_bias, np.nan)

        np.save(out_dir / f"bias_experiment_{label}.npy", exp_bias)
        np.save(out_dir / f"bias_control_{label}.npy", ctrl_bias)

        # Day-60-only snapshot (single step), alongside the day-30-60 mean above.
        ei60 = nearest_idx(era5_times, valid_times[last_step])
        exp_bias_day60 = fc_data[last_step] - era5_data[ei60] if ei60 is not None else np.full_like(exp_bias, np.nan)
        ctrl_bias_day60 = np.full_like(exp_bias, np.nan)
        if ctrl_data is not None and ei60 is not None:
            ci60 = nearest_idx(ctrl_dts_np, np.datetime64(valid_times[last_step]))
            if ci60 is not None:
                ctrl_bias_day60 = ctrl_data[ci60] - era5_data[ei60]
        np.save(out_dir / f"bias_experiment_{label}_day60.npy", exp_bias_day60)
        np.save(out_dir / f"bias_control_{label}_day60.npy", ctrl_bias_day60)

        exp_stats  = region_stats(exp_bias)
        ctrl_stats = region_stats(ctrl_bias)
        exp_stats_day60  = region_stats(exp_bias_day60)
        ctrl_stats_day60 = region_stats(ctrl_bias_day60)
        row = {"experiment": exp_name, "variable": label}
        row.update({f"control_{k}": v for k, v in ctrl_stats.items()})
        row.update({f"experiment_{k}": v for k, v in exp_stats.items()})
        row.update({f"control_day60_{k}": v for k, v in ctrl_stats_day60.items()})
        row.update({f"experiment_day60_{k}": v for k, v in exp_stats_day60.items()})
        summary_rows.append(row)
        print(f"  [{exp_name}] {label}: control global_mean_bias={ctrl_stats['global_mean_bias']:+.4f}  "
              f"experiment global_mean_bias={exp_stats['global_mean_bias']:+.4f}  "
              f"(polar: ctrl={ctrl_stats['polar_mean_bias']:+.4f} exp={exp_stats['polar_mean_bias']:+.4f})")

        # Write the CSV after every variable (not just once at the end of the
        # experiment) so the numeric results survive even if the plotting
        # step below crashes -- this is exactly what was lost last run.
        df = pd.DataFrame([r for r in summary_rows if r["experiment"] == exp_name])
        out_csv = out_dir / f"spatial_bias_{exp_name}.csv"
        df.to_csv(out_csv, index=False)

        # Bias-difference map (experiment - control): where did insertion change the bias character?
        diff_map = exp_bias - ctrl_bias
        vmax = np.nanpercentile(np.abs(diff_map), 98)
        fig = plt.figure(figsize=(10, 5))
        ax = plt.axes(projection=ccrs.PlateCarree())
        im = ax.pcolormesh(FCN_LONS, FCN_LATS, diff_map, transform=ccrs.PlateCarree(),
                            cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
        ax.coastlines(linewidth=0.5)
        ax.set_title(f"{exp_name}  {label}  bias(experiment) - bias(control)\n"
                      f"post-injection mean (day 30-60)", fontsize=10)
        plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.05, shrink=0.7, label=label)
        fig.savefig(out_dir / f"spatial_bias_map_{exp_name}_{label}.png", dpi=130, bbox_inches="tight")
        plt.close(fig)

        # Same, but day-60-only snapshot instead of the day-30-60 mean.
        diff_map_day60 = exp_bias_day60 - ctrl_bias_day60
        vmax60 = np.nanpercentile(np.abs(diff_map_day60), 98)
        fig = plt.figure(figsize=(10, 5))
        ax = plt.axes(projection=ccrs.PlateCarree())
        im = ax.pcolormesh(FCN_LONS, FCN_LATS, diff_map_day60, transform=ccrs.PlateCarree(),
                            cmap="RdBu_r", vmin=-vmax60, vmax=vmax60, shading="auto")
        ax.coastlines(linewidth=0.5)
        ax.set_title(f"{exp_name}  {label}  bias(experiment) - bias(control)\n"
                      f"day 60 only", fontsize=10)
        plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.05, shrink=0.7, label=label)
        fig.savefig(out_dir / f"spatial_bias_map_{exp_name}_{label}_day60.png", dpi=130, bbox_inches="tight")
        plt.close(fig)

    print(f"  Saved: {out_dir / f'spatial_bias_{exp_name}.csv'}")
    exp_ds.close()

ctrl_ds.close()
print("\nDone.")
