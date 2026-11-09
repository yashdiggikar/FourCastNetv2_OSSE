#!/usr/bin/env python3
"""
FCNv2 OSSE Analysis
====================
Computes for 6 insertion experiments (land_temp, land_uv, ocean_temp, ocean_uv,
coastal_temp, coastal_uv):
  1. Cosine-lat-weighted global RMSE time series vs ERA5  →  CSV + plot per experiment dir
  2. Spatial propagation maps (experiment - control) at day 30 / 45 / 60  →  plot per exp dir
  3. % RMSE improvement bar chart across all experiments  →  fcn_inference/rmse_improvement_fcn.png

FCN uses 73-channel prediction array (time, channel, lat, lon).
Truth: ERA5 GRIB (same files as Aurora, same period: Feb–Apr 2023).

Run (CPU only):
    source /umbc/rs/pi_halem/users/yashd1/venvs/aurora310/bin/activate
    python analyze_osse.py
"""

import warnings; warnings.filterwarnings("ignore")
import gc
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr
import netCDF4 as nc4
import h5py

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE      = Path("/umbc/rs/pi_halem/users/yashd1")
FCN_DIR   = BASE / "fcn_inference"
ERA5_DIR  = BASE / "AURORA_DATA" / "rmse_data"
CTRL_FILE = BASE / "FourCastNet" / "forecast_normal_60days_12hr.nc"
LSM_PATH  = BASE / "FCN3" / "makani" / "stat-files" / "land_mask.nc"

# ── Experiments (name, nc_file, out_dir, insertion_type) ─────────────────────
EXPERIMENTS = [
    ("land_temp",    FCN_DIR / "forecast_t_landonly_day30_60.nc",         FCN_DIR / "t_land",    "temp"),
    ("land_uv",      FCN_DIR / "uv_land/forecast_uv_landonly_day30_60.nc", FCN_DIR / "uv_land",  "uv"),
    ("ocean_temp",   FCN_DIR / "t_ocean/forecast_t_ocean_day30_60.nc",     FCN_DIR / "t_ocean",  "temp"),
    ("ocean_uv",     FCN_DIR / "uv_ocean/forecast_uv_ocean_day30_60.nc",   FCN_DIR / "uv_ocean", "uv"),
    ("coastal_temp", FCN_DIR / "t_coastal/forecast_t_coastal_day30_60.nc", FCN_DIR / "t_coastal","temp"),
    ("coastal_uv",   FCN_DIR / "uv_coastal/forecast_uv_coastal_day30_60.nc",FCN_DIR / "uv_coastal","uv"),
]

# ── Constants ─────────────────────────────────────────────────────────────────
N_STEPS      = 240
INJECT_STEP  = 120      # injection starts at step index 120 (day 30)
PLOT_LEVEL   = 500      # hPa

ERA5_PRES = [ERA5_DIR / f"era5_pressure_2023_{m:02d}.grib" for m in [2, 3, 4]]
ERA5_SURF = [ERA5_DIR / f"era5_surface_2023_{m:02d}.grib"  for m in [2, 3, 4]]

LABEL_MAP = {
    "land_temp": "Land T Ins", "land_uv": "Land UV Ins",
    "ocean_temp": "Ocean T Ins", "ocean_uv": "Ocean UV Ins",
    "coastal_temp": "Coastal T Ins", "coastal_uv": "Coastal UV Ins",
}
COLOR_MAP = {
    "land_temp": "#d62728", "land_uv": "#ff7f0e",
    "ocean_temp": "#1f77b4", "ocean_uv": "#17becf",
    "coastal_temp": "#2ca02c", "coastal_uv": "#9467bd",
}

# ── FCN channel order ─────────────────────────────────────────────────────────
# Load once to build channel → index mapping
def build_channel_index(nc_file):
    with nc4.Dataset(str(nc_file)) as ds:
        n_ch = ds.dimensions["channel"].size
        return {str(ds.variables["channel"][i]): i for i in range(n_ch)}

# Use t_ocean file (always present)
CH_IDX = build_channel_index(FCN_DIR / "t_ocean/forecast_t_ocean_day30_60.nc")
print(f"FCN channels (total {len(CH_IDX)}): sample = {list(CH_IDX.keys())[:8]}")

T500_CH   = CH_IDX.get("t500",  -1)
U500_CH   = CH_IDX.get("u500",  -1)
V500_CH   = CH_IDX.get("v500",  -1)
T2M_CH    = CH_IDX.get("t2m",   -1)
U10M_CH   = CH_IDX.get("u10m",  -1)
V10M_CH   = CH_IDX.get("v10m",  -1)

# ── FCN time parsing ──────────────────────────────────────────────────────────
import cftime as cft

def fcn_valid_times(nc_file):
    """Return list of datetime objects for each forecast step."""
    with nc4.Dataset(str(nc_file)) as ds:
        tv = ds.variables["time"]
        nums = tv[:].data
        units, cal = tv.units, tv.calendar
    return [cft.num2date(n, units, cal, only_use_cftime_datetimes=False)
            for n in nums]

# FCN control time mapping (units: hours since 2023-02-01 00:00:00, values 0,12,24,...)
def ctrl_time_to_dt(hours_since, ref=datetime(2023, 2, 1)):
    return ref + timedelta(hours=float(hours_since))

# ── Grid / mask ───────────────────────────────────────────────────────────────
# FCN grid: 721 lats (90→-90), 1440 lons (0→359.75)
FCN_LATS = np.linspace(90.0, -90.0, 721)
FCN_LONS = np.arange(0.0, 360.0, 0.25)
cos_w_fcn = np.cos(np.deg2rad(FCN_LATS))[:, np.newaxis]   # (721,1)

print("Loading FCN land-sea mask ...")
with h5py.File(str(LSM_PATH), "r") as f:
    lsm = f["LSM"][0].astype(np.float32)   # (721,1440) or (720,1440)
if lsm.shape[0] == 720:
    lsm = np.concatenate([lsm, lsm[-1:, :]], axis=0)  # pad to 721
land_mask_fcn  = lsm > 0.5
ocean_mask_fcn = ~land_mask_fcn
from scipy.ndimage import binary_dilation
land_dil   = binary_dilation(land_mask_fcn,  iterations=4)
ocean_dil  = binary_dilation(ocean_mask_fcn, iterations=4)
coastal_mask_fcn = land_dil & ocean_dil

# ── Helpers ───────────────────────────────────────────────────────────────────
def global_rmse(fc, truth, w2d):
    w_total = float(w2d.sum()) * fc.shape[1]
    return float(np.sqrt(np.sum(w2d * (fc - truth) ** 2) / w_total))

def load_era5_var(shortname, level_hpa, grib_files):
    print(f"  Loading ERA5 {shortname} @ {level_hpa} hPa ...")
    dsets = []
    for f in grib_files:
        if not Path(f).exists():
            continue
        ds = xr.open_dataset(str(f), engine="cfgrib",
                             filter_by_keys={"shortName": shortname,
                                             "typeOfLevel": "isobaricInhPa",
                                             "level": int(level_hpa)})
        dsets.append(ds)
    if not dsets:
        return None, None
    combined = xr.concat(dsets, dim="time").sortby("time")
    times = combined.time.values
    data  = combined[shortname].values.astype(np.float32)  # (n_t, 721, 1440)
    for ds in dsets: ds.close()
    del combined
    print(f"    → {data.shape}")
    return times, data

# Map xarray variable names → GRIB shortNames for surface fields
_SFC_GRIB_SN = {"t2m": "2t", "u10": "10u", "v10": "10v", "msl": "msl"}

def load_era5_sfc(shortname, grib_files):
    print(f"  Loading ERA5 sfc {shortname} ...")
    grib_sn = _SFC_GRIB_SN.get(shortname, shortname)
    dsets = []
    for f in grib_files:
        if not Path(f).exists():
            continue
        ds = xr.open_dataset(str(f), engine="cfgrib",
                             filter_by_keys={"shortName": grib_sn})
        dsets.append(ds)
    if not dsets:
        return None, None
    combined = xr.concat(dsets, dim="time").sortby("time")
    times = combined.time.values
    data  = combined[shortname].values.astype(np.float32)
    for ds in dsets: ds.close()
    del combined
    return times, data

def era5_idx(era5_times, valid_dt):
    if isinstance(valid_dt, (datetime,)):
        vt = np.datetime64(valid_dt)
    else:
        vt = np.datetime64(valid_dt)
    deltas = np.abs(era5_times - vt)
    idx    = int(np.argmin(deltas))
    if deltas[idx] > np.timedelta64(3, "h"):
        return None
    return idx

def discrete_cmap_norm(vmin, vmax, n=12):
    cmap  = plt.cm.get_cmap("RdBu_r", n)
    bounds = np.linspace(vmin, vmax, n + 1)
    norm  = mcolors.BoundaryNorm(bounds, cmap.N)
    return cmap, norm, bounds

def load_fcn_channel(nc_file, ch_idx):
    """Load a single channel from FCN prediction (time, 721, 1440)."""
    with nc4.Dataset(str(nc_file)) as ds:
        return ds.variables["prediction"][:, ch_idx, :, :].data.astype(np.float32)

# ── Load FCN control ──────────────────────────────────────────────────────────
print("Loading FCN control ...")
ctrl_ds  = nc4.Dataset(str(CTRL_FILE))
ctrl_t   = ctrl_ds.variables["time"][:].data   # hours since 2023-02-01 00:00
ctrl_lats = np.linspace(90.0, -90.0, 721)

# Map control times to datetime for matching
ctrl_dts = [ctrl_time_to_dt(h) for h in ctrl_t]
ctrl_dts_np = np.array([np.datetime64(d) for d in ctrl_dts])

# Find control indices for day 30/45/60
# FCN injection starts at step 120 (lead 720h from start_time)
# FCN control time 720 = index for hours=720 value
def ctrl_idx_for_hour(target_h):
    idx = np.argmin(np.abs(ctrl_t - target_h))
    return int(idx)

CTRL_D30 = ctrl_idx_for_hour(720)
CTRL_D45 = ctrl_idx_for_hour(1080)
CTRL_D60 = ctrl_idx_for_hour(1440)
print(f"  Control indices day30={CTRL_D30}, day45={CTRL_D45}, day60={CTRL_D60}")

# ── Storage ───────────────────────────────────────────────────────────────────
all_rmse = {}   # {exp_name: {varname: np.array(N_STEPS)}}

# ═══════════════════════════════════════════════════════════════════════════════
# PASS 1: T500 (temp experiments + control)
# ═══════════════════════════════════════════════════════════════════════════════
era5_t_times, era5_t500 = load_era5_var("t", PLOT_LEVEL, ERA5_PRES)

for exp_name, nc_file, out_dir, ins_type in EXPERIMENTS:
    if ins_type != "temp" or not Path(nc_file).exists():
        continue
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n--- T500 RMSE: {exp_name} ---")
    vtimes = fcn_valid_times(nc_file)
    t500_fc = load_fcn_channel(nc_file, T500_CH)   # (240,721,1440)
    rmse_arr = np.full(N_STEPS, np.nan)
    for s in range(min(N_STEPS, len(vtimes))):
        ei = era5_idx(era5_t_times, vtimes[s])
        if ei is None: continue
        rmse_arr[s] = global_rmse(t500_fc[s], era5_t500[ei], cos_w_fcn)
    if exp_name not in all_rmse: all_rmse[exp_name] = {}
    all_rmse[exp_name]["T500"] = rmse_arr
    del t500_fc

# Control T500 RMSE (injection period: ctrl time 720h..1440h)
ctrl_t500_fc = ctrl_ds.variables["prediction"][:, T500_CH, :, :].data.astype(np.float32)
ctrl_t500_rmse_inj = []
for ci in range(CTRL_D30, min(CTRL_D60 + 1, len(ctrl_t))):
    ei = era5_idx(era5_t_times, ctrl_dts[ci])
    if ei is not None:
        ctrl_t500_rmse_inj.append(global_rmse(ctrl_t500_fc[ci], era5_t500[ei], cos_w_fcn))
ctrl_t500_mean_rmse = np.mean(ctrl_t500_rmse_inj) if ctrl_t500_rmse_inj else np.nan
del ctrl_t500_fc
del era5_t500; gc.collect()
print(f"Control T500 mean RMSE (inj period): {ctrl_t500_mean_rmse:.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# PASS 2: U500, V500 (UV experiments + control)
# ═══════════════════════════════════════════════════════════════════════════════
ctrl_u500_mean_rmse = np.nan
ctrl_v500_mean_rmse = np.nan

for shortname, ch_idx, var_key in [("u", U500_CH, "U500"), ("v", V500_CH, "V500")]:
    era5_uv_times, era5_uv500 = load_era5_var(shortname, PLOT_LEVEL, ERA5_PRES)

    for exp_name, nc_file, out_dir, ins_type in EXPERIMENTS:
        if ins_type != "uv" or not Path(nc_file).exists():
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n--- {var_key} RMSE: {exp_name} ---")
        vtimes = fcn_valid_times(nc_file)
        uv500_fc = load_fcn_channel(nc_file, ch_idx)
        rmse_arr = np.full(N_STEPS, np.nan)
        for s in range(min(N_STEPS, len(vtimes))):
            ei = era5_idx(era5_uv_times, vtimes[s])
            if ei is None: continue
            rmse_arr[s] = global_rmse(uv500_fc[s], era5_uv500[ei], cos_w_fcn)
        if exp_name not in all_rmse: all_rmse[exp_name] = {}
        all_rmse[exp_name][var_key] = rmse_arr
        del uv500_fc

    # Control
    ctrl_uv_fc = ctrl_ds.variables["prediction"][:, ch_idx, :, :].data.astype(np.float32)
    ctrl_uv_rmse_inj = []
    for ci in range(CTRL_D30, min(CTRL_D60 + 1, len(ctrl_t))):
        ei = era5_idx(era5_uv_times, ctrl_dts[ci])
        if ei is not None:
            ctrl_uv_rmse_inj.append(global_rmse(ctrl_uv_fc[ci], era5_uv500[ei], cos_w_fcn))
    uv_mean = np.mean(ctrl_uv_rmse_inj) if ctrl_uv_rmse_inj else np.nan
    if shortname == "u":
        ctrl_u500_mean_rmse = uv_mean
    else:
        ctrl_v500_mean_rmse = uv_mean
    del ctrl_uv_fc
    del era5_uv500; gc.collect()

ctrl_spd500_mean_rmse = (
    np.sqrt(ctrl_u500_mean_rmse**2 + ctrl_v500_mean_rmse**2)
    if not (np.isnan(ctrl_u500_mean_rmse) or np.isnan(ctrl_v500_mean_rmse))
    else np.nan
)

# ═══════════════════════════════════════════════════════════════════════════════
# PASS 3: T2M surface (temp experiments)
# ═══════════════════════════════════════════════════════════════════════════════
era5_sfc_times, era5_t2m = load_era5_sfc("t2m", ERA5_SURF)
if era5_t2m is not None:
    for exp_name, nc_file, out_dir, ins_type in EXPERIMENTS:
        if ins_type != "temp" or not Path(nc_file).exists():
            continue
        vtimes = fcn_valid_times(nc_file)
        t2m_fc = load_fcn_channel(nc_file, T2M_CH)
        rmse_t2m = np.full(N_STEPS, np.nan)
        for s in range(min(N_STEPS, len(vtimes))):
            ei = era5_idx(era5_sfc_times, vtimes[s])
            if ei is None: continue
            rmse_t2m[s] = global_rmse(t2m_fc[s], era5_t2m[ei], cos_w_fcn)
        all_rmse[exp_name]["T2M"] = rmse_t2m
        del t2m_fc
    del era5_t2m; gc.collect()

# ═══════════════════════════════════════════════════════════════════════════════
# PASS 4: U10M, V10M surface (UV experiments)
# ═══════════════════════════════════════════════════════════════════════════════
for era5_name, fcn_ch, var_key in [("u10", U10M_CH, "U10M"), ("v10", V10M_CH, "V10M")]:
    era5_sfc_times, era5_sfc = load_era5_sfc(era5_name, ERA5_SURF)
    if era5_sfc is None:
        continue
    for exp_name, nc_file, out_dir, ins_type in EXPERIMENTS:
        if ins_type != "uv" or not Path(nc_file).exists():
            continue
        vtimes = fcn_valid_times(nc_file)
        fc_data = load_fcn_channel(nc_file, fcn_ch)
        rmse_arr = np.full(N_STEPS, np.nan)
        for s in range(min(N_STEPS, len(vtimes))):
            ei = era5_idx(era5_sfc_times, vtimes[s])
            if ei is None: continue
            rmse_arr[s] = global_rmse(fc_data[s], era5_sfc[ei], cos_w_fcn)
        all_rmse[exp_name][var_key] = rmse_arr
        del fc_data
    del era5_sfc; gc.collect()

# ═══════════════════════════════════════════════════════════════════════════════
# Save CSVs and RMSE time series plots
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Saving CSVs and RMSE time series plots ===")
for exp_name, nc_file, out_dir, ins_type in EXPERIMENTS:
    if exp_name not in all_rmse: continue
    out_dir.mkdir(parents=True, exist_ok=True)
    vtimes = fcn_valid_times(nc_file)
    # Lead hours from IC (2023-02-01 00:00 start)
    ic = datetime(2023, 2, 1)
    lead_h = np.array([(vt - ic).total_seconds() / 3600 for vt in vtimes])
    lead_days = lead_h / 24.0
    rmse_d = all_rmse[exp_name]

    df = pd.DataFrame({"lead_hour": lead_h, **rmse_d})
    csv_path = out_dir / f"rmse_{exp_name}.csv"
    df.to_csv(csv_path, index=False)
    print(f"  CSV: {csv_path}")

    fig, ax = plt.subplots(figsize=(12, 5))
    for var_key, rmse_vals in rmse_d.items():
        ax.plot(lead_days, rmse_vals, linewidth=2, label=var_key)
    ax.axvline(x=30, color="gray", linestyle="--", linewidth=1.2, alpha=0.7, label="Injection start (day 30)")
    ax.set_xlabel("Lead time (days from IC 2023-02-01 00:00 UTC)", fontsize=11)
    ax.set_ylabel("Cosine-lat-weighted global RMSE", fontsize=11)
    ax.set_title(f"FCNv2  |  {LABEL_MAP.get(exp_name, exp_name)}  |  RMSE vs ERA5", fontsize=12)
    ax.legend(fontsize=10, framealpha=0.85)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_xlim(0, 60)
    fig.tight_layout()
    fig.savefig(out_dir / f"rmse_timeseries_{exp_name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot: {out_dir}/rmse_timeseries_{exp_name}.png")

# ═══════════════════════════════════════════════════════════════════════════════
# PROPAGATION MAPS — per-grid |error| vs ERA5 at day 30 / 45 / 60
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Generating propagation maps (per-grid RMSE vs ERA5) ===")
lons_2d, lats_2d = np.meshgrid(FCN_LONS, FCN_LATS)

FCN_IC = datetime(2023, 2, 1, 0, 0)
INJ_START_IDX = 120   # first saved step with injection applied (step >= 120 in _iterate, lead=720h)

DAY_CONFIGS = [
    ("First Injection (Day 30)", INJ_START_IDX, FCN_IC + timedelta(hours=720)),
    ("Day 45", 179, FCN_IC + timedelta(hours=1080)),
    ("Day 60", 239, FCN_IC + timedelta(hours=1440)),
]

def rmse_cmap_norm(vmax, n=10):
    """Discrete sequential colormap: light (low RMSE) → dark (high RMSE)."""
    cmap   = plt.cm.get_cmap("YlOrRd", n)
    bounds = np.linspace(0.0, vmax, n + 1)
    norm   = mcolors.BoundaryNorm(bounds, cmap.N)
    return cmap, norm, bounds

# Load ERA5 T500 and U500 for the 3 map snapshots
print("  Reloading ERA5 T500 + U500 for map snapshots ...")
era5_tm, era5_t500m = load_era5_var("t", PLOT_LEVEL, ERA5_PRES)
era5_um, era5_u500m = load_era5_var("u", PLOT_LEVEL, ERA5_PRES)
t500_snaps = np.stack([era5_t500m[era5_idx(era5_tm, vt), :, :] for _, _, vt in DAY_CONFIGS]).astype(np.float32)
u500_snaps = np.stack([era5_u500m[era5_idx(era5_um, vt), :, :] for _, _, vt in DAY_CONFIGS]).astype(np.float32)
del era5_t500m, era5_u500m; gc.collect()

for exp_name, nc_file, out_dir, ins_type in EXPERIMENTS:
    if not Path(nc_file).exists():
        continue
    print(f"  Propagation map: {exp_name} ...")
    ch_idx      = T500_CH if ins_type == "temp" else U500_CH
    var_label   = "T @ 500 hPa (K)" if ins_type == "temp" else "U @ 500 hPa (m/s)"
    truth_snaps = t500_snaps if ins_type == "temp" else u500_snaps

    with nc4.Dataset(str(nc_file)) as ds:
        exp_slices = np.stack([
            ds.variables["prediction"][exp_idx, ch_idx, :, :].data.astype(np.float32)
            for _, exp_idx, _ in DAY_CONFIGS
        ])  # (3, 721, 1440)

    errors = np.abs(exp_slices - truth_snaps)   # per-grid absolute error vs ERA5

    vmax = max(float(np.nanpercentile(errors, 97)), 0.1)
    vmax = round(vmax, 1)
    cmap, norm, bounds = rmse_cmap_norm(vmax, n=10)

    fig, axes = plt.subplots(1, 3, figsize=(21, 7),
                              subplot_kw={"projection": ccrs.PlateCarree()})
    for ax, (day_label, _, _), err in zip(axes, DAY_CONFIGS, errors):
        ax.set_global()
        ax.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor="k")
        ax.add_feature(cfeature.BORDERS,   linewidth=0.3, edgecolor="gray")
        im = ax.pcolormesh(lons_2d, lats_2d, err, cmap=cmap, norm=norm,
                           transform=ccrs.PlateCarree(), rasterized=True)
        ax.set_title(day_label, fontsize=12, fontweight="bold")
        gl = ax.gridlines(draw_labels=True, linewidth=0.4, color="gray", alpha=0.5, linestyle="--")
        gl.top_labels = gl.right_labels = False
        gl.xlocator = mticker.FixedLocator([-120, -60, 0, 60, 120])
        gl.ylocator = mticker.FixedLocator([-60, -30, 0, 30, 60])

    fig.subplots_adjust(bottom=0.18, top=0.90, left=0.03, right=0.97, wspace=0.05)
    cbar_ax = fig.add_axes([0.15, 0.06, 0.70, 0.025])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal", extend="max")
    cbar.set_label(f"|Error| vs ERA5  —  {var_label}", fontsize=11)
    cbar.set_ticks(bounds[::2])
    fig.suptitle(
        f"FCNv2  |  {LABEL_MAP.get(exp_name, exp_name)}  |  Per-Grid |Error| vs ERA5  ({var_label})",
        fontsize=13, fontweight="bold")
    out_path = out_dir / f"propagation_map_{exp_name}.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {out_path}")
    del exp_slices, errors

del t500_snaps, u500_snaps, era5_tm, era5_um; gc.collect()

# ═══════════════════════════════════════════════════════════════════════════════
# % RMSE Improvement Bar Chart
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Computing % RMSE improvement ===")
improvements = {}
for exp_name, nc_file, out_dir, ins_type in EXPERIMENTS:
    if exp_name not in all_rmse:
        improvements[exp_name] = np.nan
        continue
    rmse_d = all_rmse[exp_name]
    if ins_type == "temp" and "T500" in rmse_d:
        exp_mean = np.nanmean(rmse_d["T500"][INJECT_STEP:])
        ctrl_ref  = ctrl_t500_mean_rmse
    elif ins_type == "uv" and "U500" in rmse_d and "V500" in rmse_d:
        exp_mean = np.sqrt(
            np.nanmean(rmse_d["U500"][INJECT_STEP:]) ** 2 +
            np.nanmean(rmse_d["V500"][INJECT_STEP:]) ** 2
        )
        ctrl_ref = ctrl_spd500_mean_rmse
    else:
        improvements[exp_name] = np.nan
        continue
    improvements[exp_name] = ((ctrl_ref - exp_mean) / ctrl_ref * 100.0
                               if ctrl_ref > 0 and not np.isnan(ctrl_ref) else np.nan)

print("  Improvements:", {k: f"{v:.2f}%" for k, v in improvements.items()})

imp_df = pd.DataFrame({"experiment": list(improvements.keys()),
                        "pct_rmse_improvement": list(improvements.values())})
imp_csv = FCN_DIR / "rmse_improvement_fcn.csv"
imp_df.to_csv(imp_csv, index=False)

fig, ax = plt.subplots(figsize=(11, 6))
names  = [LABEL_MAP.get(k, k) for k in improvements]
vals   = [improvements[k] for k in improvements]
colors = [COLOR_MAP.get(k, "steelblue") for k in improvements]
bars = ax.bar(names, vals, color=colors, edgecolor="k", linewidth=0.7, width=0.6)
for bar, v in zip(bars, vals):
    if not np.isnan(v):
        ax.text(bar.get_x() + bar.get_width() / 2, v + (0.5 if v >= 0 else -1.5),
                f"{v:.1f}%", ha="center", va="bottom" if v >= 0 else "top",
                fontsize=10, fontweight="bold")
ax.axhline(0, color="k", linewidth=0.8)
ax.set_ylabel("% RMSE Improvement over Control (days 30–60)", fontsize=11)
ax.set_title("FCNv2  |  % RMSE Improvement by Insertion Experiment\n"
             "(T500 for temp-ins, Wind speed 500 hPa for UV-ins)", fontsize=11)
y_vals = [v for v in vals if not np.isnan(v)]
if y_vals:
    ax.set_ylim(min(y_vals) - 5, max(y_vals) + 5)
plt.xticks(rotation=20, ha="right", fontsize=10)
ax.grid(axis="y", alpha=0.3, linestyle="--")
fig.tight_layout()
bar_path = FCN_DIR / "rmse_improvement_fcn.png"
fig.savefig(bar_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {bar_path}")

ctrl_ds.close()
print("\n=== FCN OSSE analysis complete ===")
