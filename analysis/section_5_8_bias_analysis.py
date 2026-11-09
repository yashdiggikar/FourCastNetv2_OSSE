#!/usr/bin/env python3
"""
Section 5.8 FCN Spatial Bias Analysis  (memory-efficient version)
Mature phase: lead day 30 to end of run.
Processes one FCN run at a time to keep peak RAM ~3–4 GB.
"""

import sys
import gc
import warnings
import logging
import glob
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import h5py
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE     = Path("/umbc/rs/pi_halem/users/yashd1")
FCN_DIR  = BASE / "FourCastNet"
ERA5_DIR = BASE / "AURORA_DATA/rmse_data"

CONTROL_FILE = FCN_DIR / "forecast_normal_60days_12hr.nc"
TINS_FILE    = FCN_DIR / "Prediction_Files_Fcn/t_ins/t_ins/forecast_truth_injected_60days_12hr_FINAL.nc"
WIND_FILE    = FCN_DIR / "Prediction_Files_Fcn/wind/wind_ins/forecast_uv_injected_60days_12hr_FINAL.nc"
ZINS_FILE    = FCN_DIR / "Prediction_Files_Fcn/z_ins/forecast_z_injected_60days_12hr_FINAL.nc"

# Mature phase = lead day 30 onwards = 2023-03-03 → 2023-04-01
# So we only need March + April ERA5 files (February not needed)
ERA5_MATURE_FILES = [
    ERA5_DIR / "era5_pressure_2023_03.grib",
    ERA5_DIR / "era5_pressure_2023_04.grib",
]

OUT_DIR  = FCN_DIR / "bias"
VAL_DIR  = OUT_DIR / "values"
PLOT_DIR = OUT_DIR / "plots"
LOG_DIR  = OUT_DIR / "logs"
for d in [VAL_DIR, PLOT_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─── Logging ──────────────────────────────────────────────────────────────────
LOG_FILE = LOG_DIR / "section_5_8_debug_log.txt"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

log.info("=" * 72)
log.info("Section 5.8 FCN Spatial Bias Analysis  [memory-efficient]")
log.info("=" * 72)

# FCN runs (name → file)
FCN_RUNS = {
    "control":  CONTROL_FILE,
    "t_ins":    TINS_FILE,
    "wind_ins": WIND_FILE,
    "z_ins":    ZINS_FILE,
}

# FCN channel names → h5py index lookup done at runtime
FCN_CHANNELS_NEEDED = ["t500", "u500", "v500", "z500"]

saved_csvs  = []
saved_plots = []

UNIT_MAP  = {"t500": "K", "u500": "m s⁻¹", "v500": "m s⁻¹",
             "z500": "m² s⁻²", "wind500": "m s⁻¹"}
LABEL_MAP = {"t500": "T500", "u500": "U500", "v500": "V500",
             "z500": "Z500", "wind500": "Wind500"}

# ═════════════════════════════════════════════════════════════════════════════
# STEP 1 – Identify mature-phase time indices in FCN
# ═════════════════════════════════════════════════════════════════════════════
log.info("\nSTEP 1: Identifying FCN grid, channels, and mature-phase indices")

with h5py.File(str(CONTROL_FILE), "r") as f:
    # Read channel names
    raw_channels = f["channel"][:]
    # h5py returns bytes or str depending on file
    if isinstance(raw_channels[0], (bytes, np.bytes_)):
        fcn_channels = [c.decode("utf-8") for c in raw_channels]
    else:
        fcn_channels = [str(c) for c in raw_channels]

    # Read lat / lon
    fcn_lat = f["lat"][:]        # shape (721,)
    fcn_lon = f["lon"][:]        # shape (1440,)

    # Read time (hours since 2023-02-01 00:00:00)
    time_hours = f["time"][:]    # shape (120,)

log.info(f"  FCN channels (first 8): {fcn_channels[:8]}")
log.info(f"  FCN lat: {fcn_lat[0]:.2f} → {fcn_lat[-1]:.2f} ({len(fcn_lat)} pts)")
log.info(f"  FCN lon: {fcn_lon[0]:.2f} → {fcn_lon[-1]:.2f} ({len(fcn_lon)} pts)")
log.info(f"  FCN time_hours[0]={time_hours[0]}  time_hours[-1]={time_hours[-1]}")

# Build FCN datetime array
IC = pd.Timestamp("2023-02-01 00:00:00")
fcn_times = pd.to_datetime([IC + pd.Timedelta(hours=int(h)) for h in time_hours])
log.info(f"  FCN time range: {fcn_times[0]} → {fcn_times[-1]}  ({len(fcn_times)} steps)")

# Mature phase: lead >= 30 days
mature_start = IC + pd.Timedelta(days=30)   # 2023-03-03 00:00
mature_mask  = fcn_times >= mature_start
mature_idx   = np.where(mature_mask)[0]      # indices into FCN arrays
mature_times = fcn_times[mature_idx]
log.info(f"  IC: {IC}  |  mature start: {mature_start}")
log.info(f"  Mature indices: {mature_idx[0]}…{mature_idx[-1]}  "
         f"({len(mature_idx)} steps, "
         f"{mature_times[0]} → {mature_times[-1]})")

# Channel index map
ch_idx = {}
for name in FCN_CHANNELS_NEEDED:
    if name not in fcn_channels:
        raise ValueError(f"Channel '{name}' missing from FCN file")
    ch_idx[name] = fcn_channels.index(name)
log.info(f"  Channel indices: { {k: v for k, v in ch_idx.items()} }")

# Grid 2D meshgrid for CSV export
LAT2D, LON2D = np.meshgrid(fcn_lat, fcn_lon, indexing="ij")

# ═════════════════════════════════════════════════════════════════════════════
# STEP 2 – Load ERA5 truth for the mature phase (March + April only)
# ═════════════════════════════════════════════════════════════════════════════
log.info("\nSTEP 2: Loading ERA5 truth (mature phase: March–April 2023)")

# ERA5 times that match FCN 12-hourly (00 and 12 UTC) in mature window
target_era5_times = mature_times  # align exactly

def load_era5_var(short_name: str, level: int, target_times: pd.DatetimeIndex) -> np.ndarray:
    """Load ERA5 variable at a single pressure level, subset to target_times.
    Returns (n_times, lat, lon) float32 numpy array aligned to FCN lat/lon."""
    parts = []
    for f in ERA5_MATURE_FILES:
        try:
            ds = xr.open_dataset(
                str(f), engine="cfgrib",
                backend_kwargs={"filter_by_keys": {
                    "shortName": short_name,
                    "typeOfLevel": "isobaricInhPa",
                    "level": level,
                }},
            )
            # Drop scalar coords that interfere with concat
            drop_these = [c for c in ds.coords
                          if ds.coords[c].dims == () or
                          c in ("step", "number", "isobaricInhPa", "valid_time",
                                "surface", "heightAboveGround")]
            ds = ds.drop_vars(drop_these, errors="ignore")
            dvar = list(ds.data_vars)[0]
            da   = ds[dvar]
            if "time" not in da.dims and "valid_time" in da.dims:
                da = da.rename({"valid_time": "time"})
            # Rename lat/lon dims
            rmap = {}
            if "latitude" in da.dims:
                rmap["latitude"] = "lat"
            if "longitude" in da.dims:
                rmap["longitude"] = "lon"
            if rmap:
                da = da.rename(rmap)
            parts.append(da)
            log.info(f"    {short_name}@{level}: {Path(f).name} shape={da.shape}  "
                     f"t0={str(da.time.values[0])[:16]} t1={str(da.time.values[-1])[:16]}")
        except Exception as exc:
            log.warning(f"    SKIP {Path(f).name} for {short_name}: {exc}")

    combined = xr.concat(parts, dim="time").sortby("time")
    # Select only the target times (match FCN 12-hourly)
    era5_pd  = pd.DatetimeIndex(combined.time.values)
    hit      = era5_pd.isin(target_times)
    selected = combined.isel(time=hit)
    # Sort to match target_times order
    selected = selected.sel(time=target_times)
    arr = selected.values.astype(np.float32)
    log.info(f"    {short_name}@{level} combined & selected: shape={arr.shape}  "
             f"range=[{arr.min():.4f}, {arr.max():.4f}]")
    return arr

log.info("  Loading T500 …")
era5_t500 = load_era5_var("t",  500, target_era5_times)
log.info("  Loading U500 …")
era5_u500 = load_era5_var("u",  500, target_era5_times)
log.info("  Loading V500 …")
era5_v500 = load_era5_var("v",  500, target_era5_times)
log.info("  Loading Z500 …")
era5_z500 = load_era5_var("z",  500, target_era5_times)

# Derived wind speed
era5_wind500 = np.sqrt(era5_u500**2 + era5_v500**2).astype(np.float32)
log.info(f"  ERA5 wind500: range=[{era5_wind500.min():.3f}, {era5_wind500.max():.3f}]")

log.info("  Unit check:")
log.info(f"    T500  ERA5: {era5_t500.min():.2f}–{era5_t500.max():.2f} K  (FCN same)")
log.info(f"    Z500  ERA5: {era5_z500.min():.0f}–{era5_z500.max():.0f} m²/s²  (FCN same)")
log.info(f"    ERA5 lat order: {era5_t500.shape} → matches FCN lat 90→-90")

# ═════════════════════════════════════════════════════════════════════════════
# STEP 3 – Load each FCN run, compute mature-phase bias, free memory
# ═════════════════════════════════════════════════════════════════════════════
log.info("\nSTEP 3: Computing spatial bias for each FCN run (one at a time)")

def load_fcn_mature(fpath: Path, channel_name: str) -> np.ndarray:
    """Load single FCN channel at mature-phase time indices via h5py.
    Returns (n_mature, lat, lon) float32 array."""
    cidx = ch_idx[channel_name]
    with h5py.File(str(fpath), "r") as f:
        # Slice: [time_indices, channel, :, :]
        arr = f["prediction"][mature_idx, cidx, :, :]
    return arr.astype(np.float32)

def calc_bias(fcn_arr, era5_arr):
    """Mean(FCN − ERA5) over axis 0 → (lat, lon)."""
    return np.nanmean(fcn_arr - era5_arr, axis=0).astype(np.float64)

# era5 reference dict (already numpy)
ERA5_TRUTH = {
    "t500":    era5_t500,
    "u500":    era5_u500,
    "v500":    era5_v500,
    "z500":    era5_z500,
    "wind500": era5_wind500,
}

# bias[run][var] = 2D (lat, lon) numpy array
bias = {}

for run_name, fpath in FCN_RUNS.items():
    log.info(f"\n  ── Run: {run_name}  ({fpath.name}) ──")
    bias[run_name] = {}
    for var in FCN_CHANNELS_NEEDED:
        log.info(f"    Loading {var} …")
        fcn_arr = load_fcn_mature(fpath, var)
        b = calc_bias(fcn_arr, ERA5_TRUTH[var])
        bias[run_name][var] = b
        log.info(f"    {var} bias: min={b.min():.4f}  max={b.max():.4f}  "
                 f"mean={b.mean():.4f}  MAB={np.abs(b).mean():.4f}")
        del fcn_arr

    # Derive wind500 bias using already-computed FCN wind speed
    log.info(f"    Deriving wind500 …")
    fcn_u = load_fcn_mature(fpath, "u500")
    fcn_v = load_fcn_mature(fpath, "v500")
    fcn_ws = np.sqrt(fcn_u**2 + fcn_v**2).astype(np.float32)
    del fcn_u, fcn_v
    b_ws = calc_bias(fcn_ws, ERA5_TRUTH["wind500"])
    bias[run_name]["wind500"] = b_ws
    log.info(f"    wind500 bias: min={b_ws.min():.4f}  max={b_ws.max():.4f}  "
             f"mean={b_ws.mean():.4f}  MAB={np.abs(b_ws).mean():.4f}")
    del fcn_ws
    gc.collect()

# Release ERA5 full arrays (no longer needed)
del era5_t500, era5_u500, era5_v500, era5_z500, era5_wind500
gc.collect()
log.info("\n  ERA5 full arrays released. Proceeding to bias reduction + plots.")

# ═════════════════════════════════════════════════════════════════════════════
# STEP 4 – Save individual bias CSVs
# ═════════════════════════════════════════════════════════════════════════════
log.info("\nSTEP 4: Saving individual bias CSV files")

def save_bias_csv(arr, run, var, suffix="bias"):
    fpath = str(VAL_DIR / f"FCN_{run}_{var}_{suffix}.csv")
    pd.DataFrame({
        "lat": LAT2D.ravel(), "lon": LON2D.ravel(), suffix: arr.ravel()
    }).to_csv(fpath, index=False)
    log.info(f"  Saved: {fpath}")
    saved_csvs.append(fpath)

for run in FCN_RUNS:
    for var in bias[run]:
        save_bias_csv(bias[run][var], run, var, "bias")

# ═════════════════════════════════════════════════════════════════════════════
# STEP 5 – Bias reduction  (|ctrl| - |ins|)
# ═════════════════════════════════════════════════════════════════════════════
log.info("\nSTEP 5: Calculating bias reduction")

bias_red = {}   # bias_red[run][var] = 2D numpy array

for run in ["t_ins", "wind_ins", "z_ins"]:
    bias_red[run] = {}
    for var in bias.get(run, {}):
        if var not in bias.get("control", {}):
            continue
        cb = bias["control"][var]
        ib = bias[run][var]
        br = np.abs(cb) - np.abs(ib)
        bias_red[run][var] = br
        improved = br.mean() > 0
        log.info(f"  [{run}] {var}: "
                 f"ctrl MAB={np.abs(cb).mean():.4f}  "
                 f"ins MAB={np.abs(ib).mean():.4f}  "
                 f"mean BR={br.mean():.4f}  "
                 f"→ {'↑ IMPROVED' if improved else '↓ WORSENED'}")
        # Save combined CSV
        fpath = str(VAL_DIR / f"FCN_{run}_{var}_combined.csv")
        pd.DataFrame({
            "lat": LAT2D.ravel(), "lon": LON2D.ravel(),
            "control_bias":   cb.ravel(),
            "insertion_bias": ib.ravel(),
            "bias_reduction": br.ravel(),
        }).to_csv(fpath, index=False)
        log.info(f"    Saved: {fpath}")
        saved_csvs.append(fpath)

# ═════════════════════════════════════════════════════════════════════════════
# PLOTTING UTILITIES
# ═════════════════════════════════════════════════════════════════════════════
PROJ = ccrs.PlateCarree()

def _panel(ax, data, lat, lon, title, vmax=None, cmap="RdBu_r"):
    if vmax is None:
        vmax = float(np.nanpercentile(np.abs(data), 97))
    vmax = max(vmax, 1e-9)
    im = ax.pcolormesh(lon, lat, data, transform=PROJ,
                       cmap=cmap, vmin=-vmax, vmax=vmax, shading="auto")
    ax.add_feature(cfeature.COASTLINE,  linewidth=0.5)
    ax.add_feature(cfeature.BORDERS,    linewidth=0.3, linestyle=":")
    ax.set_global()
    ax.gridlines(draw_labels=False, linewidth=0.3, color="gray",
                 alpha=0.5, linestyle="--")
    ax.set_title(title, fontsize=9, fontweight="bold", pad=4)
    return im

def _cbar(fig, ax, im, label):
    cb = fig.colorbar(im, ax=ax, orientation="horizontal",
                      pad=0.04, shrink=0.8, aspect=30)
    cb.set_label(label, fontsize=8)
    cb.ax.tick_params(labelsize=7)

def _save(fig, stem):
    for ext in ("png", "pdf"):
        fp = str(PLOT_DIR / f"{stem}.{ext}")
        fig.savefig(fp, dpi=150, bbox_inches="tight")
        log.info(f"  Saved: {fp}")
        saved_plots.append(fp)
    plt.close(fig)

COL_TTLS = ["Control Bias", "Insertion Bias", "Bias Reduction  (+= improved)"]

mature_label = (f"Mature phase: {mature_times[0].strftime('%b %d')}–"
                f"{mature_times[-1].strftime('%b %d %Y')}  "
                f"({len(mature_times)} × 12-hr steps)")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 5.43 – Control mature-phase bias maps
# ═════════════════════════════════════════════════════════════════════════════
log.info("\nFIGURE 5.43: Control run mature-phase bias maps")

plot_543 = [v for v in ["t500", "u500", "z500", "wind500"]
            if v in bias.get("control", {})]

fig, axes = plt.subplots(
    len(plot_543), 1, figsize=(15, 4.5 * len(plot_543)),
    subplot_kw={"projection": PROJ},
)
if len(plot_543) == 1:
    axes = [axes]

for ax, var in zip(axes, plot_543):
    b    = bias["control"][var]
    unit = UNIT_MAP[var]
    im   = _panel(ax, b, fcn_lat, fcn_lon,
                  f"Control – {LABEL_MAP[var]} Mature-Phase Bias  [{unit}]")
    _cbar(fig, ax, im, unit)
    log.info(f"  {var}: mean bias={b.mean():.4f}  MAB={np.abs(b).mean():.4f} {unit}")

fig.suptitle(f"Figure 5.43 – FCN Control Run: Mature-Phase Mean Bias\n{mature_label}",
             fontsize=12, fontweight="bold", y=1.01)
plt.tight_layout()
_save(fig, "Figure_5_43_FCN_control_mature_phase_bias_maps")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 5.44 – Temperature-insertion: U500 & Z500
# ═════════════════════════════════════════════════════════════════════════════
log.info("\nFIGURE 5.44: Temperature-insertion bias reduction (U500, Z500)")

plot_544 = [v for v in ["u500", "z500"]
            if v in bias.get("control", {}) and v in bias.get("t_ins", {})]

fig, axes = plt.subplots(
    len(plot_544), 3, figsize=(19, 4.5 * len(plot_544)),
    subplot_kw={"projection": PROJ},
)
if len(plot_544) == 1:
    axes = axes[np.newaxis, :]

for row, var in enumerate(plot_544):
    cb_, ib_, br_ = bias["control"][var], bias["t_ins"][var], bias_red["t_ins"][var]
    unit = UNIT_MAP[var];  lbl = LABEL_MAP[var]
    vb   = max(float(np.nanpercentile(np.abs(cb_), 97)),
               float(np.nanpercentile(np.abs(ib_), 97)))
    vr   = float(np.nanpercentile(np.abs(br_), 97))
    im0  = _panel(axes[row,0], cb_, fcn_lat, fcn_lon, f"Control {lbl}  [{unit}]", vmax=vb)
    im1  = _panel(axes[row,1], ib_, fcn_lat, fcn_lon, f"T-Ins {lbl}  [{unit}]",   vmax=vb)
    im2  = _panel(axes[row,2], br_, fcn_lat, fcn_lon, f"{lbl} Bias Reduction  [{unit}]", vmax=vr)
    for im_, ax_ in zip([im0, im1, im2], axes[row]):
        _cbar(fig, ax_, im_, unit)
    log.info(f"  {var}: ctrl MAB={np.abs(cb_).mean():.4f}  t_ins MAB={np.abs(ib_).mean():.4f}  "
             f"mean BR={br_.mean():.4f} ({'↑' if br_.mean()>0 else '↓'})")

for ax_, ct in zip(axes[0], COL_TTLS):
    ax_.set_title(ct, fontsize=10, fontweight="bold", pad=6)

fig.suptitle(f"Figure 5.44 – FCN Temperature-Insertion vs. Control\n"
             f"Mature-Phase Bias: U500 & Z500  |  {mature_label}",
             fontsize=12, fontweight="bold", y=1.01)
plt.tight_layout()
_save(fig, "Figure_5_44_FCN_temperature_insertion_bias_reduction")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 5.45 – Wind-insertion: T500 & Z500
# ═════════════════════════════════════════════════════════════════════════════
log.info("\nFIGURE 5.45: Wind-insertion bias reduction (T500, Z500)")

plot_545 = [v for v in ["t500", "z500"]
            if v in bias.get("control", {}) and v in bias.get("wind_ins", {})]

fig, axes = plt.subplots(
    len(plot_545), 3, figsize=(19, 4.5 * len(plot_545)),
    subplot_kw={"projection": PROJ},
)
if len(plot_545) == 1:
    axes = axes[np.newaxis, :]

for row, var in enumerate(plot_545):
    cb_, ib_, br_ = bias["control"][var], bias["wind_ins"][var], bias_red["wind_ins"][var]
    unit = UNIT_MAP[var];  lbl = LABEL_MAP[var]
    vb   = max(float(np.nanpercentile(np.abs(cb_), 97)),
               float(np.nanpercentile(np.abs(ib_), 97)))
    vr   = float(np.nanpercentile(np.abs(br_), 97))
    im0  = _panel(axes[row,0], cb_, fcn_lat, fcn_lon, f"Control {lbl}  [{unit}]", vmax=vb)
    im1  = _panel(axes[row,1], ib_, fcn_lat, fcn_lon, f"Wind-Ins {lbl}  [{unit}]", vmax=vb)
    im2  = _panel(axes[row,2], br_, fcn_lat, fcn_lon, f"{lbl} Bias Reduction  [{unit}]", vmax=vr)
    for im_, ax_ in zip([im0, im1, im2], axes[row]):
        _cbar(fig, ax_, im_, unit)
    log.info(f"  {var}: ctrl MAB={np.abs(cb_).mean():.4f}  wind_ins MAB={np.abs(ib_).mean():.4f}  "
             f"mean BR={br_.mean():.4f} ({'↑' if br_.mean()>0 else '↓'})")

for ax_, ct in zip(axes[0], COL_TTLS):
    ax_.set_title(ct, fontsize=10, fontweight="bold", pad=6)

fig.suptitle(f"Figure 5.45 – FCN Wind-Insertion vs. Control\n"
             f"Mature-Phase Bias: T500 & Z500  |  {mature_label}",
             fontsize=12, fontweight="bold", y=1.01)
plt.tight_layout()
_save(fig, "Figure_5_45_FCN_wind_insertion_bias_reduction")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 5.46 – Geopotential-insertion: T500 & Wind500
# ═════════════════════════════════════════════════════════════════════════════
log.info("\nFIGURE 5.46: Geopotential-insertion bias reduction (T500, Wind500)")

plot_546 = [v for v in ["t500", "wind500"]
            if v in bias.get("control", {}) and v in bias.get("z_ins", {})]

fig, axes = plt.subplots(
    len(plot_546), 3, figsize=(19, 4.5 * len(plot_546)),
    subplot_kw={"projection": PROJ},
)
if len(plot_546) == 1:
    axes = axes[np.newaxis, :]

for row, var in enumerate(plot_546):
    cb_, ib_, br_ = bias["control"][var], bias["z_ins"][var], bias_red["z_ins"][var]
    unit = UNIT_MAP[var];  lbl = LABEL_MAP[var]
    vb   = max(float(np.nanpercentile(np.abs(cb_), 97)),
               float(np.nanpercentile(np.abs(ib_), 97)))
    vr   = float(np.nanpercentile(np.abs(br_), 97))
    im0  = _panel(axes[row,0], cb_, fcn_lat, fcn_lon, f"Control {lbl}  [{unit}]", vmax=vb)
    im1  = _panel(axes[row,1], ib_, fcn_lat, fcn_lon, f"Z-Ins {lbl}  [{unit}]",   vmax=vb)
    im2  = _panel(axes[row,2], br_, fcn_lat, fcn_lon, f"{lbl} Bias Reduction  [{unit}]", vmax=vr)
    for im_, ax_ in zip([im0, im1, im2], axes[row]):
        _cbar(fig, ax_, im_, unit)
    log.info(f"  {var}: ctrl MAB={np.abs(cb_).mean():.4f}  z_ins MAB={np.abs(ib_).mean():.4f}  "
             f"mean BR={br_.mean():.4f} ({'↑' if br_.mean()>0 else '↓'})")

for ax_, ct in zip(axes[0], COL_TTLS):
    ax_.set_title(ct, fontsize=10, fontweight="bold", pad=6)

fig.suptitle(f"Figure 5.46 – FCN Geopotential-Insertion vs. Control\n"
             f"Mature-Phase Bias: T500 & Wind500  |  {mature_label}",
             fontsize=12, fontweight="bold", y=1.01)
plt.tight_layout()
_save(fig, "Figure_5_46_FCN_geopotential_insertion_bias_reduction")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 5.47 – Failure-mode summary map
# ═════════════════════════════════════════════════════════════════════════════
log.info("\nFIGURE 5.47: Failure-mode summary map")

diag_var = "z500" if "z500" in bias.get("control", {}) else "t500"
log.info(f"  Diagnostic variable: {diag_var}")

cb_diag = bias["control"][diag_var]

# Improvement per insertion (positive = abs bias reduced)
t_impr = bias_red.get("t_ins",    {}).get(diag_var, np.zeros_like(cb_diag))
w_impr = bias_red.get("wind_ins", {}).get(diag_var, np.zeros_like(cb_diag))
g_impr = bias_red.get("z_ins",    {}).get(diag_var, np.zeros_like(cb_diag))

# Threshold: 10% of 90th-percentile of control absolute bias
T_thresh = 0.10 * float(np.percentile(np.abs(cb_diag), 90))
log.info(f"  Classification threshold: {T_thresh:.4f}")

ny, nx = cb_diag.shape
class_arr = np.zeros((ny, nx), dtype=np.int8)

stk      = np.stack([t_impr, w_impr, g_impr], axis=-1)
best_idx = np.argmax(stk, axis=-1)
worst_idx= np.argmin(stk, axis=-1)
max_val  = stk.max(axis=-1)
min_val  = stk.min(axis=-1)

all_improve = (t_impr > T_thresh) & (w_impr > T_thresh) & (g_impr > T_thresh)
all_worsen  = (t_impr < -T_thresh) & (w_impr < -T_thresh) & (g_impr < -T_thresh)

class_arr[(best_idx == 0) & (max_val >  T_thresh)] = 3   # temp best
class_arr[(best_idx == 1) & (max_val >  T_thresh)] = 4   # wind best
class_arr[(best_idx == 2) & (max_val >  T_thresh)] = 5   # geo best
class_arr[(worst_idx== 0) & (min_val < -T_thresh)] = 6   # temp worst
class_arr[(worst_idx== 1) & (min_val < -T_thresh)] = 7   # wind worst
class_arr[(worst_idx== 2) & (min_val < -T_thresh)] = 8   # geo worst
class_arr[all_improve]  = 1
class_arr[all_worsen]   = 2
# 0 = neutral / mixed

CLASS_LABELS = {
    0: "Neutral / mixed",
    1: "All insertions improve",
    2: "All insertions worsen",
    3: "Temp-insertion best",
    4: "Wind-insertion best",
    5: "Geo-insertion best",
    6: "Temp-insertion worst",
    7: "Wind-insertion worst",
    8: "Geo-insertion worst",
}

# Area-weighted percentages
cos_lat = np.cos(np.deg2rad(fcn_lat))
wt_2d   = np.tile(cos_lat[:, np.newaxis], (1, nx))
total_w = float(wt_2d.sum())

class_stats = {}
for cid, cname in CLASS_LABELS.items():
    pct = 100.0 * float(wt_2d[class_arr == cid].sum()) / total_w
    class_stats[cid] = {"name": cname, "pct": pct}
    log.info(f"  Class {cid} ({cname:30s}): {pct:6.2f}%")

# Save failure-mode map CSV
fm_csv = str(VAL_DIR / "FCN_failure_mode_map.csv")
pd.DataFrame({
    "lat": LAT2D.ravel(), "lon": LON2D.ravel(),
    "class_id":   class_arr.ravel(),
    "class_name": [CLASS_LABELS[c] for c in class_arr.ravel()],
}).to_csv(fm_csv, index=False)
log.info(f"  Saved: {fm_csv}")
saved_csvs.append(fm_csv)

# Save area summary CSV
sum_csv = str(VAL_DIR / "FCN_failure_mode_area_summary.csv")
pd.DataFrame([
    {"class_id": cid, "class_name": v["name"], "area_pct": round(v["pct"], 4)}
    for cid, v in class_stats.items()
]).to_csv(sum_csv, index=False)
log.info(f"  Saved: {sum_csv}")
saved_csvs.append(sum_csv)

# Plot
COLORS = ["#b0b0b0", "#2ecc71", "#e74c3c",
          "#3498db", "#9b59b6", "#f39c12",
          "#1a5276", "#7d3c98", "#784212"]
cmap_fm = ListedColormap(COLORS)
norm_fm = BoundaryNorm(np.arange(-0.5, 9.5, 1), cmap_fm.N)

fig, ax = plt.subplots(1, 1, figsize=(16, 7), subplot_kw={"projection": PROJ})
im = ax.pcolormesh(fcn_lon, fcn_lat, class_arr,
                   cmap=cmap_fm, norm=norm_fm, transform=PROJ, shading="auto")
ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
ax.add_feature(cfeature.BORDERS,   linewidth=0.3, linestyle=":")
ax.set_global()
ax.gridlines(draw_labels=False, linewidth=0.3, color="gray", alpha=0.5)

cb = plt.colorbar(im, ax=ax, orientation="vertical", pad=0.02, shrink=0.8)
cb.set_ticks(range(9))
cb.set_ticklabels([f"{i}: {CLASS_LABELS[i]}" for i in range(9)], fontsize=7.5)
cb.set_label("Failure Mode Class", fontsize=9)

ax.set_title(
    f"Figure 5.47 – FCN Insertion Failure-Mode Summary ({LABEL_MAP[diag_var]})\n"
    f"{mature_label}  |  threshold={T_thresh:.3f}",
    fontsize=11, fontweight="bold",
)
ax.text(0.01, 0.01,
        "\n".join([f"{v['name']}: {v['pct']:.1f}%" for v in class_stats.values()]),
        transform=ax.transAxes, fontsize=7, va="bottom", ha="left",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.75))
plt.tight_layout()
_save(fig, "Figure_5_47_FCN_failure_mode_summary_map")

# ═════════════════════════════════════════════════════════════════════════════
# FINAL LOG SUMMARY
# ═════════════════════════════════════════════════════════════════════════════
log.info("\n" + "=" * 72)
log.info("ALL SAVED CSV FILES:")
for f in saved_csvs:
    log.info(f"  {f}")

log.info("\nALL SAVED PLOT FILES:")
for f in sorted(saved_plots):
    log.info(f"  {f}")

log.info("\nBIAS-REDUCTION SUMMARY (mean abs bias vs. control):")
for run in ["t_ins", "wind_ins", "z_ins"]:
    for var in bias_red.get(run, {}):
        br_mean = float(bias_red[run][var].mean())
        ctrl_mab = float(np.abs(bias["control"][var]).mean())
        ins_mab  = float(np.abs(bias[run][var]).mean())
        log.info(f"  {run:12s} {LABEL_MAP[var]:8s}: "
                 f"ctrl MAB={ctrl_mab:.4f}  ins MAB={ins_mab:.4f}  "
                 f"mean BR={br_mean:+.4f}  "
                 f"{'↑ IMPROVED' if br_mean > 0 else '↓ WORSENED'}")

log.info("\n" + "=" * 72)
log.info("Section 5.8 FCN spatial bias analysis completed successfully.")
log.info("=" * 72)

print("\nSection 5.8 FCN spatial bias analysis completed successfully.")
