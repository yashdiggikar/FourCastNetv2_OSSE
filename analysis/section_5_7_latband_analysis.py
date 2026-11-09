#!/usr/bin/env python3
"""
Section 5.7  Latitude-Band / Dynamical-Regime RMSE Analysis — FCNv2
Figures 5.35 (setup), 5.36 (T-ins), 5.37 (Wind-ins), 5.38 (Z-ins), 5.39 (bar summary)

All RMSE values are cosine-latitude weighted within each band.
Time series x-axis: lead day 0–60 (12-hourly FCN steps).
"""

import warnings; warnings.filterwarnings("ignore")
import gc, sys
from pathlib import Path
from datetime import datetime, timedelta

import h5py
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE     = Path("/umbc/rs/pi_halem/users/yashd1/FourCastNet")
ERA5DIR  = Path("/umbc/rs/pi_halem/users/yashd1/AURORA_DATA/rmse_data")
OUTDIR   = BASE / "lats"
VALDIR   = OUTDIR / "values"
PLOTDIR  = OUTDIR / "plots"
LOGFILE  = OUTDIR / "logs" / "section_5_7_debug_log.txt"

for d in [OUTDIR, VALDIR, PLOTDIR, OUTDIR / "logs"]:
    d.mkdir(parents=True, exist_ok=True)

log = open(LOGFILE, "w", buffering=1)
def L(msg):
    print(msg); log.write(msg + "\n")

# ─── FCN run files ─────────────────────────────────────────────────────────────
FCN_FILES = {
    "control":  BASE / "forecast_normal_60days_12hr.nc",
    "t_ins":    BASE / "Prediction_Files_Fcn/t_ins/t_ins/forecast_truth_injected_60days_12hr_FINAL.nc",
    "wind_ins": BASE / "Prediction_Files_Fcn/wind/wind_ins/forecast_uv_injected_60days_12hr_FINAL.nc",
    "z_ins":    BASE / "Prediction_Files_Fcn/z_ins/forecast_z_injected_60days_12hr_FINAL.nc",
}

# FCN channel indices (verified from file)
CH = {"u10m": 0, "v10m": 1, "u500": 15, "v500": 28, "z500": 41, "t500": 54}

# ERA5 file lists
ERA5_PRES = [ERA5DIR / f"era5_pressure_2023_{m:02d}.grib" for m in [2, 3, 4]]
ERA5_SURF = [ERA5DIR / f"era5_surface_2023_{m:02d}.grib"  for m in [2, 3, 4]]

for p in ERA5_PRES + ERA5_SURF:
    L(f"{'FOUND' if p.exists() else 'MISSING'}: {p.name}")

# ─── Time axis ─────────────────────────────────────────────────────────────────
FCN_IC   = datetime(2023, 2, 1, 0, 0)
N_STEPS  = 120
fcn_times  = np.array([FCN_IC + timedelta(hours=12*k) for k in range(N_STEPS)])
lead_days  = np.array([0.5 * k for k in range(N_STEPS)])   # 0.0, 0.5, … 59.5

INIT_STEPS   = np.where(lead_days <  30)[0]   # day  0 – 29.5
MATURE_STEPS = np.where(lead_days >= 30)[0]   # day 30 – 59.5

# ─── Grid ──────────────────────────────────────────────────────────────────────
NLAT, NLON = 721, 1440
lats_1d    = np.linspace(90, -90, NLAT)    # 90 → -90
lons_1d    = np.linspace(0, 359.75, NLON)
cos_w_1d   = np.cos(np.deg2rad(lats_1d))  # (721,)
cos_w_2d   = cos_w_1d[:, None] * np.ones((1, NLON))  # (721, 1440)

# ─── Latitude bands ────────────────────────────────────────────────────────────
BANDS = {
    "Tropics":      [(-30,  30)],
    "Midlatitudes": [(-60, -30), (30,  60)],
    "Polar":        [(-90, -60), (60,  90)],
}
BAND_COLORS_MAP = {"Tropics": "#e74c3c", "Midlatitudes": "#27ae60", "Polar": "#2980b9"}

SLICE_CENTERS = [2, 14, 40, 60]   # NH representative slice centres

def _band_mask(lat_ranges):
    m = np.zeros(NLAT, bool)
    for lo, hi in lat_ranges:
        m |= (lats_1d >= lo) & (lats_1d < hi)
    return m

BAND_MASKS = {name: _band_mask(ranges) for name, ranges in BANDS.items()}

def weighted_rmse(fcn_t, era5_t, lat_mask):
    sq   = (fcn_t - era5_t) ** 2
    w    = cos_w_2d * lat_mask[:, None]
    wsum = w.sum()
    return float(np.sqrt((sq * w).sum() / wsum)) if wsum > 1e-10 else np.nan

# ─── ERA5 loading ──────────────────────────────────────────────────────────────
def _load_era5_month(fpath, shortname, type_of_level, level):
    """Load one month of ERA5, return xr.DataArray with 'time' dim."""
    kw = {"typeOfLevel": type_of_level, "shortName": shortname}
    if level is not None:
        kw["level"] = int(level)
    ds = xr.open_dataset(str(fpath), engine="cfgrib",
                         filter_by_keys=kw,
                         backend_kwargs={"indexpath": ""})
    drop = [c for c in ds.coords
            if ds.coords[c].dims == () or
            c in ("step", "number", "isobaricInhPa", "surface",
                  "valid_time", "heightAboveGround")]
    ds = ds.drop_vars(drop, errors="ignore")
    varname = list(ds.data_vars)[0]
    return ds[varname]   # DataArray with dims (time, latitude, longitude)

def load_era5_for_fcn(shortname, type_of_level, level, era5_files, target_times):
    """
    Load ERA5 from a list of monthly GRIB files.
    Match each FCN step to the nearest ERA5 timestamp (tolerance ≤1 h).
    Returns ndarray of shape (N_STEPS, NLAT, NLON), NaN where no match.
    """
    result = np.full((N_STEPS, NLAT, NLON), np.nan, dtype=np.float32)

    for fpath in era5_files:
        if not fpath.exists():
            continue
        L(f"  Loading {fpath.name}  var={shortname}")
        try:
            da = _load_era5_month(fpath, shortname, type_of_level, level)
        except Exception as e:
            L(f"    SKIP ({e})")
            continue

        era5_ts = pd.to_datetime(da.time.values)

        # Ensure lat is 90→-90 (descending)
        lat_vals = da.latitude.values if "latitude" in da.coords else da.lat.values
        if lat_vals[0] < lat_vals[-1]:           # ascending → flip
            da = da.isel(latitude=slice(None, None, -1))

        arr = da.values.astype(np.float32)       # (T_era5, 721, 1440)
        da.close(); gc.collect()

        for k, tgt in enumerate(target_times):
            if not np.all(np.isnan(result[k])):   # already filled
                continue
            tgt_pd = pd.Timestamp(tgt)
            deltas  = np.abs(era5_ts - tgt_pd)
            idx     = deltas.argmin()
            if deltas[idx].total_seconds() <= 3600:
                result[k] = arr[idx]

        del arr; gc.collect()

    matched = int(np.sum(~np.all(np.isnan(result), axis=(1, 2))))
    L(f"  ERA5 {shortname} (lev={level}): matched {matched}/{N_STEPS} steps")
    return result

# ─── FCN surgical loading ──────────────────────────────────────────────────────
def load_fcn_channel(run_name, ch_key):
    """Load full (N_STEPS, 721, 1440) array for one channel from an FCN run."""
    fpath = FCN_FILES[run_name]
    idx   = CH[ch_key]
    with h5py.File(str(fpath), "r") as f:
        arr = f["prediction"][:, idx, :, :]
    return arr.astype(np.float32)

# ─── Variable definitions ──────────────────────────────────────────────────────
# Each entry: (label, unit, era5_short, era5_level, era5_type, fcn_ch or None)
# fcn_ch=None means it's a derived quantity (wind magnitude)
VARIABLES = {
    "t500":    ("T500",    "K",       "t",   500, "isobaricInhPa"),
    "u500":    ("U500",    "m s⁻¹",   "u",   500, "isobaricInhPa"),
    "z500":    ("Z500",    "m²s⁻²",   "z",   500, "isobaricInhPa"),
    "wind500": ("Wind500", "m s⁻¹",   None,  500, None),
    "wind10":  ("Wind10",  "m s⁻¹",   None,  None, None),
}

RUNS       = ["control", "t_ins", "wind_ins", "z_ins"]
RUN_LABEL  = {"control":"Control", "t_ins":"T-Ins",
              "wind_ins":"Wind-Ins", "z_ins":"Z-Ins"}
RUN_COLOR  = {"control":"#2c3e50", "t_ins":"#e74c3c",
              "wind_ins":"#3498db", "z_ins":"#f39c12"}
RUN_LS     = {"control":"-", "t_ins":"--", "wind_ins":"--", "z_ins":"--"}
RUN_LW     = {"control":2.0, "t_ins":1.7, "wind_ins":1.7, "z_ins":1.7}

# ─── Compute RMSE time series ──────────────────────────────────────────────────
L("\n" + "="*70)
L("COMPUTING RMSE TIME SERIES")
L("="*70)

ALL_RMSE = {}   # (run, var_key) → DataFrame with cols: step, lead_day, Tropics, Midlatitudes, Polar

for var_key, (lbl, unit, era5_short, era5_level, era5_type) in VARIABLES.items():
    L(f"\n{'─'*60}")
    L(f"Variable: {var_key}  ({lbl})")

    # Load ERA5 for this variable
    if var_key == "wind500":
        era5_u = load_era5_for_fcn("u", "isobaricInhPa", 500, ERA5_PRES, fcn_times)
        era5_v = load_era5_for_fcn("v", "isobaricInhPa", 500, ERA5_PRES, fcn_times)
        era5_arr = np.sqrt(era5_u**2 + era5_v**2)
        del era5_u, era5_v; gc.collect()
    elif var_key == "wind10":
        era5_u = load_era5_for_fcn("10u", "surface", None, ERA5_SURF, fcn_times)
        era5_v = load_era5_for_fcn("10v", "surface", None, ERA5_SURF, fcn_times)
        era5_arr = np.sqrt(era5_u**2 + era5_v**2)
        del era5_u, era5_v; gc.collect()
    else:
        era5_arr = load_era5_for_fcn(era5_short, era5_type, era5_level,
                                     ERA5_PRES, fcn_times)

    for run in RUNS:
        L(f"  Run: {run}")

        if var_key == "wind500":
            fcn_u  = load_fcn_channel(run, "u500")
            fcn_v  = load_fcn_channel(run, "v500")
            fcn_arr = np.sqrt(fcn_u**2 + fcn_v**2)
            del fcn_u, fcn_v; gc.collect()
        elif var_key == "wind10":
            fcn_u  = load_fcn_channel(run, "u10m")
            fcn_v  = load_fcn_channel(run, "v10m")
            fcn_arr = np.sqrt(fcn_u**2 + fcn_v**2)
            del fcn_u, fcn_v; gc.collect()
        else:
            fcn_arr = load_fcn_channel(run, var_key)

        rows = []
        for k in range(N_STEPS):
            row = {"step": k, "lead_day": lead_days[k]}
            if np.all(np.isnan(era5_arr[k])):
                for bn in BANDS:
                    row[bn] = np.nan
            else:
                for bn, mask in BAND_MASKS.items():
                    row[bn] = weighted_rmse(fcn_arr[k], era5_arr[k], mask)
            rows.append(row)

        df = pd.DataFrame(rows)
        df["run"] = run
        csv_path = VALDIR / f"FCN_{run}_{var_key}_latband_rmse.csv"
        df.to_csv(csv_path, index=False)
        L(f"    → {csv_path.name}")

        for bn in BANDS:
            mature_rmse = df.loc[df["step"].isin(MATURE_STEPS), bn].mean()
            init_rmse   = df.loc[df["step"].isin(INIT_STEPS),   bn].mean()
            L(f"       {bn:14s}  init={init_rmse:.3f}  mature={mature_rmse:.3f}")

        ALL_RMSE[(run, var_key)] = df
        del fcn_arr; gc.collect()

    del era5_arr; gc.collect()

L("\nAll RMSE time series computed and saved.")

# ─── Pct improvement summary CSV ──────────────────────────────────────────────
INS_RUNS = ["t_ins", "wind_ins", "z_ins"]
pct_rows = []
for ins_run in INS_RUNS:
    for var_key in VARIABLES:
        df_c = ALL_RMSE.get(("control", var_key))
        df_i = ALL_RMSE.get((ins_run, var_key))
        if df_c is None or df_i is None:
            continue
        for phase, steps in [("initial", INIT_STEPS), ("mature", MATURE_STEPS)]:
            for bn in BANDS:
                ctrl_mean = df_c.loc[df_c["step"].isin(steps), bn].mean()
                ins_mean  = df_i.loc[df_i["step"].isin(steps), bn].mean()
                pct = 100.0 * (ctrl_mean - ins_mean) / ctrl_mean if ctrl_mean > 0 else np.nan
                pct_rows.append({
                    "run": ins_run, "variable": var_key, "band": bn,
                    "phase": phase, "ctrl_rmse": ctrl_mean,
                    "ins_rmse": ins_mean, "pct_improvement": pct,
                })

df_pct = pd.DataFrame(pct_rows)
pct_csv = VALDIR / "FCN_latband_pct_improvement_summary.csv"
df_pct.to_csv(pct_csv, index=False)
L(f"\nSaved pct summary: {pct_csv}")

# ─── Plot helpers ──────────────────────────────────────────────────────────────
PROJ = ccrs.PlateCarree()
MATURE_LABEL = "Mature phase: day 30–60  (grey shading)"

def _ts_panel(ax, var_key, band_name, run_focus):
    """Render RMSE time series for all 4 runs on one panel."""
    lbl, unit = VARIABLES[var_key][0], VARIABLES[var_key][1]
    any_data = False
    for run in RUNS:
        df = ALL_RMSE.get((run, var_key))
        if df is None:
            continue
        y = df[band_name].values
        if np.all(np.isnan(y)):
            continue
        ls  = "-" if run in ("control", run_focus) else ":"
        lw  = RUN_LW[run] if run in ("control", run_focus) else 1.0
        ax.plot(lead_days, y,
                color=RUN_COLOR[run], linewidth=lw, linestyle=ls,
                label=RUN_LABEL[run], zorder=4)
        any_data = True

    ax.axvspan(30, 60, alpha=0.10, color="gray", zorder=1)
    ax.axvline(30, color="gray", linewidth=0.8, linestyle="--", alpha=0.7, zorder=2)
    ax.set_xlim(0, 60)
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.grid(True, which="major", alpha=0.3, linewidth=0.5)
    ax.set_xlabel("Lead day", fontsize=8.5)
    ax.set_ylabel(f"RMSE  [{unit}]", fontsize=8.5)
    ax.tick_params(labelsize=8)
    ax.set_title(f"{lbl}  —  {band_name}", fontsize=9.5, fontweight="bold", pad=3)

def _make_ts_figure(fig_num, fig_stem, run_focus, vars_list, title_suffix):
    """2 rows (variables) × 3 cols (bands) RMSE time series figure."""
    nrows   = len(vars_list)
    bnames  = list(BANDS.keys())
    figH    = 4.8 * nrows + 2.2

    fig, axes = plt.subplots(nrows, 3, figsize=(19, figH),
                             gridspec_kw={"hspace": 0.55, "wspace": 0.28})
    if nrows == 1:
        axes = [axes]

    for row, var_key in enumerate(vars_list):
        for col, bn in enumerate(bnames):
            _ts_panel(axes[row][col], var_key, bn, run_focus)

    # Single legend at bottom
    handles, labels = axes[0][0].get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = h
    # Add mature-phase patch manually
    mp = mpatches.Patch(color="gray", alpha=0.3, label="Mature phase (day 30–60)")
    seen["Mature phase (day 30–60)"] = mp
    fig.legend(seen.values(), seen.keys(),
               loc="lower center", ncol=6, fontsize=9,
               framealpha=0.95, bbox_to_anchor=(0.5, -0.015))

    var_str = " & ".join(VARIABLES[v][0] for v in vars_list)
    fig.suptitle(
        f"Figure {fig_num} — FCN {RUN_LABEL[run_focus]}: "
        f"Lat-Band RMSE Time Series\n"
        f"Variables: {var_str}  |  {title_suffix}\n"
        "Cosine-latitude weighted  |  All 4 runs shown",
        fontsize=11, fontweight="bold", y=1.01,
    )

    out = str(PLOTDIR / f"{fig_stem}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    L(f"  Saved: {out}")
    plt.close(fig)

# ─── Figure 5.35 — Latitude-Band Setup Diagram ────────────────────────────────
L("\nFigure 5.35 …")

fig, ax = plt.subplots(figsize=(15, 6.5),
                       subplot_kw={"projection": PROJ})
ax.set_global()
ax.add_feature(cfeature.LAND,      facecolor="#f0f0f0",  zorder=0)
ax.add_feature(cfeature.OCEAN,     facecolor="#dceeff",  zorder=0)
ax.add_feature(cfeature.COASTLINE, linewidth=0.5,         zorder=4)

band_shade = {
    "Tropics":      ("#ffcccc", 0.55, -30, 30),
    "Midlatitudes": ("#ccffcc", 0.50, None, None),
    "Polar":        ("#cce5ff", 0.55, None, None),
}

# Fill tropics
ax.fill_between([-180, 180], -30, 30, transform=PROJ,
                color="#ffcccc", alpha=0.55, zorder=1,
                label="Tropics  (|lat| < 30°)")
# Fill midlats (both hemispheres)
ax.fill_between([-180, 180], 30, 60, transform=PROJ,
                color="#ccffcc", alpha=0.50, zorder=1,
                label="Midlatitudes  (30°–60°)")
ax.fill_between([-180, 180], -60, -30, transform=PROJ,
                color="#ccffcc", alpha=0.50, zorder=1)
# Fill polar
ax.fill_between([-180, 180], 60, 90, transform=PROJ,
                color="#cce5ff", alpha=0.55, zorder=1,
                label="Polar  (|lat| > 60°)")
ax.fill_between([-180, 180], -90, -60, transform=PROJ,
                color="#cce5ff", alpha=0.55, zorder=1)

# Band boundary lines (use ax.plot — axhline doesn't accept transform)
for lat in [-60, -30, 30, 60]:
    ax.plot([-180, 180], [lat, lat], color="#555555", linewidth=0.8,
            linestyle="-", alpha=0.6, transform=PROJ, zorder=5)

# Latitude slice centres (NH)
slice_styles = [
    (2,  "#c0392b", "--", "Centre 2°N (deep tropics)"),
    (14, "#8e44ad", "-.", "Centre 14°N (subtropics)"),
    (40, "#27ae60", ":", "Centre 40°N (midlat storm tracks)"),
    (60, "#2980b9", "--", "Centre 60°N (subpolar)"),
]
for lat, clr, ls, lbl in slice_styles:
    ax.plot([-180, 180], [lat,  lat],  color=clr, linewidth=2.0,
            linestyle=ls, transform=PROJ, zorder=6, label=lbl)
    ax.plot([-180, 180], [-lat, -lat], color=clr, linewidth=1.0,
            linestyle=ls, alpha=0.45, transform=PROJ, zorder=6)

gl = ax.gridlines(draw_labels=True, linewidth=0.4, color="gray",
                  alpha=0.5, linestyle="--")
gl.top_labels = False; gl.right_labels = False

ax.set_title(
    "Figure 5.35 — FCN Section 5.7: Latitude-Band and Slice Setup\n"
    "Broad dynamical bands (shaded)  |  Latitude slice centres (coloured dashed lines)",
    fontsize=11, fontweight="bold", pad=6,
)
ax.legend(loc="lower left", fontsize=8.5, ncol=2, framealpha=0.92,
          edgecolor="#aaaaaa")

out = str(PLOTDIR / "Figure_5_35_FCN_latband_setup.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
L(f"  Saved: {out}")
plt.close(fig)

# ─── Figures 5.36–5.38 — Time Series per Insertion ────────────────────────────
L("\nFigure 5.36 …")
_make_ts_figure(
    "5.36", "Figure_5_36_FCN_t_insertion_latband_rmse",
    run_focus="t_ins",
    vars_list=["u500", "z500"],
    title_suffix="Probes thermal-wind & hypsometric coupling by latitude",
)

L("Figure 5.37 …")
_make_ts_figure(
    "5.37", "Figure_5_37_FCN_wind_insertion_latband_rmse",
    run_focus="wind_ins",
    vars_list=["t500", "z500"],
    title_suffix="Probes horizontal advection & geostrophic coupling by latitude",
)

L("Figure 5.38 …")
_make_ts_figure(
    "5.38", "Figure_5_38_FCN_z_insertion_latband_rmse",
    run_focus="z_ins",
    vars_list=["t500", "wind500"],
    title_suffix="Probes geostrophic balance & hypsometric relationship by latitude",
)

# ─── Figure 5.39 — Grouped Bar Chart Summary ──────────────────────────────────
L("\nFigure 5.39 …")

VARS_BAR = ["t500", "u500", "z500", "wind500", "wind10"]
bnames   = list(BANDS.keys())
phases   = ["initial", "mature"]
PHASE_HATCH = {"initial": "///", "mature": ""}
PHASE_ALPHA = {"initial": 0.65, "mature": 1.0}

n_vars = len(VARS_BAR)
fig, axes = plt.subplots(2, n_vars, figsize=(5 * n_vars, 10),
                         gridspec_kw={"hspace": 0.55, "wspace": 0.35})

x       = np.arange(len(bnames))
width   = 0.26
ins_off = {"t_ins": -width, "wind_ins": 0, "z_ins": width}

for phase_row, phase in enumerate(phases):
    for col, var_key in enumerate(VARS_BAR):
        ax   = axes[phase_row][col]
        lbl  = VARIABLES[var_key][0]
        unit = VARIABLES[var_key][1]

        for ins_run in INS_RUNS:
            vals = []
            for bn in bnames:
                row_sel = df_pct[(df_pct.run==ins_run) &
                                 (df_pct.variable==var_key) &
                                 (df_pct.band==bn) &
                                 (df_pct.phase==phase)]
                v = row_sel["pct_improvement"].values
                vals.append(float(v[0]) if len(v) else np.nan)

            bars = ax.bar(x + ins_off[ins_run], vals, width,
                          color=RUN_COLOR[ins_run],
                          label=RUN_LABEL[ins_run],
                          alpha=PHASE_ALPHA[phase],
                          hatch=PHASE_HATCH[phase],
                          edgecolor="white", linewidth=0.6, zorder=3)

            for bar_obj, v in zip(bars, vals):
                if np.isnan(v):
                    continue
                ypos = bar_obj.get_height() + (0.4 if v >= 0 else -1.5)
                ax.text(bar_obj.get_x() + bar_obj.get_width() / 2,
                        ypos, f"{v:.1f}%",
                        ha="center", va="bottom", fontsize=6.5,
                        rotation=90, color="#222222")

        ax.axhline(0, color="black", linewidth=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(bnames, fontsize=8.5, rotation=10)
        ax.set_ylabel("% RMSE Reduction\nvs Control", fontsize=8.5)
        ax.set_title(f"{lbl}  [{unit}]", fontsize=9.5, fontweight="bold")
        ax.tick_params(labelsize=8)
        ax.grid(axis="y", alpha=0.3, linewidth=0.5)

        if col == 0:
            phase_label = "Early phase (day 0–30)" if phase == "initial" else "Mature phase (day 30–60)"
            ax.set_ylabel(f"{phase_label}\n% RMSE Reduction vs Control", fontsize=8.5)

        if phase_row == 0 and col == n_vars - 1:
            ax.legend(fontsize=8, loc="upper right")

fig.suptitle(
    "Figure 5.39 — FCN Insertion % RMSE Improvement by Latitude Band\n"
    "Top row = Early phase (day 0–30)  |  Bottom row = Mature phase (day 30–60)\n"
    "Positive = RMSE reduction (improvement) vs control  |  Cosine-latitude weighted",
    fontsize=11, fontweight="bold", y=1.01,
)

out = str(PLOTDIR / "Figure_5_39_FCN_latband_bar_summary.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
L(f"  Saved: {out}")
plt.close(fig)

# ─── Finish ────────────────────────────────────────────────────────────────────
L("\n" + "="*70)
L("All Section 5.7 figures and CSVs complete.")
L("="*70)
log.close()
