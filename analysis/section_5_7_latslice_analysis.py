#!/usr/bin/env python3
"""
Section 5.7 — Latitude-Slice RMSE Analysis (4 specific slices: 2°, 14°, 40°, 60°N)
Each slice is a ±5° band centred on the specified latitude (NH only).
Output → FourCastNet/lat_bands/
"""

import warnings; warnings.filterwarnings("ignore")
import gc
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
BASE    = Path("/umbc/rs/pi_halem/users/yashd1/FourCastNet")
ERA5DIR = Path("/umbc/rs/pi_halem/users/yashd1/AURORA_DATA/rmse_data")
OUTDIR  = BASE / "lat_bands"
VALDIR  = OUTDIR / "values"
PLOTDIR = OUTDIR / "plots"
LOGFILE = OUTDIR / "logs" / "section_5_7_latslice_debug_log.txt"

for d in [OUTDIR, VALDIR, PLOTDIR, OUTDIR / "logs"]:
    d.mkdir(parents=True, exist_ok=True)

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
fcn_times = np.array([FCN_IC + timedelta(hours=12 * k) for k in range(N_STEPS)])
lead_days = np.array([0.5 * k for k in range(N_STEPS)])

MATURE_STEPS = np.where(lead_days >= 30)[0]

# ─── Grid ──────────────────────────────────────────────────────────────────────
NLAT, NLON = 721, 1440
lats_1d    = np.linspace(90, -90, NLAT)
lons_1d    = np.linspace(0, 359.75, NLON)
cos_w_2d   = np.cos(np.deg2rad(lats_1d))[:, None] * np.ones((1, NLON))

# ─── Latitude slice definitions (centre ± 5°, NH) ──────────────────────────────
HALF_WIDTH = 5   # degrees

SLICES = {
    "2°N":  ( 2 - HALF_WIDTH,  2 + HALF_WIDTH),   # -3 to  7
    "14°N": (14 - HALF_WIDTH, 14 + HALF_WIDTH),   #  9 to 19
    "40°N": (40 - HALF_WIDTH, 40 + HALF_WIDTH),   # 35 to 45
    "60°N": (60 - HALF_WIDTH, 60 + HALF_WIDTH),   # 55 to 65
}
SLICE_FULL = {
    "2°N":  "2°N  (−3° – 7°)",
    "14°N": "14°N  (9° – 19°)",
    "40°N": "40°N  (35° – 45°)",
    "60°N": "60°N  (55° – 65°)",
}
SLICE_COLORS = {
    "2°N":  "#e74c3c",   # red — deep tropics
    "14°N": "#e67e22",   # orange — subtropics
    "40°N": "#27ae60",   # green — midlatitudes
    "60°N": "#2980b9",   # blue — subpolar
}

def _slice_mask(lo, hi):
    return (lats_1d >= lo) & (lats_1d < hi)

SLICE_MASKS = {name: _slice_mask(lo, hi) for name, (lo, hi) in SLICES.items()}

def weighted_rmse(fcn_t, era5_t, mask):
    sq   = (fcn_t - era5_t) ** 2
    w    = cos_w_2d * mask[:, None]
    wsum = w.sum()
    return float(np.sqrt((sq * w).sum() / wsum)) if wsum > 1e-10 else np.nan

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

def load_era5_for_fcn(shortname, type_of_level, level, era5_files, target_times):
    result = np.full((N_STEPS, NLAT, NLON), np.nan, dtype=np.float32)
    for fpath in era5_files:
        if not fpath.exists():
            continue
        L(f"  Loading {fpath.name}  var={shortname}")
        try:
            da = _load_era5_month(fpath, shortname, type_of_level, level)
        except Exception as e:
            L(f"    SKIP ({e})"); continue
        era5_ts = pd.to_datetime(da.time.values)
        lat_vals = da.latitude.values if "latitude" in da.coords else da.lat.values
        if lat_vals[0] < lat_vals[-1]:
            da = da.isel(latitude=slice(None, None, -1))
        arr = da.values.astype(np.float32)
        da.close(); gc.collect()
        for k, tgt in enumerate(target_times):
            if not np.all(np.isnan(result[k])):
                continue
            deltas = np.abs(era5_ts - pd.Timestamp(tgt))
            idx    = deltas.argmin()
            if deltas[idx].total_seconds() <= 3600:
                result[k] = arr[idx]
        del arr; gc.collect()
    matched = int(np.sum(~np.all(np.isnan(result), axis=(1, 2))))
    L(f"  ERA5 {shortname} (lev={level}): matched {matched}/{N_STEPS} steps")
    return result

def load_fcn_channel(run_name, ch_key):
    with h5py.File(str(FCN_FILES[run_name]), "r") as f:
        arr = f["prediction"][:, CH[ch_key], :, :]
    return arr.astype(np.float32)

# ─── Variable definitions ──────────────────────────────────────────────────────
VARIABLES = {
    "t500":    ("T500",    "K",       "t",   500, "isobaricInhPa"),
    "u500":    ("U500",    "m s⁻¹",   "u",   500, "isobaricInhPa"),
    "z500":    ("Z500",    "m²s⁻²",   "z",   500, "isobaricInhPa"),
    "wind500": ("Wind500", "m s⁻¹",   None,  None, None),
    "wind10":  ("Wind10",  "m s⁻¹",   None,  None, None),
}
RUNS      = ["control", "t_ins", "wind_ins", "z_ins"]
RUN_LABEL = {"control": "Control", "t_ins": "T-Ins",
             "wind_ins": "Wind-Ins", "z_ins": "Z-Ins"}
RUN_COLOR = {"control": "#2c3e50", "t_ins": "#e74c3c",
             "wind_ins": "#3498db", "z_ins": "#f39c12"}
RUN_LW    = {"control": 2.2, "t_ins": 1.8, "wind_ins": 1.8, "z_ins": 1.8}
INS_RUNS  = ["t_ins", "wind_ins", "z_ins"]

# ─── Compute RMSE time series ──────────────────────────────────────────────────
L("=" * 70 + "\nCOMPUTING LAT-SLICE RMSE TIME SERIES\n" + "=" * 70)

ALL_RMSE = {}

for var_key, (lbl, unit, era5_short, era5_level, era5_type) in VARIABLES.items():
    L(f"\n{'─'*60}\nVariable: {var_key}")

    if var_key == "wind500":
        era5_u   = load_era5_for_fcn("u", "isobaricInhPa", 500, ERA5_PRES, fcn_times)
        era5_v   = load_era5_for_fcn("v", "isobaricInhPa", 500, ERA5_PRES, fcn_times)
        era5_arr = np.sqrt(era5_u**2 + era5_v**2)
        del era5_u, era5_v; gc.collect()
    elif var_key == "wind10":
        era5_u   = load_era5_for_fcn("10u", "surface", None, ERA5_SURF, fcn_times)
        era5_v   = load_era5_for_fcn("10v", "surface", None, ERA5_SURF, fcn_times)
        era5_arr = np.sqrt(era5_u**2 + era5_v**2)
        del era5_u, era5_v; gc.collect()
    else:
        era5_arr = load_era5_for_fcn(era5_short, era5_type, era5_level,
                                     ERA5_PRES, fcn_times)

    for run in RUNS:
        L(f"  Run: {run}")
        if var_key == "wind500":
            fcn_u   = load_fcn_channel(run, "u500")
            fcn_v   = load_fcn_channel(run, "v500")
            fcn_arr = np.sqrt(fcn_u**2 + fcn_v**2)
            del fcn_u, fcn_v; gc.collect()
        elif var_key == "wind10":
            fcn_u   = load_fcn_channel(run, "u10m")
            fcn_v   = load_fcn_channel(run, "v10m")
            fcn_arr = np.sqrt(fcn_u**2 + fcn_v**2)
            del fcn_u, fcn_v; gc.collect()
        else:
            fcn_arr = load_fcn_channel(run, var_key)

        rows = []
        for k in range(N_STEPS):
            row = {"step": k, "lead_day": lead_days[k]}
            if np.all(np.isnan(era5_arr[k])):
                for sn in SLICES: row[sn] = np.nan
            else:
                for sn, mask in SLICE_MASKS.items():
                    row[sn] = weighted_rmse(fcn_arr[k], era5_arr[k], mask)
            rows.append(row)

        df = pd.DataFrame(rows)
        df["run"] = run
        csv_path = VALDIR / f"FCN_{run}_{var_key}_latslice_rmse.csv"
        df.to_csv(csv_path, index=False)
        L(f"    → {csv_path.name}")
        for sn in SLICES:
            m = df.loc[df["step"].isin(MATURE_STEPS), sn].mean()
            L(f"       {sn:6s}  mature={m:.4f}")

        ALL_RMSE[(run, var_key)] = df
        del fcn_arr; gc.collect()

    del era5_arr; gc.collect()

# ─── Pct improvement CSV ───────────────────────────────────────────────────────
pct_rows = []
for ins_run in INS_RUNS:
    for var_key in VARIABLES:
        df_c = ALL_RMSE.get(("control", var_key))
        df_i = ALL_RMSE.get((ins_run,   var_key))
        if df_c is None or df_i is None: continue
        for phase, steps in [("mature", MATURE_STEPS)]:
            for sn in SLICES:
                ctrl_m = df_c.loc[df_c["step"].isin(steps), sn].mean()
                ins_m  = df_i.loc[df_i["step"].isin(steps), sn].mean()
                pct = 100.0 * (ctrl_m - ins_m) / ctrl_m if ctrl_m > 0 else np.nan
                pct_rows.append({"run": ins_run, "variable": var_key,
                                 "slice": sn, "phase": phase,
                                 "ctrl_rmse": ctrl_m, "ins_rmse": ins_m,
                                 "pct_improvement": pct})

df_pct = pd.DataFrame(pct_rows)
df_pct.to_csv(VALDIR / "FCN_latslice_pct_improvement_summary.csv", index=False)
L("\nPct summary saved.")

# ─── Plot helpers ──────────────────────────────────────────────────────────────
PROJ = ccrs.PlateCarree()

def _ts_panel(ax, var_key, slice_name, run_focus):
    lbl, unit = VARIABLES[var_key][0], VARIABLES[var_key][1]
    for run in RUNS:
        df = ALL_RMSE.get((run, var_key))
        if df is None: continue
        y = df[slice_name].values
        if np.all(np.isnan(y)): continue
        ls = "-" if run in ("control", run_focus) else ":"
        lw = RUN_LW[run] if run in ("control", run_focus) else 1.2
        ax.plot(lead_days, y, color=RUN_COLOR[run],
                lw=lw, ls=ls, label=RUN_LABEL[run], zorder=4)
    ax.axvspan(30, 60, alpha=0.10, color="gray", zorder=1)
    ax.axvline(30, color="gray", lw=0.9, ls="--", alpha=0.7, zorder=2)
    ax.set_xlim(0, 60)
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.grid(True, which="major", alpha=0.3, lw=0.5)
    ax.set_xlabel("Lead day", fontsize=8.5)
    ax.set_ylabel(f"RMSE  [{unit}]", fontsize=8.5)
    ax.tick_params(labelsize=8)
    ax.set_title(f"{lbl}  —  {SLICE_FULL[slice_name]}",
                 fontsize=9.5, fontweight="bold", pad=4)

def _make_ts(fig_num, fig_stem, run_focus, vars_list, title_suffix):
    snames = list(SLICES.keys())
    nrows  = len(vars_list)
    fig, axes = plt.subplots(nrows, 4, figsize=(24, 5.2 * nrows + 2.5),
                             gridspec_kw={"hspace": 0.55, "wspace": 0.30})
    if nrows == 1: axes = [axes]
    for row, var_key in enumerate(vars_list):
        for col, sn in enumerate(snames):
            _ts_panel(axes[row][col], var_key, sn, run_focus)

    handles, labels = axes[0][0].get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        if l not in seen: seen[l] = h
    seen["Mature phase (day 30–60)"] = mpatches.Patch(
        color="gray", alpha=0.25, label="Mature phase (day 30–60)")
    fig.legend(seen.values(), seen.keys(),
               loc="lower center", ncol=6, fontsize=9.5,
               framealpha=0.95, bbox_to_anchor=(0.5, -0.015))

    var_str = " & ".join(VARIABLES[v][0] for v in vars_list)
    fig.suptitle(
        f"Figure {fig_num} — FCN {RUN_LABEL[run_focus]}: "
        f"Latitude-Slice RMSE Time Series\n"
        f"Variables: {var_str}  |  {title_suffix}\n"
        "Each slice = ±5° around centre (NH)  |  "
        "Cosine-latitude weighted  |  Insertion activates at day 30",
        fontsize=11.5, fontweight="bold", y=1.01,
    )
    out = str(PLOTDIR / f"{fig_stem}.png")
    fig.savefig(out, dpi=175, bbox_inches="tight")
    L(f"  Saved: {out}"); plt.close(fig)

# ─── Figure LS.1 — Slice Setup Diagram ────────────────────────────────────────
L("\nFigure LS.1 (setup) ...")
fig = plt.figure(figsize=(18, 8))
ax  = fig.add_subplot(1, 1, 1, projection=PROJ)
ax.set_global()
ax.add_feature(cfeature.LAND,      facecolor="#f4f4f4", zorder=0)
ax.add_feature(cfeature.OCEAN,     facecolor="#d6ecf7", zorder=0)
ax.add_feature(cfeature.COASTLINE, linewidth=0.6,        zorder=5)
ax.add_feature(cfeature.BORDERS,   linewidth=0.3, linestyle=":", zorder=5)

slice_fill_alpha = 0.45
for sn, (lo, hi) in SLICES.items():
    clr = SLICE_COLORS[sn]
    ax.fill_between([-180, 180], lo, hi, transform=PROJ,
                    color=clr, alpha=slice_fill_alpha, zorder=2,
                    label=f"{sn}  ({SLICE_FULL[sn].split('(')[1].rstrip(')')})")
    # Centre line
    centre = (lo + hi) / 2
    ax.plot([-180, 180], [centre, centre], color=clr, lw=2.5,
            ls="--", transform=PROJ, zorder=6)
    # SH mirror (half opacity)
    ax.fill_between([-180, 180], -hi, -lo, transform=PROJ,
                    color=clr, alpha=slice_fill_alpha * 0.4, zorder=2)
    ax.plot([-180, 180], [-centre, -centre], color=clr, lw=1.2,
            ls="--", alpha=0.45, transform=PROJ, zorder=6)
    # Right-edge label
    ax.text(178, centre + 0.8, sn, transform=PROJ,
            color=clr, fontsize=9, fontweight="bold", ha="right", va="bottom", zorder=8)

gl = ax.gridlines(draw_labels=True, linewidth=0.4, color="gray", alpha=0.5, linestyle="--")
gl.top_labels = False; gl.right_labels = False
gl.xlabel_style = {"size": 9}; gl.ylabel_style = {"size": 9}

ax.set_title(
    "FCN Section 5.7: Latitude-Slice Setup  (4 slices, each ±5° around centre)\n"
    "Solid colours = NH slices used for RMSE  |  Faded = SH mirror for context",
    fontsize=13, fontweight="bold", pad=10,
)
ax.legend(loc="lower left", fontsize=9.5, ncol=2,
          framealpha=0.95, edgecolor="#aaaaaa", borderpad=0.8)
fig.tight_layout(pad=1.5)
out = str(PLOTDIR / "Figure_LS1_FCN_latslice_setup.png")
fig.savefig(out, dpi=200, bbox_inches="tight")
L(f"  Saved: {out}"); plt.close(fig)

# ─── Figures LS.2–LS.4 — Time Series per Insertion ────────────────────────────
L("\nFigure LS.2 (T-insertion) ...")
_make_ts("LS.2", "Figure_LS2_FCN_t_insertion_latslice_rmse",
         run_focus="t_ins", vars_list=["u500", "z500"],
         title_suffix="Probes thermal-wind & hypsometric coupling per latitude slice")

L("Figure LS.3 (Wind-insertion) ...")
_make_ts("LS.3", "Figure_LS3_FCN_wind_insertion_latslice_rmse",
         run_focus="wind_ins", vars_list=["t500", "z500"],
         title_suffix="Probes advective & geostrophic coupling per latitude slice")

L("Figure LS.4 (Z-insertion) ...")
_make_ts("LS.4", "Figure_LS4_FCN_z_insertion_latslice_rmse",
         run_focus="z_ins", vars_list=["t500", "wind500"],
         title_suffix="Probes geostrophic balance & hypsometric coupling per latitude slice")

# ─── Figure LS.5 — Bar Chart (3 panels × experiment, x = variables, bars = slices)
L("\nFigure LS.5 (bar chart) ...")

EXP_CONFIG = {
    "t_ins":    {"title": "① Temperature Insertion\n(T500 forced = ERA5 truth)",
                 "vars":  ["u500", "z500", "wind500", "wind10"]},
    "wind_ins": {"title": "② Wind Insertion\n(U500+V500 forced = ERA5 truth)",
                 "vars":  ["t500", "z500", "wind10"]},
    "z_ins":    {"title": "③ Geopotential Insertion\n(Z500 forced = ERA5 truth)",
                 "vars":  ["t500", "u500", "wind500", "wind10"]},
}
snames = list(SLICES.keys())
width  = 0.18
n_slices = len(snames)
slice_off = {sn: (i - (n_slices - 1) / 2) * width
             for i, sn in enumerate(snames)}

fig, axes = plt.subplots(1, 3, figsize=(24, 7),
                          gridspec_kw={"wspace": 0.35})

for col, (ins_run, cfg) in enumerate(EXP_CONFIG.items()):
    ax       = axes[col]
    var_keys = cfg["vars"]
    x        = np.arange(len(var_keys))
    ymax     = 0.0

    for sn in snames:
        vals = []
        for var_key in var_keys:
            row = df_pct[(df_pct.run == ins_run) &
                         (df_pct.variable == var_key) &
                         (df_pct["slice"] == sn)]
            vals.append(float(row["pct_improvement"].values[0])
                        if len(row) else float("nan"))

        bars = ax.bar(x + slice_off[sn], vals, width,
                      color=SLICE_COLORS[sn],
                      label=SLICE_FULL[sn],
                      edgecolor="white", linewidth=0.7, zorder=3)

        for bar_obj, v in zip(bars, vals):
            if np.isnan(v): continue
            ymax = max(ymax, v)
            ax.text(bar_obj.get_x() + bar_obj.get_width() / 2,
                    v + 1.2, f"{v:.1f}%",
                    ha="center", va="bottom",
                    fontsize=7, fontweight="bold", color="#1a1a1a")

    xlabels = [f"{VARIABLES[v][0]}\n[{VARIABLES[v][1]}]" for v in var_keys]
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=10, fontweight="bold")
    ax.axhline(0, color="black", lw=1.0, zorder=4)
    ax.set_ylim(0, min(ymax * 1.22 + 8, 115))
    ax.set_ylabel("% RMSE Reduction vs Control", fontsize=10)
    ax.tick_params(axis="y", labelsize=9)
    ax.grid(axis="y", alpha=0.3, lw=0.5, zorder=0)
    ax.set_title(cfg["title"], fontsize=11.5, fontweight="bold", pad=8)
    ax.legend(title="Latitude slice", title_fontsize=9,
              fontsize=9, loc="upper right", framealpha=0.92)

fig.suptitle(
    "FCN: Downstream RMSE Improvement by Insertion Type and Latitude Slice\n"
    "Mature phase (day 30–60)  |  Each panel = one insertion experiment  |  "
    "Bars = lat slices (±5° around centre)  |  Cosine-latitude weighted",
    fontsize=12, fontweight="bold", y=1.02,
)
out = str(PLOTDIR / "Figure_LS5_FCN_latslice_bar_summary.png")
fig.savefig(out, dpi=200, bbox_inches="tight")
L(f"  Saved: {out}"); plt.close(fig)

L("\n" + "=" * 70 + "\nAll lat-slice figures and CSVs complete.\n" + "=" * 70)
log.close()
