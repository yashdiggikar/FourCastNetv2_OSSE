#!/usr/bin/env python3
"""
Replot Section 5.7 Figures 5.35–5.39 with fixes:
  1. Fig 5.35  — higher DPI, larger fonts, more visible slice lines
  2. Figs 5.36–38 — panel titles include lat range, e.g. "Tropics (|lat|<30°)"
  3. Fig 5.39  — mature phase only (insertion activates at day 30), clean bar layout
"""

import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator
import cartopy.crs as ccrs
import cartopy.feature as cfeature

VALDIR  = Path("/umbc/rs/pi_halem/users/yashd1/FourCastNet/lats/values")
PLOTDIR = Path("/umbc/rs/pi_halem/users/yashd1/FourCastNet/lats/plots")

BANDS = {
    "Tropics":      [(-30,  30)],
    "Midlatitudes": [(-60, -30), (30,  60)],
    "Polar":        [(-90, -60), (60,  90)],
}
BAND_FULL = {
    "Tropics":      "Tropics  (|lat| < 30°)",
    "Midlatitudes": "Midlatitudes  (30° – 60°)",
    "Polar":        "Polar  (|lat| > 60°)",
}
VARIABLES = {
    "t500":    ("T500",    "K"),
    "u500":    ("U500",    "m s⁻¹"),
    "z500":    ("Z500",    "m²s⁻²"),
    "wind500": ("Wind500", "m s⁻¹"),
    "wind10":  ("Wind10",  "m s⁻¹"),
}
RUNS      = ["control", "t_ins", "wind_ins", "z_ins"]
RUN_LABEL = {"control": "Control", "t_ins": "T-Ins",
             "wind_ins": "Wind-Ins", "z_ins": "Z-Ins"}
RUN_COLOR = {"control": "#2c3e50", "t_ins": "#e74c3c",
             "wind_ins": "#3498db", "z_ins": "#f39c12"}
RUN_LW    = {"control": 2.2, "t_ins": 1.8, "wind_ins": 1.8, "z_ins": 1.8}
INS_RUNS  = ["t_ins", "wind_ins", "z_ins"]

N_STEPS   = 120
lead_days = np.array([0.5 * k for k in range(N_STEPS)])
PROJ      = ccrs.PlateCarree()

# Load all RMSE CSVs
ALL_RMSE = {}
for run in RUNS:
    for var_key in VARIABLES:
        f = VALDIR / f"FCN_{run}_{var_key}_latband_rmse.csv"
        if f.exists():
            ALL_RMSE[(run, var_key)] = pd.read_csv(f)

df_pct = pd.read_csv(VALDIR / "FCN_latband_pct_improvement_summary.csv")


# ════════════════════════════════════════════════════════════════════════════════
# FIGURE 5.35 — Latitude Band Setup (high-quality version)
# ════════════════════════════════════════════════════════════════════════════════
print("Figure 5.35 ...")

fig = plt.figure(figsize=(18, 8))
ax  = fig.add_subplot(1, 1, 1, projection=PROJ)
ax.set_global()

ax.add_feature(cfeature.LAND,      facecolor="#f4f4f4", zorder=0)
ax.add_feature(cfeature.OCEAN,     facecolor="#d6ecf7", zorder=0)
ax.add_feature(cfeature.COASTLINE, linewidth=0.6,        zorder=5)
ax.add_feature(cfeature.BORDERS,   linewidth=0.3, linestyle=":", zorder=5)

# Band fills (both hemispheres)
fill_kw = dict(transform=PROJ, zorder=1)
ax.fill_between([-180, 180], -30,  30, color="#ff9999", alpha=0.40, **fill_kw,
                label="Tropics  (|lat| < 30°)")
ax.fill_between([-180, 180],  30,  60, color="#77dd77", alpha=0.40, **fill_kw,
                label="Midlatitudes  (30°–60°)")
ax.fill_between([-180, 180], -60, -30, color="#77dd77", alpha=0.40, **fill_kw)
ax.fill_between([-180, 180],  60,  90, color="#77aaff", alpha=0.45, **fill_kw,
                label="Polar  (|lat| > 60°)")
ax.fill_between([-180, 180], -90, -60, color="#77aaff", alpha=0.45, **fill_kw)

# Band boundary lines
for lat in [-60, -30, 30, 60]:
    ax.plot([-180, 180], [lat, lat], color="#333333", lw=1.0,
            ls="-", alpha=0.7, transform=PROJ, zorder=6)

# Latitude slice centre lines (NH solid, SH half-opacity)
slices = [
    ( 2, "#c0392b", (8, 4),  "Centre 2°N  —  deep tropics / ITCZ"),
    (14, "#8e44ad", (6, 2),  "Centre 14°N  —  Hadley cell / subtropics"),
    (40, "#27ae60", (4, 2),  "Centre 40°N  —  midlatitude storm tracks"),
    (60, "#1a5276", (3, 1),  "Centre 60°N  —  subpolar / polar front"),
]
for lat, clr, dash, lbl in slices:
    ax.plot([-180, 180], [ lat,  lat], color=clr, lw=2.5, ls=(0, dash),
            transform=PROJ, zorder=7, label=lbl)
    ax.plot([-180, 180], [-lat, -lat], color=clr, lw=1.2, ls=(0, dash),
            alpha=0.45, transform=PROJ, zorder=7)

# Inline lat labels on the right edge of the map
for lat, clr, _, _ in slices:
    ax.text(178, lat + 0.8, f"{lat}°N", transform=PROJ,
            color=clr, fontsize=8, fontweight="bold", ha="right",
            va="bottom", zorder=8)

gl = ax.gridlines(draw_labels=True, linewidth=0.4, color="gray",
                  alpha=0.5, linestyle="--")
gl.top_labels    = False
gl.right_labels  = False
gl.xlabel_style  = {"size": 9}
gl.ylabel_style  = {"size": 9}

ax.set_title(
    "Figure 5.35 — FCN Section 5.7: Latitude-Band and Slice-Centre Setup\n"
    "Shaded = broad dynamical bands  |  Dashed lines = latitude slice centres",
    fontsize=13, fontweight="bold", pad=10,
)
ax.legend(loc="lower left", fontsize=9.5, ncol=2,
          framealpha=0.95, edgecolor="#aaaaaa",
          borderpad=0.8, labelspacing=0.5)

fig.tight_layout(pad=1.5)
out = str(PLOTDIR / "Figure_5_35_FCN_latband_setup.png")
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"  Saved: {out}")
plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════════
# FIGURES 5.36–5.38 — Time Series (with lat range in panel titles)
# ════════════════════════════════════════════════════════════════════════════════

def _ts_panel(ax, var_key, band_name, run_focus):
    lbl, unit = VARIABLES[var_key]
    band_title = BAND_FULL[band_name]

    for run in RUNS:
        df = ALL_RMSE.get((run, var_key))
        if df is None:
            continue
        y = df[band_name].values
        if np.all(np.isnan(y)):
            continue
        ls = "-"  if run in ("control", run_focus) else ":"
        lw = RUN_LW[run] if run in ("control", run_focus) else 1.2
        ax.plot(lead_days, y,
                color=RUN_COLOR[run], lw=lw, ls=ls,
                label=RUN_LABEL[run], zorder=4)

    ax.axvspan(30, 60, alpha=0.10, color="gray", zorder=1)
    ax.axvline(30, color="gray", lw=0.9, ls="--", alpha=0.7, zorder=2)
    ax.set_xlim(0, 60)
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.grid(True, which="major", alpha=0.3, lw=0.5)
    ax.set_xlabel("Lead day", fontsize=9)
    ax.set_ylabel(f"RMSE  [{unit}]", fontsize=9)
    ax.tick_params(labelsize=8.5)
    ax.set_title(f"{lbl}  —  {band_title}", fontsize=10, fontweight="bold", pad=4)

def _make_ts(fig_num, fig_stem, run_focus, vars_list, title_suffix):
    nrows  = len(vars_list)
    bnames = list(BANDS.keys())
    fig, axes = plt.subplots(nrows, 3, figsize=(20, 5.2 * nrows + 2.5),
                             gridspec_kw={"hspace": 0.55, "wspace": 0.30})
    if nrows == 1:
        axes = [axes]

    for row, var_key in enumerate(vars_list):
        for col, bn in enumerate(bnames):
            _ts_panel(axes[row][col], var_key, bn, run_focus)

    # Unified legend at bottom
    handles, labels = axes[0][0].get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = h
    seen["Mature phase (day 30–60)"] = mpatches.Patch(
        color="gray", alpha=0.25, label="Mature phase (day 30–60)")
    fig.legend(seen.values(), seen.keys(),
               loc="lower center", ncol=6, fontsize=9.5,
               framealpha=0.95, bbox_to_anchor=(0.5, -0.015))

    var_str = " & ".join(VARIABLES[v][0] for v in vars_list)
    fig.suptitle(
        f"Figure {fig_num} — FCN {RUN_LABEL[run_focus]}: "
        f"Latitude-Band RMSE Time Series\n"
        f"Variables: {var_str}  |  {title_suffix}\n"
        "Cosine-latitude weighted within each band  |  "
        "Insertion activates at day 30 (mature phase)  |  All 4 runs shown",
        fontsize=11.5, fontweight="bold", y=1.01,
    )
    out = str(PLOTDIR / f"{fig_stem}.png")
    fig.savefig(out, dpi=175, bbox_inches="tight")
    print(f"  Saved: {out}")
    plt.close(fig)


print("Figure 5.36 ...")
_make_ts("5.36", "Figure_5_36_FCN_t_insertion_latband_rmse",
         run_focus="t_ins",
         vars_list=["u500", "z500"],
         title_suffix="Probes thermal-wind & hypsometric coupling by latitude")

print("Figure 5.37 ...")
_make_ts("5.37", "Figure_5_37_FCN_wind_insertion_latband_rmse",
         run_focus="wind_ins",
         vars_list=["t500", "z500"],
         title_suffix="Probes horizontal advection & geostrophic coupling by latitude")

print("Figure 5.38 ...")
_make_ts("5.38", "Figure_5_38_FCN_z_insertion_latband_rmse",
         run_focus="z_ins",
         vars_list=["t500", "wind500"],
         title_suffix="Probes geostrophic balance & hypsometric relationship by latitude")


# ════════════════════════════════════════════════════════════════════════════════
# FIGURE 5.39 — 3 panels (one per experiment), x = downstream variables,
#               bars = latitude bands.  Self-inserted variable excluded per panel.
# ════════════════════════════════════════════════════════════════════════════════
print("Figure 5.39 ...")

df_m = df_pct[df_pct["phase"] == "mature"].copy()

# Variables shown in each insertion panel (exclude the inserted variable itself)
EXP_CONFIG = {
    "t_ins": {
        "title":    "① Temperature Insertion\n(T500 forced = ERA5 truth)",
        "vars":     ["u500", "z500", "wind500", "wind10"],   # skip t500
    },
    "wind_ins": {
        "title":    "② Wind Insertion\n(U500 + V500 forced = ERA5 truth)",
        "vars":     ["t500", "z500", "wind10"],              # skip u500, wind500
    },
    "z_ins": {
        "title":    "③ Geopotential Insertion\n(Z500 forced = ERA5 truth)",
        "vars":     ["t500", "u500", "wind500", "wind10"],   # skip z500
    },
}

# Bar colours = lat bands (warm→cool matches tropics→polar intuitively)
BAND_COLORS = {
    "Tropics":      "#e74c3c",
    "Midlatitudes": "#27ae60",
    "Polar":        "#2980b9",
}
bnames = list(BANDS.keys())
width  = 0.22

fig, axes = plt.subplots(1, 3, figsize=(22, 7),
                          gridspec_kw={"wspace": 0.35})

for col, (ins_run, cfg) in enumerate(EXP_CONFIG.items()):
    ax       = axes[col]
    var_keys = cfg["vars"]
    n_vars   = len(var_keys)
    x        = np.arange(n_vars)
    band_off = {bnames[0]: -width, bnames[1]: 0.0, bnames[2]: width}

    ymax = 0.0

    for bn in bnames:
        vals = []
        for var_key in var_keys:
            row = df_m[(df_m.run == ins_run) &
                       (df_m.variable == var_key) &
                       (df_m.band == bn)]
            vals.append(float(row["pct_improvement"].values[0])
                        if len(row) else float("nan"))

        bars = ax.bar(x + band_off[bn], vals, width,
                      color=BAND_COLORS[bn],
                      label=f"{bn}  ({BAND_FULL[bn].split('(')[1].rstrip(')')})",
                      edgecolor="white", linewidth=0.8, zorder=3)

        for bar_obj, v in zip(bars, vals):
            if np.isnan(v):
                continue
            ymax = max(ymax, v)
            ax.text(bar_obj.get_x() + bar_obj.get_width() / 2,
                    v + 1.2,
                    f"{v:.1f}%",
                    ha="center", va="bottom",
                    fontsize=7.5, fontweight="bold", color="#1a1a1a")

    # x-tick labels: variable name + unit
    xlabels = [f"{VARIABLES[v][0]}\n[{VARIABLES[v][1]}]" for v in var_keys]
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=10, fontweight="bold")
    ax.axhline(0, color="black", lw=1.0, zorder=4)
    ax.set_ylim(0, min(ymax * 1.22 + 8, 115))
    ax.set_ylabel("% RMSE Reduction vs Control", fontsize=10)
    ax.tick_params(axis="y", labelsize=9)
    ax.grid(axis="y", alpha=0.3, lw=0.5, zorder=0)
    ax.set_title(cfg["title"], fontsize=11.5, fontweight="bold", pad=8)
    ax.legend(title="Latitude band", title_fontsize=9,
              fontsize=9, loc="upper right", framealpha=0.92)

fig.suptitle(
    "Figure 5.39 — FCN: Downstream RMSE Improvement by Insertion Type and Latitude Band\n"
    "Mature phase (day 30–60)  |  Each panel = one insertion experiment  |  "
    "Bars = latitude bands  |  Cosine-latitude weighted",
    fontsize=12, fontweight="bold", y=1.02,
)

out = str(PLOTDIR / "Figure_5_39_FCN_latband_bar_summary.png")
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"  Saved: {out}")
plt.close(fig)

print("\nAll figures replotted.")
