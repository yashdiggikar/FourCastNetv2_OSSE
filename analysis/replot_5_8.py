#!/usr/bin/env python3
"""
Replot Figures 5.43-5.47 with corrected, unambiguous labeling.
Loads pre-computed bias values from CSV — no raw data needed.

Fixes:
  - Each panel title clearly states variable name AND run type
  - Column headers added via fig.text (do NOT overwrite panel titles)
  - Row labels added on the left margin
  - MAB / mean-BR annotated in each panel corner
  - Figure 5.47 colorbar and legend re-laid out with more space
"""

import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import BoundaryNorm, ListedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ─── Paths ────────────────────────────────────────────────────────────────────
BIAS_DIR = Path("/umbc/rs/pi_halem/users/yashd1/FourCastNet/bias")
VAL_DIR  = BIAS_DIR / "values"
PLOT_DIR = BIAS_DIR / "plots"
PLOT_DIR.mkdir(exist_ok=True)

# Grid dimensions
NLAT, NLON = 721, 1440

UNIT_MAP  = {"t500": "K", "u500": "m s⁻¹", "v500": "m s⁻¹",
             "z500": "m² s⁻²", "wind500": "m s⁻¹"}
LABEL_MAP = {"t500": "T500", "u500": "U500", "v500": "V500",
             "z500": "Z500", "wind500": "Wind500"}
MATURE_LABEL = "Mature phase: Mar 03–Apr 01 2023  (60 × 12-hr steps)"

# ─── Helpers ──────────────────────────────────────────────────────────────────
PROJ = ccrs.PlateCarree()

def _load_bias(run, var):
    """Load 2D bias array from individual bias CSV."""
    f = VAL_DIR / f"FCN_{run}_{var}_bias.csv"
    df = pd.read_csv(f)
    return df.pivot(index="lat", columns="lon", values="bias").values  # (721,1440)

def _load_combined(run, var):
    """Load control_bias, insertion_bias, bias_reduction from combined CSV."""
    f = VAL_DIR / f"FCN_{run}_{var}_combined.csv"
    df = pd.read_csv(f)
    shape = (NLAT, NLON)
    ctrl = df.pivot(index="lat", columns="lon", values="control_bias").values
    ins  = df.pivot(index="lat", columns="lon", values="insertion_bias").values
    br   = df.pivot(index="lat", columns="lon", values="bias_reduction").values
    return ctrl, ins, br

def _get_latlons():
    """Recover lat/lon 1D arrays from any bias CSV."""
    df = pd.read_csv(VAL_DIR / "FCN_control_t500_bias.csv")
    lats = np.sort(df["lat"].unique())[::-1]   # 90 → -90
    lons = np.sort(df["lon"].unique())          # 0 → 359.75
    return lats, lons

LAT, LON = _get_latlons()

def _panel(ax, data, title, vmax=None, cmap="RdBu_r"):
    if vmax is None:
        vmax = float(np.nanpercentile(np.abs(data), 97))
    vmax = max(vmax, 1e-9)
    im = ax.pcolormesh(LON, LAT, data, transform=PROJ,
                       cmap=cmap, vmin=-vmax, vmax=vmax, shading="auto")
    ax.add_feature(cfeature.COASTLINE,  linewidth=0.5, zorder=3)
    ax.add_feature(cfeature.BORDERS,    linewidth=0.25, linestyle=":", zorder=3)
    ax.set_global()
    ax.gridlines(draw_labels=False, linewidth=0.25, color="gray",
                 alpha=0.5, linestyle="--")
    ax.set_title(title, fontsize=9.5, fontweight="bold", pad=4)
    return im

def _annotate(ax, text):
    """Small stat box in bottom-left of each panel."""
    ax.text(0.015, 0.04, text, transform=ax.transAxes,
            fontsize=7.5, va="bottom", ha="left",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray",
                      alpha=0.80, lw=0.5))

def _cbar(fig, ax, im, label):
    cb = fig.colorbar(im, ax=ax, orientation="horizontal",
                      pad=0.04, shrink=0.82, aspect=28)
    cb.set_label(label, fontsize=8)
    cb.ax.tick_params(labelsize=7)

def _col_header(fig, positions_x, y, labels, fontsize=10):
    """Write column-header text above the axes without touching any ax.set_title."""
    for x, lbl in zip(positions_x, labels):
        fig.text(x, y, lbl, ha="center", va="bottom",
                 fontsize=fontsize, fontweight="bold",
                 color="#1a1a2e")

def _row_label(fig, positions_y, x, labels, fontsize=10):
    """Write row-label text to the left of the axes."""
    for y, lbl in zip(positions_y, labels):
        fig.text(x, y, lbl, ha="center", va="center",
                 fontsize=fontsize, fontweight="bold",
                 rotation=90, color="#1a1a2e")

def _save(fig, stem):
    for ext in ("png", "pdf"):
        fp = str(PLOT_DIR / f"{stem}.{ext}")
        fig.savefig(fp, dpi=150, bbox_inches="tight")
        print(f"  Saved: {fp}")
    plt.close(fig)

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 5.43 — Control run bias maps
# ═════════════════════════════════════════════════════════════════════════════
print("\nFigure 5.43 …")

VARS_543 = ["t500", "u500", "z500", "wind500"]
fig, axes = plt.subplots(
    len(VARS_543), 1, figsize=(14, 4.4 * len(VARS_543)),
    subplot_kw={"projection": PROJ},
    gridspec_kw={"hspace": 0.45},
)

for ax, var in zip(axes, VARS_543):
    b    = _load_bias("control", var)
    unit = UNIT_MAP[var]
    lbl  = LABEL_MAP[var]
    mab  = float(np.abs(b).mean())
    mb   = float(b.mean())
    im   = _panel(ax, b,
                  title=f"FCN Control — {lbl} Mean Bias  [{unit}]")
    _annotate(ax, f"MAB = {mab:.3f} {unit}\nMean = {mb:+.3f} {unit}")
    _cbar(fig, ax, im, f"{lbl} Bias  [{unit}]")

fig.suptitle(
    "Figure 5.43 — FCN Control Run: Mature-Phase Mean Bias Maps\n"
    f"{MATURE_LABEL}",
    fontsize=12, fontweight="bold", y=1.005,
)
_save(fig, "Figure_5_43_FCN_control_mature_phase_bias_maps")

# ═════════════════════════════════════════════════════════════════════════════
# SHARED LAYOUT HELPER for 6-panel figures (2 rows × 3 cols)
# ═════════════════════════════════════════════════════════════════════════════
def plot_6panel(vars_list, run_name, run_label, ins_short,
                fig_num, fig_stem, fig_title_suffix):
    """
    vars_list  : list of 2 variable keys, e.g. ['u500','z500']
    run_name   : key into bias dict, e.g. 't_ins'
    run_label  : long label for figure, e.g. 'Temperature Insertion'
    ins_short  : short label for panel, e.g. 'T-Ins'
    """
    COL_LABELS = [
        "① Control Run\n(free forecast, no correction)",
        f"② {run_label}\n(inserted variable = ERA5 truth)",
        "③ Bias Reduction\n(red = improved  |  blue = worsened)",
    ]
    ROW_LABELS = [LABEL_MAP[v] + "\n[" + UNIT_MAP[v] + "]" for v in vars_list]

    fig = plt.figure(figsize=(20, 4.6 * len(vars_list) + 1.8))

    # Leave left margin for row labels and top margin for column headers
    gs = fig.add_gridspec(
        len(vars_list), 3,
        left=0.07, right=0.97,
        top=0.82,  bottom=0.08,
        hspace=0.55, wspace=0.06,
    )
    axes = [[fig.add_subplot(gs[r, c], projection=PROJ)
             for c in range(3)]
            for r in range(len(vars_list))]

    for row, var in enumerate(vars_list):
        ctrl, ins, br = _load_combined(run_name, var)
        unit = UNIT_MAP[var]
        lbl  = LABEL_MAP[var]

        mab_ctrl = float(np.abs(ctrl).mean())
        mab_ins  = float(np.abs(ins).mean())
        mean_br  = float(br.mean())
        pct      = 100.0 * mean_br / mab_ctrl if mab_ctrl > 0 else 0.0

        # Shared vmax for cols 1&2 so colours are directly comparable
        vb = max(float(np.nanpercentile(np.abs(ctrl), 97)),
                 float(np.nanpercentile(np.abs(ins),  97)))
        vr = float(np.nanpercentile(np.abs(br), 97))

        im0 = _panel(axes[row][0], ctrl,
                     title=f"Control — {lbl} Bias  [{unit}]", vmax=vb)
        im1 = _panel(axes[row][1], ins,
                     title=f"{ins_short} — {lbl} Bias  [{unit}]", vmax=vb)
        im2 = _panel(axes[row][2], br,
                     title=f"{lbl} Bias Reduction  [{unit}]", vmax=vr)

        _annotate(axes[row][0], f"MAB = {mab_ctrl:.3f} {unit}")
        _annotate(axes[row][1], f"MAB = {mab_ins:.3f} {unit}")
        _annotate(axes[row][2],
                  f"Mean BR = {mean_br:+.3f} {unit}\n({pct:.1f}% reduction)")

        _cbar(fig, axes[row][0], im0, f"Bias  [{unit}]")
        _cbar(fig, axes[row][1], im1, f"Bias  [{unit}]")
        _cbar(fig, axes[row][2], im2, f"Bias Reduction  [{unit}]")

    # ── Column headers (above axes, not touching any set_title) ──────────────
    # Get axes positions in figure coords to place text centred above each col
    fig.canvas.draw()   # needed for get_position() to be accurate
    for col_idx, col_lbl in enumerate(COL_LABELS):
        ax_pos = axes[0][col_idx].get_position()
        cx = ax_pos.x0 + ax_pos.width / 2
        fig.text(cx, 0.845, col_lbl,
                 ha="center", va="bottom", fontsize=10, fontweight="bold",
                 color="#0a0a2a",
                 bbox=dict(boxstyle="round,pad=0.3", fc="#eef2ff",
                           ec="#8888bb", lw=0.8))

    # ── Row labels (left of axes) ─────────────────────────────────────────────
    for row_idx, row_lbl in enumerate(ROW_LABELS):
        ax_pos = axes[row_idx][0].get_position()
        cy = ax_pos.y0 + ax_pos.height / 2
        fig.text(0.012, cy, row_lbl,
                 ha="center", va="center", fontsize=10, fontweight="bold",
                 rotation=90, color="#0a0a2a",
                 bbox=dict(boxstyle="round,pad=0.3", fc="#fff8ee",
                           ec="#bb8800", lw=0.8))

    # ── Suptitle ──────────────────────────────────────────────────────────────
    vars_str = " & ".join(LABEL_MAP[v] for v in vars_list)
    fig.suptitle(
        f"Figure {fig_num} — FCN {run_label}: Bias Comparison vs. Control\n"
        f"Variables: {vars_str}  |  {fig_title_suffix}\n"
        f"{MATURE_LABEL}",
        fontsize=11.5, fontweight="bold", y=0.985,
    )
    _save(fig, fig_stem)

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 5.44 — Temperature insertion: U500, Z500
# ═════════════════════════════════════════════════════════════════════════════
print("Figure 5.44 …")
plot_6panel(
    vars_list    = ["u500", "z500"],
    run_name     = "t_ins",
    run_label    = "Temperature Insertion",
    ins_short    = "T-Ins",
    fig_num      = "5.44",
    fig_stem     = "Figure_5_44_FCN_temperature_insertion_bias_reduction",
    fig_title_suffix = "Probes thermal-wind & hypsometric coupling",
)

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 5.45 — Wind insertion: T500, Z500
# ═════════════════════════════════════════════════════════════════════════════
print("Figure 5.45 …")
plot_6panel(
    vars_list    = ["t500", "z500"],
    run_name     = "wind_ins",
    run_label    = "Wind Insertion",
    ins_short    = "Wind-Ins",
    fig_num      = "5.45",
    fig_stem     = "Figure_5_45_FCN_wind_insertion_bias_reduction",
    fig_title_suffix = "Probes horizontal advection & geostrophic coupling",
)

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 5.46 — Geopotential insertion: T500, Wind500
# ═════════════════════════════════════════════════════════════════════════════
print("Figure 5.46 …")
plot_6panel(
    vars_list    = ["t500", "wind500"],
    run_name     = "z_ins",
    run_label    = "Geopotential Insertion",
    ins_short    = "Z-Ins",
    fig_num      = "5.46",
    fig_stem     = "Figure_5_46_FCN_geopotential_insertion_bias_reduction",
    fig_title_suffix = "Probes geostrophic balance & hypsometric relationship",
)

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 5.47 — Failure-mode summary map (wider layout, clearer legend)
# ═════════════════════════════════════════════════════════════════════════════
print("Figure 5.47 …")

df_fm  = pd.read_csv(VAL_DIR / "FCN_failure_mode_map.csv")
df_sum = pd.read_csv(VAL_DIR / "FCN_failure_mode_area_summary.csv")

class_arr = df_fm.pivot(index="lat", columns="lon", values="class_id").values

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
COLORS = ["#c8c8c8",  # 0 neutral
          "#27ae60",  # 1 all improve
          "#c0392b",  # 2 all worsen
          "#2980b9",  # 3 temp best
          "#8e44ad",  # 4 wind best
          "#e67e22",  # 5 geo best
          "#1a4f7a",  # 6 temp worst
          "#6c3483",  # 7 wind worst
          "#6e2c00"]  # 8 geo worst

cmap_fm = ListedColormap(COLORS)
norm_fm = BoundaryNorm(np.arange(-0.5, 9.5, 1), cmap_fm.N)

# Wide figure: map on the left, legend panel on the right
fig = plt.figure(figsize=(19, 7.5))
ax_map = fig.add_axes([0.03, 0.10, 0.72, 0.78], projection=PROJ)
ax_leg = fig.add_axes([0.77, 0.10, 0.21, 0.78])   # plain axes for legend

# Map
im = ax_map.pcolormesh(LON, LAT, class_arr,
                        cmap=cmap_fm, norm=norm_fm,
                        transform=PROJ, shading="auto")
ax_map.add_feature(cfeature.COASTLINE,  linewidth=0.6, zorder=3)
ax_map.add_feature(cfeature.BORDERS,    linewidth=0.3, linestyle=":", zorder=3)
ax_map.set_global()
ax_map.gridlines(draw_labels=False, linewidth=0.3, color="gray", alpha=0.5)

ax_map.set_title(
    "Figure 5.47 — FCN Insertion Failure-Mode Summary Map  (diagnostic: Z500)\n"
    f"{MATURE_LABEL}  |  classification threshold = 146.09 m² s⁻²",
    fontsize=10.5, fontweight="bold", pad=6,
)

# Legend panel (right side)
ax_leg.set_xlim(0, 1)
ax_leg.set_ylim(0, 1)
ax_leg.axis("off")
ax_leg.set_title("Failure-Mode Classes\n(area-weighted %)",
                 fontsize=10, fontweight="bold", pad=4)

# Sort by area pct descending for readability
pct_map = {row["class_id"]: row["area_pct"]
           for _, row in df_sum.iterrows()}

y_positions = np.linspace(0.92, 0.05, 9)
for i, (cid, clabel) in enumerate(CLASS_LABELS.items()):
    pct = pct_map.get(cid, 0.0)
    y   = y_positions[i]
    # Colour swatch
    rect = mpatches.FancyBboxPatch(
        (0.03, y - 0.032), 0.12, 0.062,
        boxstyle="round,pad=0.005",
        facecolor=COLORS[cid], edgecolor="gray", linewidth=0.6,
    )
    ax_leg.add_patch(rect)
    # Class number
    ax_leg.text(0.09, y, str(cid),
                ha="center", va="center", fontsize=8,
                fontweight="bold", color="white")
    # Class name
    ax_leg.text(0.19, y + 0.015, clabel,
                ha="left", va="center", fontsize=8.5, fontweight="bold",
                color="#1a1a2e")
    # Area percentage
    ax_leg.text(0.19, y - 0.018, f"{pct:.2f}% of global area",
                ha="left", va="center", fontsize=8, color="#444444")

# Divider line between map and legend areas (subtle)
fig.add_artist(
    plt.Line2D([0.765, 0.765], [0.05, 0.95],
               transform=fig.transFigure,
               color="#aaaaaa", linewidth=0.8, linestyle="--")
)

fig.suptitle(
    "FCN — Section 5.8 | Where Does Each Insertion Type Help or Hurt Z500?",
    fontsize=12, fontweight="bold", y=0.995,
)
_save(fig, "Figure_5_47_FCN_failure_mode_summary_map")

print("\nAll figures replotted successfully.")
