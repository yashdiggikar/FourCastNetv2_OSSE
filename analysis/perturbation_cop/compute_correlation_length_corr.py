#!/usr/bin/env python3
"""
Empirical correlation-length verification for the FCNv2 Correlated OSSE
Perturbation experiments (t_perturbed_corr, uv_perturbed_corr).

See ../CORRELATION_LENGTH_VERIFICATION.md for the full method description
and rationale. Short version: the perturbation field is never saved on its
own, but since insertion is a hard replacement, it can be recovered exactly
as `experiment_output[injection_step] - truth[same_valid_time]`. This script
does that at the first post-injection step, then checks:
  1. Horizontal correlation length (zonal + meridional, 3 latitude bands),
     at 3 representative pressure levels (300/500/1000 hPa), against the
     nominal 250 km target.
  2. Vertical level-to-level correlation matrix (all 13 native levels)
     against the theoretical exp(-|ln(p_i)-ln(p_j)|/0.5) matrix from
     METHOD_OVERVIEW.md Section 2.2.

Full variable grid (matching this project's other 9 "vs truth" scripts,
per user request): T, U, V, Z, R (relative humidity) -- computed for BOTH
experiments regardless of which family was directly inserted, so this also
reports cross-variable coupling in the correlation-length sense (e.g. is
the T-insertion experiment's Z-response field itself still spatially
correlated at ~250km, or does cross-variable coupling scramble that
structure?). R is known to be absent from this project's ERA5 archive at
every level (see compute_global_rmse_corr.py's own discovery of this) --
kept in the variable list anyway for structural parity with Aurora/Prithvi;
it will simply skip via the same defensive try/except used everywhere else
in this project.

Output per experiment (into PERTURBATION_EXPERIMENTS/FCN/<experiment>/):
    corr_length_horizontal_<experiment>.csv
    corr_length_vertical_<experiment>.csv
    corr_length_horizontal_map_<experiment>.png
    corr_length_vertical_map_<experiment>.png

Run:
    source /umbc/rs/pi_halem/users/yashd1/venvs/aurora310/bin/activate
    python compute_correlation_length_corr.py
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

BASE      = Path("/umbc/rs/pi_halem/users/yashd1")
FCN_DIR   = BASE / "fcn_inference"
ERA5_DIR  = BASE / "AURORA_DATA" / "rmse_data"
OUT_DIR   = BASE / "PERTURBATION_EXPERIMENTS" / "FCN"

EXPERIMENTS = [
    ("t_perturbed_corr",  FCN_DIR / "t_perturbed_corr"  / "forecast_t_perturbed_corr_global_day30_60.nc",  "temp"),
    ("uv_perturbed_corr", FCN_DIR / "uv_perturbed_corr" / "forecast_uv_perturbed_corr_global_day30_60.nc", "uv"),
]

TARGET_LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]  # hPa
REPR_HPA      = [300, 500, 1000]   # representative levels for the horizontal check
REPR_IDX      = {hpa: TARGET_LEVELS.index(hpa) for hpa in REPR_HPA}
INJECT_STEP   = 120   # first post-injection step (0-indexed), matches every spatial_bias script
MAX_LAG_PX    = 40
VERT_L_LOGP   = 0.5   # natural-log-pressure decay scale, METHOD_OVERVIEW.md Section 2.2

ERA5_PRES = [ERA5_DIR / f"era5_pressure_2023_{m:02d}.grib" for m in [2, 3, 4]]
ERA5_SURF = [ERA5_DIR / f"era5_surface_2023_{m:02d}.grib"  for m in [2, 3, 4]]

FCN_LATS = np.linspace(90.0, -90.0, 721)
FCN_LONS = np.arange(0.0, 360.0, 0.25)
GRID_DEG = 0.25
KM_PER_DEG_LAT = 111.0

# Nearest actual grid row to each representative latitude band
BANDS = {"equator_5N": 5.0, "midlat_45N": 45.0, "highlat_70N": 70.0}
BAND_ROWS = {name: int(np.argmin(np.abs(FCN_LATS - lat))) for name, lat in BANDS.items()}

# Full variable grid: label -> (ERA5 shortName == FCN channel prefix, derived-non-divergent flag)
UPPER_VARS = {"T": ("t", False), "U": ("u", True), "V": ("v", True), "Z": ("z", False), "R": ("r", False)}
# label -> (FCN own channel name, ERA5 surface shortName)
SFC_VARS = {"T2M": ("t2m", "2t"), "U10": ("u10m", "10u"), "V10": ("v10m", "10v")}
_SFC_ERA5_VARNAME = {"2t": "t2m", "10u": "u10", "10v": "v10"}


def build_channel_index(nc_file):
    with nc4.Dataset(str(nc_file)) as ds:
        n_ch = ds.dimensions["channel"].size
        return {str(ds.variables["channel"][i]): i for i in range(n_ch)}


def fcn_valid_times(nc_file):
    with nc4.Dataset(str(nc_file)) as ds:
        tv = ds.variables["time"]
        nums, units, cal = tv[:].data, tv.units, tv.calendar
    return [cft.num2date(n, units, cal, only_use_cftime_datetimes=False) for n in nums]


def load_era5_full_levels(shortname, grib_files, level_list=TARGET_LEVELS):
    """All requested levels for one variable, one time axis, read one level
    at a time (the per-level-filtered cfgrib open is the pattern already
    proven fast/reliable elsewhere in this project). Returns (times,
    levels_hpa_ascending, data[time, level, lat, lon])."""
    per_level_times = None
    data_by_level = []
    good_levels = []
    for lv in level_list:
        dsets = []
        try:
            for f in grib_files:
                if not f.exists():
                    continue
                dsets.append(xr.open_dataset(str(f), engine="cfgrib",
                             filter_by_keys={"shortName": shortname, "typeOfLevel": "isobaricInhPa",
                                              "level": int(lv)}, indexpath=None))
            if not dsets:
                raise RuntimeError("no grib files opened")
            combined = xr.concat(dsets, dim="time").sortby("time")
            if per_level_times is None:
                per_level_times = combined.time.values
            data_by_level.append(combined[shortname].values.astype(np.float32))
            good_levels.append(lv)
        except Exception as e:
            print(f"    WARNING: level {lv}hPa unavailable for '{shortname}' ({e!r}), skipping this level.")
        finally:
            for ds in dsets: ds.close()
    if not data_by_level:
        return None, [], None
    data = np.stack(data_by_level, axis=1)   # (time, level, lat, lon)
    return per_level_times, good_levels, data


def nearest_time_idx(times, target_dt, tol_hours=3):
    vt = np.datetime64(target_dt)
    deltas = np.abs(times - vt)
    idx = int(np.argmin(deltas))
    return idx if deltas[idx] <= np.timedelta64(tol_hours, "h") else None


def lag_correlation_1d(line, max_lag, wrap):
    """Pearson correlation of `line` against itself shifted by k, k=0..max_lag."""
    n = len(line)
    line = line - line.mean()
    denom = np.sum(line ** 2)
    corrs = np.empty(max_lag + 1)
    for k in range(max_lag + 1):
        if wrap:
            shifted = np.roll(line, -k)
            num = np.sum(line * shifted)
            corrs[k] = num / denom if denom > 0 else np.nan
        else:
            a, b = line[:n - k] if k > 0 else line, line[k:] if k > 0 else line
            num = np.sum(a * b)
            corrs[k] = num / denom if denom > 0 else np.nan
    return corrs


def efold_length_km(lags_km, corrs):
    """Lag (km) at which corrs first drops to 1/e, via linear interpolation."""
    target = 1.0 / np.e
    below = np.where(corrs <= target)[0]
    if len(below) == 0:
        return np.nan   # never dropped below 1/e within MAX_LAG_PX
    i = below[0]
    if i == 0:
        return 0.0
    x0, x1 = lags_km[i - 1], lags_km[i]
    y0, y1 = corrs[i - 1], corrs[i]
    return x0 + (target - y0) * (x1 - x0) / (y1 - y0)


def horizontal_check(field2d, label, level_hpa, derived_field, out_rows, ax, plot_it):
    """field2d: (721, 1440) perturbation field at one representative level."""
    for band_name, row in BAND_ROWS.items():
        lat = FCN_LATS[row]
        # Zonal: shift along the row (periodic, matches generator's mode="wrap")
        zonal_line = field2d[row, :]
        zonal_corr = lag_correlation_1d(zonal_line, MAX_LAG_PX, wrap=True)
        km_per_px_zonal = KM_PER_DEG_LAT * np.cos(np.deg2rad(lat)) * GRID_DEG
        zonal_lags_km = np.arange(MAX_LAG_PX + 1) * km_per_px_zonal
        L_zonal = efold_length_km(zonal_lags_km, zonal_corr)

        # Meridional: shift along a column through this row (not periodic,
        # matches generator's mode="nearest" at the poles)
        col = 0   # any fixed longitude column; field is a random realization, no preferred column
        merid_line = field2d[max(0, row - MAX_LAG_PX):min(721, row + MAX_LAG_PX + 1), col]
        merid_corr = lag_correlation_1d(merid_line[len(merid_line) // 2:], MAX_LAG_PX, wrap=False)
        km_per_px_merid = KM_PER_DEG_LAT * GRID_DEG
        merid_lags_km = np.arange(len(merid_corr)) * km_per_px_merid
        L_merid = efold_length_km(merid_lags_km, merid_corr)

        for direction, lags_km, corr, L in [
            ("zonal", zonal_lags_km, zonal_corr, L_zonal),
            ("meridional", merid_lags_km, merid_corr, L_merid),
        ]:
            out_rows.append({
                "variable": label, "level_hpa": level_hpa, "band": band_name, "latitude_deg": lat,
                "direction": direction, "derived_field": derived_field,
                "empirical_L_km": L, "target_L_km": 250.0,
                "ratio_empirical_over_target": L / 250.0 if not np.isnan(L) else np.nan,
            })
            if ax is not None and plot_it and band_name == "midlat_45N":
                ax.plot(lags_km, corr, marker="o", ms=3,
                        label=f"{label} {direction} (L≈{L:.0f} km)" if not np.isnan(L) else f"{label} {direction}")


def vertical_check(cube, levels_hpa, label, out_rows):
    """cube: (n_levels, lat, lon) perturbation cube."""
    n = len(levels_hpa)
    flat = cube.reshape(n, -1)
    emp_corr = np.corrcoef(flat)
    logp = np.log(levels_hpa)
    theo_corr = np.exp(-np.abs(logp[:, None] - logp[None, :]) / VERT_L_LOGP)
    for i in range(n):
        for j in range(n):
            out_rows.append({
                "variable": label, "level_i": levels_hpa[i], "level_j": levels_hpa[j],
                "empirical_corr": emp_corr[i, j], "theoretical_corr": theo_corr[i, j],
                "diff": emp_corr[i, j] - theo_corr[i, j],
            })
    return emp_corr, theo_corr


for exp_name, nc_file, ins_type in EXPERIMENTS:
    out_dir = OUT_DIR / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if not nc_file.exists():
        print(f"[{exp_name}] SKIP — output not yet present at {nc_file} (job still running?).")
        continue

    print(f"\n=== {exp_name} ===")
    ch_idx = build_channel_index(nc_file)
    valid_times = fcn_valid_times(nc_file)
    inject_dt = valid_times[INJECT_STEP]
    exp_ds = nc4.Dataset(str(nc_file))

    horiz_rows, vert_rows = [], []
    fig_h, ax_h = plt.subplots(figsize=(8, 5))
    theta = np.linspace(0, MAX_LAG_PX * KM_PER_DEG_LAT * GRID_DEG, 200)
    ax_h.plot(theta, np.exp(-theta / 250.0), "k--", lw=1, label="theoretical exp(-d/250km)")
    ax_h.plot(theta, np.exp(-theta / (250.0 * np.sqrt(2))), "k:", lw=1,
              label="theoretical exp(-d/(250*sqrt(2))km)")

    def read_level_cube(prefix, levels):
        cube = np.empty((len(levels), 721, 1440), dtype=np.float32)
        for i, lv in enumerate(levels):
            ch = f"{prefix}{lv}"
            if ch not in ch_idx:
                cube[i] = np.nan
                continue
            cube[i] = exp_ds.variables["prediction"][INJECT_STEP, ch_idx[ch], :, :]
        return cube

    # Every upper-air variable/level is evaluated for every experiment (not
    # just the directly-inserted family), matching the cross-variable-
    # coupling convention already used in compute_global_rmse_corr.py etc.
    for label, (grib_sn, derived) in UPPER_VARS.items():
        era5_times, era5_levels, era5_cube_all = load_era5_full_levels(grib_sn, ERA5_PRES)
        if era5_times is None:
            print(f"  [{exp_name}] {label}: no ERA5 data available at any level, skipping.")
            continue
        ti = nearest_time_idx(era5_times, inject_dt)
        if ti is None:
            print(f"  [{exp_name}] {label}: no ERA5 truth near injection time, skipping.")
            continue
        truth_cube = era5_cube_all[ti]   # (n_levels_available, 721, 1440), ascending pressure

        exp_cube = read_level_cube(grib_sn, era5_levels)
        pert_cube = exp_cube - truth_cube   # perturbation, recovered exactly

        for hpa in REPR_HPA:
            if hpa not in era5_levels:
                continue
            lvl_idx = era5_levels.index(hpa)
            horizontal_check(pert_cube[lvl_idx], label, hpa, derived, horiz_rows, ax_h, plot_it=(hpa == 500))
        if len(era5_levels) >= 2:
            vertical_check(pert_cube, era5_levels, label, vert_rows)
        print(f"  [{exp_name}] {label}: horizontal ({[h for h in REPR_HPA if h in era5_levels]}hPa) "
              f"+ vertical ({len(era5_levels)} levels) checks done")

        # Save right after the upper-air check, before the surface block --
        # so upper-air results survive even if the surface read fails.
        pd.DataFrame(horiz_rows).to_csv(out_dir / f"corr_length_horizontal_{exp_name}.csv", index=False)
        pd.DataFrame(vert_rows).to_csv(out_dir / f"corr_length_vertical_{exp_name}.csv", index=False)

    # Surface variables: horizontal check only (no vertical structure)
    for sfc_label, (sfc_ch, sfc_grib_sn) in SFC_VARS.items():
        if sfc_ch not in ch_idx:
            print(f"  [{exp_name}] {sfc_label} (surface): channel '{sfc_ch}' not in experiment output, skipping.")
            continue
        sfc_exp = exp_ds.variables["prediction"][INJECT_STEP, ch_idx[sfc_ch], :, :]
        sfc_dsets = [xr.open_dataset(str(f), engine="cfgrib",
                     filter_by_keys={"shortName": sfc_grib_sn}, indexpath=None)
                     for f in ERA5_SURF if f.exists()]
        if not sfc_dsets:
            print(f"  [{exp_name}] {sfc_label} (surface): no ERA5 surface grib available, skipping.")
            continue
        sfc_combined = xr.concat(sfc_dsets, dim="time").sortby("time")
        sfc_varname = _SFC_ERA5_VARNAME[sfc_grib_sn]
        sfc_ti = nearest_time_idx(sfc_combined.time.values, inject_dt)
        for ds in sfc_dsets: ds.close()
        if sfc_ti is not None:
            sfc_truth = sfc_combined[sfc_varname].values[sfc_ti].astype(np.float32)
            sfc_pert = np.asarray(sfc_exp) - sfc_truth
            horizontal_check(sfc_pert, sfc_label, None, False, horiz_rows, None, plot_it=False)
            print(f"  [{exp_name}] {sfc_label} (surface): horizontal check done")

    # Save incrementally after every variable (survives any later crash)
    pd.DataFrame(horiz_rows).to_csv(out_dir / f"corr_length_horizontal_{exp_name}.csv", index=False)
    pd.DataFrame(vert_rows).to_csv(out_dir / f"corr_length_vertical_{exp_name}.csv", index=False)

    ax_h.set_xlabel("lag distance (km)"); ax_h.set_ylabel("correlation")
    ax_h.set_title(f"{exp_name}: horizontal lag-correlation at 45N, level=500hPa")
    ax_h.legend(fontsize=7); ax_h.grid(alpha=0.3)
    fig_h.savefig(out_dir / f"corr_length_horizontal_map_{exp_name}.png", dpi=130, bbox_inches="tight")
    plt.close(fig_h)

    # Vertical heatmap: use the first-processed variable's matrices (each var already saved to CSV above)
    if vert_rows:
        df_v = pd.DataFrame(vert_rows)
        first_var = df_v["variable"].iloc[0]
        sub = df_v[df_v["variable"] == first_var]
        levels_sorted = sorted(sub["level_i"].unique())
        n = len(levels_sorted)
        emp_m = sub.pivot(index="level_i", columns="level_j", values="empirical_corr").loc[levels_sorted, levels_sorted].values
        theo_m = sub.pivot(index="level_i", columns="level_j", values="theoretical_corr").loc[levels_sorted, levels_sorted].values
        fig_v, axes = plt.subplots(1, 3, figsize=(14, 4.2))
        for a, m, title in zip(axes, [emp_m, theo_m, emp_m - theo_m],
                                [f"empirical ({first_var})", "theoretical", "difference"]):
            im = a.imshow(m, vmin=-1 if title != "difference" else -0.3,
                          vmax=1 if title != "difference" else 0.3, cmap="RdBu_r")
            a.set_title(title, fontsize=10)
            a.set_xticks(range(n)); a.set_xticklabels(levels_sorted, rotation=90, fontsize=6)
            a.set_yticks(range(n)); a.set_yticklabels(levels_sorted, fontsize=6)
            plt.colorbar(im, ax=a, shrink=0.8)
        fig_v.suptitle(f"{exp_name}: vertical correlation matrix, levels in hPa")
        fig_v.savefig(out_dir / f"corr_length_vertical_map_{exp_name}.png", dpi=130, bbox_inches="tight")
        plt.close(fig_v)

    print(f"  Saved: {out_dir / f'corr_length_horizontal_{exp_name}.csv'}")
    print(f"  Saved: {out_dir / f'corr_length_vertical_{exp_name}.csv'}")
    exp_ds.close()

print("\nDone.")
