#!/usr/bin/env python3
"""
Spectral blurring diagnostic for the FCNv2 Correlated OSSE Perturbation
experiments (t_perturbed_corr, uv_perturbed_corr).

See ../SPECTRAL_BLURRING_ANALYSIS.md for the full method description and
rationale. Short version: at several lead times spanning the post-injection
window, compute experiment-minus-control, take its 1D zonal (longitude)
power spectrum in 3 latitude bands, and track how fast high-wavenumber
(fine-scale) power decays as lead time increases -- the blurring signature.

Full variable/level grid (matching this project's other "vs truth" scripts,
per user request): T, U, V, Z, R (relative humidity) at 300/500/1000 hPa,
plus surface T2M/U10/V10 -- computed for BOTH experiments regardless of
which family was directly inserted (cross-variable coupling included). Unlike
the correlation-length script, no ERA5 truth is needed here at all (this
diagnostic is purely experiment-minus-control, both FCN's own saved output),
so FCN's known ERA5-side R-at-every-level gap does not apply -- R300/500/1000
compute normally as long as FCN's own model output has those channels.

Output per experiment (into PERTURBATION_EXPERIMENTS/FCN/<experiment>/):
    spectral_blur_<experiment>.csv
    spectral_blur_summary_<experiment>.csv
    spectral_blur_map_<experiment>_<band>.png

Run:
    source /umbc/rs/pi_halem/users/yashd1/venvs/aurora310/bin/activate
    python compute_spectral_blur_corr.py
"""
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import netCDF4 as nc4
import cftime as cft
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE      = Path("/umbc/rs/pi_halem/users/yashd1")
FCN_DIR   = BASE / "fcn_inference"
CTRL_FILE = BASE / "FourCastNet" / "forecast_normal_60days_12hr.nc"
OUT_DIR   = BASE / "PERTURBATION_EXPERIMENTS" / "FCN"

EXPERIMENTS = [
    ("t_perturbed_corr",  FCN_DIR / "t_perturbed_corr"  / "forecast_t_perturbed_corr_global_day30_60.nc",  "temp"),
    ("uv_perturbed_corr", FCN_DIR / "uv_perturbed_corr" / "forecast_uv_perturbed_corr_global_day30_60.nc", "uv"),
]

SNAPSHOT_STEPS = [120, 140, 160, 180, 200, 220, 239]
LEVELS = [300, 500, 1000]

FCN_LATS = np.linspace(90.0, -90.0, 721)
FCN_LONS = np.arange(0.0, 360.0, 0.25)
GRID_DEG = 0.25
KM_PER_DEG_LAT = 111.0
N_LON = len(FCN_LONS)

BANDS = {"equator_5N": 5.0, "midlat_45N": 45.0, "highlat_70N": 70.0}
BAND_ROWS = {name: int(np.argmin(np.abs(FCN_LATS - lat))) for name, lat in BANDS.items()}
WINDOW_DEG = 5.0

UPPER_PREFIXES = {"T": "t", "U": "u", "V": "v", "Z": "z", "R": "r"}
SFC_CHANNELS = {"T2M": "t2m", "U10": "u10m", "V10": "v10m"}


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


def nearest_idx(times, target_dt, tol_hours=6):
    # tol_hours=6, not the usual 3: the control file is 12-hourly cadence
    # (see below) while the experiment is 6-hourly, so a snapshot step can
    # legitimately be up to 6h from the nearest available control step.
    vt = np.datetime64(target_dt)
    deltas = np.abs(times - vt)
    idx = int(np.argmin(deltas))
    return idx if deltas[idx] <= np.timedelta64(tol_hours, "h") else None


def zonal_power_spectrum(field2d, row_center, window_rows):
    """Average 1D real-FFT power spectrum over a latitude window, in cycles
    per global longitude ring (wavenumber 0..N_LON//2)."""
    r0 = max(0, row_center - window_rows)
    r1 = min(field2d.shape[0], row_center + window_rows + 1)
    powers = []
    for r in range(r0, r1):
        line = field2d[r, :] - field2d[r, :].mean()
        fft = np.fft.rfft(line)
        powers.append(np.abs(fft) ** 2)
    return np.mean(powers, axis=0)   # (N_LON//2 + 1,)


def effective_resolution_wavenumber(power, frac=0.01):
    """Wavenumber at which power first drops to `frac` of the power at
    wavenumber 1 (skip wavenumber 0 = the DC/mean component)."""
    ref = power[1]
    if ref <= 0:
        return np.nan
    below = np.where(power[1:] <= frac * ref)[0]
    return float(below[0] + 1) if len(below) else np.nan


# Full label -> channel name map, built once (same for every experiment,
# since this diagnostic evaluates every variable regardless of which family
# was directly inserted -- cross-variable coupling, matching
# compute_global_rmse_corr.py's convention).
CH_NAMES = {f"{label}{lev}": f"{prefix}{lev}" for label, prefix in UPPER_PREFIXES.items() for lev in LEVELS}
CH_NAMES.update(SFC_CHANNELS)
ALL_LABELS = list(CH_NAMES)

for exp_name, nc_file, ins_type in EXPERIMENTS:
    out_dir = OUT_DIR / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if not nc_file.exists():
        print(f"[{exp_name}] SKIP — output not yet present at {nc_file} (job still running?).")
        continue

    print(f"\n=== {exp_name} ===")
    ch_idx = build_channel_index(nc_file)
    ctrl_ch_idx = build_channel_index(CTRL_FILE)
    exp_ds = nc4.Dataset(str(nc_file))
    ctrl_ds = nc4.Dataset(str(CTRL_FILE))

    # The control file (FourCastNet/forecast_normal_60days_12hr.nc) is
    # 12-hourly/120 steps, NOT the same cadence as the 6-hourly/240-step
    # experiment output -- raw step-index alignment is wrong here (it either
    # crashes once step > 120, or for a coarser/longer control like Aurora's,
    # silently pulls the wrong valid time). Match by actual valid datetime
    # instead, same pattern already proven correct in
    # compute_global_rmse_corr.py.
    valid_times = fcn_valid_times(nc_file)
    ctrl_t = ctrl_ds.variables["time"][:].data
    ctrl_dts_np = np.array([np.datetime64(ctrl_time_to_dt(h)) for h in ctrl_t])

    spectrum_rows, summary_rows = [], []
    band_figs = {b: plt.subplots(figsize=(7, 5)) for b in BANDS}
    cmap = plt.get_cmap("viridis")
    first_label_plotted = None

    for label in ALL_LABELS:
        ch_name = CH_NAMES[label]
        if ch_name not in ch_idx or ch_name not in ctrl_ch_idx:
            print(f"  [{exp_name}] {label}: channel unavailable, skipping.")
            continue
        if first_label_plotted is None:
            first_label_plotted = label

        for si, step in enumerate(SNAPSHOT_STEPS):
            ci = nearest_idx(ctrl_dts_np, valid_times[step])
            if ci is None:
                print(f"  [{exp_name}] {label} step {step}: no control valid time within tolerance, skipping.")
                continue
            exp_field = np.asarray(exp_ds.variables["prediction"][step, ch_idx[ch_name], :, :])
            ctrl_field = np.asarray(ctrl_ds.variables["prediction"][ci, ctrl_ch_idx[ch_name], :, :])
            diff = exp_field - ctrl_field

            for band_name, row in BAND_ROWS.items():
                window_rows = int(WINDOW_DEG / GRID_DEG)
                power = zonal_power_spectrum(diff, row, window_rows)
                wavenumbers = np.arange(len(power))
                for wn, p in zip(wavenumbers, power):
                    spectrum_rows.append({"variable": label, "step": step, "band": band_name,
                                           "wavenumber": int(wn), "power": float(p)})
                eff_res = effective_resolution_wavenumber(power)
                summary_rows.append({"variable": label, "step": step, "band": band_name,
                                      "effective_resolution_wavenumber": eff_res})

                fig, ax = band_figs[band_name]
                color = cmap(si / max(1, len(SNAPSHOT_STEPS) - 1))
                ax.loglog(wavenumbers[1:], power[1:], color=color, lw=1.2,
                          label=f"{label} step {step}" if label == first_label_plotted else None)

        print(f"  [{exp_name}] {label}: spectral blur computed for {len(SNAPSHOT_STEPS)} lead times x {len(BANDS)} bands")

        pd.DataFrame(spectrum_rows).to_csv(out_dir / f"spectral_blur_{exp_name}.csv", index=False)
        pd.DataFrame(summary_rows).to_csv(out_dir / f"spectral_blur_summary_{exp_name}.csv", index=False)

    for band_name, (fig, ax) in band_figs.items():
        ax.set_xlabel("zonal wavenumber (cycles / global ring)")
        ax.set_ylabel("power")
        ax.set_title(f"{exp_name}: zonal power spectrum, band={band_name}\n(experiment - control, color = lead time)")
        ax.legend(fontsize=6, ncol=2)
        ax.grid(alpha=0.3, which="both")
        fig.savefig(out_dir / f"spectral_blur_map_{exp_name}_{band_name}.png", dpi=130, bbox_inches="tight")
        plt.close(fig)

    print(f"  Saved: {out_dir / f'spectral_blur_{exp_name}.csv'}")
    print(f"  Saved: {out_dir / f'spectral_blur_summary_{exp_name}.csv'}")
    exp_ds.close()
    ctrl_ds.close()

print("\nDone.")
