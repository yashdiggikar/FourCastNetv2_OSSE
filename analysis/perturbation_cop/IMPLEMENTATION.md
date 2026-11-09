# FourCastNetv2 — Correlated OSSE Perturbation Implementation

See `../METHOD_OVERVIEW.md` for the cross-model method description and
citations. This file documents the FCN-specific code, paths, and
hyperparameters.

---

## 1. Location

| | Path |
|---|---|
| Temperature experiment | `/umbc/rs/pi_halem/users/yashd1/fcn_inference/t_perturbed_corr/` |
| Wind experiment | `/umbc/rs/pi_halem/users/yashd1/fcn_inference/uv_perturbed_corr/` |
| SLURM job scripts | `run_t_perturbed_corr.slurm`, `run_uv_perturbed_corr.slurm` (same dirs) |
| Base template (uncorrelated baseline, unmodified) | `fcn_inference/t_perturbed/` |

Both new directories are **full copies** of `fcn_inference/t_perturbed/`'s
`OSSE-FourCastNetv2/` package (the whole earth2mip fork), following this
project's existing per-experiment-copy convention (see
`hydro`-adjacent conversation history / `project_osse_tristudy` memory:
"each experiment subdirectory has its own full copy of the
OSSE-FourCastNetv2 repo, hand-edited per copy" — the FCN insertion mechanism
has no shared/reusable rollout function across experiments, unlike Aurora's).

## 2. Where the perturbation logic actually lives

`OSSE-FourCastNetv2/earth2mip/networks/__init__.py`, inside the
`_TimeLoopImpl._iterate()` generator (the same class/method the
uncorrelated `t_perturbed/` uses — **not** the `Inference`/`_SimpleModelAdapter`
class used by the "official"/root-level `OSSE-FourCastNetv2` copy; `t_perturbed/`'s
copy was already independently simplified before this work started).

New helper function added: `_correlated_level_fields(n_lev, shape_hw,
vert_chol, rng)` — spatial Gaussian-filter smoothing + vertical
Cholesky-mixing (see METHOD_OVERVIEW.md Section 2.1-2.2).

For the wind experiment, two additional helpers:
`_streamfunction_to_uv(psi, lat_deg, lon_deg)` and
`_generate_correlated_wind_perturbation(...)` (Section 2.3).

`inf.py` in each directory is unchanged in structure from `t_perturbed/`'s
`inf.py` — only the output path and experiment-description string differ.

## 3. Grid and channel conventions (FCNv2-sm, verified from the model package)

- Grid: 721 (lat, descending 90 -> -90, includes both poles) x 1440 (lon,
  ascending 0 -> 360 exclusive), 0.25 deg spacing.
- Channel layout (fixed, verified via `uv_land`/`t_lower`/`tuv_global`'s
  existing hardcoded indices, reused unchanged here):
  - T (13 levels, 50->1000 hPa ascending): channels `47:60`
  - U (13 levels): channels `8:21`
  - V (13 levels): channels `21:34`
- `TARGET_LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850,
  925, 1000]` hPa (same order for T, U, V).

## 4. Hyperparameters used

```
HORIZ_CORR_LENGTH_KM = 250.0
GRID_SPACING_DEG     = 0.25
KM_PER_DEG_LAT        = 111.0
SIGMA_PX              = 250.0 / (111.0 * 0.25) ≈ 9.0   grid cells

VERT_L_LOGP           = 0.5    (natural-log-pressure decay scale)
```

Temperature sigma-by-level (K), Ingleby (2017)-style, **unchanged from
`t_perturbed/`**:

```
[50,100,150,200,250,300,400,500,600,700,850,925,1000] hPa ->
[2.0,1.5,1.5,1.2,1.0,0.8,0.7,0.7,0.8,0.8,1.0,1.0,1.2] K
```

Wind sigma-by-level (m/s) — **newly constructed for this work** (no
uncorrelated wind-perturbation baseline existed previously to reuse), larger
near the jet-stream levels (200-300 hPa) where both wind speed and
representativeness error are largest:

```
[50,100,150,200,250,300,400,500,600,700,850,925,1000] hPa ->
[1.5,1.6,1.8,2.2,2.0,1.8,1.4,1.2,1.1,1.0,1.0,1.1,1.3] m/s
```

## 5. IC and truth source

- IC: same as `t_perturbed/` — the control forecast file
  `/umbc/rs/pi_halem/users/yashd1/FourCastNet/forecast_normal_60days_12hr.nc`,
  step index 0 (= 2023-02-01 00:00 UTC), **not** the CDS API used by the
  non-perturbed land/ocean/coastal experiments. This IC choice was
  inherited as-is from `t_perturbed/` (kept for exact IC-comparability
  between the uncorrelated and correlated perturbation experiments).
- Truth: ERA5 pressure-level GRIB,
  `/umbc/rs/pi_halem/users/yashd1/AURORA_DATA/rmse_data/era5_pressure_2023_{02,03,04}.grib`
  (per `Organized_results/file_paths.md` Section 2 — Aurora & FCN share this
  truth source). Temperature experiment reads `shortName: "t"`; wind
  experiment reads `shortName: "u"` and `"v"` as two separate `cfgrib` opens
  (GRIB requires per-shortName filtering).

## 6. Environment note

`fcn_inference` runs in a **separate venv from Aurora/Prithvi**:
`/umbc/rs/pi_halem/users/yashd1/envs/fcn3_env`. This venv did **not** have
`scipy` installed (needed for `scipy.ndimage.gaussian_filter`) — installed
during this work (`pip install scipy` -> 1.13.1). This is the only
environment change made across all three models for this implementation.

## 7. SLURM submission (2026-07-22)

```
sbatch fcn_inference/t_perturbed_corr/run_t_perturbed_corr.slurm    # job 203458
sbatch fcn_inference/uv_perturbed_corr/run_uv_perturbed_corr.slurm  # job 203459
```

Both requested 128G mem, 1 GPU, 8h walltime, `chip-gpu` cluster,
`gpu` partition, `pi_halem` account — same resource request as
`t_perturbed/`'s original job.
