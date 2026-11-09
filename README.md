# FourCastNetv2_OSSE

Observing System Simulation Experiment (OSSE) **truth-insertion** framework for NVIDIA
**FourCastNet v2-small** (SFNO, 0.25°, 73 channels, 13 pressure levels), implemented as a fork of
NVIDIA **earth2mip** (`github.com/yashdiggikar/OSSE-FourCastNetv2`). Released with the
manuscript *An AI-Based OSSE Framework to Evaluate Learned Physical Consistency: A Multi-Model Study Using FourCastNet, Aurora, Prithvi WxC, and GraphCast* (Diggikar, Halem, and Raman) (submitted to AGU *JGR: Machine Learning and Computation*).

FCNv2 is rolled out autoregressively for 60 days (240 × 6 h steps) from
**2023-02-01 00:00 UTC** (ERA5 IC). Steps 0–119 run free. From step 120 (day 30) onward,
selected channels are **hard-replaced with ERA5 "truth"** inside the model's iteration loop,
and the corrected state is fed back into the model. Forecasts are verified against ERA5.

This is one of five companion repositories: `Aurora_OSSE`, **`FourCastNetv2_OSSE`**,
`GraphCast_OSSE`, `PrithviWxC_OSSE`, `OSSE_CrossModel_Analysis`.

---

## Repository layout

```
FourCastNetv2_OSSE/
├── model/                      Full earth2mip fork at commit 12b374a (global U+V injection)
│   ├── earth2mip/networks/__init__.py   iteration loop where truth is injected
│   └── fcnv2_sm_package/       FCNv2-small package metadata (weights not included)
├── experiments/
│   ├── 00_global_runs/         control and global T / Z insertion (overrides from git history)
│   ├── 05_t_insertion_land/   06_t_insertion_ocean/   07_t_insertion_coastal/
│   ├── 08_t_insertion_lower3/  09_t_insertion_12h/  10_t_insertion_perturbed_gaussian/
│   ├── 11_t_insertion_perturbed_cop/
│   ├── 12_uv_insertion_land/  13_uv_insertion_ocean/  14_uv_insertion_coastal/
│   ├── 15_uv_insertion_lower3/  16_uv_insertion_12h/  17_uv_insertion_perturbed_cop/
│   └── 18_tuv_insertion_global/
│       └── model_overrides/earth2mip/...   files that differ from model/ for this run
├── preprocessing/              ERA5 truth download
├── analysis/
│   ├── osse_core/              RMSE time series, propagation maps, % improvement (analyze_osse.py)
│   ├── nrmse/ lats/ lat_bands/ bias/ coupling/ section_5_*.py  Global-run analyses
│   └── perturbation_cop/       COP analyses (global, lat-band, bias, correlation length, spectra)
├── environment/  tools/assemble_experiment.sh  docs/
```

### How the per-experiment model code is stored
* **Global runs** (control, T, U+V, Z insertion; run May–June 2025): the injection code lived
  in the committed history of the fork. `model/` is the fork at HEAD (`12b374a`, global U+V
  injection). The T and Z variants of `earth2mip/networks/__init__.py` are taken directly from
  commits `a3acc65` (last T-injection commit) and `c0eeb1e` (last Z-injection commit). The
  control uses the upstream file from `bddb959`, the commit before any injection edits.
* **Extended experiments** (05–18; 2026): each experiment ran with its own copy of the fork
  carrying uncommitted local edits. `model_overrides/` holds those exact files
  (`networks/__init__.py`, plus `initial_conditions/__init__.py`, `networks/fcnv2/sfnonet.py`
  and, in some experiments, `networks/fcnv2_sm.py`).
* `tools/assemble_experiment.sh experiments/<exp> <work_dir>` rebuilds the full code tree for
  any experiment.

`docs/PROVENANCE.tsv` records each file's origin (path or `git:<repo>@<commit>`) and MD5 checksum.

---

## Experiments

| Folder | Inserted | Levels | Region / mask | Window |
|---|---|---|---|---|
| `00_global_runs/control` | — | — | — | none |
| `00_global_runs/t_insertion_global` | T | 13 | global | steps 120–239 |
| `00_global_runs/z_insertion_global` | Z | 13 | global | steps 120–239 |
| *(uv_insertion_global = `model/` unchanged)* | U, V | 13 | global | steps 120–239 |
| `05/12_*_land` | T / U,V | 13 | land (LSM > 0.5) | 120–239 |
| `06/13_*_ocean` | T / U,V | 13 | ocean | 120–239 |
| `07/14_*_coastal` | T / U,V | 13 | ~100 km coastal buffer | 120–239 |
| `08/15_*_lower3` | T / U,V | 1000/925/850 hPa | global | 120–239 |
| `09/16_*_12h` | T / U,V | 13 | global, every 12 h | 120, 122, … |
| `10_t_insertion_perturbed_gaussian` | T + i.i.d. Gaussian obs. error | 13 | global | 120–239 |
| `11/17_*_perturbed_cop` | T / U,V + correlated obs. error (COP) | 13 | global | 120–239 |
| `18_tuv_insertion_global` | T, U, V | 13 | global | 120–239 |

## Provenance gaps (please read)
* The **driver scripts for the global runs** (control, T, U+V, Z) were not preserved. Only the
  model-side injection code survives, and it is included from git history as described above.
  The extended-experiment drivers (`experiments/*/inf.py`) follow the same pattern
  (`fcnv2_sm` load → iterate 240 steps from 2023-02-01 → write NetCDF).
* The original global-run outputs were saved every 12 h (`forecast_*_60days_12hr*.nc`).

---

## Reproducing the results
1. **Environment:** `pip install -r environment/requirements-frozen.txt` (model runs, torch
   2.5.1+cu121), then `pip install -e ./model`. Analysis used
   `environment/requirements-analysis-frozen.txt`. Production runs used this Python venv
   (`fcn3_env`). `environment/pull_image.slurm` (pulls NVIDIA Modulus 23.08 via Apptainer) is
   kept only for completeness.
2. **Weights (not redistributed):** FCNv2-small (`fcnv2_sm`) from the NVIDIA model registry
   used by earth2mip. Place `weights.tar`, `global_means.npy` and `global_stds.npy` next to
   `model/fcnv2_sm_package/metadata.json`.
3. **Data:** ERA5 (Hersbach et al., 2020) pressure and surface levels, Feb–Apr 2023
   (`preprocessing/download_era5_truth.py`). Set `CDSAPI_KEY` or `~/.cdsapirc`.
4. **Run:** `tools/assemble_experiment.sh experiments/06_t_insertion_ocean /scratch/fcn_t_ocean`,
   then `sbatch run_t_ocean.slurm` (the job sets `PYTHONPATH` to the fork). Scripts contain
   the original absolute cluster paths; see `docs/PATHS.md`.
5. **Analysis:** `analysis/osse_core/analyze_osse.py`, the global-run analyses in `analysis/`,
   and `analysis/perturbation_cop/`.

## License
Apache-2.0 (`LICENSE`, inherited from NVIDIA earth2mip). See `NOTICE` for the modifications.
Model weights are not included.

## Citation
See `CITATION.cff`. Please also cite: Bonev, B., et al. (2023), Spherical Fourier Neural
Operators: Learning stable dynamics on the sphere, *ICML 2023*, PMLR 202, 2806–2823.
