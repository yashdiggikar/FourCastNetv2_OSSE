# Hard-coded paths

The scripts in this repository are released **exactly as they were run** on the UMBC HPC
cluster (chip / `pi_halem` allocation), so they contain absolute paths to the original
locations of data, model weights and outputs. To re-run them elsewhere, replace the
prefixes below with your own locations (e.g. with `sed` or by editing the constants at the
top of each script). No other code changes are needed.

Root prefix used everywhere: `/umbc/rs/pi_halem/users/yashd1` (some legacy Aurora drivers use `/xace/d1/yashd1`).

| Original path prefix | Occurrences | Example file |
|---|---|---|
| `/umbc/rs/pi_halem/users/yashd1/fcn_inference` | 133 | `analysis/osse_core/run_analysis.slurm` |
| `/umbc/rs/pi_halem/users/yashd1/AURORA_DATA` | 23 | `analysis/nrmse/compute_global_nrmse.py` |
| `/umbc/rs/pi_halem/users/yashd1` | 21 | `analysis/osse_core/analyze_osse.py` |
| `/umbc/rs/pi_halem/users/yashd1/envs` | 19 | `experiments/05_t_insertion_land/run_t_insertion.slurm` |
| `/umbc/rs/pi_halem/users/yashd1/FourCastNet` | 13 | `analysis/coupling/compute_coupling.py` |
| `/umbc/rs/pi_halem/users/yashd1/venvs` | 7 | `analysis/osse_core/analyze_osse.py` |
| `/umbc/rs/pi_halem/users/yashd1/FCN3` | 6 | `experiments/05_t_insertion_land/inf.py` |
| `/umbc/rs/pi_halem/users/yashd1/fcn_rmse` | 2 | `analysis/coupling/compute_coupling.py` |
| `/umbc/rs/pi_halem/users/yashd1/.apptainer_cache` | 1 | `environment/pull_image.slurm` |
