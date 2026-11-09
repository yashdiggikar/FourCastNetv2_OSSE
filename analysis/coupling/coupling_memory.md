# Section 5.6 — Cross-Variable Coupling Analysis — Memory File
## Reference for replicating the coupling analysis on other models (Aurora, Pangu, etc.)

---

## Overview

Quantifies causal cross-variable interactions by measuring how inserting ERA5 truth
for one variable group (A) reduces forecast error in another variable group (B).
Computed across **all 13 pressure levels** (1000–50 hPa) for 5 variable groups.

Data source: `/umbc/rs/pi_halem/users/yashd1/fcn_rmse/FCNv2/`
Script: `FourCastNet/coupling/compute_coupling.py`

---

## Directory Structure

```
FourCastNet/coupling/
├── values/
│   ├── FCN_coupling_matrix.csv           # long-form: all ins × group combinations
│   ├── FCN_coupling_per_level.csv        # per-level coupling for each combination
│   └── FCN_coupling_summary.csv          # top/second/weakest couplings per experiment
├── plots/
│   ├── Figure_5_30_FCN_coupling_heatmap.png
│   ├── Figure_FCN_coupling_bar.png
│   ├── Figure_FCN_coupling_vertical_profiles.png
│   └── Table_5_8_FCN_coupling_summary.png
├── logs/
│   └── compute_coupling_log.txt
├── compute_coupling.py
├── coupling_results_analysis.txt
└── coupling_memory.md   ← this file
```

---

## Coupling Score Formula

```python
# For each pressure level of response group B:
pct_level = 100 * (ctrl_mature_mean(B_level) - ins_mature_mean(B_level)) / ctrl_mature_mean(B_level)

# Average across all 13 pressure levels:
C(A→B) = np.nanmean([pct_level for each level in B])
```

- Mature phase = steps 60–119 = columns 60–119 in the CSV (day 30–60)
- Positive = inserting A reduces B error (coupling)
- Negative = inserting A worsens B (analysis shock)
- ~100% = self-insertion (trivial, forced = ERA5)

---

## Data Source Format (`fcn_rmse/FCNv2/`)

Each run directory contains `rmse_all_vars.csv`:
- **Rows** = 65 variables: t1000…t50, u1000…u50, v1000…v50, z1000…z50, r1000…r50
- **Columns** = 120 timestamps (2023-02-01 00Z to 2023-04-01 12Z, 12-hourly)
- **Values** = global cosine-lat weighted RMSE at each timestep

```python
df = pd.read_csv('rmse_all_vars.csv', index_col=0)
# Mature phase columns:
mature = df.iloc[:, 60:120]
```

Runs used: `Control`, `temp_ins`, `wind_ins`, `geo_ins`
(Do NOT edit files in fcn_rmse/ — read-only source)

---

## Variable Groups and Levels

```python
LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
GROUPS = {
    "T": [f"t{l}" for l in LEVELS],   # Temperature
    "U": [f"u{l}" for l in LEVELS],   # Zonal wind
    "V": [f"v{l}" for l in LEVELS],   # Meridional wind
    "Z": [f"z{l}" for l in LEVELS],   # Geopotential
    "R": [f"r{l}" for l in LEVELS],   # Relative humidity
}
```

---

## Insertion → Trivial Variable Map

```python
INS_INSERTED = {
    "temp_ins":  ["T"],       # All T levels forced to ERA5 truth
    "wind_ins":  ["U", "V"],  # All U and V levels forced to ERA5 truth
    "geo_ins":   ["Z"],       # All Z levels forced to ERA5 truth
}
```

---

## Actual Coupling Matrix (computed values)

| | T | U | V | Z | R |
|---|---|---|---|---|---|
| T-Ins | trivial | 18.0% | 16.0% | 29.9% | 6.4% |
| Wind-Ins | 55.4% | trivial | trivial | **71.8%** | 43.1% |
| Geo-Ins | **65.8%** | 65.4% | **68.1%** | trivial | 28.4% |

### Per-level peak couplings (key values):

| Experiment → Response | Avg | Peak level | Peak % | Min level | Min % |
|---|---|---|---|---|---|
| T-Ins → U | 18.0% | 200 hPa | 31.6% | 50 hPa | 5.4% |
| T-Ins → Z | 29.9% | 250 hPa | 44.2% | 925 hPa | 12.1% |
| T-Ins → R | 6.4% | 50 hPa | 49.3% | 925 hPa | **−12.0%** |
| Wind-Ins → T | 55.4% | 200 hPa | 69.2% | 50 hPa | 6.2% |
| Wind-Ins → Z | 71.8% | 700 hPa | 82.9% | 50 hPa | 31.7% |
| Wind-Ins → R | 43.1% | 250 hPa | 53.8% | 50 hPa | 8.4% |
| Geo-Ins → T | 65.8% | 500 hPa | 77.8% | 50 hPa | 23.0% |
| Geo-Ins → U | 65.4% | 250 hPa | 66.6% | 50 hPa | 62.5% (very uniform) |
| Geo-Ins → R | 28.4% | 200 hPa | 41.9% | 50 hPa | 3.9% |

---

## Figures Produced

| Figure | What it shows |
|--------|--------------|
| Figure 5.30 — Heatmap | 3×5 coupling matrix, colour-coded, trivial cells hatched |
| Coupling Bar Chart | Non-trivial couplings per response variable, grouped bars |
| Vertical Profiles | Coupling vs pressure level for each response variable |
| Table 5.8 | Top/second/weakest coupling per insertion experiment |

---

## Key Physical Results Summary

- **Wind-Ins → Z (71.8%)**: strongest non-trivial coupling; confirms baroclinic instability drives Z errors
- **Geo-Ins → T (65.8%)**: hypsometric balance Z→T is strong; Z encodes thermal structure
- **Geo-Ins → U/V (65–68%)**: geostrophic balance vertically uniform — FCNv2 learned it at all levels
- **T-Ins is weakest**: max 29.9% (Z); negative at 925 hPa for R (analysis shock)
- **Wind→Z > Z→Wind**: 71.8% vs 65–68%, wind field is slightly more primary in FCNv2
- **R always least responsive**: moisture physics semi-independent of dynamical state

---

## How to Replicate for Aurora

1. Ensure Aurora RMSE CSVs exist in same format (rows=vars, cols=timesteps).
   Variable naming convention: same `t1000`...`r50` pattern if possible.
2. Update `RMSE_BASE` path and run directory names.
3. Update `INS_RUNS` dict with Aurora's insertion run names and trivial groups.
4. Change figure numbering (5.31 for Aurora if FCN=5.30).
5. `LEVELS`, `GROUPS`, coupling formula — all reusable unchanged.

---

## Run Order

```bash
# Coupling script is standalone — reads only from fcn_rmse CSVs
source /umbc/rs/pi_halem/users/yashd1/venvs/aurora310/bin/activate
python3 -u /umbc/rs/pi_halem/users/yashd1/FourCastNet/coupling/compute_coupling.py
```

Runtime: < 1 minute (no GRIB/h5 reading, pure CSV arithmetic).

---

## Common Pitfalls

| Issue | Fix |
|-------|-----|
| "No such file" on RMSE CSV | Check run name matches directory name exactly (case-sensitive: `Control` not `control`) |
| Trivial cell shows <100% | Floating-point; values ~0.0001% — mark as trivial regardless |
| Negative coupling | Analysis shock (real result) — do not clip to zero |
| wind_ins has 2 trivial groups | Both U and V are forced; Wind500 = sqrt(U²+V²) so also trivial if plotting wind speed |
