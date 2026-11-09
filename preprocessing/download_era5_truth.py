#!/usr/bin/env python3
"""
Download ERA5 for Aurora RMSE verification.

Period : 2023-02-01 → 2023-04-15  (6-hourly, 0.25°)
Pressure: t u v z q  at 13 levels (50–1000 hPa)
Surface : t2m u10 v10 msl sp

Outputs (AURORA_DATA/rmse_data/):
  era5_pressure_2023_02.grib
  era5_pressure_2023_03.grib
  era5_pressure_2023_04.grib   ← days 1-15 only
  era5_surface_2023_02.grib
  era5_surface_2023_03.grib
  era5_surface_2023_04.grib    ← days 1-15 only
"""

import cdsapi
import os
from pathlib import Path

CDS_URL = "https://cds.climate.copernicus.eu/api"
CDS_KEY = os.environ["CDSAPI_KEY"]  # REDACTED for release: personal CDS API key removed

OUT_DIR = Path("/umbc/rs/pi_halem/users/yashd1/AURORA_DATA/rmse_data")
OUT_DIR.mkdir(parents=True, exist_ok=True)

client = cdsapi.Client(url=CDS_URL, key=CDS_KEY)

PRESSURE_LEVELS = [
    "50", "100", "150", "200", "250", "300",
    "400", "500", "600", "700", "850", "925", "1000"
]

DAYS_FEB = [f"{d:02d}" for d in range(1, 29)]   # 1-28
DAYS_MAR = [f"{d:02d}" for d in range(1, 32)]   # 1-31
DAYS_APR = [f"{d:02d}" for d in range(1, 16)]   # 1-15

MONTHS = [
    ("02", "2023", DAYS_FEB),
    ("03", "2023", DAYS_MAR),
    ("04", "2023", DAYS_APR),
]

TIMES = ["00:00", "06:00", "12:00", "18:00"]

# ── Pressure levels ───────────────────────────────────────────────────────────
for month, year, days in MONTHS:
    out_file = OUT_DIR / f"era5_pressure_{year}_{month}.grib"
    if out_file.exists():
        print(f"[SKIP] {out_file.name} already exists.")
        continue
    print(f"[DOWN] Pressure {year}-{month} ({len(days)} days) → {out_file.name}")
    client.retrieve(
        "reanalysis-era5-pressure-levels",
        {
            "product_type": ["reanalysis"],
            "variable": [
                "geopotential",
                "specific_humidity",
                "temperature",
                "u_component_of_wind",
                "v_component_of_wind",
            ],
            "year": [year],
            "month": [month],
            "day": days,
            "time": TIMES,
            "pressure_level": PRESSURE_LEVELS,
            "data_format": "grib",
            "download_format": "unarchived",
        },
    ).download(str(out_file))
    print(f"[DONE] {out_file.name}  ({out_file.stat().st_size / 1e9:.2f} GB)")

# ── Single levels (surface) ───────────────────────────────────────────────────
for month, year, days in MONTHS:
    out_file = OUT_DIR / f"era5_surface_{year}_{month}.grib"
    if out_file.exists():
        print(f"[SKIP] {out_file.name} already exists.")
        continue
    print(f"[DOWN] Surface  {year}-{month} ({len(days)} days) → {out_file.name}")
    client.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": ["reanalysis"],
            "variable": [
                "2m_temperature",
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",
                "mean_sea_level_pressure",
                "surface_pressure",
            ],
            "year": [year],
            "month": [month],
            "day": days,
            "time": TIMES,
            "data_format": "grib",
            "download_format": "unarchived",
        },
    ).download(str(out_file))
    print(f"[DONE] {out_file.name}  ({out_file.stat().st_size / 1e9:.2f} GB)")

print("All downloads complete.")
