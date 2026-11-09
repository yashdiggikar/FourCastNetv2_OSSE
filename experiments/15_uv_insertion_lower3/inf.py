import datetime
import numpy as np
import torch
import xarray as xr
import netCDF4 as nc4
from cftime import date2num
from pathlib import Path

from earth2mip.networks.fcnv2_sm import load as fcnv2_sm_load
from earth2mip.initial_conditions import cds

# ── Config ────────────────────────────────────────────────────────────────────
device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name}, VRAM: {props.total_memory/1e9:.1f} GB")
start_time  = datetime.datetime(2023, 2, 1)
num_steps   = 240
output_file = "/umbc/rs/pi_halem/users/yashd1/fcn_inference/uv_lower/forecast_uv_lower3_global_day30_60.nc"

WEIGHTS_DIR = Path("/umbc/rs/pi_halem/users/yashd1/fcn_inference/fcnv2_sm")
ERA5_DIR    = Path("/umbc/rs/pi_halem/users/yashd1/AURORA_DATA/rmse_data")

# ── ERA5 U and V wind ─────────────────────────────────────────────────────────
print("📂 Opening ERA5 U/V wind GRIB files...")
era5_files = [
    ERA5_DIR / "era5_pressure_2023_02.grib",
    ERA5_DIR / "era5_pressure_2023_03.grib",
    ERA5_DIR / "era5_pressure_2023_04.grib",
]
u_datasets, v_datasets = [], []
for fpath in era5_files:
    u_datasets.append(xr.open_dataset(
        str(fpath), engine="cfgrib",
        filter_by_keys={"shortName": "u", "typeOfLevel": "isobaricInhPa"},
        indexpath=None,
    ))
    v_datasets.append(xr.open_dataset(
        str(fpath), engine="cfgrib",
        filter_by_keys={"shortName": "v", "typeOfLevel": "isobaricInhPa"},
        indexpath=None,
    ))
era5_u_ds = xr.concat(u_datasets, dim="time").sortby("time")
era5_v_ds = xr.concat(v_datasets, dim="time").sortby("time")
print(f"✅ ERA5 U/V: {dict(era5_u_ds.dims)}, levels: {era5_u_ds.isobaricInhPa.values.tolist()}")

# ── Model ─────────────────────────────────────────────────────────────────────
print(f"📂 Loading FCNv2-sm onto {device}...")
package = {
    "global_means.npy": str(WEIGHTS_DIR / "global_means.npy"),
    "global_stds.npy":  str(WEIGHTS_DIR / "global_stds.npy"),
    "weights.tar":      str(WEIGHTS_DIR / "weights.tar"),
    "config.json":      str(WEIGHTS_DIR / "metadata.json"),
}
sfno_model = fcnv2_sm_load(package, device=device)
sfno_model.truth_u_ds = era5_u_ds   # no land_bool needed — global insertion
sfno_model.truth_v_ds = era5_v_ds
print("✅ Model loaded.")

# ── Initial condition from CDS ────────────────────────────────────────────────
print("📡 Fetching initial condition from CDS API...")
data_source = cds.DataSource(sfno_model.in_channel_names)
x0 = data_source[start_time]
x0 = torch.from_numpy(x0).float().unsqueeze(0).unsqueeze(0).to(device)
print(f"✅ IC shape: {x0.shape}")

# ── Create output NetCDF (incremental write) ──────────────────────────────────
channel_names = sfno_model.out_channel_names
lats  = sfno_model.grid.lat
lons  = sfno_model.grid.lon

print(f"📦 Creating output file: {output_file}")
ncf = nc4.Dataset(output_file, "w", format="NETCDF4")
ncf.createDimension("time",    None)
ncf.createDimension("channel", len(channel_names))
ncf.createDimension("lat",     len(lats))
ncf.createDimension("lon",     len(lons))

tv   = ncf.createVariable("time",    "f8", ("time",))
tv.units    = "hours since 1900-01-01 00:00:00"
tv.calendar = "standard"
cv   = ncf.createVariable("channel", str,  ("channel",))
lv   = ncf.createVariable("lat",     "f4", ("lat",)); lv.units = "degrees_north"
lonv = ncf.createVariable("lon",     "f4", ("lon",)); lonv.units = "degrees_east"
pv   = ncf.createVariable("prediction", "f4", ("time", "channel", "lat", "lon"),
                           zlib=True, complevel=4)
lv[:]   = lats
lonv[:] = lons
for i, ch in enumerate(channel_names):
    cv[i] = ch

# ── Inference ─────────────────────────────────────────────────────────────────
print(f"🚀 Running inference for {num_steps} steps ({num_steps//4} days)...")
for step_idx, (step_time, out, _) in enumerate(sfno_model(start_time, x0)):
    if step_idx >= num_steps:
        break
    tv[step_idx] = date2num(step_time, units=tv.units, calendar=tv.calendar)
    pv[step_idx] = out.cpu().numpy()[0]
    if step_idx % 20 == 0:
        print(f"  step {step_idx:3d}/{num_steps} | {step_time}")

ncf.close()
print(f"✅ Saved: {output_file}")
