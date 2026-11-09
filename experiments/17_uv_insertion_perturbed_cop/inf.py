"""UV_PERTURBED_CORR: ERA5 U+V all-13-level GLOBAL insertion, with spatially-
and vertically-CORRELATED, non-divergent (streamfunction-derived) wind
perturbation noise.

See PERTURBATION_EXPERIMENTS/FCN/IMPLEMENTATION.md for the full method
writeup (citations, hyperparameter justification). The perturbation-
generation code lives in
OSSE-FourCastNetv2/earth2mip/networks/__init__.py::_generate_correlated_wind_perturbation()
and is applied inside _TimeLoopImpl._iterate() at every injection step.
"""
import datetime
import numpy as np
import torch
import xarray as xr
import netCDF4 as nc4
from cftime import date2num
from pathlib import Path

from earth2mip.networks.fcnv2_sm import load as fcnv2_sm_load

device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name}, VRAM: {props.total_memory/1e9:.1f} GB")
start_time  = datetime.datetime(2023, 2, 1)
num_steps   = 240
output_file = "/umbc/rs/pi_halem/users/yashd1/fcn_inference/uv_perturbed_corr/forecast_uv_perturbed_corr_global_day30_60.nc"

WEIGHTS_DIR = Path("/umbc/rs/pi_halem/users/yashd1/fcn_inference/fcnv2_sm")
ERA5_DIR    = Path("/umbc/rs/pi_halem/users/yashd1/AURORA_DATA/rmse_data")

print("Opening ERA5 U/V wind GRIB files ...")
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
    ))
    v_datasets.append(xr.open_dataset(
        str(fpath), engine="cfgrib",
        filter_by_keys={"shortName": "v", "typeOfLevel": "isobaricInhPa"},
    ))
era5_u_ds = xr.concat(u_datasets, dim="time").sortby("time")
era5_v_ds = xr.concat(v_datasets, dim="time").sortby("time")
print(f"ERA5 U: {dict(era5_u_ds.dims)}, levels: {era5_u_ds.isobaricInhPa.values.tolist()}")
print(f"ERA5 V: {dict(era5_v_ds.dims)}, levels: {era5_v_ds.isobaricInhPa.values.tolist()}")

print(f"Loading FCNv2-sm onto {device} ...")
package = {
    "global_means.npy": str(WEIGHTS_DIR / "global_means.npy"),
    "global_stds.npy":  str(WEIGHTS_DIR / "global_stds.npy"),
    "weights.tar":      str(WEIGHTS_DIR / "weights.tar"),
    "config.json":      str(WEIGHTS_DIR / "metadata.json"),
}
sfno_model = fcnv2_sm_load(package, device=device)
sfno_model.truth_u_ds = era5_u_ds   # global insertion — no land_bool needed
sfno_model.truth_v_ds = era5_v_ds
print("Model loaded.")

print("Loading IC from control forecast (step 0 = 2023-02-01 00:00) ...")
CTRL_FILE = Path("/umbc/rs/pi_halem/users/yashd1/FourCastNet/forecast_normal_60days_12hr.nc")
with nc4.Dataset(str(CTRL_FILE)) as _ctrl:
    x0_np = _ctrl.variables["prediction"][0, :, :, :].astype(np.float32)  # (73, 721, 1440)
x0 = torch.from_numpy(x0_np).float().unsqueeze(0).unsqueeze(0).to(device)
print(f"IC shape: {x0.shape}")

channel_names = sfno_model.out_channel_names
lats  = sfno_model.grid.lat
lons  = sfno_model.grid.lon

print(f"Creating output file: {output_file}")
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
ncf.experiment = ("UV_PERTURBED_CORR: ERA5 U+V all-13-level global insertion with "
                   "spatially- and vertically-CORRELATED, non-divergent "
                   "(streamfunction-derived) wind perturbation noise "
                   "(250km horizontal length scale, exponential log-pressure "
                   "vertical decay, Ingleby 2017-style sigma-by-level)")

print(f"Running {num_steps}-step inference ...")
for step_idx, (step_time, out, _) in enumerate(sfno_model(start_time, x0)):
    if step_idx >= num_steps:
        break
    tv[step_idx] = date2num(step_time, units=tv.units, calendar=tv.calendar)
    pv[step_idx] = out.cpu().numpy()[0]
    if step_idx % 20 == 0:
        print(f"  step {step_idx:3d}/{num_steps} | {step_time}")

ncf.close()
print(f"Saved: {output_file}")
