# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch

import datetime
import sys
import urllib
import warnings
from typing import Any, Callable, Iterator, Optional, Tuple
import numpy as np
import xarray as xr
import datetime


import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter


import earth2mip.grid
from earth2mip import (
    ModelRegistry,
    loaders,
    model_registry,
    registry,
    schema,
    time_loop,
)
from earth2mip.loaders import LoaderProtocol

if sys.version_info < (3, 10):
    from importlib_metadata import EntryPoint, entry_points
else:
    from importlib.metadata import EntryPoint, entry_points


__all__ = ["get_model"]


def depends_on_time(f):
    if isinstance(f, torch.jit.ScriptModule):
        return False
    else:
        import inspect
        signature = inspect.signature(f)
        parameters = signature.parameters
        return "time" in parameters


class Wrapper(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


def _load_package(package, pretrained=True):
    from earth2mip.networks.fcnv2_sm import load
    return load(package)


class TimeLoop(torch.nn.Module):
    time_step: datetime.timedelta
    in_channel_names: list
    out_channel_names: list
    grid: earth2mip.grid.LatLonGrid

    def __call__(self, time, x, restart=None):
        return self.forward(time, x, restart)


_packages = {}


def get_model(model_name, registry=None, device="cpu"):
    if model_name in _packages:
        return _packages[model_name]
    package = registry.get_model(model_name)
    model = _load_package(package)
    _packages[model_name] = model
    return model


# ═════════════════════════════════════════════════════════════════════════
# CORRELATED OSSE PERTURBATION (COP) SCHEME — WIND (U, V) VARIANT
# ═════════════════════════════════════════════════════════════════════════
# Upgrade over independent-per-component Gaussian noise. u and v are NOT
# perturbed independently here — both are derived from ONE random
# streamfunction field psi (itself horizontally- and vertically-correlated,
# same construction as the T variant in t_perturbed_corr/), via the
# standard non-divergent/rotational-wind relation:
#     u' = -(1/R) dpsi/dphi         v' = (1/(R cos(phi))) dpsi/dlambda
# (Holton, "An Introduction to Dynamic Meteorology" — streamfunction wind
# relation). This avoids injecting spurious pointwise divergence/convergence
# that two independently-perturbed u,v fields would otherwise create, and
# mirrors the standard DA practice of representing wind background/
# observation error via streamfunction/velocity-potential rather than raw
# u,v components (Parrish & Derber 1992; Ingleby 2001).
# After differentiation the raw u',v' fields are rescaled per level so
# their empirical std matches the target sigma_u/sigma_v (Ingleby
# 2017-style wind-error-by-level table) — differentiation changes the
# physical magnitude of a unit-variance streamfunction in a way that
# depends on the correlation length, so this calibration step is required.
# See PERTURBATION_EXPERIMENTS/FCN/IMPLEMENTATION.md for full details.
# ═════════════════════════════════════════════════════════════════════════

TARGET_LEVELS_UV = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]

# Wind observation error by level (m/s), Ingleby (2017)-style radiosonde wind
# error assessment: larger near the jet-stream levels (200-300 hPa, where
# wind speed and representativeness error are both largest), smaller near
# the surface and in the upper stratosphere. u and v assumed to share the
# same magnitude table (isotropic turbulence assumption) — documented
# simplification, see IMPLEMENTATION.md.
SIGMA_UV_BY_LEVEL = np.array([
    1.5,  # 50  hPa
    1.6,  # 100 hPa
    1.8,  # 150 hPa
    2.2,  # 200 hPa  (near jet core — largest wind + representativeness error)
    2.0,  # 250 hPa
    1.8,  # 300 hPa
    1.4,  # 400 hPa
    1.2,  # 500 hPa
    1.1,  # 600 hPa
    1.0,  # 700 hPa
    1.0,  # 850 hPa
    1.1,  # 925 hPa
    1.3,  # 1000 hPa (surface friction / representativeness)
], dtype=np.float32)  # shape (13,)

HORIZ_CORR_LENGTH_KM = 250.0
GRID_SPACING_DEG     = 0.25
KM_PER_DEG_LAT        = 111.0
SIGMA_PX = HORIZ_CORR_LENGTH_KM / (KM_PER_DEG_LAT * GRID_SPACING_DEG)   # ≈ 9.0 grid cells

VERT_L_LOGP = 0.5
_logp = np.log(np.array(TARGET_LEVELS_UV, dtype=np.float64))
_vert_dist = np.abs(_logp[:, None] - _logp[None, :])
VERT_CORR_UV = np.exp(-_vert_dist / VERT_L_LOGP)
VERT_CHOL_UV = np.linalg.cholesky(VERT_CORR_UV).astype(np.float32)   # (13,13)

EARTH_RADIUS_M = 6371000.0


def _correlated_level_fields(n_lev, shape_hw, vert_chol, rng):
    """Return (n_lev, H, W) fields, each unit-variance, spatially smooth
    (Gaussian-filtered white noise, correlation length SIGMA_PX pixels) AND
    correlated across levels via vert_chol."""
    H, W = shape_hw
    raw = np.stack([
        gaussian_filter(rng.standard_normal((H, W)).astype(np.float32),
                         sigma=SIGMA_PX, mode=("nearest", "wrap"))
        for _ in range(n_lev)
    ], axis=0)
    raw = raw / raw.std(axis=(1, 2), keepdims=True)
    mixed = np.einsum("ij,jhw->ihw", vert_chol, raw)
    mixed = mixed / mixed.std(axis=(1, 2), keepdims=True)
    return mixed.astype(np.float32)


def _streamfunction_to_uv(psi, lat_deg, lon_deg):
    """psi: (n_lev, H, W) random streamfunction field (arbitrary units).
    Returns non-divergent (u, v) via u = -(1/R) dpsi/dphi,
    v = (1/(R cos(phi))) dpsi/dlambda (Holton, streamfunction/rotational-
    wind relation). Longitude derivative uses a periodic central
    difference; latitude derivative uses np.gradient with the true
    (possibly non-uniform) latitude coordinate."""
    lat_rad = np.deg2rad(np.asarray(lat_deg, dtype=np.float64))
    lon_rad = np.deg2rad(np.asarray(lon_deg, dtype=np.float64))
    dlambda = float(np.mean(np.diff(lon_rad)))

    dpsi_dlambda = (np.roll(psi, -1, axis=2) - np.roll(psi, 1, axis=2)) / (2.0 * dlambda)
    dpsi_dphi    = np.gradient(psi, lat_rad, axis=1)

    cosphi = np.clip(np.cos(lat_rad), np.cos(np.deg2rad(85.0)), None)[None, :, None]

    u = -(1.0 / EARTH_RADIUS_M) * dpsi_dphi
    v = (1.0 / (EARTH_RADIUS_M * cosphi)) * dpsi_dlambda
    return u.astype(np.float32), v.astype(np.float32)


def _generate_correlated_wind_perturbation(n_lev, lat_deg, lon_deg, vert_chol, sigma_by_level, rng):
    """Full pipeline: correlated random streamfunction -> non-divergent u,v
    -> rescale per level so empirical std matches sigma_by_level (m/s)."""
    psi = _correlated_level_fields(n_lev, (len(lat_deg), len(lon_deg)), vert_chol, rng)
    u_raw, v_raw = _streamfunction_to_uv(psi, lat_deg, lon_deg)
    u_std = u_raw.std(axis=(1, 2), keepdims=True)
    v_std = v_raw.std(axis=(1, 2), keepdims=True)
    u = (u_raw / u_std) * sigma_by_level[:, None, None]
    v = (v_raw / v_std) * sigma_by_level[:, None, None]
    return u.astype(np.float32), v.astype(np.float32)


class _TimeLoopImpl(TimeLoop):
    def __init__(self, model, center, scale, grid, in_channel_names, out_channel_names,
                 time_step, n_history=0, time_dependent=False):
        super().__init__()
        self.model             = model
        self.center            = center
        self.scale             = scale
        self.grid              = grid
        self.in_channel_names  = in_channel_names
        self.out_channel_names = out_channel_names
        self.time_step         = time_step
        self.n_history         = n_history
        self.time_dependent    = time_dependent

    def __call__(self, time, x, restart=None):
        if restart:
            yield from self._iterate(**restart)
        else:
            yield from self._iterate(x=x, time=time)

    def _iterate(self, x, normalize=True, time=None):
        if self.time_dependent and not time:
            raise ValueError("Time dependent models require time.")
        time = time or datetime.datetime(1900, 1, 1)

        if not isinstance(self.center, torch.Tensor):
            self.center = torch.from_numpy(np.array(self.center)).to(x.device)
            self.scale  = torch.from_numpy(np.array(self.scale)).to(x.device)
        elif self.center.device != x.device:
            self.center = self.center.to(x.device)
            self.scale  = self.scale.to(x.device)

        with torch.no_grad():
            _, n_time_levels, n_channels, _, _ = x.shape
            assert n_time_levels == self.n_history + 1

            if normalize:
                x = (x - self.center) / self.scale

            restart = dict(x=x, normalize=False, time=time)
            yield time, self.scale * x[:, -1] + self.center, restart

            step = 0

            while True:
                x = self.model(x[:, -1]).unsqueeze(1)
                time = time + self.time_step
                step += 1

                out = self.scale * x[:, -1] + self.center

                # ── Correlated-perturbation global U+V insertion, steps 120+ ────────
                if hasattr(self, "truth_u_ds") and step >= 120:
                    try:
                        t_coord = np.datetime64(time, "ns")
                        u_np = self.truth_u_ds["u"].sel(
                            time=t_coord, isobaricInhPa=TARGET_LEVELS_UV, method="nearest",
                        ).values.astype(np.float32)  # (13, lat, lon)
                        v_np = self.truth_v_ds["v"].sel(
                            time=t_coord, isobaricInhPa=TARGET_LEVELS_UV, method="nearest",
                        ).values.astype(np.float32)

                        if u_np.shape[-2] == 720:
                            u_np = np.concatenate([u_np, u_np[:, -1:, :]], axis=-2)
                            v_np = np.concatenate([v_np, v_np[:, -1:, :]], axis=-2)

                        rng = np.random.default_rng(seed=step)
                        lat_deg = self.grid.lat
                        lon_deg = self.grid.lon
                        u_noise, v_noise = _generate_correlated_wind_perturbation(
                            13, lat_deg, lon_deg, VERT_CHOL_UV, SIGMA_UV_BY_LEVEL, rng
                        )
                        u_np_perturbed = u_np + u_noise
                        v_np_perturbed = v_np + v_noise

                        center_u = self.center[:, 8:21].view(1, 13, 1, 1)
                        scale_u  = self.scale[:, 8:21].view(1, 13, 1, 1)
                        center_v = self.center[:, 21:34].view(1, 13, 1, 1)
                        scale_v  = self.scale[:, 21:34].view(1, 13, 1, 1)

                        u_tensor = torch.from_numpy(u_np_perturbed).float().unsqueeze(0).to(x.device)
                        v_tensor = torch.from_numpy(v_np_perturbed).float().unsqueeze(0).to(x.device)
                        norm_u = (u_tensor - center_u) / scale_u
                        norm_v = (v_tensor - center_v) / scale_v

                        x[:, -1, 8:21]  = norm_u
                        x[:, -1, 21:34] = norm_v
                        out[:, 8:21]    = scale_u * norm_u + center_u
                        out[:, 21:34]   = scale_v * norm_v + center_v

                        if step % 40 == 0:
                            print(f"[UV-PERTURBED-CORR] step={step}  time={time}  "
                                  f"u_noise_rms={float(np.sqrt(np.mean(u_noise**2))):.3f}  "
                                  f"v_noise_rms={float(np.sqrt(np.mean(v_noise**2))):.3f} m/s")

                    except Exception as e:
                        print(f"[UV-PERTURBED-CORR] injection error step={step} | {time} → {e}")

                restart = dict(x=x, normalize=False, time=time)
                yield time, out, restart


def _default_inference(package, metadata: schema.Model, device):
    if metadata.architecture == "pickle":
        loader = loaders.pickle
    elif metadata.architecture_entrypoint:
        ep = EntryPoint(name=None, group=None, value=metadata.architecture_entrypoint)
        loader: LoaderProtocol = ep.load()
    else:
        raise NotImplementedError()
