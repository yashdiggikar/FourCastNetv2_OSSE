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


# ── Pressure-level observation error σ_T (Ingleby 2017 / ECMWF radiosonde) ───
# FCNv2 T channels 47–59 → levels [50,100,150,200,250,300,400,500,600,700,850,925,1000] hPa
# Stratosphere (50–150 hPa): larger errors due to solar heating & instrument bias
# Free troposphere (400–600 hPa): minimum errors — best retrieval quality
# Lower troposphere / BL (850–1000 hPa): surface effects & humidity coupling
SIGMA_T_BY_LEVEL = np.array([
    2.0,  # 50  hPa
    1.5,  # 100 hPa
    1.5,  # 150 hPa
    1.2,  # 200 hPa
    1.0,  # 250 hPa
    0.8,  # 300 hPa
    0.7,  # 400 hPa
    0.7,  # 500 hPa
    0.8,  # 600 hPa
    0.8,  # 700 hPa
    1.0,  # 850 hPa
    1.0,  # 925 hPa
    1.2,  # 1000 hPa
], dtype=np.float32)  # shape (13,)


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

        # Convert center/scale to tensors on x's device (needed for GPU inference)
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
                x = self.model(x[:, -1]).unsqueeze(1)  # FCNv2 forward(x): 4D in/out; re-add time dim
                time = time + self.time_step
                step += 1

                out = self.scale * x[:, -1] + self.center

                # ── Perturbed global T insertion, all 13 levels, steps 120+ ─────────
                # Method: ERA5 truth T is perturbed with spatially-uncorrelated
                # Gaussian noise ε ~ N(0, σ²_lev) before insertion (Errico et al. 2013).
                # σ per level from Ingleby (2017) ECMWF radiosonde error assessment.
                # Seed is step-deterministic for exact reproducibility.
                if hasattr(self, "truth_ds") and step >= 120:
                    TARGET_LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
                    try:
                        t_np = self.truth_ds["t"].sel(
                            time=np.datetime64(time, "ns"),
                            isobaricInhPa=TARGET_LEVELS,
                            method="nearest",
                        ).values.astype(np.float32)  # (13, lat, lon)

                        if t_np.shape[-2] == 720:
                            t_np = np.concatenate([t_np, t_np[:, -1:, :]], axis=-2)

                        # Perturbation in physical space (Kelvin) — spatially uncorrelated
                        rng = np.random.default_rng(seed=step)
                        noise = np.stack([
                            rng.normal(0.0, float(SIGMA_T_BY_LEVEL[li]), size=t_np.shape[1:]).astype(np.float32)
                            for li in range(13)
                        ], axis=0)  # (13, lat, lon)
                        t_np_perturbed = t_np + noise

                        center_t     = self.center[:, 47:60].view(1, 13, 1, 1)
                        scale_t      = self.scale[:, 47:60].view(1, 13, 1, 1)
                        t_tensor     = torch.from_numpy(t_np_perturbed).float().unsqueeze(0).to(x.device)
                        normalized_t = (t_tensor - center_t) / scale_t

                        x[:, -1, 47:60] = normalized_t
                        out[:, 47:60]   = scale_t * normalized_t + center_t

                        if step % 40 == 0:
                            noise_rms = float(np.sqrt(np.mean(noise ** 2)))
                            print(f"[T-PERTURBED] step={step}  time={time}  noise_rms={noise_rms:.3f}K")

                    except Exception as e:
                        print(f"[T-PERTURBED] injection error step={step} | {time} → {e}")

                restart = dict(x=x, normalize=False, time=time)
                yield time, out, restart
