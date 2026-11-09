# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import torch
import datetime
import sys
import warnings
from typing import Any, Iterator, Optional, Tuple
import numpy as np
import xarray as xr

import earth2mip.grid
from earth2mip import (ModelRegistry, loaders, model_registry, registry, schema, time_loop)
from earth2mip.loaders import LoaderProtocol

if sys.version_info < (3, 10):
    from importlib_metadata import EntryPoint, entry_points
else:
    from importlib.metadata import EntryPoint, entry_points

__all__ = ["get_model"]


def depends_on_time(f):
    if isinstance(f, torch.jit.ScriptModule):
        return False
    import inspect
    return "time" in inspect.signature(f).parameters


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

        with torch.no_grad():
            _, n_time_levels, n_channels, _, _ = x.shape
            assert n_time_levels == self.n_history + 1

            if normalize:
                x = (x - self.center) / self.scale

            restart = dict(x=x, normalize=False, time=time)
            yield time, self.scale * x[:, -1] + self.center, restart

            step = 0
            INJ_START = 120   # day 30

            while True:
                x = self.model(x, time)
                time = time + self.time_step
                step += 1

                out = self.scale * x[:, -1] + self.center

                # ── T global insertion every 12 h (every other 6-h step) ──────────
                # Inject at steps 120, 122, 124, ... (step - INJ_START) % 2 == 0
                # Odd steps keep the model's own forecast unmodified.
                if hasattr(self, "truth_ds") and step >= INJ_START and (step - INJ_START) % 2 == 0:
                    TARGET_LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
                    try:
                        t_np = self.truth_ds["t"].sel(
                            time=np.datetime64(time, "ns"),
                            isobaricInhPa=TARGET_LEVELS,
                            method="nearest",
                        ).values.astype(np.float32)  # (13, lat, lon)

                        if t_np.shape[-2] == 720:
                            t_np = np.concatenate([t_np, t_np[:, -1:, :]], axis=-2)

                        center_t     = self.center[47:60].view(1, 13, 1, 1)
                        scale_t      = self.scale[47:60].view(1, 13, 1, 1)
                        t_tensor     = torch.from_numpy(t_np).float().unsqueeze(0).to(x.device)
                        normalized_t = (t_tensor - center_t) / scale_t

                        x[:, -1, 47:60] = normalized_t
                        out[:, 47:60]   = scale_t * normalized_t + center_t

                        if step % 40 == 0:
                            print(f"[T-12H] step={step} (inject)  time={time}")
                    except Exception as e:
                        print(f"[T-12H] error step={step} | {time} → {e}")
                elif hasattr(self, "truth_ds") and step >= INJ_START:
                    if step % 40 == 0:
                        print(f"[T-12H] step={step} (skip)    time={time}")

                restart = dict(x=x, normalize=False, time=time)
                yield time, out, restart
