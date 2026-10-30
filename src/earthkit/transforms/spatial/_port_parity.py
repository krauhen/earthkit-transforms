# Copyright 2024-, European Centre for Medium Range Weather Forecasts.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Port-only parity extras for the neighbourhood operators. NOT FOR UPSTREAM.

This module reproduces the single precision arithmetic, the summation order and the float64
return type of the legacy operators that the ported products are verified against. The clean,
upstream-ready functions live in ``_neighbourhood.py``; the two functions here wrap them and only
add what is port-only:

- ``disk_mean(..., accumulate_float32=True)``: sum in float32, x offset outer and y offset inner.
- ``fill_missing``: float32 weights and arithmetic.
- both return float64, whatever the input dtype.
"""

import numpy as np
import xarray as xr
from earthkit.utils.decorators import format_handler

from earthkit.transforms.spatial import _neighbourhood as _clean
from earthkit.transforms.spatial._neighbourhood import _disk_offsets


def _disk_mean_f32_2d(field, radius):
    dtype = np.float32
    f = np.asarray(field, dtype=dtype)
    n = int(np.floor(radius))
    ny, nx = f.shape
    ok = np.isfinite(f)
    a = np.pad(np.where(ok, f, dtype(0.0)), n, mode="constant")
    o = np.pad(ok.astype(dtype), n, mode="constant")
    total = np.zeros(f.shape, dtype)
    count = np.zeros(f.shape, dtype)
    for dx, dy in _disk_offsets(radius):
        total += a[n + dy : n + dy + ny, n + dx : n + dx + nx]
        count += o[n + dy : n + dy + ny, n + dx : n + dx + nx]
    out = np.full(f.shape, np.nan)
    m = ok & (count > 0)
    out[m] = (total[m] / count[m]).astype(np.float64)
    return out


def _nint(x):
    """Fortran NINT: round half away from zero (for non-negative ``x``), in the dtype of ``x``."""
    return np.floor(x + np.float32(0.5))


def _fill_weights(radius, weight):
    """Stencil ``(dx, dy, w)`` of the disk ``dx**2 + dy**2 <= radius**2``, float32 weights."""
    r = int(radius)
    out = []
    for dx in range(-r, r + 1):
        for dy in range(-r, r + 1):
            if dx * dx + dy * dy > r * r:
                continue
            d = np.sqrt(np.float32(dx * dx + dy * dy))
            if weight == "exp":
                w = np.exp(-d)
            elif weight == "lin":
                w = np.float32(1.0) if dx * dx + dy * dy <= 1 else (np.float32(r) - d) / np.float32(r - 1)
            elif weight == "cst":
                w = np.float32(1.0)
            else:
                raise ValueError(f"weight must be one of 'exp', 'lin', 'cst', got {weight!r}")
            out.append((dx, dy, np.float32(w)))
    return out


def _fill_missing_f32_2d(field, radius, max_missing_fraction, weight):
    f = np.asarray(field, dtype=np.float32)
    ny, nx = f.shape
    r = int(radius)
    stencil = _fill_weights(r, weight)
    # stencil points inside the domain, per cell
    n_in = np.zeros(f.shape, np.float32)
    for dx, dy, _ in stencil:
        n_in[max(0, -dy) : min(ny, ny - dy), max(0, -dx) : min(nx, nx - dx)] += np.float32(1.0)
    max_missing = _nint(np.float32(max_missing_fraction) * n_in)
    ok = np.isfinite(f)
    a = np.pad(np.where(ok, f, np.float32(0.0)), r, mode="constant")
    o = np.pad(ok.astype(np.float32), r, mode="constant")
    sum_f = np.zeros(f.shape, np.float32)
    sum_w = np.zeros(f.shape, np.float32)
    n_def = np.zeros(f.shape, np.float32)
    for dx, dy, w in stencil:
        av = a[r + dy : r + dy + ny, r + dx : r + dx + nx]
        ov = o[r + dy : r + dy + ny, r + dx : r + dx + nx]
        sum_f += w * av
        sum_w += w * ov
        n_def += ov
    fill = ~ok & (sum_w > 0) & (n_in - n_def <= max_missing)
    out = f.copy()
    out[fill] = sum_f[fill] / sum_w[fill]
    return out.astype(np.float64)


def _per_slice(func, *args):
    """Return a function applying the 2-D ``func`` over the last two axes, looping over leading axes."""

    def apply(values):
        values = np.asarray(values)
        flat = values.reshape((-1,) + values.shape[-2:])
        return np.stack([func(v, *args) for v in flat]).reshape(values.shape)

    return apply


def _as_float64(data):
    """Cast the data to float64, the dtype the port-only kernels return."""
    if isinstance(data, (xr.DataArray, xr.Dataset)):
        return data.astype(np.float64)
    return np.asarray(data, dtype=np.float64)


@format_handler()
def disk_mean(data, radius, *, accumulate_float32=False, lat_key=None, lon_key=None):
    """Port version of :func:`_neighbourhood.disk_mean`: float64 result, optional float32 sums."""
    data = _as_float64(data)
    if not accumulate_float32:
        return _clean.disk_mean(data, radius, lat_key=lat_key, lon_key=lon_key)
    if radius < 0:
        raise ValueError(f"radius must be non-negative, got {radius}")
    return _clean._dispatch(_per_slice(_disk_mean_f32_2d, radius), data, lat_key=lat_key, lon_key=lon_key)


@format_handler()
def fill_missing(data, radius, max_missing_fraction, *, weight="exp", lat_key=None, lon_key=None):
    """Port version of :func:`_neighbourhood.fill_missing`: float32 arithmetic, float64 result."""
    if int(radius) != radius or radius < 1:
        raise ValueError(f"radius must be a positive integer, got {radius}")
    _fill_weights(1, weight)  # validate weight before looping
    func = _per_slice(_fill_missing_f32_2d, int(radius), max_missing_fraction, weight)
    return _clean._dispatch(func, _as_float64(data), lat_key=lat_key, lon_key=lon_key)
