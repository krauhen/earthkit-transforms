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

"""Neighbourhood operators on regular 2-D grids."""

import numpy as np
import xarray as xr
from earthkit.utils.decorators import format_handler

from earthkit.transforms._tools import get_spatial_info


def _disk_offsets(radius):
    """Offsets ``(dx, dy)`` of the disk ``dx**2 + dy**2 <= radius**2``, ``dx`` outer, ``dy`` inner."""
    n = int(np.floor(radius))
    return [(dx, dy) for dx in range(-n, n + 1) for dy in range(-n, n + 1) if dx * dx + dy * dy <= radius * radius]


def _disk_mean_2d(values, radius):
    """Disk mean over the last two axes of ``values``; leading axes are carried along.

    The sum is accumulated in float64 and the result is cast back to the input dtype
    (float64 for non-floating input).
    """
    values = np.asarray(values)
    out_dtype = values.dtype if np.issubdtype(values.dtype, np.floating) else np.dtype("float64")
    f = values.astype(np.float64)
    n = int(np.floor(radius))
    ny, nx = f.shape[-2:]
    ok = np.isfinite(f)
    pad = [(0, 0)] * (f.ndim - 2) + [(n, n), (n, n)]
    a = np.pad(np.where(ok, f, 0.0), pad, mode="constant")
    o = np.pad(ok.astype(np.float64), pad, mode="constant")
    total = np.zeros(f.shape)
    count = np.zeros(f.shape)
    for dx, dy in _disk_offsets(radius):
        window = (Ellipsis, slice(n + dy, n + dy + ny), slice(n + dx, n + dx + nx))
        total += a[window]
        count += o[window]
    valid = ok & (count > 0)
    out = np.full(f.shape, np.nan)
    np.divide(total, count, out=out, where=valid)
    return out.astype(out_dtype)


def _apply_spatial(func, data, *args, lat_key=None, lon_key=None):
    """Apply ``func(values, *args)`` over the two spatial dimensions of a DataArray.

    ``func`` receives the spatial dimensions as the last two axes and must return an array of the
    same shape. Dask-backed input is processed lazily; the spatial dimensions are rechunked to a
    single chunk each.
    """
    info = get_spatial_info(data, lat_key=lat_key, lon_key=lon_key)
    if not info["regular"] or len(info["spatial_dims"]) != 2:
        raise ValueError("Neighbourhood operators require a regular 2-D latitude/longitude grid")
    ydim, xdim = info["spatial_dims"]
    result = xr.apply_ufunc(
        func,
        data,
        *args,
        input_core_dims=[[ydim, xdim]] + [[]] * len(args),
        output_core_dims=[[ydim, xdim]],
        keep_attrs=True,
        dask="parallelized",
        output_dtypes=[data.dtype if np.issubdtype(data.dtype, np.floating) else np.float64],
        dask_gufunc_kwargs={"allow_rechunk": True},
    )
    return result.transpose(*data.dims)


def _dispatch(func, data, *args, lat_key=None, lon_key=None):
    """Apply ``func`` to a Dataset, DataArray or array; arrays use their last two axes."""
    if isinstance(data, xr.Dataset):
        out = data.copy()
        for name in data.data_vars:
            out[name] = _apply_spatial(func, data[name], *args, lat_key=lat_key, lon_key=lon_key)
        return out
    if isinstance(data, xr.DataArray):
        return _apply_spatial(func, data, *args, lat_key=lat_key, lon_key=lon_key)
    data = np.asarray(data)
    if data.ndim < 2:
        raise ValueError(f"Expected at least 2 dimensions (y, x), got {data.ndim}")
    return func(data, *args)


@format_handler()
def disk_mean(
    data: xr.DataArray | xr.Dataset | np.ndarray,
    radius: float,
    *,
    lat_key: str | None = None,
    lon_key: str | None = None,
) -> xr.DataArray | xr.Dataset | np.ndarray:
    r"""Mean of the defined values on a disk around each grid cell.

    For each cell the mean is taken over the cells with offsets ``dx**2 + dy**2 <= radius**2``,
    with ``radius`` in grid cells. Missing (NaN) cells are left out of the mean, and cells that
    are missing in the input stay missing. The disk is truncated at the domain edge.

    .. math::

        \bar{f}(y, x) = \frac{1}{N} \sum_{dx^2 + dy^2 \le r^2} f(y + dy, x + dx)

    where :math:`N` is the number of defined cells on the disk.

    Parameters
    ----------
    data : xarray.DataArray | xarray.Dataset | numpy.ndarray
        Data on a regular grid. For a numpy array the last two dimensions are taken as
        (y, x) and any leading dimensions are carried along.
    radius : float
        Disk radius in grid cells. Must be non-negative.
    lat_key : str, optional
        Name of the latitude coordinate of xarray input. Detected if not given.
    lon_key : str, optional
        Name of the longitude coordinate of xarray input. Detected if not given.

    Returns
    -------
    xarray.DataArray | xarray.Dataset | numpy.ndarray
        The disk mean, with the same type, shape, dimensions and attributes as ``data``.
        Floating point input keeps its dtype; other input gives float64.

    Raises
    ------
    ValueError
        If ``radius`` is negative, or xarray input is not on a regular 2-D latitude/longitude grid.

    Notes
    -----
    The sum is accumulated in double precision and the result is cast to the input dtype.
    Dask-backed input stays lazy, but the two spatial dimensions are rechunked to a single chunk
    each (``allow_rechunk=True``) because every cell needs its neighbours.

    Examples
    --------
    >>> import numpy as np
    >>> from earthkit.transforms.spatial import disk_mean
    >>> disk_mean(np.ones((3, 3)), 1.0)
    array([[1., 1., 1.],
           [1., 1., 1.],
           [1., 1., 1.]])
    """
    if radius < 0:
        raise ValueError(f"radius must be non-negative, got {radius}")
    return _dispatch(_disk_mean_2d, data, radius, lat_key=lat_key, lon_key=lon_key)
