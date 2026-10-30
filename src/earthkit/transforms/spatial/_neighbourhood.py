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

import functools
import typing as T

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


def _window_sums(f, ok, n, stencil):
    """Weighted sums over a stencil around every cell of the last two axes of ``f``.

    Parameters
    ----------
    f : numpy.ndarray
        Values (float64); entries where ``ok`` is False are ignored.
    ok : numpy.ndarray
        Boolean mask of the defined values.
    n : int
        Padding, at least the largest ``|dx|`` or ``|dy|`` in ``stencil``.
    stencil : list of (int, int, float)
        Offsets ``(dx, dy)`` with their weights.

    Returns
    -------
    tuple of numpy.ndarray
        Sum of ``w * f``, sum of ``w`` and the number of defined cells, over the defined stencil
        cells inside the domain; each has the shape of ``f``.
    """
    ny, nx = f.shape[-2:]
    pad = [(0, 0)] * (f.ndim - 2) + [(n, n), (n, n)]
    a = np.pad(np.where(ok, f, 0.0), pad, mode="constant")
    o = np.pad(ok.astype(np.float64), pad, mode="constant")
    sum_f = np.zeros(f.shape)
    sum_w = np.zeros(f.shape)
    count = np.zeros(f.shape)
    for dx, dy, w in stencil:
        window = (Ellipsis, slice(n + dy, n + dy + ny), slice(n + dx, n + dx + nx))
        sum_f += w * a[window]
        sum_w += w * o[window]
        count += o[window]
    return sum_f, sum_w, count


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


_FILL_WEIGHTS = ("exp", "lin", "cst")


def _fill_stencil(radius, weight):
    """Stencil ``(dx, dy, w)`` of the disk of integer ``radius`` for a named weight function."""
    stencil = []
    for dx, dy in _disk_offsets(radius):
        d = np.hypot(dx, dy)
        if weight == "exp":
            w = np.exp(-d)
        elif weight == "lin":
            # d > 1 only happens for radius >= 2, so the division is safe
            w = 1.0 if d <= 1 else (radius - d) / (radius - 1)
        else:
            w = 1.0
        stencil.append((dx, dy, float(w)))
    return stencil


def _fill_missing_2d(values, radius, max_missing_fraction, weight):
    """Fill NaN cells of the last two axes of ``values``; see :func:`fill_missing`."""
    values = np.asarray(values)
    out_dtype = values.dtype if np.issubdtype(values.dtype, np.floating) else np.dtype("float64")
    f = values.astype(np.float64)
    ok = np.isfinite(f)
    stencil = _fill_stencil(radius, weight)
    sum_f, sum_w, n_def = _window_sums(f, ok, radius, stencil)
    inside = np.ones(f.shape[-2:], dtype=bool)
    n_in = _window_sums(inside.astype(np.float64), inside, radius, stencil)[2]
    max_missing = np.floor(max_missing_fraction * n_in + 0.5)
    fill = ~ok & (sum_w > 0) & (n_in - n_def <= max_missing)
    out = f.copy()
    np.divide(sum_f, sum_w, out=out, where=fill)
    return out.astype(out_dtype)


@format_handler()
def fill_missing(
    data: xr.DataArray | xr.Dataset | np.ndarray,
    radius: int,
    max_missing_fraction: float,
    *,
    weight: T.Literal["exp", "lin", "cst"] = "exp",
    lat_key: str | None = None,
    lon_key: str | None = None,
) -> xr.DataArray | xr.Dataset | np.ndarray:
    r"""Fill missing cells with a distance-weighted mean of their defined neighbours.

    The stencil is the disk ``dx**2 + dy**2 <= radius**2`` (centre included), truncated at the
    domain edge, with :math:`d = \sqrt{dx^2 + dy^2}` in grid cells. A missing cell is filled with

    .. math::

        f(y, x) = \frac{\sum w(d) f(y + dy, x + dx)}{\sum w(d)}

    over the defined stencil cells. It is only filled if the number of missing stencil cells
    (centre included) is at most ``max_missing_fraction * n`` rounded half up, with ``n`` the number
    of stencil cells inside the domain. Defined cells are copied unchanged and only one pass is made,
    so filled values are not used to fill other cells.

    Parameters
    ----------
    data : xarray.DataArray | xarray.Dataset | numpy.ndarray
        Data on a regular grid, missing values as NaN. For a numpy array the last two dimensions
        are taken as (y, x) and any leading dimensions are carried along.
    radius : int
        Stencil radius in grid cells, a whole number of at least 1.
    max_missing_fraction : float
        Largest fraction of missing stencil cells, between 0 and 1, for which a cell is still filled.
    weight : {"exp", "lin", "cst"}, optional
        Weight function of the distance :math:`d`: ``"exp"`` (default) is :math:`e^{-d}`; ``"lin"`` is
        1 for :math:`d \le 1` and :math:`(r - d) / (r - 1)` beyond; ``"cst"`` is 1.
    lat_key : str, optional
        Name of the latitude coordinate of xarray input. Detected if not given.
    lon_key : str, optional
        Name of the longitude coordinate of xarray input. Detected if not given.

    Returns
    -------
    xarray.DataArray | xarray.Dataset | numpy.ndarray
        The filled data, with the same type, shape, dimensions and attributes as ``data``.
        Floating point input keeps its dtype; other input gives float64.

    Raises
    ------
    ValueError
        If ``radius``, ``max_missing_fraction`` or ``weight`` is invalid, or xarray input is not
        on a regular 2-D latitude/longitude grid.

    Notes
    -----
    The sums are accumulated in double precision and the result is cast to the input dtype.
    Dask-backed input stays lazy, but the two spatial dimensions are rechunked to a single chunk
    each (``allow_rechunk=True``) because every cell needs its neighbours.

    Examples
    --------
    >>> import numpy as np
    >>> from earthkit.transforms.spatial import fill_missing
    >>> field = np.ones((3, 3))
    >>> field[1, 1] = np.nan
    >>> fill_missing(field, 1, 0.5)
    array([[1., 1., 1.],
           [1., 1., 1.],
           [1., 1., 1.]])
    """
    if int(radius) != radius or radius < 1:
        raise ValueError(f"radius must be a whole number of at least 1, got {radius}")
    if not 0 <= max_missing_fraction <= 1:
        raise ValueError(f"max_missing_fraction must be between 0 and 1, got {max_missing_fraction}")
    if weight not in _FILL_WEIGHTS:
        raise ValueError(f"weight must be one of {_FILL_WEIGHTS}, got {weight!r}")
    func = functools.partial(
        _fill_missing_2d, radius=int(radius), max_missing_fraction=max_missing_fraction, weight=weight
    )
    return _dispatch(func, data, lat_key=lat_key, lon_key=lon_key)
