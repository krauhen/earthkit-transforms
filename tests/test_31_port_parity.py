"""Tests of the port-only parity extras (float32 arithmetic, float64 return). Not for upstream."""

import numpy as np
import pytest
import xarray as xr

from earthkit.transforms import spatial
from earthkit.transforms.spatial import _neighbourhood as _nb


def _disk_mean_f32_reference(field, radius):
    """Plain float32 loop: x offset outer, y offset inner, as the legacy operator sums it."""
    f = np.asarray(field, dtype=np.float32)
    n = int(np.floor(radius))
    ny, nx = f.shape
    out = np.full(f.shape, np.nan)
    for y in range(ny):
        for x in range(nx):
            if not np.isfinite(f[y, x]):
                continue
            total = np.float32(0.0)
            count = np.float32(0.0)
            for dx in range(-n, n + 1):
                for dy in range(-n, n + 1):
                    if dx * dx + dy * dy > radius * radius:
                        continue
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < ny and 0 <= xx < nx and np.isfinite(f[yy, xx]):
                        total = np.float32(total + f[yy, xx])
                        count = np.float32(count + np.float32(1.0))
            out[y, x] = np.float64(total / count)
    return out


@pytest.mark.parametrize("accumulate_float32", [False, True])
def test_disk_mean_constant_field(accumulate_float32):
    field = np.full((7, 9), 3.5)
    result = spatial.disk_mean(field, 2.0, accumulate_float32=accumulate_float32)
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, field)


def test_disk_mean_float32_bit_for_bit():
    rng = np.random.default_rng(0)
    field = (1.0e5 + rng.normal(0.0, 500.0, (12, 15))).astype(np.float32)
    field[3, 4] = np.nan
    result = spatial.disk_mean(field, 3.0, accumulate_float32=True)
    np.testing.assert_array_equal(result, _disk_mean_f32_reference(field, 3.0))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("accumulate_float32", [False, True])
def test_disk_mean_returns_float64(dtype, accumulate_float32):
    field = np.random.default_rng(1).normal(size=(2, 8, 9)).astype(dtype)
    result = spatial.disk_mean(field, 2.0, accumulate_float32=accumulate_float32)
    assert result.dtype == np.float64
    assert result.shape == field.shape


def test_disk_mean_default_matches_clean_float64():
    field = np.random.default_rng(2).normal(size=(8, 9))
    np.testing.assert_array_equal(spatial.disk_mean(field, 2.0), _nb.disk_mean(field, 2.0))


def test_disk_mean_dataarray_float32_path():
    field = np.random.default_rng(3).normal(size=(2, 6, 8)).astype(np.float32)
    da = xr.DataArray(
        field,
        dims=["time", "lat", "lon"],
        coords={"lat": np.arange(6.0), "lon": np.arange(8.0)},
        attrs={"units": "K"},
    )
    result = spatial.disk_mean(da, 1.5, accumulate_float32=True)
    assert result.dims == da.dims
    assert result.attrs == da.attrs
    np.testing.assert_array_equal(result.values[1], spatial.disk_mean(field[1], 1.5, accumulate_float32=True))


def test_fill_missing_single_cell_float32_expectation():
    yy, xx = np.mgrid[0:7, 0:7]
    field = (0.1 * xx + 0.2 * yy).astype(float)
    field[3, 3] = np.nan
    result = spatial.fill_missing(field, 1, 0.5)
    expected = np.mean([field[2, 3], field[4, 3], field[3, 2], field[3, 4]])
    assert result[3, 3] == pytest.approx(expected, rel=1e-6)
    mask = np.ones(field.shape, bool)
    mask[3, 3] = False
    np.testing.assert_array_equal(result[mask], field.astype(np.float32).astype(np.float64)[mask])


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_fill_missing_returns_float64(dtype):
    field = np.ones((2, 6, 6), dtype=dtype)
    field[:, 3, 3] = np.nan
    result = spatial.fill_missing(field, 2, 0.5)
    assert result.dtype == np.float64
    assert np.isfinite(result).all()


def test_fill_missing_float32_close_to_clean():
    field = np.random.default_rng(4).normal(size=(9, 9))
    field[4, 4] = np.nan
    np.testing.assert_allclose(spatial.fill_missing(field, 2, 0.5), _nb.fill_missing(field, 2, 0.5), rtol=1e-5)


def test_fill_missing_invalid_arguments():
    for kwargs in ({"radius": 0}, {"radius": 1.5}, {"weight": "gauss"}):
        args = {"radius": 1, "max_missing_fraction": 0.5, **kwargs}
        with pytest.raises(ValueError):
            spatial.fill_missing(np.ones((4, 4)), **args)
