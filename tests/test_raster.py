from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from lunar_pathfinder.raster import (
    RasterWindow,
    load_elevation_raster,
)

LUNAR_CRS = "+proj=stere +lat_0=-90 +lon_0=0 +a=1737400 +b=1737400 +units=m +no_defs"


def write_test_raster(
    path: Path,
    elevation: np.ndarray,
    *,
    crs: str | None = LUNAR_CRS,
    nodata: float | None = np.nan,
) -> None:
    with rasterio.open(
        path,
        mode="w",
        driver="GTiff",
        height=elevation.shape[0],
        width=elevation.shape[1],
        count=1,
        dtype="float32",
        crs=crs,
        transform=from_origin(
            west=-100.0,
            north=100.0,
            xsize=5.0,
            ysize=5.0,
        ),
        nodata=nodata,
    ) as dataset:
        dataset.write(
            elevation.astype(np.float32),
            1,
        )


def test_full_elevation_raster_is_loaded(
    tmp_path: Path,
) -> None:
    path = tmp_path / "elevation.tif"
    elevation = np.arange(20).reshape(4, 5)
    write_test_raster(path, elevation)

    result = load_elevation_raster(path)

    np.testing.assert_allclose(
        result.elevation_m,
        elevation,
    )
    assert result.elevation_m.dtype == np.float64
    assert result.elevation_m.shape == (4, 5)
    assert result.cell_size_m == 5.0
    assert result.bounds.left == -100.0
    assert result.bounds.top == 100.0
    assert result.valid_fraction == 1.0
    assert result.crs_wkt


def test_raster_window_updates_values_and_metadata(
    tmp_path: Path,
) -> None:
    path = tmp_path / "elevation.tif"
    elevation = np.arange(36).reshape(6, 6)
    write_test_raster(path, elevation)

    result = load_elevation_raster(
        path,
        RasterWindow(
            row_offset=1,
            column_offset=2,
            height=3,
            width=2,
        ),
    )

    np.testing.assert_allclose(
        result.elevation_m,
        elevation[1:4, 2:4],
    )
    assert result.elevation_m.shape == (3, 2)
    assert result.bounds.left == -90.0
    assert result.bounds.top == 95.0


def test_nan_nodata_is_preserved(
    tmp_path: Path,
) -> None:
    path = tmp_path / "elevation.tif"
    elevation = np.zeros((4, 4))
    elevation[1, 2] = np.nan
    write_test_raster(path, elevation)

    result = load_elevation_raster(path)

    assert np.isnan(result.elevation_m[1, 2])
    assert result.valid_fraction == pytest.approx(15 / 16)


def test_missing_coordinate_system_is_rejected(
    tmp_path: Path,
) -> None:
    path = tmp_path / "elevation.tif"
    write_test_raster(
        path,
        np.zeros((4, 4)),
        crs=None,
    )

    with pytest.raises(
        ValueError,
        match="must define a coordinate system",
    ):
        load_elevation_raster(path)


def test_window_outside_raster_is_rejected(
    tmp_path: Path,
) -> None:
    path = tmp_path / "elevation.tif"
    write_test_raster(path, np.zeros((4, 4)))

    window = RasterWindow(
        row_offset=2,
        column_offset=2,
        height=4,
        width=4,
    )

    with pytest.raises(
        ValueError,
        match="extends beyond dataset bounds",
    ):
        load_elevation_raster(path, window)


@pytest.mark.parametrize(
    ("row_offset", "column_offset", "height", "width"),
    [
        (-1, 0, 4, 4),
        (0, -1, 4, 4),
        (0, 0, 0, 4),
        (0, 0, 4, 0),
    ],
)
def test_invalid_window_is_rejected(
    row_offset: int,
    column_offset: int,
    height: int,
    width: int,
) -> None:
    with pytest.raises(ValueError):
        RasterWindow(
            row_offset=row_offset,
            column_offset=column_offset,
            height=height,
            width=width,
        )
