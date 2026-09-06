from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from lunar_pathfinder.raster import RasterWindow
from lunar_pathfinder.validation import validate_slope_against_reference

LUNAR_CRS = "+proj=stere +lat_0=-90 +lon_0=0 +a=1737400 +b=1737400 +units=m +no_defs"


def write_test_raster(
    path: Path,
    values: np.ndarray,
) -> None:
    with rasterio.open(
        path,
        mode="w",
        driver="GTiff",
        height=values.shape[0],
        width=values.shape[1],
        count=1,
        dtype="float32",
        crs=LUNAR_CRS,
        transform=from_origin(
            west=-100.0,
            north=100.0,
            xsize=5.0,
            ysize=5.0,
        ),
    ) as dataset:
        dataset.write(
            values.astype(np.float32),
            1,
        )


def test_identical_flat_slope_has_zero_error(
    tmp_path: Path,
) -> None:
    elevation_path = tmp_path / "elevation.tif"
    slope_path = tmp_path / "slope.tif"

    write_test_raster(
        elevation_path,
        np.zeros((20, 20)),
    )

    write_test_raster(
        slope_path,
        np.zeros((20, 20)),
    )

    result = validate_slope_against_reference(
        elevation_path,
        slope_path,
    )

    assert result.derived_mean_degrees == 0.0
    assert result.reference_mean_degrees == 0.0
    assert result.mae_degrees == 0.0
    assert result.rmse_degrees == 0.0
    assert result.median_absolute_error_degrees == 0.0
    assert result.percentile_95_absolute_error_degrees == 0.0


def test_validation_supports_matching_window(
    tmp_path: Path,
) -> None:
    elevation_path = tmp_path / "elevation.tif"
    slope_path = tmp_path / "slope.tif"

    write_test_raster(
        elevation_path,
        np.zeros((30, 30)),
    )

    write_test_raster(
        slope_path,
        np.zeros((30, 30)),
    )

    result = validate_slope_against_reference(
        elevation_path,
        slope_path,
        window=RasterWindow(
            row_offset=5,
            column_offset=5,
            height=10,
            width=10,
        ),
    )

    assert result.mae_degrees == 0.0
