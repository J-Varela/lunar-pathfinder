from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from lunar_pathfinder.analysis import (
    analyze_elevation_raster,
    analyze_route,
)
from lunar_pathfinder.traversability import TraversabilityClass

LUNAR_CRS = "+proj=stere +lat_0=-90 +lon_0=0 +a=1737400 +b=1737400 +units=m +no_defs"


def write_test_raster(
    path: Path,
    elevation: np.ndarray,
) -> None:
    with rasterio.open(
        path,
        mode="w",
        driver="GTiff",
        height=elevation.shape[0],
        width=elevation.shape[1],
        count=1,
        dtype="float32",
        crs=LUNAR_CRS,
        transform=from_origin(
            west=-100.0,
            north=100.0,
            xsize=5.0,
            ysize=5.0,
        ),
        nodata=np.nan,
    ) as dataset:
        dataset.write(
            elevation.astype(np.float32),
            1,
        )


def test_flat_raster_is_fully_preferred(
    tmp_path: Path,
) -> None:
    path = tmp_path / "flat-lunar-terrain.tif"
    elevation = np.full((10, 10), 100.0)
    write_test_raster(path, elevation)

    result = analyze_elevation_raster(path)

    np.testing.assert_allclose(
        result.slope_degrees,
        0.0,
    )
    assert result.raster.cell_size_m == 5.0
    assert result.mean_slope_degrees == 0.0
    assert result.max_slope_degrees == 0.0

    assert result.class_counts[TraversabilityClass.PREFERRED] == 100
    assert result.class_counts[TraversabilityClass.CAUTION] == 0
    assert result.class_counts[TraversabilityClass.HAZARDOUS] == 0
    assert result.class_counts[TraversabilityClass.BLOCKED] == 0


def test_raster_with_nodata_requires_explicit_handling(
    tmp_path: Path,
) -> None:
    path = tmp_path / "nodata-lunar-terrain.tif"
    elevation = np.zeros((10, 10))
    elevation[4, 5] = np.nan
    write_test_raster(path, elevation)

    with pytest.raises(
        ValueError,
        match="1 non-finite elevation cells",
    ):
        analyze_elevation_raster(path)


def test_route_analysis_reports_detour_and_classes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "terrain.tif"

    elevation = np.zeros((20, 20))
    write_test_raster(path, elevation)

    analysis = analyze_elevation_raster(path)

    result = analyze_route(
        analysis,
        start=(1, 1),
        destination=(18, 18),
    )

    assert result.route.distance_m > 0.0
    assert result.straight_line_distance_m > 0.0
    assert result.detour_ratio == pytest.approx(1.0)

    assert result.terrain_counts[TraversabilityClass.PREFERRED] == len(
        result.route.path
    )
