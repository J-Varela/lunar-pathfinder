from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from lunar_pathfinder.analysis import (
    analyze_elevation_raster,
    analyze_route,
)
from lunar_pathfinder.visualization import save_route_visualization

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
    ) as dataset:
        dataset.write(
            elevation.astype(np.float32),
            1,
        )


def test_route_visualization_is_written(
    tmp_path: Path,
) -> None:
    raster_path = tmp_path / "terrain.tif"
    image_path = tmp_path / "route.png"

    write_test_raster(
        raster_path,
        np.zeros((20, 20)),
    )

    analysis = analyze_elevation_raster(raster_path)

    route = analyze_route(
        analysis,
        start=(1, 1),
        destination=(18, 18),
    )

    result = save_route_visualization(
        analysis,
        route,
        image_path,
    )

    assert result == image_path
    assert image_path.is_file()
    assert image_path.stat().st_size > 0
