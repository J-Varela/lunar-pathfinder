"""GeoTIFF ingestion for lunar elevation products."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from numpy.typing import NDArray
from rasterio.coords import BoundingBox
from rasterio.transform import array_bounds
from rasterio.windows import Window

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class RasterWindow:
    """Integer raster window expressed in rows and columns."""

    row_offset: int
    column_offset: int
    height: int
    width: int

    def __post_init__(self) -> None:
        values = (
            self.row_offset,
            self.column_offset,
            self.height,
            self.width,
        )

        if not all(
            isinstance(value, int) and not isinstance(value, bool) for value in values
        ):
            raise ValueError("raster window values must be integers")

        if self.row_offset < 0 or self.column_offset < 0:
            raise ValueError("raster window offsets cannot be negative")

        if self.height <= 0 or self.width <= 0:
            raise ValueError("raster window dimensions must be positive")


@dataclass(frozen=True)
class LunarElevationRaster:
    """Elevation values and spatial metadata from a lunar raster."""

    elevation_m: FloatArray
    cell_size_m: float
    transform: Affine
    bounds: BoundingBox
    crs_wkt: str
    nodata: float | None

    @property
    def valid_fraction(self) -> float:
        """Return the fraction of finite elevation cells."""
        return float(np.mean(np.isfinite(self.elevation_m)))


def load_elevation_raster(
    path: str | Path,
    window: RasterWindow | None = None,
) -> LunarElevationRaster:
    """Load a complete lunar elevation raster or integer crop."""
    source = Path(path)

    if not source.is_file():
        raise FileNotFoundError(f"elevation raster not found: {source}")

    with rasterio.open(source) as dataset:
        if dataset.count != 1:
            raise ValueError("elevation raster must contain exactly one band")

        if dataset.crs is None:
            raise ValueError("elevation raster must define a coordinate system")

        if not np.isclose(dataset.transform.b, 0.0) or not np.isclose(
            dataset.transform.d,
            0.0,
        ):
            raise ValueError("rotated rasters are not supported")

        resolution_x, resolution_y = dataset.res

        if not np.isclose(resolution_x, resolution_y):
            raise ValueError("raster cells must be square")

        rasterio_window = None

        if window is not None:
            if (
                window.row_offset + window.height > dataset.height
                or window.column_offset + window.width > dataset.width
            ):
                raise ValueError("raster window extends beyond dataset bounds")

            rasterio_window = Window(
                col_off=window.column_offset,
                row_off=window.row_offset,
                width=window.width,
                height=window.height,
            )

        elevation = dataset.read(
            1,
            window=rasterio_window,
            out_dtype=np.float64,
        )
        transform = (
            dataset.transform
            if rasterio_window is None
            else dataset.window_transform(rasterio_window)
        )
        left, bottom, right, top = array_bounds(
            elevation.shape[0],
            elevation.shape[1],
            transform,
        )

        return LunarElevationRaster(
            elevation_m=elevation,
            cell_size_m=float(resolution_x),
            transform=transform,
            bounds=BoundingBox(
                left=left,
                bottom=bottom,
                right=right,
                top=top,
            ),
            crs_wkt=dataset.crs.to_wkt(),
            nodata=dataset.nodata,
        )
