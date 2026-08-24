"""Terrain-analysis operations for lunar elevation grids."""

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def calculate_slope(
    elevation: NDArray[np.floating],
    cell_size_m: float,
) -> FloatArray:
    """Calculate terrain slope in degrees from a 2D elevation grid.

    Parameters
    ----------
    elevation:
        Two-dimensional grid of surface elevations in meters.
    cell_size_m:
        Horizontal width and height of each grid cell in meters.

    Returns
    -------
    numpy.ndarray
        Terrain slope for every cell, measured in degrees.
    """
    elevation_array = np.asarray(elevation, dtype=np.float64)

    if elevation_array.ndim != 2:
        raise ValueError("elevation must be a two-dimensional grid")

    if min(elevation_array.shape) < 2:
        raise ValueError("elevation must contain at least two rows and columns")

    if not np.all(np.isfinite(elevation_array)):
        raise ValueError("elevation must contain only finite values")

    if not np.isfinite(cell_size_m) or cell_size_m <= 0:
        raise ValueError("cell_size_m must be positive and finite")

    gradient_y, gradient_x = np.gradient(
        elevation_array,
        cell_size_m,
    )
    rise_over_run = np.hypot(gradient_x, gradient_y)

    return np.degrees(np.arctan(rise_over_run))
