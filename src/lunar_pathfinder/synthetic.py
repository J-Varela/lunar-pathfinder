"""Deterministic synthetic terrain for algorithm development."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class SyntheticTerrain:
    """A generated elevation grid and its spatial resolution."""

    elevation_m: FloatArray
    cell_size_m: float


@dataclass(frozen=True)
class Crater:
    """Parameters describing a synthetic impact crater."""

    center_x_m: float
    center_y_m: float
    radius_m: float
    depth_m: float
    rim_height_m: float


def generate_lunar_terrain(
    size: int = 160,
    cell_size_m: float = 20.0,
    seed: int = 42,
) -> SyntheticTerrain:
    """Generate a repeatable cratered lunar test surface."""
    if size < 16:
        raise ValueError("size must be at least 16")

    if not np.isfinite(cell_size_m) or cell_size_m <= 0:
        raise ValueError("cell_size_m must be positive and finite")

    rng = np.random.default_rng(seed)
    coordinates = np.arange(size, dtype=np.float64) * cell_size_m
    x, y = np.meshgrid(coordinates, coordinates)

    center_m = coordinates[-1] / 2.0

    # Broad regional relief with a gentle overall tilt.
    elevation = (
        70.0 * np.sin(x / 550.0)
        + 45.0 * np.cos(y / 700.0)
        + 0.012 * (x - center_m)
        - 0.008 * (y - center_m)
    )

    craters = (
        Crater(700.0, 850.0, 310.0, 180.0, 38.0),
        Crater(2_200.0, 700.0, 230.0, 130.0, 28.0),
        Crater(1_650.0, 2_100.0, 390.0, 220.0, 45.0),
        Crater(650.0, 2_450.0, 180.0, 95.0, 22.0),
        Crater(2_650.0, 2_450.0, 260.0, 150.0, 32.0),
    )

    for crater in craters:
        distance = np.hypot(
            x - crater.center_x_m,
            y - crater.center_y_m,
        )

        bowl = -crater.depth_m * np.exp(-((distance / (0.72 * crater.radius_m)) ** 4))
        rim = crater.rim_height_m * np.exp(
            -(((distance - crater.radius_m) / (0.16 * crater.radius_m)) ** 2)
        )
        elevation += bowl + rim

    # Low-amplitude roughness represents smaller unresolved features.
    elevation += rng.normal(loc=0.0, scale=2.0, size=(size, size))

    return SyntheticTerrain(
        elevation_m=np.asarray(elevation, dtype=np.float64),
        cell_size_m=float(cell_size_m),
    )
