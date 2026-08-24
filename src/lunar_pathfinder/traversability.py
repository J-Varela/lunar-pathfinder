"""Rover traversability classification for lunar terrain."""

from dataclasses import dataclass
from enum import IntEnum

import numpy as np
from numpy.typing import NDArray

UInt8Array = NDArray[np.uint8]


class TraversabilityClass(IntEnum):
    """Operational terrain classes used by the route planner."""

    PREFERRED = 0
    CAUTION = 1
    HAZARDOUS = 2
    BLOCKED = 3


@dataclass(frozen=True)
class TraversabilityThresholds:
    """Slope thresholds defining rover terrain classes."""

    preferred_max_degrees: float = 8.0
    caution_max_degrees: float = 15.0
    hazardous_max_degrees: float = 25.0

    def __post_init__(self) -> None:
        values = (
            self.preferred_max_degrees,
            self.caution_max_degrees,
            self.hazardous_max_degrees,
        )

        if not all(np.isfinite(value) for value in values):
            raise ValueError("traversability thresholds must be finite")

        if not (
            0.0
            < self.preferred_max_degrees
            < self.caution_max_degrees
            < self.hazardous_max_degrees
        ):
            raise ValueError("thresholds must be positive and strictly increasing")


def classify_traversability(
    slope_degrees: NDArray[np.floating],
    thresholds: TraversabilityThresholds | None = None,
) -> UInt8Array:
    """Classify every terrain cell from its slope.

    The default intervals are:

    - Preferred: 0 <= slope < 8 degrees
    - Caution: 8 <= slope < 15 degrees
    - Hazardous: 15 <= slope <= 25 degrees
    - Blocked: slope > 25 degrees
    """
    slope = np.asarray(slope_degrees, dtype=np.float64)
    limits = thresholds or TraversabilityThresholds()

    if slope.ndim != 2:
        raise ValueError("slope must be a two-dimensional grid")

    if not np.all(np.isfinite(slope)):
        raise ValueError("slope must contain only finite values")

    if np.any(slope < 0.0):
        raise ValueError("slope values cannot be negative")

    classes = np.full(
        slope.shape,
        TraversabilityClass.BLOCKED,
        dtype=np.uint8,
    )

    classes[slope < limits.preferred_max_degrees] = TraversabilityClass.PREFERRED

    caution = (slope >= limits.preferred_max_degrees) & (
        slope < limits.caution_max_degrees
    )
    classes[caution] = TraversabilityClass.CAUTION

    hazardous = (slope >= limits.caution_max_degrees) & (
        slope <= limits.hazardous_max_degrees
    )
    classes[hazardous] = TraversabilityClass.HAZARDOUS

    return classes


def count_traversability_classes(
    classes: NDArray[np.integer],
) -> dict[TraversabilityClass, int]:
    """Count terrain cells in each traversability class."""
    class_grid = np.asarray(classes)

    if class_grid.ndim != 2:
        raise ValueError("traversability classes must be a two-dimensional grid")

    valid_values = np.array(
        [terrain_class.value for terrain_class in TraversabilityClass]
    )

    if not np.all(np.isin(class_grid, valid_values)):
        raise ValueError("traversability grid contains unknown classes")

    return {
        terrain_class: int(np.count_nonzero(class_grid == terrain_class.value))
        for terrain_class in TraversabilityClass
    }
