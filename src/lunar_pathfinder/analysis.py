"""End-to-end terrain analysis for lunar elevation rasters."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from lunar_pathfinder.raster import (
    LunarElevationRaster,
    RasterWindow,
    load_elevation_raster,
)
from lunar_pathfinder.routing import RouteResult, TerrainCosts, find_route
from lunar_pathfinder.terrain import calculate_slope
from lunar_pathfinder.traversability import (
    TraversabilityClass,
    TraversabilityThresholds,
    classify_traversability,
    count_traversability_classes,
)

FloatArray = NDArray[np.float64]
UInt8Array = NDArray[np.uint8]


@dataclass(frozen=True)
class TerrainAnalysisResult:
    """Terrain-analysis products derived from an elevation raster."""

    raster: LunarElevationRaster
    slope_degrees: FloatArray
    traversability: UInt8Array
    class_counts: dict[TraversabilityClass, int]

    @property
    def mean_slope_degrees(self) -> float:
        """Return the mean terrain slope."""
        return float(np.mean(self.slope_degrees))

    @property
    def max_slope_degrees(self) -> float:
        """Return the maximum terrain slope."""
        return float(np.max(self.slope_degrees))


def analyze_elevation_raster(
    path: str | Path,
    *,
    window: RasterWindow | None = None,
    thresholds: TraversabilityThresholds | None = None,
) -> TerrainAnalysisResult:
    """Load a lunar DEM and derive slope and rover traversability."""
    raster = load_elevation_raster(
        path,
        window=window,
    )

    if raster.valid_fraction < 1.0:
        invalid_cells = int(np.count_nonzero(~np.isfinite(raster.elevation_m)))
        raise ValueError(
            "elevation raster contains "
            f"{invalid_cells} non-finite elevation cells; "
            "nodata handling is required before terrain analysis"
        )

    slope = calculate_slope(
        raster.elevation_m,
        raster.cell_size_m,
    )

    traversability = classify_traversability(
        slope,
        thresholds=thresholds,
    )

    return TerrainAnalysisResult(
        raster=raster,
        slope_degrees=slope,
        traversability=traversability,
        class_counts=count_traversability_classes(traversability),
    )


@dataclass(frozen=True)
class RouteAnalysisResult:
    """Route and summary statistics across analyzed lunar terrain."""

    route: RouteResult
    straight_line_distance_m: float
    detour_ratio: float
    terrain_counts: dict[TraversabilityClass, int]


def analyze_route(
    analysis: TerrainAnalysisResult,
    start: tuple[int, int],
    destination: tuple[int, int],
    *,
    costs: TerrainCosts | None = None,
) -> RouteAnalysisResult:
    """Plan a rover route and summarize the terrain encountered."""
    route = find_route(
        analysis.traversability,
        start=start,
        destination=destination,
        cell_size_m=analysis.raster.cell_size_m,
        costs=costs,
    )

    straight_line_distance_m = (
        np.hypot(
            destination[0] - start[0],
            destination[1] - start[1],
        )
        * analysis.raster.cell_size_m
    )

    route_classes = np.asarray(
        [analysis.traversability[cell] for cell in route.path],
        dtype=np.uint8,
    )

    terrain_counts = {
        terrain_class: int(np.count_nonzero(route_classes == terrain_class.value))
        for terrain_class in TraversabilityClass
    }

    return RouteAnalysisResult(
        route=route,
        straight_line_distance_m=float(straight_line_distance_m),
        detour_ratio=route.distance_m / straight_line_distance_m,
        terrain_counts=terrain_counts,
    )


def sample_straight_line(
    traversability: np.ndarray,
    start: tuple[int, int],
    destination: tuple[int, int],
) -> tuple[tuple[int, int], ...]:
    """Return raster cells approximating a straight line between two points."""
    row0, col0 = start
    row1, col1 = destination

    steps = max(
        abs(row1 - row0),
        abs(col1 - col0),
    )

    rows = np.rint(np.linspace(row0, row1, steps + 1)).astype(int)

    cols = np.rint(np.linspace(col0, col1, steps + 1)).astype(int)

    cells = tuple(dict.fromkeys(zip(rows.tolist(), cols.tolist(), strict=True)))

    return cells


def count_route_classes(
    traversability: np.ndarray,
    path: tuple[tuple[int, int], ...],
) -> dict[TraversabilityClass, int]:
    """Count traversability classes along a path."""
    values = np.asarray(
        [traversability[cell] for cell in path],
        dtype=np.uint8,
    )

    return {
        terrain_class: int(np.count_nonzero(values == terrain_class.value))
        for terrain_class in TraversabilityClass
    }
