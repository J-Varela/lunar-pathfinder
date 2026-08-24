"""Generate the v0.1.0-alpha.3 rover mission route."""

import json
from math import hypot
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from lunar_pathfinder.routing import find_route
from lunar_pathfinder.synthetic import generate_lunar_terrain
from lunar_pathfinder.terrain import calculate_slope
from lunar_pathfinder.traversability import (
    TraversabilityClass,
    classify_traversability,
)
from lunar_pathfinder.visualization import save_route_overview

FIGURE_PATH = Path("outputs/figures/synthetic_rover_route.png")
SUMMARY_PATH = Path("outputs/missions/synthetic_rover_route.json")

GridCell = tuple[int, int]


def nearest_preferred_cell(
    classes: NDArray[np.integer],
    target: GridCell,
) -> GridCell:
    """Select the preferred cell nearest a requested target."""
    preferred_cells = np.argwhere(classes == TraversabilityClass.PREFERRED)

    if preferred_cells.size == 0:
        raise ValueError("terrain contains no preferred cells")

    offsets = preferred_cells - np.asarray(target)
    squared_distances = np.sum(offsets**2, axis=1)
    nearest = preferred_cells[np.argmin(squared_distances)]

    return int(nearest[0]), int(nearest[1])


def main() -> None:
    terrain = generate_lunar_terrain()
    slope = calculate_slope(
        terrain.elevation_m,
        terrain.cell_size_m,
    )
    classes = classify_traversability(slope)

    requested_start = (10, 10)
    requested_destination = (145, 145)

    start = nearest_preferred_cell(classes, requested_start)
    destination = nearest_preferred_cell(
        classes,
        requested_destination,
    )

    route = find_route(
        traversability=classes,
        start=start,
        destination=destination,
        cell_size_m=terrain.cell_size_m,
    )

    direct_distance_m = (
        hypot(
            destination[0] - start[0],
            destination[1] - start[1],
        )
        * terrain.cell_size_m
    )

    route_class_counts = {
        terrain_class.name.lower(): int(
            sum(classes[cell] == terrain_class.value for cell in route.path)
        )
        for terrain_class in TraversabilityClass
    }

    summary = {
        "dataset": "deterministic synthetic lunar terrain",
        "seed": 42,
        "requested_start_cell": list(requested_start),
        "requested_destination_cell": list(requested_destination),
        "selected_start_cell": list(start),
        "selected_destination_cell": list(destination),
        "route_cells": len(route.path),
        "route_distance_m": round(route.distance_m, 2),
        "direct_distance_m": round(direct_distance_m, 2),
        "detour_factor": round(
            route.distance_m / direct_distance_m,
            4,
        ),
        "total_weighted_cost": round(route.total_cost, 2),
        "terrain_costs": {
            "preferred": 1.0,
            "caution": 2.0,
            "hazardous": 8.0,
            "blocked": "impassable",
        },
        "route_class_counts": route_class_counts,
        "blocked_cells_entered": route_class_counts["blocked"],
    }

    figure_path = save_route_overview(
        elevation_m=terrain.elevation_m,
        traversability=classes,
        route=route.path,
        cell_size_m=terrain.cell_size_m,
        output_path=FIGURE_PATH,
    )

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Figure: {figure_path}")
    print(f"Summary: {SUMMARY_PATH}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
