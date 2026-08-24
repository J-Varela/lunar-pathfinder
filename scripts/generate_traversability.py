"""Generate the v0.1.0-alpha.2 traversability artifact."""

import json
from pathlib import Path

from lunar_pathfinder.synthetic import generate_lunar_terrain
from lunar_pathfinder.terrain import calculate_slope
from lunar_pathfinder.traversability import (
    TraversabilityClass,
    classify_traversability,
    count_traversability_classes,
)
from lunar_pathfinder.visualization import (
    save_traversability_overview,
)

FIGURE_PATH = Path("outputs/figures/synthetic_traversability_overview.png")
SUMMARY_PATH = Path("outputs/missions/synthetic_traversability_summary.json")


def main() -> None:
    terrain = generate_lunar_terrain()
    slope = calculate_slope(
        terrain.elevation_m,
        terrain.cell_size_m,
    )
    classes = classify_traversability(slope)
    counts = count_traversability_classes(classes)

    total_cells = classes.size
    cell_area_km2 = terrain.cell_size_m**2 / 1_000_000.0

    class_summary = {
        terrain_class.name.lower(): {
            "class_id": terrain_class.value,
            "cells": counts[terrain_class],
            "fraction": round(
                counts[terrain_class] / total_cells,
                6,
            ),
            "area_km2": round(
                counts[terrain_class] * cell_area_km2,
                4,
            ),
        }
        for terrain_class in TraversabilityClass
    }

    summary = {
        "dataset": "deterministic synthetic lunar terrain",
        "seed": 42,
        "grid_shape": list(classes.shape),
        "cell_size_m": terrain.cell_size_m,
        "thresholds_degrees": {
            "preferred": "[0, 8)",
            "caution": "[8, 15)",
            "hazardous": "[15, 25]",
            "blocked": "(25, infinity)",
        },
        "classes": class_summary,
    }

    figure_path = save_traversability_overview(
        elevation_m=terrain.elevation_m,
        slope_degrees=slope,
        traversability=classes,
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
