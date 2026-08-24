"""Generate the v0.1.0-alpha.1 terrain-analysis artifact."""

import json
from pathlib import Path

import numpy as np

from lunar_pathfinder.synthetic import generate_lunar_terrain
from lunar_pathfinder.terrain import calculate_slope
from lunar_pathfinder.visualization import save_terrain_overview

FIGURE_PATH = Path("outputs/figures/synthetic_terrain_overview.png")
SUMMARY_PATH = Path("outputs/missions/synthetic_terrain_summary.json")


def main() -> None:
    terrain = generate_lunar_terrain()
    slope = calculate_slope(
        terrain.elevation_m,
        terrain.cell_size_m,
    )

    figure_path = save_terrain_overview(
        elevation_m=terrain.elevation_m,
        slope_degrees=slope,
        cell_size_m=terrain.cell_size_m,
        output_path=FIGURE_PATH,
    )

    summary = {
        "dataset": "deterministic synthetic lunar terrain",
        "seed": 42,
        "shape": list(terrain.elevation_m.shape),
        "cell_size_m": terrain.cell_size_m,
        "area_km2": (terrain.elevation_m.size * terrain.cell_size_m**2 / 1_000_000.0),
        "elevation_min_m": float(np.min(terrain.elevation_m)),
        "elevation_max_m": float(np.max(terrain.elevation_m)),
        "elevation_relief_m": float(np.ptp(terrain.elevation_m)),
        "slope_min_degrees": float(np.min(slope)),
        "slope_max_degrees": float(np.max(slope)),
        "slope_mean_degrees": float(np.mean(slope)),
        "cells_above_15_degrees": int(np.count_nonzero(slope > 15.0)),
        "fraction_above_15_degrees": float(np.mean(slope > 15.0)),
    }

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
