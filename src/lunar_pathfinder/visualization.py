"""Visualization functions for lunar terrain products."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from numpy.typing import NDArray


def save_terrain_overview(
    elevation_m: NDArray[np.floating],
    slope_degrees: NDArray[np.floating],
    cell_size_m: float,
    output_path: str | Path,
) -> Path:
    """Save a presentation-ready elevation and slope overview."""
    elevation = np.asarray(elevation_m, dtype=np.float64)
    slope = np.asarray(slope_degrees, dtype=np.float64)

    if elevation.shape != slope.shape:
        raise ValueError("elevation and slope must have matching shapes")

    if elevation.ndim != 2:
        raise ValueError("terrain products must be two-dimensional")

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    height_km = elevation.shape[0] * cell_size_m / 1_000.0
    width_km = elevation.shape[1] * cell_size_m / 1_000.0
    extent = (0.0, width_km, 0.0, height_km)

    figure, axes = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(16, 5.5),
        constrained_layout=True,
    )
    figure.patch.set_facecolor("#080b12")
    figure.suptitle(
        "Lunar Pathfinder — Synthetic South-Polar Test Site",
        color="white",
        fontsize=16,
        fontweight="bold",
    )

    elevation_image = axes[0].imshow(
        elevation,
        cmap="gray",
        origin="lower",
        extent=extent,
    )
    axes[0].set_title("Elevation", color="white")
    axes[0].set_xlabel("Easting (km)")
    axes[0].set_ylabel("Northing (km)")
    elevation_bar = figure.colorbar(
        elevation_image,
        ax=axes[0],
        shrink=0.82,
    )
    elevation_bar.set_label("Elevation (m)")

    slope_image = axes[1].imshow(
        slope,
        cmap="magma",
        origin="lower",
        extent=extent,
        vmin=0.0,
        vmax=float(np.percentile(slope, 98)),
    )
    axes[1].set_title("Terrain Slope", color="white")
    axes[1].set_xlabel("Easting (km)")
    axes[1].set_ylabel("Northing (km)")
    slope_bar = figure.colorbar(
        slope_image,
        ax=axes[1],
        shrink=0.82,
    )
    slope_bar.set_label("Slope (degrees)")

    axes[2].hist(
        slope.ravel(),
        bins=40,
        color="#e8c547",
        edgecolor="#080b12",
    )
    axes[2].axvline(
        15.0,
        color="#ff4d6d",
        linestyle="--",
        linewidth=2,
        label="Example rover limit: 15°",
    )
    axes[2].set_title("Slope Distribution", color="white")
    axes[2].set_xlabel("Slope (degrees)")
    axes[2].set_ylabel("Grid cells")
    axes[2].legend()

    for axis in axes:
        axis.set_facecolor("#111827")
        axis.tick_params(colors="white")

        for spine in axis.spines.values():
            spine.set_color("#667085")

        axis.xaxis.label.set_color("white")
        axis.yaxis.label.set_color("white")

    for colorbar in (elevation_bar, slope_bar):
        colorbar.ax.tick_params(colors="white")
        colorbar.ax.yaxis.label.set_color("white")

    figure.savefig(
        destination,
        dpi=180,
        facecolor=figure.get_facecolor(),
    )
    plt.close(figure)

    return destination


def save_traversability_overview(
    elevation_m: NDArray[np.floating],
    slope_degrees: NDArray[np.floating],
    traversability: NDArray[np.integer],
    cell_size_m: float,
    output_path: str | Path,
) -> Path:
    """Save elevation, slope, and rover traversability maps."""
    elevation = np.asarray(elevation_m, dtype=np.float64)
    slope = np.asarray(slope_degrees, dtype=np.float64)
    classes = np.asarray(traversability, dtype=np.uint8)

    if not (elevation.ndim == 2 and elevation.shape == slope.shape == classes.shape):
        raise ValueError("terrain products must be matching 2D grids")

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    height_km = elevation.shape[0] * cell_size_m / 1_000.0
    width_km = elevation.shape[1] * cell_size_m / 1_000.0
    extent = (0.0, width_km, 0.0, height_km)

    class_colors = [
        "#3ddc97",
        "#f4d35e",
        "#f28f3b",
        "#d7263d",
    ]
    class_labels = [
        "Preferred (<8°)",
        "Caution (8–15°)",
        "Hazardous (15–25°)",
        "Blocked (>25°)",
    ]

    class_cmap = ListedColormap(class_colors)
    class_norm = BoundaryNorm(
        boundaries=[-0.5, 0.5, 1.5, 2.5, 3.5],
        ncolors=class_cmap.N,
    )

    figure, axes = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(16, 5.6),
        constrained_layout=True,
    )
    figure.patch.set_facecolor("#080b12")
    figure.suptitle(
        "Lunar Pathfinder — Rover Traversability Analysis",
        color="white",
        fontsize=16,
        fontweight="bold",
    )

    elevation_image = axes[0].imshow(
        elevation,
        cmap="gray",
        origin="lower",
        extent=extent,
    )
    axes[0].set_title("Synthetic Elevation", color="white")
    elevation_bar = figure.colorbar(
        elevation_image,
        ax=axes[0],
        shrink=0.82,
    )
    elevation_bar.set_label("Elevation (m)")

    slope_image = axes[1].imshow(
        slope,
        cmap="magma",
        origin="lower",
        extent=extent,
        vmin=0.0,
        vmax=float(np.percentile(slope, 98)),
    )
    axes[1].set_title("Calculated Slope", color="white")
    slope_bar = figure.colorbar(
        slope_image,
        ax=axes[1],
        shrink=0.82,
    )
    slope_bar.set_label("Slope (degrees)")

    class_image = axes[2].imshow(
        classes,
        cmap=class_cmap,
        norm=class_norm,
        origin="lower",
        extent=extent,
        interpolation="nearest",
    )
    axes[2].set_title("Rover Traversability", color="white")
    class_bar = figure.colorbar(
        class_image,
        ax=axes[2],
        shrink=0.82,
        ticks=[0, 1, 2, 3],
    )
    class_bar.ax.set_yticklabels(class_labels)

    for axis in axes:
        axis.set_facecolor("#111827")
        axis.set_xlabel("Easting (km)")
        axis.set_ylabel("Northing (km)")
        axis.tick_params(colors="white")

        for spine in axis.spines.values():
            spine.set_color("#667085")

        axis.xaxis.label.set_color("white")
        axis.yaxis.label.set_color("white")

    for colorbar in (elevation_bar, slope_bar, class_bar):
        colorbar.ax.tick_params(colors="white")
        colorbar.ax.yaxis.label.set_color("white")

    figure.savefig(
        destination,
        dpi=180,
        facecolor=figure.get_facecolor(),
    )
    plt.close(figure)

    return destination
