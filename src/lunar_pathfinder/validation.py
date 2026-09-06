"""Validation utilities for lunar terrain products."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lunar_pathfinder.analysis import analyze_elevation_raster
from lunar_pathfinder.raster import RasterWindow, load_elevation_raster


@dataclass(frozen=True)
class SlopeValidationResult:
    """Summary statistics comparing derived and reference slope."""

    derived_mean_degrees: float
    reference_mean_degrees: float
    mae_degrees: float
    rmse_degrees: float
    median_absolute_error_degrees: float
    percentile_95_absolute_error_degrees: float
    correlation: float


def _correlation(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    """Return Pearson correlation, or NaN for constant inputs."""
    first_values = first.ravel()
    second_values = second.ravel()

    if np.std(first_values) == 0.0 or np.std(second_values) == 0.0:
        return float("nan")

    return float(
        np.corrcoef(
            first_values,
            second_values,
        )[0, 1]
    )


def validate_slope_against_reference(
    elevation_path: str | Path,
    reference_slope_path: str | Path,
    *,
    window: RasterWindow | None = None,
) -> SlopeValidationResult:
    """Compare derived DEM slope against a reference slope raster."""
    analysis = analyze_elevation_raster(
        elevation_path,
        window=window,
    )

    reference = load_elevation_raster(
        reference_slope_path,
        window=window,
    )

    derived = analysis.slope_degrees
    reference_slope = reference.elevation_m

    if derived.shape != reference_slope.shape:
        raise ValueError(
            "derived and reference slope rasters must have matching shapes"
        )

    if not np.all(np.isfinite(reference_slope)):
        raise ValueError("reference slope raster contains non-finite values")

    error = derived - reference_slope
    absolute_error = np.abs(error)

    return SlopeValidationResult(
        derived_mean_degrees=float(np.mean(derived)),
        reference_mean_degrees=float(np.mean(reference_slope)),
        mae_degrees=float(np.mean(absolute_error)),
        rmse_degrees=float(np.sqrt(np.mean(error**2))),
        median_absolute_error_degrees=float(np.median(absolute_error)),
        percentile_95_absolute_error_degrees=float(np.percentile(absolute_error, 95)),
        correlation=_correlation(
            derived,
            reference_slope,
        ),
    )
