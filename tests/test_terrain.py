import numpy as np
import pytest

from lunar_pathfinder.terrain import calculate_slope


def test_flat_terrain_has_zero_slope() -> None:
    elevation = np.full((5, 5), 100.0)

    slope = calculate_slope(elevation, cell_size_m=10.0)

    np.testing.assert_allclose(slope, 0.0)


def test_uniform_rise_produces_expected_slope() -> None:
    elevation = np.tile(
        np.arange(5, dtype=np.float64) * 10.0,
        (5, 1),
    )

    slope = calculate_slope(elevation, cell_size_m=10.0)

    np.testing.assert_allclose(slope, 45.0)


@pytest.mark.parametrize("cell_size_m", [0.0, -1.0, np.inf, np.nan])
def test_invalid_cell_size_is_rejected(cell_size_m: float) -> None:
    elevation = np.zeros((5, 5))

    with pytest.raises(
        ValueError,
        match="cell_size_m must be positive and finite",
    ):
        calculate_slope(elevation, cell_size_m)


def test_non_2d_elevation_is_rejected() -> None:
    elevation = np.zeros(5)

    with pytest.raises(
        ValueError,
        match="elevation must be a two-dimensional grid",
    ):
        calculate_slope(elevation, cell_size_m=10.0)


def test_nonfinite_elevation_is_rejected() -> None:
    elevation = np.zeros((5, 5))
    elevation[2, 2] = np.nan

    with pytest.raises(
        ValueError,
        match="elevation must contain only finite values",
    ):
        calculate_slope(elevation, cell_size_m=10.0)
