import numpy as np
import pytest

from lunar_pathfinder.synthetic import generate_lunar_terrain


def test_generated_terrain_has_requested_shape() -> None:
    terrain = generate_lunar_terrain(size=64, cell_size_m=20.0)

    assert terrain.elevation_m.shape == (64, 64)
    assert terrain.cell_size_m == 20.0
    assert np.all(np.isfinite(terrain.elevation_m))


def test_generated_terrain_is_reproducible() -> None:
    first = generate_lunar_terrain(seed=17)
    second = generate_lunar_terrain(seed=17)

    np.testing.assert_array_equal(
        first.elevation_m,
        second.elevation_m,
    )


def test_different_seeds_produce_different_roughness() -> None:
    first = generate_lunar_terrain(seed=1)
    second = generate_lunar_terrain(seed=2)

    assert not np.array_equal(
        first.elevation_m,
        second.elevation_m,
    )


def test_small_grid_is_rejected() -> None:
    with pytest.raises(ValueError, match="size must be at least 16"):
        generate_lunar_terrain(size=15)


@pytest.mark.parametrize("cell_size_m", [0.0, -20.0, np.inf, np.nan])
def test_invalid_cell_size_is_rejected(cell_size_m: float) -> None:
    with pytest.raises(
        ValueError,
        match="cell_size_m must be positive and finite",
    ):
        generate_lunar_terrain(cell_size_m=cell_size_m)
