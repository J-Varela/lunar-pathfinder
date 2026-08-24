import numpy as np
import pytest

from lunar_pathfinder.routing import (
    RouteResult,
    TerrainCosts,
    find_route,
)
from lunar_pathfinder.traversability import TraversabilityClass

PREFERRED = TraversabilityClass.PREFERRED
HAZARDOUS = TraversabilityClass.HAZARDOUS
BLOCKED = TraversabilityClass.BLOCKED


def test_straight_route_across_preferred_terrain() -> None:
    grid = np.full((3, 5), PREFERRED, dtype=np.uint8)

    route = find_route(
        grid,
        start=(1, 0),
        destination=(1, 4),
        cell_size_m=10.0,
    )

    assert isinstance(route, RouteResult)
    assert route.path == (
        (1, 0),
        (1, 1),
        (1, 2),
        (1, 3),
        (1, 4),
    )
    assert route.distance_m == pytest.approx(40.0)
    assert route.total_cost == pytest.approx(40.0)


def test_diagonal_route_uses_physical_distance() -> None:
    grid = np.full((3, 3), PREFERRED, dtype=np.uint8)

    route = find_route(
        grid,
        start=(0, 0),
        destination=(2, 2),
        cell_size_m=10.0,
    )

    assert route.path == ((0, 0), (1, 1), (2, 2))
    assert route.distance_m == pytest.approx(20.0 * np.sqrt(2.0))
    assert route.total_cost == pytest.approx(20.0 * np.sqrt(2.0))


def test_route_passes_through_barrier_gap() -> None:
    grid = np.full((7, 7), PREFERRED, dtype=np.uint8)
    grid[:, 3] = BLOCKED
    grid[5, 3] = PREFERRED

    route = find_route(
        grid,
        start=(1, 1),
        destination=(1, 5),
        cell_size_m=10.0,
    )

    assert (5, 3) in route.path
    assert all(grid[cell] != BLOCKED for cell in route.path)


def test_safer_route_is_preferred_over_hazardous_shortcut() -> None:
    grid = np.full((3, 5), PREFERRED, dtype=np.uint8)
    grid[1, 1:4] = HAZARDOUS

    route = find_route(
        grid,
        start=(1, 0),
        destination=(1, 4),
        cell_size_m=10.0,
    )

    assert all(grid[cell] != HAZARDOUS for cell in route.path)
    assert route.distance_m > 40.0
    assert route.total_cost < 250.0


def test_no_route_is_reported() -> None:
    grid = np.full((3, 3), PREFERRED, dtype=np.uint8)
    grid[1, :] = BLOCKED

    with pytest.raises(
        ValueError,
        match="no traversable route exists",
    ):
        find_route(
            grid,
            start=(0, 0),
            destination=(2, 2),
            cell_size_m=10.0,
        )


@pytest.mark.parametrize(
    ("start", "destination", "message"),
    [
        ((0, 0), (2, 2), "start cell is blocked"),
        ((2, 2), (0, 0), "destination cell is blocked"),
    ],
)
def test_blocked_endpoints_are_rejected(
    start: tuple[int, int],
    destination: tuple[int, int],
    message: str,
) -> None:
    grid = np.full((3, 3), PREFERRED, dtype=np.uint8)
    grid[0, 0] = BLOCKED

    with pytest.raises(ValueError, match=message):
        find_route(
            grid,
            start=start,
            destination=destination,
            cell_size_m=10.0,
        )


@pytest.mark.parametrize(
    ("start", "destination", "message"),
    [
        ((-1, 0), (1, 1), "start cell is outside"),
        ((0, 0), (3, 1), "destination cell is outside"),
    ],
)
def test_out_of_bounds_endpoints_are_rejected(
    start: tuple[int, int],
    destination: tuple[int, int],
    message: str,
) -> None:
    grid = np.full((3, 3), PREFERRED, dtype=np.uint8)

    with pytest.raises(ValueError, match=message):
        find_route(
            grid,
            start=start,
            destination=destination,
            cell_size_m=10.0,
        )


def test_stationary_route_has_zero_distance_and_cost() -> None:
    grid = np.full((3, 3), PREFERRED, dtype=np.uint8)

    route = find_route(
        grid,
        start=(1, 1),
        destination=(1, 1),
        cell_size_m=10.0,
    )

    assert route.path == ((1, 1),)
    assert route.distance_m == 0.0
    assert route.total_cost == 0.0


@pytest.mark.parametrize(
    ("preferred", "caution", "hazardous"),
    [
        (0.5, 2.0, 8.0),
        (2.0, 1.0, 8.0),
        (1.0, 9.0, 8.0),
        (1.0, 2.0, np.inf),
        (1.0, np.nan, 8.0),
    ],
)
def test_invalid_terrain_costs_are_rejected(
    preferred: float,
    caution: float,
    hazardous: float,
) -> None:
    with pytest.raises(ValueError):
        TerrainCosts(
            preferred=preferred,
            caution=caution,
            hazardous=hazardous,
        )
