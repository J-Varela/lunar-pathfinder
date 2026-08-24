"""A* route planning across classified lunar terrain."""

from dataclasses import dataclass
from heapq import heappop, heappush
from math import hypot, sqrt

import numpy as np
from numpy.typing import NDArray

from lunar_pathfinder.traversability import TraversabilityClass

GridCell = tuple[int, int]


@dataclass(frozen=True)
class TerrainCosts:
    """Movement-cost multipliers for traversable terrain."""

    preferred: float = 1.0
    caution: float = 2.0
    hazardous: float = 8.0

    def __post_init__(self) -> None:
        values = (self.preferred, self.caution, self.hazardous)

        if not all(np.isfinite(value) for value in values):
            raise ValueError("terrain costs must be finite")

        if not (1.0 <= self.preferred <= self.caution <= self.hazardous):
            raise ValueError("terrain costs must be at least 1 and nondecreasing")


@dataclass(frozen=True)
class RouteResult:
    """Result returned by the rover route planner."""

    path: tuple[GridCell, ...]
    total_cost: float
    distance_m: float


def find_route(
    traversability: NDArray[np.integer],
    start: GridCell,
    destination: GridCell,
    cell_size_m: float,
    costs: TerrainCosts | None = None,
) -> RouteResult:
    """Find a minimum-cost route using eight-direction A* search."""
    grid = np.asarray(traversability)
    terrain_costs = costs or TerrainCosts()

    _validate_inputs(
        grid,
        start,
        destination,
        cell_size_m,
    )

    if start == destination:
        return RouteResult(
            path=(start,),
            total_cost=0.0,
            distance_m=0.0,
        )

    multipliers = {
        TraversabilityClass.PREFERRED.value: terrain_costs.preferred,
        TraversabilityClass.CAUTION.value: terrain_costs.caution,
        TraversabilityClass.HAZARDOUS.value: terrain_costs.hazardous,
    }

    distances = np.full(grid.shape, np.inf, dtype=np.float64)
    distances[start] = 0.0

    previous: dict[GridCell, GridCell] = {}
    queue: list[tuple[float, float, GridCell]] = []

    heappush(
        queue,
        (
            _heuristic(start, destination, cell_size_m),
            0.0,
            start,
        ),
    )

    while queue:
        _, current_cost, current = heappop(queue)

        if current_cost > distances[current]:
            continue

        if current == destination:
            path = _reconstruct_path(previous, destination)
            return RouteResult(
                path=path,
                total_cost=current_cost,
                distance_m=_path_distance(path, cell_size_m),
            )

        for neighbor, step_distance in _neighbors(
            grid,
            current,
            cell_size_m,
        ):
            terrain_class = int(grid[neighbor])
            movement_cost = step_distance * multipliers[terrain_class]
            candidate_cost = current_cost + movement_cost

            if candidate_cost >= distances[neighbor]:
                continue

            distances[neighbor] = candidate_cost
            previous[neighbor] = current

            priority = candidate_cost + _heuristic(
                neighbor,
                destination,
                cell_size_m,
            )
            heappush(
                queue,
                (priority, candidate_cost, neighbor),
            )

    raise ValueError("no traversable route exists")


def _neighbors(
    grid: NDArray[np.integer],
    current: GridCell,
    cell_size_m: float,
) -> list[tuple[GridCell, float]]:
    row, column = current
    neighbors: list[tuple[GridCell, float]] = []

    for row_offset in (-1, 0, 1):
        for column_offset in (-1, 0, 1):
            if row_offset == 0 and column_offset == 0:
                continue

            neighbor = (
                row + row_offset,
                column + column_offset,
            )

            if not _is_in_bounds(grid, neighbor):
                continue

            if _is_blocked(grid, neighbor):
                continue

            diagonal = row_offset != 0 and column_offset != 0

            if diagonal:
                horizontal = (row, column + column_offset)
                vertical = (row + row_offset, column)

                if _is_blocked(grid, horizontal) or _is_blocked(grid, vertical):
                    continue

            step_distance = cell_size_m * sqrt(2.0) if diagonal else cell_size_m
            neighbors.append((neighbor, step_distance))

    return neighbors


def _validate_inputs(
    grid: NDArray[np.integer],
    start: GridCell,
    destination: GridCell,
    cell_size_m: float,
) -> None:
    if grid.ndim != 2:
        raise ValueError("traversability must be a two-dimensional grid")

    valid_classes = np.array(
        [terrain_class.value for terrain_class in TraversabilityClass]
    )
    if not np.all(np.isin(grid, valid_classes)):
        raise ValueError("traversability grid contains unknown classes")

    if not np.isfinite(cell_size_m) or cell_size_m <= 0.0:
        raise ValueError("cell_size_m must be positive and finite")

    for name, cell in (
        ("start", start),
        ("destination", destination),
    ):
        if not _is_in_bounds(grid, cell):
            raise ValueError(f"{name} cell is outside the terrain grid")

        if _is_blocked(grid, cell):
            raise ValueError(f"{name} cell is blocked")


def _is_in_bounds(
    grid: NDArray[np.integer],
    cell: GridCell,
) -> bool:
    row, column = cell
    return 0 <= row < grid.shape[0] and 0 <= column < grid.shape[1]


def _is_blocked(
    grid: NDArray[np.integer],
    cell: GridCell,
) -> bool:
    return bool(grid[cell] == TraversabilityClass.BLOCKED.value)


def _heuristic(
    cell: GridCell,
    destination: GridCell,
    cell_size_m: float,
) -> float:
    return (
        hypot(
            destination[0] - cell[0],
            destination[1] - cell[1],
        )
        * cell_size_m
    )


def _reconstruct_path(
    previous: dict[GridCell, GridCell],
    destination: GridCell,
) -> tuple[GridCell, ...]:
    path = [destination]
    current = destination

    while current in previous:
        current = previous[current]
        path.append(current)

    path.reverse()
    return tuple(path)


def _path_distance(
    path: tuple[GridCell, ...],
    cell_size_m: float,
) -> float:
    distance = 0.0

    for first, second in zip(path, path[1:], strict=False):
        distance += (
            hypot(
                second[0] - first[0],
                second[1] - first[1],
            )
            * cell_size_m
        )

    return distance
