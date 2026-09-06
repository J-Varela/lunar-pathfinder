"""Generate Lunar Pathfinder demo outputs for Site 11 de Gerlache Rim."""

from pathlib import Path

from lunar_pathfinder.analysis import (
    analyze_elevation_raster,
    analyze_route,
    count_route_classes,
    sample_straight_line,
)
from lunar_pathfinder.raster import RasterWindow
from lunar_pathfinder.traversability import TraversabilityClass
from lunar_pathfinder.validation import validate_slope_against_reference
from lunar_pathfinder.visualization import (
    save_route_comparison_visualization,
    save_route_visualization,
    save_traversability_route_visualization,
)

DATA_PATH = Path("data/raw/Site11_final_adj_5mpp_surf.tif")
SLOPE_PATH = Path("data/raw/Site11_final_adj_5mpp_slp.tif")
OUTPUT_DIR = Path("outputs")

SITE11_WINDOW = RasterWindow(
    row_offset=1200,
    column_offset=1200,
    height=800,
    width=800,
)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    analysis = analyze_elevation_raster(
        DATA_PATH,
        window=SITE11_WINDOW,
    )

    validation = validate_slope_against_reference(
        DATA_PATH,
        SLOPE_PATH,
        window=SITE11_WINDOW,
    )

    route = analyze_route(
        analysis,
        start=(50, 50),
        destination=(750, 750),
    )

    elevation_output = save_route_visualization(
        analysis,
        route,
        OUTPUT_DIR / "site11-route.png",
    )

    traversability_output = save_traversability_route_visualization(
        analysis,
        route,
        OUTPUT_DIR / "site11-traversability-route.png",
    )

    comparison_output = save_route_comparison_visualization(
        analysis,
        route,
        OUTPUT_DIR / "site11-route-comparison.png",
    )

    straight_path = sample_straight_line(
        analysis.traversability,
        route.route.path[0],
        route.route.path[-1],
    )

    straight_counts = count_route_classes(
        analysis.traversability,
        straight_path,
    )

    total_cells = analysis.traversability.size
    route_cells = len(route.route.path)

    print("Lunar Pathfinder — Site 11 de Gerlache Rim")
    print("=" * 48)
    print()

    print(f"Raster shape: {analysis.raster.elevation_m.shape}")
    print(f"Cell size: {analysis.raster.cell_size_m:.1f} m")
    print(f"Mean slope: {analysis.mean_slope_degrees:.3f}°")
    print(f"Max slope: {analysis.max_slope_degrees:.3f}°")

    print()
    print("Terrain composition:")

    for terrain_class in TraversabilityClass:
        count = analysis.class_counts[terrain_class]
        percentage = 100.0 * count / total_cells

        print(
            f"  {terrain_class.name.lower():10s} {count:7d} cells ({percentage:6.2f}%)"
        )

    print()
    print("Slope validation against NASA reference:")
    print(f"  Derived mean: {validation.derived_mean_degrees:.3f}°")
    print(f"  Reference mean: {validation.reference_mean_degrees:.3f}°")
    print(f"  MAE: {validation.mae_degrees:.3f}°")
    print(f"  RMSE: {validation.rmse_degrees:.3f}°")
    print(f"  Median absolute error: {validation.median_absolute_error_degrees:.3f}°")
    print(
        "  95th percentile absolute error: "
        f"{validation.percentile_95_absolute_error_degrees:.3f}°"
    )
    print(f"  Correlation: {validation.correlation:.6f}")

    print()
    print("Route summary:")
    print(f"  Start: {route.route.path[0]}")
    print(f"  Destination: {route.route.path[-1]}")
    print(f"  Path cells: {route_cells}")
    print(f"  Route distance: {route.route.distance_m:.1f} m")
    print(f"  Straight-line distance: {route.straight_line_distance_m:.1f} m")
    print(f"  Detour ratio: {route.detour_ratio:.4f}")
    print(f"  Weighted cost: {route.route.total_cost:.1f}")

    print()
    print("Route terrain:")

    for terrain_class in TraversabilityClass:
        count = route.terrain_counts[terrain_class]
        percentage = 100.0 * count / route_cells

        print(
            f"  {terrain_class.name.lower():10s} "
            f"{count:5d} route cells "
            f"({percentage:6.2f}%)"
        )

    print()
    print("Straight-line terrain:")

    for terrain_class in TraversabilityClass:
        count = straight_counts[terrain_class]
        percentage = 100.0 * count / len(straight_path)

        print(
            f"  {terrain_class.name.lower():10s} {count:5d} cells ({percentage:6.2f}%)"
        )

    print()
    print(f"Saved: {elevation_output}")
    print(f"Saved: {traversability_output}")
    print(f"Saved: {comparison_output}")


if __name__ == "__main__":
    main()
