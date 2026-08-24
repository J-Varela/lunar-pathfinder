import numpy as np
import pytest

from lunar_pathfinder.traversability import (
    TraversabilityClass,
    TraversabilityThresholds,
    classify_traversability,
    count_traversability_classes,
)


def test_default_classification_boundaries() -> None:
    slope = np.array([[0.0, 7.999, 8.0, 14.999, 15.0, 25.0, 25.001]])

    classes = classify_traversability(slope)

    expected = np.array(
        [
            [
                TraversabilityClass.PREFERRED,
                TraversabilityClass.PREFERRED,
                TraversabilityClass.CAUTION,
                TraversabilityClass.CAUTION,
                TraversabilityClass.HAZARDOUS,
                TraversabilityClass.HAZARDOUS,
                TraversabilityClass.BLOCKED,
            ]
        ],
        dtype=np.uint8,
    )

    np.testing.assert_array_equal(classes, expected)


def test_classification_preserves_grid_shape() -> None:
    slope = np.zeros((12, 17))

    classes = classify_traversability(slope)

    assert classes.shape == slope.shape
    assert classes.dtype == np.uint8


def test_custom_thresholds_are_applied() -> None:
    slope = np.array([[4.0, 7.0, 11.0, 16.0]])
    thresholds = TraversabilityThresholds(
        preferred_max_degrees=5.0,
        caution_max_degrees=10.0,
        hazardous_max_degrees=15.0,
    )

    classes = classify_traversability(slope, thresholds)

    expected = np.array(
        [
            [
                TraversabilityClass.PREFERRED,
                TraversabilityClass.CAUTION,
                TraversabilityClass.HAZARDOUS,
                TraversabilityClass.BLOCKED,
            ]
        ],
        dtype=np.uint8,
    )
    np.testing.assert_array_equal(classes, expected)


def test_non_2d_slope_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="slope must be a two-dimensional grid",
    ):
        classify_traversability(np.zeros(5))


def test_negative_slope_is_rejected() -> None:
    slope = np.zeros((5, 5))
    slope[2, 2] = -1.0

    with pytest.raises(
        ValueError,
        match="slope values cannot be negative",
    ):
        classify_traversability(slope)


@pytest.mark.parametrize("invalid_value", [np.nan, np.inf])
def test_nonfinite_slope_is_rejected(invalid_value: float) -> None:
    slope = np.zeros((5, 5))
    slope[2, 2] = invalid_value

    with pytest.raises(
        ValueError,
        match="slope must contain only finite values",
    ):
        classify_traversability(slope)


@pytest.mark.parametrize(
    "thresholds",
    [
        TraversabilityThresholds,
    ],
)
def test_threshold_type_is_available(
    thresholds: type[TraversabilityThresholds],
) -> None:
    assert thresholds().preferred_max_degrees == 8.0


@pytest.mark.parametrize(
    ("preferred", "caution", "hazardous"),
    [
        (8.0, 8.0, 25.0),
        (15.0, 8.0, 25.0),
        (8.0, 15.0, 15.0),
        (-1.0, 15.0, 25.0),
        (8.0, np.nan, 25.0),
    ],
)
def test_invalid_thresholds_are_rejected(
    preferred: float,
    caution: float,
    hazardous: float,
) -> None:
    with pytest.raises(ValueError):
        TraversabilityThresholds(
            preferred_max_degrees=preferred,
            caution_max_degrees=caution,
            hazardous_max_degrees=hazardous,
        )


def test_traversability_classes_are_counted() -> None:
    classes = np.array(
        [
            [
                TraversabilityClass.PREFERRED,
                TraversabilityClass.PREFERRED,
                TraversabilityClass.CAUTION,
            ],
            [
                TraversabilityClass.HAZARDOUS,
                TraversabilityClass.BLOCKED,
                TraversabilityClass.PREFERRED,
            ],
        ],
        dtype=np.uint8,
    )

    counts = count_traversability_classes(classes)

    assert counts == {
        TraversabilityClass.PREFERRED: 3,
        TraversabilityClass.CAUTION: 1,
        TraversabilityClass.HAZARDOUS: 1,
        TraversabilityClass.BLOCKED: 1,
    }


def test_unknown_traversability_class_is_rejected() -> None:
    classes = np.array([[0, 1, 2, 4]], dtype=np.uint8)

    with pytest.raises(
        ValueError,
        match="traversability grid contains unknown classes",
    ):
        count_traversability_classes(classes)
