import math

import pytest
from lbqa_analysis.ellipse_axes import StableEllipseAxes


@pytest.mark.parametrize("azimuth", [90.0, -90.0, 270.0, 450.0])
def test_initial_vertical_major_assigns_minor_to_x(azimuth: float) -> None:
    result = StableEllipseAxes().update(12.0, 8.0, azimuth)

    assert result["width_x"] == 8.0
    assert result["width_y"] == 12.0
    assert result["angle_x_deg"] == pytest.approx(0.0)
    assert result["angle_y_deg"] == pytest.approx(-90.0)
    assert result["orientation_uncertain"] is False
    assert result["raw_major_diameter"] == 12.0
    assert result["raw_minor_diameter"] == 8.0
    assert result["raw_azimuth_deg"] == azimuth


def test_width_crossing_through_circle_keeps_x_identity_without_ninety_degree_swap() -> None:
    tracker = StableEllipseAxes()
    widths_x = [12.0, 10.1, 10.0, 9.9, 8.0]

    results = [
        tracker.update(max(width, 10.0), min(width, 10.0), 0.0 if width >= 10.0 else 90.0)
        for width in widths_x
    ]

    assert [result["width_x"] for result in results] == pytest.approx(widths_x)
    assert [result["width_y"] for result in results] == pytest.approx([10.0] * 5)
    assert [result["angle_x_deg"] for result in results] == pytest.approx([0.0] * 5)
    assert [result["orientation_uncertain"] for result in results] == [
        False,
        True,
        True,
        True,
        False,
    ]
    assert results[-1]["width_x"] == results[-1]["raw_minor_diameter"]


def test_gradual_rotation_keeps_identity_even_after_x_is_closer_to_vertical() -> None:
    tracker = StableEllipseAxes()

    for angle in range(0, 401, 20):
        result = tracker.update(20.0, 8.0, float(angle))
        assert result["width_x"] == 20.0
        assert result["width_y"] == 8.0
        assert result["angle_x_deg"] == pytest.approx((angle + 90.0) % 180.0 - 90.0)
        assert result["orientation_uncertain"] is False


def test_rotation_tracks_initial_minor_axis_across_angle_wrap() -> None:
    tracker = StableEllipseAxes()

    for angle in range(80, 301, 20):
        result = tracker.update(20.0, 8.0, float(angle))
        assert result["width_x"] == 8.0
        assert result["width_y"] == 20.0
        assert result["angle_x_deg"] == pytest.approx((angle + 180.0) % 180.0 - 90.0)


def test_modulo_180_angles_do_not_change_axis_assignment() -> None:
    tracker = StableEllipseAxes()

    first = tracker.update(12.0, 6.0, 179.0)
    second = tracker.update(13.0, 7.0, 1.0)
    third = tracker.update(14.0, 8.0, -177.0)

    assert [first["angle_x_deg"], second["angle_x_deg"], third["angle_x_deg"]] == [-1.0, 1.0, 3.0]
    assert [first["width_x"], second["width_x"], third["width_x"]] == [12.0, 13.0, 14.0]


@pytest.mark.parametrize("raw_angle", [0.0, 70.0, 110.0, 179.0])
def test_near_circle_uses_current_covariance_projection_onto_held_direction(
    raw_angle: float,
) -> None:
    tracker = StableEllipseAxes()
    tracker.update(12.0, 8.0, 30.0)

    result = tracker.update(10.0, 9.9, raw_angle)

    delta = math.radians(30.0 - raw_angle)
    expected_x = math.sqrt(10.0**2 * math.cos(delta) ** 2 + 9.9**2 * math.sin(delta) ** 2)
    expected_y = math.sqrt(10.0**2 * math.sin(delta) ** 2 + 9.9**2 * math.cos(delta) ** 2)
    assert result["angle_x_deg"] == 30.0
    assert result["orientation_uncertain"] is True
    assert result["width_x"] == pytest.approx(expected_x)
    assert result["width_y"] == pytest.approx(expected_y)
    assert result["width_x"] != pytest.approx(10.0)
    assert result["width_x"] ** 2 + result["width_y"] ** 2 == pytest.approx(10.0**2 + 9.9**2)
    assert result["width_mode"] == "held_direction_projection"
    assert result["raw_azimuth_deg"] == raw_angle


def test_multiple_uncertain_updates_hold_direction_without_freezing_widths() -> None:
    tracker = StableEllipseAxes()
    tracker.update(12.0, 8.0, 15.0)

    first = tracker.update(10.0, 10.0, 80.0)
    second = tracker.update(20.0, 20.0, -70.0)
    third = tracker.update(21.0, 12.0, 105.0)

    assert first["width_x"] == pytest.approx(10.0)
    assert second["width_x"] == pytest.approx(20.0)
    assert first["angle_x_deg"] == second["angle_x_deg"] == third["angle_x_deg"] == 15.0
    assert third["width_x"] == 12.0
    assert third["width_y"] == 21.0


@pytest.mark.parametrize("major,minor", [(10.0, 10.0), (10.0, 9.9)])
def test_initial_near_circle_is_uncertain(major: float, minor: float) -> None:
    result = StableEllipseAxes().update(major, minor, 80.0)

    assert result["orientation_uncertain"] is True
    assert result["angle_x_deg"] == -10.0
    assert result["width_x"] == pytest.approx(minor)
    assert result["width_y"] == pytest.approx(major)


def test_near_circle_threshold_includes_exact_two_percent_gap() -> None:
    tracker = StableEllipseAxes()
    tracker.update(120.0, 80.0, 15.0)

    result = tracker.update(100.0, 98.0, 40.0)

    assert result["orientation_uncertain"] is True
    assert result["angle_x_deg"] == 15.0


@pytest.mark.parametrize("scale", [1e-100, 1.0, 1000.0, 1e100])
def test_width_units_are_generic_and_no_diameter_smoothing_is_applied(scale: float) -> None:
    tracker = StableEllipseAxes()
    tracker.update(12.0 * scale, 8.0 * scale, 0.0)

    result = tracker.update(40.0 * scale, 5.0 * scale, 5.0)

    assert result["width_x"] == 40.0 * scale
    assert result["width_y"] == 5.0 * scale
    assert result["angle_x_deg"] == 5.0


@pytest.mark.parametrize(
    "values",
    [
        (math.nan, 1.0, 0.0),
        (2.0, math.inf, 0.0),
        (2.0, 1.0, math.nan),
        (2.0, 1.0, math.inf),
        (0.0, 0.0, 0.0),
        (2.0, -1.0, 0.0),
        (1.0, 2.0, 0.0),
    ],
)
def test_invalid_input_leaves_tracker_state_unchanged(values: tuple[float, float, float]) -> None:
    tracker = StableEllipseAxes()
    tracker.update(12.0, 8.0, 15.0)

    with pytest.raises(ValueError):
        tracker.update(*values)

    result = tracker.update(10.0, 10.0, 80.0)
    assert result["angle_x_deg"] == 15.0


def test_independent_trackers_do_not_share_state() -> None:
    first = StableEllipseAxes()
    first.update(12.0, 8.0, 0.0)
    first.update(12.0, 8.0, 40.0)

    result = StableEllipseAxes().update(10.0, 10.0, 80.0)

    assert result["angle_x_deg"] == -10.0
