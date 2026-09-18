from __future__ import annotations

import numpy as np
import pytest
from lbqa_data.beam_visualization import (
    BeamAxes,
    BeamDisplayOptions,
    beam_axes_from_measurement,
    crop_beam_region,
    estimate_beam_axes,
    render_beam_spot_image,
    stabilize_beam_axes,
)
from lbqa_data.report_generator import _report_options_for_image


def test_manual_zoom_uses_requested_center_not_auto_beam_tracking() -> None:
    image = np.zeros((64, 64), dtype=np.float32)
    image[2:6, 2:6] = 10.0

    crop = crop_beam_region(image, options=BeamDisplayOptions(zoom_factor=2.0))

    assert crop.image.shape == (32, 32)
    assert crop.x_offset_px == 16
    assert crop.y_offset_px == 16


def test_render_uses_rgb_thermal_colormap() -> None:
    image = np.arange(32 * 32, dtype=np.float32).reshape(32, 32)

    rendered = render_beam_spot_image(
        image,
        options=BeamDisplayOptions(colormap="thermal", overlay_mode="none"),
    )

    assert rendered.mode == "RGB"
    pixels = np.asarray(rendered)
    assert pixels.ndim == 3
    assert not np.array_equal(pixels[..., 0], pixels[..., 1])


def test_render_can_use_shared_intensity_range_across_frames() -> None:
    bright = np.zeros((32, 32), dtype=np.float32)
    dim = np.zeros((32, 32), dtype=np.float32)
    bright[12:20, 12:20] = 1.0
    dim[12:20, 12:20] = 0.25
    options = BeamDisplayOptions(
        colormap="gray",
        overlay_mode="none",
        intensity_min=0.0,
        intensity_max=1.0,
    )

    bright_pixels = np.asarray(render_beam_spot_image(bright, options=options))
    dim_pixels = np.asarray(render_beam_spot_image(dim, options=options))

    assert int(np.max(bright_pixels)) == 255
    assert 50 <= int(np.max(dim_pixels)) <= 70


def test_estimate_beam_axes_finds_rotated_major_axis() -> None:
    y_px, x_px = np.indices((96, 96), dtype=np.float64)
    angle_rad = np.deg2rad(30.0)
    dx_px = x_px - 48.0
    dy_px = y_px - 48.0
    major_px = dx_px * np.cos(angle_rad) + dy_px * np.sin(angle_rad)
    minor_px = -dx_px * np.sin(angle_rad) + dy_px * np.cos(angle_rad)
    image = np.exp(-((major_px**2) / (2.0 * 14.0**2) + (minor_px**2) / (2.0 * 4.0**2)))

    axes = estimate_beam_axes(image, threshold_percent=10.0)

    assert axes is not None
    assert axes.azimuth_deg == pytest.approx(30.0, abs=2.0)
    assert axes.major_sigma_px > axes.minor_sigma_px


def test_stabilize_beam_axes_prevents_major_minor_label_swap() -> None:
    first = stabilize_beam_axes(
        BeamAxes(48.0, 48.0, 10.0, 12.0, 4.0, 50.0),
        None,
    )
    swapped = stabilize_beam_axes(
        BeamAxes(48.0, 48.0, -80.0, 12.0, 4.0, 50.0),
        first,
    )

    assert swapped.x_axis_azimuth_deg == pytest.approx(10.0)
    assert swapped.axis_assignment == "minor_to_x"
    assert swapped.orientation_status == "tracked"


def test_first_continuous_x_axis_is_nearest_horizontal_even_when_minor() -> None:
    axes = stabilize_beam_axes(BeamAxes(48.0, 48.0, 80.0, 12.0, 4.0, 50.0), None)
    assert axes.x_axis_azimuth_deg == pytest.approx(-10.0)
    assert axes.x_axis_sigma_px == 4.0
    assert axes.y_axis_sigma_px == 12.0
    assert axes.axis_assignment == "minor_to_x"


def test_ellipse_below_circularity_threshold_keeps_tracking() -> None:
    first = stabilize_beam_axes(BeamAxes(48.0, 48.0, 12.0, 10.0, 4.0, 50.0), None)
    tracked = stabilize_beam_axes(BeamAxes(48.0, 48.0, 55.0, 10.0, 9.6, 50.0), first)
    assert tracked.x_axis_azimuth_deg == pytest.approx(55.0)
    assert tracked.orientation_status == "tracked"


def test_stabilize_beam_axes_holds_direction_for_near_circular_beam() -> None:
    first = stabilize_beam_axes(
        BeamAxes(48.0, 48.0, 12.0, 10.0, 4.0, 50.0),
        None,
    )
    near_circular = stabilize_beam_axes(
        BeamAxes(48.0, 48.0, 55.0, 10.0, 9.9, 50.0),
        first,
    )

    assert near_circular.x_axis_azimuth_deg == pytest.approx(12.0)
    assert near_circular.orientation_status == "held_near_circular"
    delta = np.deg2rad(12.0 - 55.0)
    assert near_circular.x_axis_sigma_px == pytest.approx(
        np.hypot(10.0 * np.cos(delta), 9.9 * np.sin(delta))
    )
    assert near_circular.y_axis_sigma_px == pytest.approx(
        np.hypot(10.0 * np.sin(delta), 9.9 * np.cos(delta))
    )


def test_render_can_rotate_report_image_without_rotating_title() -> None:
    image = np.zeros((48, 64), dtype=np.float32)
    image[20:28, 10:54] = 10.0

    rendered = render_beam_spot_image(
        image,
        title="z=0",
        options=BeamDisplayOptions(colormap="thermal", overlay_mode="box"),
        rotation_deg=-25.0,
    )

    assert rendered.mode == "RGB"
    assert rendered.size[0] > 0
    assert rendered.size[1] > 0


def test_render_draws_axis_overlay_when_axes_are_available(monkeypatch) -> None:
    image = np.zeros((64, 64), dtype=np.float32)
    image[24:40, 18:46] = 10.0
    options = BeamDisplayOptions(colormap="thermal", overlay_mode="box")

    without_axes = np.asarray(render_beam_spot_image(image, options=options))
    monkeypatch.setattr(
        "lbqa_data.beam_visualization._draw_threshold_box",
        lambda *args, **kwargs: pytest.fail("Gaussian ellipse must not duplicate threshold box"),
    )
    with_axes = np.asarray(
        render_beam_spot_image(
            image,
            options=options,
            axes=BeamAxes(
                centroid_x_px=32.0,
                centroid_y_px=32.0,
                azimuth_deg=20.0,
                major_sigma_px=8.0,
                minor_sigma_px=4.0,
                threshold_percent=50.0,
            ),
        )
    )

    assert not np.array_equal(with_axes, without_axes)


def test_report_options_auto_crop_full_frame_beam_images() -> None:
    y_px, x_px = np.indices((2040, 2040), dtype=np.float64)
    image = np.exp(-0.5 * (((x_px - 900.0) / 16.0) ** 2 + ((y_px - 1100.0) / 24.0) ** 2))

    options = _report_options_for_image(
        image,
        BeamDisplayOptions(colormap="thermal", overlay_mode="box"),
    )

    assert options.zoom_factor > 1.0
    assert options.center_x_px == pytest.approx(900.0, abs=2.0)
    assert options.center_y_px == pytest.approx(1100.0, abs=2.0)


def test_report_resize_and_rotation_do_not_interpolate_thermal_pixels() -> None:
    frame = np.zeros((8, 8), dtype=np.float32)
    frame[2:4, 2:6] = 0.5
    frame[4:6, 2:6] = 1.0
    original = render_beam_spot_image(frame, options=BeamDisplayOptions(
        min_display_px=8, max_display_px=8, overlay_mode="none",
    ))
    enlarged = render_beam_spot_image(frame, rotation_deg=17.0, options=BeamDisplayOptions(
        min_display_px=64, max_display_px=64, overlay_mode="none",
    ))
    palette = {tuple(pixel) for pixel in np.asarray(original).reshape(-1, 3)}
    rendered_palette = {tuple(pixel) for pixel in np.asarray(enlarged).reshape(-1, 3)}
    assert rendered_palette <= palette


def _measurement_fields() -> dict[str, float]:
    return {
        "centroid_x_um": 5.0, "centroid_y_um": 7.0,
        "major_diameter_um": 16.0, "minor_diameter_um": 8.0, "major_azimuth_deg": 45.0,
        "x_diameter_um": 12.0, "y_diameter_um": 10.0, "x_azimuth_deg": 30.0,
        "pixel_x_um": 0.5, "pixel_y_um": 1.0,
    }


def _ellipse_covariance(axes: BeamAxes) -> np.ndarray:
    angle = np.deg2rad(axes.azimuth_deg)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return rotation @ np.diag([axes.major_sigma_px**2, axes.minor_sigma_px**2]) @ rotation.T


def test_measurement_axes_transform_raw_covariance_and_each_marker_direction() -> None:
    axes = beam_axes_from_measurement(**_measurement_fields(), source="unit_test")
    # Raw physical covariance [[10, 6], [6, 10]] scaled by 2x horizontally.
    np.testing.assert_allclose(_ellipse_covariance(axes), [[40.0, 12.0], [12.0, 10.0]])
    assert axes.centroid_x_px == 10.0
    assert axes.centroid_y_px == 7.0
    assert axes.x_axis_sigma_px == pytest.approx(3 * np.sqrt(3.25))
    assert axes.y_axis_sigma_px == pytest.approx(2.5 * np.sqrt(1.75))
    assert axes.x_axis_azimuth_deg == pytest.approx(np.rad2deg(np.arctan2(0.5, np.sqrt(3))))
    assert axes.y_axis_azimuth_deg == pytest.approx(np.rad2deg(np.arctan2(np.sqrt(3) / 2, -1)))
    assert axes.y_axis_azimuth_deg != pytest.approx(axes.x_axis_azimuth_deg + 90)
    assert axes.major_sigma_px >= axes.minor_sigma_px > 0
    assert axes.source == "unit_test"


def test_measurement_axes_keep_native_near_circle_angle_and_held_x_projection() -> None:
    values = _measurement_fields()
    delta = np.deg2rad(12.0 - 55.0)
    diameter_x = np.hypot(20.0 * np.cos(delta), 19.8 * np.sin(delta))
    diameter_y = np.hypot(20.0 * np.sin(delta), 19.8 * np.cos(delta))
    values.update({
        "major_diameter_um": 20.0, "minor_diameter_um": 19.8, "major_azimuth_deg": 55.0,
        "x_diameter_um": diameter_x, "y_diameter_um": diameter_y, "x_azimuth_deg": 12.0,
        "pixel_x_um": 0.5, "pixel_y_um": 0.5,
    })
    axes = beam_axes_from_measurement(**values, threshold_percent=25.0)
    assert axes.azimuth_deg % 180 == pytest.approx(55.0)
    assert axes.major_sigma_px == pytest.approx(10.0)
    assert axes.minor_sigma_px == pytest.approx(9.9)
    assert axes.x_axis_azimuth_deg == pytest.approx(12.0)
    assert axes.y_axis_azimuth_deg == pytest.approx(102.0)
    assert axes.x_axis_sigma_px == pytest.approx(diameter_x / 2)
    assert axes.y_axis_sigma_px == pytest.approx(diameter_y / 2)
    assert axes.threshold_percent == 25.0


@pytest.mark.parametrize("field,value", [
    ("pixel_x_um", 0.0), ("pixel_y_um", -1.0),
    ("major_diameter_um", 0.0), ("minor_diameter_um", -1.0),
    ("x_diameter_um", 0.0), ("y_diameter_um", -np.inf),
    ("centroid_x_um", np.nan), ("centroid_y_um", np.inf),
    ("major_azimuth_deg", np.nan), ("x_azimuth_deg", np.inf),
])
def test_measurement_axes_reject_invalid_physical_values(field, value) -> None:
    values = _measurement_fields()
    values[field] = value
    with pytest.raises(ValueError, match="Finite measurements and positive"):
        beam_axes_from_measurement(**values)
