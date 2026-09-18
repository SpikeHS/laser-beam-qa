import numpy as np
import pytest
from lbqa_analysis.synthetic import (
    generate_gaussian_beam_image,
    generate_z_scan_sequence,
    max_count_for_bit_depth,
)


def _centroid_px(image_counts: np.ndarray, background_count: float = 0.0) -> tuple[float, float]:
    signal_counts = image_counts.astype(float) - background_count
    signal_counts[signal_counts < 0.0] = 0.0
    y_px, x_px = np.indices(signal_counts.shape, dtype=float)
    total_count = signal_counts.sum()
    return (
        float((x_px * signal_counts).sum() / total_count),
        float((y_px * signal_counts).sum() / total_count),
    )


def test_gaussian_image_has_requested_size() -> None:
    image_counts = generate_gaussian_beam_image(
        width_px=64,
        height_px=48,
        centroid_x_px=31.0,
        centroid_y_px=22.0,
        sigma_x_px=5.0,
        sigma_y_px=7.0,
        angle_deg=0.0,
        amplitude=1000.0,
        background=10.0,
        noise_std=0.0,
        bit_depth=12,
    )

    assert image_counts.shape == (48, 64)


def test_gaussian_peak_is_near_expected_center() -> None:
    image_counts = generate_gaussian_beam_image(
        width_px=80,
        height_px=70,
        centroid_x_px=35.4,
        centroid_y_px=41.6,
        sigma_x_px=4.0,
        sigma_y_px=6.0,
        angle_deg=25.0,
        amplitude=3000.0,
        background=0.0,
        noise_std=0.0,
        bit_depth=12,
    )

    peak_y_px, peak_x_px = np.unravel_index(np.argmax(image_counts), image_counts.shape)

    assert peak_x_px == pytest.approx(35.4, abs=1.0)
    assert peak_y_px == pytest.approx(41.6, abs=1.0)


def test_noiseless_centroid_matches_expected_center() -> None:
    image_counts = generate_gaussian_beam_image(
        width_px=96,
        height_px=88,
        centroid_x_px=45.2,
        centroid_y_px=39.7,
        sigma_x_px=5.0,
        sigma_y_px=6.5,
        angle_deg=12.0,
        amplitude=3500.0,
        background=0.0,
        noise_std=0.0,
        bit_depth=12,
    )

    centroid_x_px, centroid_y_px = _centroid_px(image_counts)

    assert centroid_x_px == pytest.approx(45.2, abs=0.05)
    assert centroid_y_px == pytest.approx(39.7, abs=0.05)


def test_z_scan_spot_diameter_increases_away_from_waist() -> None:
    frames = generate_z_scan_sequence(
        z_positions_mm=[-1.0, 0.0, 1.0],
        waist_z_mm=0.0,
        waist_diameter_x_um=10.0,
        waist_diameter_y_um=12.0,
        full_angle_x_mrad=20.0,
        full_angle_y_mrad=24.0,
        effective_pixel_x_um=0.1375,
        effective_pixel_y_um=0.1375,
        width_px=128,
        height_px=128,
        noise_std=0.0,
    )

    assert frames[0].d4sigma_x_um > frames[1].d4sigma_x_um
    assert frames[2].d4sigma_x_um > frames[1].d4sigma_x_um
    assert frames[0].sigma_y_px > frames[1].sigma_y_px
    assert frames[2].sigma_y_px > frames[1].sigma_y_px


def test_saturation_simulation_produces_max_count_pixels() -> None:
    bit_depth = 12
    image_counts = generate_gaussian_beam_image(
        width_px=40,
        height_px=40,
        centroid_x_px=20.0,
        centroid_y_px=20.0,
        sigma_x_px=3.0,
        sigma_y_px=3.0,
        angle_deg=0.0,
        amplitude=500.0,
        background=20.0,
        noise_std=0.0,
        bit_depth=bit_depth,
        saturation=True,
    )

    assert image_counts.max() == max_count_for_bit_depth(bit_depth)


def test_boundary_truncation_is_reproducible() -> None:
    kwargs = {
        "width_px": 48,
        "height_px": 48,
        "centroid_x_px": 1.0,
        "centroid_y_px": 24.0,
        "sigma_x_px": 8.0,
        "sigma_y_px": 5.0,
        "angle_deg": 0.0,
        "amplitude": 3000.0,
        "background": 0.0,
        "noise_std": 0.0,
        "bit_depth": 12,
    }

    image_counts = generate_gaussian_beam_image(**kwargs)
    repeated_image_counts = generate_gaussian_beam_image(**kwargs)
    centroid_x_px, _ = _centroid_px(image_counts)

    assert np.array_equal(image_counts, repeated_image_counts)
    assert centroid_x_px > kwargs["centroid_x_px"]
