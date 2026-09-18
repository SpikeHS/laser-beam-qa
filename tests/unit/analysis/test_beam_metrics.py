import numpy as np
import pytest
from lbqa_analysis import (
    analyze_beam_plane,
    calculate_centroid,
    calculate_d4sigma_2d,
    calculate_fwhm_xy,
    dark_subtract,
    estimate_gaussian_2d,
)
from lbqa_contracts.models import OpticalCalibration


def _gaussian_image(
    width_px: int,
    height_px: int,
    centroid_x_px: float,
    centroid_y_px: float,
    sigma_x_px: float,
    sigma_y_px: float,
    amplitude: float = 5000.0,
) -> np.ndarray:
    y_px, x_px = np.indices((height_px, width_px), dtype=np.float64)
    return amplitude * np.exp(
        -0.5
        * (
            ((x_px - centroid_x_px) / sigma_x_px) ** 2
            + ((y_px - centroid_y_px) / sigma_y_px) ** 2
        )
    )


def test_dark_subtract_clips_negative_values() -> None:
    raw = np.array([[10, 20], [30, 40]], dtype=np.uint16)
    dark = np.array([[12, 5], [25, 50]], dtype=np.uint16)

    corrected = dark_subtract(raw, dark)

    assert corrected.dtype == np.float64
    assert corrected.tolist() == [[0.0, 15.0], [5.0, 0.0]]


def test_centroid_matches_synthetic_gaussian() -> None:
    effective_pixel_um = 0.1375
    image = _gaussian_image(
        width_px=160,
        height_px=144,
        centroid_x_px=76.3,
        centroid_y_px=61.8,
        sigma_x_px=6.0,
        sigma_y_px=8.0,
    )

    centroid = calculate_centroid(image, effective_pixel_um, effective_pixel_um)

    assert centroid["centroid_x_um"] == pytest.approx(76.3 * effective_pixel_um, abs=1e-4)
    assert centroid["centroid_y_um"] == pytest.approx(61.8 * effective_pixel_um, abs=1e-4)


def test_d4sigma_matches_synthetic_gaussian_widths() -> None:
    effective_pixel_um = 0.1375
    sigma_x_px = 7.0
    sigma_y_px = 11.0
    image = _gaussian_image(
        width_px=192,
        height_px=192,
        centroid_x_px=96.0,
        centroid_y_px=94.0,
        sigma_x_px=sigma_x_px,
        sigma_y_px=sigma_y_px,
    )

    metrics = calculate_d4sigma_2d(image, effective_pixel_um, effective_pixel_um)

    assert metrics["d4sigma_x_um"] == pytest.approx(4.0 * sigma_x_px * effective_pixel_um)
    assert metrics["d4sigma_y_um"] == pytest.approx(4.0 * sigma_y_px * effective_pixel_um)
    assert metrics["d4sigma_major_um"] == pytest.approx(4.0 * sigma_y_px * effective_pixel_um)
    assert metrics["d4sigma_minor_um"] == pytest.approx(4.0 * sigma_x_px * effective_pixel_um)
    assert metrics["ellipticity"] == pytest.approx(sigma_x_px / sigma_y_px)


def test_fwhm_centerline_matches_gaussian_width() -> None:
    effective_pixel_um = 0.1375
    sigma_x_px = 8.0
    sigma_y_px = 10.0
    image = _gaussian_image(
        width_px=192,
        height_px=192,
        centroid_x_px=96.0,
        centroid_y_px=96.0,
        sigma_x_px=sigma_x_px,
        sigma_y_px=sigma_y_px,
    )

    fwhm = calculate_fwhm_xy(image, effective_pixel_um, effective_pixel_um)

    factor = 2.0 * np.sqrt(2.0 * np.log(2.0))
    assert fwhm["fwhm_x_um"] == pytest.approx(factor * sigma_x_px * effective_pixel_um, rel=0.01)
    assert fwhm["fwhm_y_um"] == pytest.approx(factor * sigma_y_px * effective_pixel_um, rel=0.01)


def test_analyze_beam_plane_returns_contract_result() -> None:
    effective_pixel_um = 0.1375
    image = _gaussian_image(
        width_px=160,
        height_px=160,
        centroid_x_px=80.0,
        centroid_y_px=78.0,
        sigma_x_px=7.0,
        sigma_y_px=9.0,
        amplitude=3000.0,
    )
    calibration = OpticalCalibration(
        magnification=40.0,
        camera_pixel_size_um=5.5,
        effective_pixel_x_um=effective_pixel_um,
        effective_pixel_y_um=effective_pixel_um,
        field_of_view_x_um=283.0,
        field_of_view_y_um=283.0,
        working_distance_mm=0.6,
    )

    result = analyze_beam_plane(
        image,
        calibration=calibration,
        z_actual_mm=0.25,
        quality_limits={
            "bit_depth": 12,
            "edge_margin_px": 4,
            "edge_energy_limit_percent": 2.0,
            "saturation_threshold_percent": 99.9,
        },
    )

    assert result.valid
    assert result.z_actual_mm == pytest.approx(0.25)
    assert result.d4sigma_x_um == pytest.approx(4.0 * 7.0 * effective_pixel_um)


def test_gaussian_fit_width_uses_beam_core_instead_of_broad_halo() -> None:
    effective_pixel_um = 0.1375
    core_sigma_x_px = 6.0
    core_sigma_y_px = 10.0
    image = _gaussian_image(
        width_px=220,
        height_px=220,
        centroid_x_px=110.0,
        centroid_y_px=104.0,
        sigma_x_px=core_sigma_x_px,
        sigma_y_px=core_sigma_y_px,
        amplitude=1.0,
    )
    image += _gaussian_image(
        width_px=220,
        height_px=220,
        centroid_x_px=110.0,
        centroid_y_px=104.0,
        sigma_x_px=45.0,
        sigma_y_px=50.0,
        amplitude=0.04,
    )

    second_moment = calculate_d4sigma_2d(image, effective_pixel_um, effective_pixel_um)
    gaussian = estimate_gaussian_2d(image, effective_pixel_um, effective_pixel_um)

    assert second_moment["d4sigma_x_um"] > 4.0 * core_sigma_x_px * effective_pixel_um * 1.5
    assert gaussian["d4sigma_x_um"] == pytest.approx(
        4.0 * core_sigma_x_px * effective_pixel_um,
        rel=0.08,
    )
    assert gaussian["d4sigma_y_um"] == pytest.approx(
        4.0 * core_sigma_y_px * effective_pixel_um,
        rel=0.08,
    )


def test_analyze_beam_plane_can_use_gaussian_fit_width_method() -> None:
    effective_pixel_um = 0.1375
    image = _gaussian_image(
        width_px=180,
        height_px=180,
        centroid_x_px=91.0,
        centroid_y_px=87.0,
        sigma_x_px=5.0,
        sigma_y_px=8.0,
        amplitude=0.8,
    )
    calibration = OpticalCalibration(
        magnification=40.0,
        camera_pixel_size_um=5.5,
        effective_pixel_x_um=effective_pixel_um,
        effective_pixel_y_um=effective_pixel_um,
        field_of_view_x_um=24.75,
        field_of_view_y_um=24.75,
        working_distance_mm=0.6,
    )

    result = analyze_beam_plane(
        image,
        calibration=calibration,
        z_actual_mm=0.0,
        quality_limits={
            "width_method": "gaussian_fit",
            "bit_depth": 1,
            "low_signal_threshold_percent": 1.0,
            "edge_energy_limit_percent": 5.0,
            "saturation_threshold_percent": 99.9,
        },
    )

    assert result.valid
    assert result.d4sigma_x_um == pytest.approx(4.0 * 5.0 * effective_pixel_um, rel=0.05)
    assert result.d4sigma_y_um == pytest.approx(4.0 * 8.0 * effective_pixel_um, rel=0.05)
