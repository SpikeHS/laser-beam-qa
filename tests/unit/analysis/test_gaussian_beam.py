from __future__ import annotations

import math

import pytest
from lbqa_analysis.gaussian_beam import (
    calculate_measured_m2,
    predict_gaussian_divergence,
)


def test_predict_gaussian_divergence_from_waist_diameter() -> None:
    prediction = predict_gaussian_divergence(
        waist_diameter_x_um=20.0,
        waist_diameter_y_um=10.0,
        wavelength_nm=1064.0,
        m2=1.2,
    )

    assert prediction["valid"] is True
    expected_x = 4.0 * 1.2 * 1064.0 / (math.pi * 20.0)
    expected_y = 4.0 * 1.2 * 1064.0 / (math.pi * 10.0)
    assert prediction["predicted_full_angle_x_mrad"] == pytest.approx(expected_x)
    assert prediction["predicted_full_angle_y_mrad"] == pytest.approx(expected_y)
    assert prediction["predicted_half_angle_x_mrad"] == pytest.approx(expected_x / 2.0)
    assert prediction["predicted_half_angle_y_mrad"] == pytest.approx(expected_y / 2.0)
    expected_fraunhofer_x = 2000.0 * math.asin(2.0 * 1.064 / (math.pi * 20.0))
    expected_fraunhofer_y = 2000.0 * math.asin(2.0 * 1.064 / (math.pi * 10.0))
    assert prediction["fraunhofer_ideal_full_angle_x_mrad"] == pytest.approx(
        expected_fraunhofer_x
    )
    assert prediction["fraunhofer_ideal_full_angle_y_mrad"] == pytest.approx(
        expected_fraunhofer_y
    )


def test_predict_gaussian_divergence_requires_explicit_wavelength() -> None:
    prediction = predict_gaussian_divergence(
        waist_diameter_x_um=20.0,
        waist_diameter_y_um=10.0,
        wavelength_nm=None,
    )

    assert prediction["valid"] is False
    assert prediction["invalid_reason"] == "wavelength_nm_not_provided"
    assert prediction["predicted_full_angle_x_mrad"] is None


def test_calculate_measured_m2_uses_dual_axis_full_angle_identity() -> None:
    wavelength_nm = 1064.0
    waist_x_um = 20.0
    waist_y_um = 10.0
    full_x_mrad = 4.0 * wavelength_nm / (math.pi * waist_x_um)
    full_y_mrad = 4.0 * 1.3 * wavelength_nm / (math.pi * waist_y_um)

    measured = calculate_measured_m2(
        waist_diameter_x_um=waist_x_um,
        waist_diameter_y_um=waist_y_um,
        full_angle_x_mrad=full_x_mrad,
        full_angle_y_mrad=full_y_mrad,
        wavelength_nm=wavelength_nm,
    )

    assert measured["valid"] is True
    assert measured["m2_x"] == pytest.approx(1.0)
    assert measured["m2_y"] == pytest.approx(1.3)
