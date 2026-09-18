import math
from dataclasses import replace

import numpy as np
import pytest
from lbqa_analysis import evaluate_frame_quality, fit_zscan, judge_result
from lbqa_contracts.errors import ErrorCode
from lbqa_contracts.models import BeamPlaneResult


def _plane(
    z_actual_mm: float,
    d4sigma_x_um: float,
    d4sigma_y_um: float,
    *,
    valid: bool = True,
    invalid_reason: str | None = None,
    outlier: bool = False,
    outlier_reason: str | None = None,
) -> BeamPlaneResult:
    d4sigma_major_um = max(d4sigma_x_um, d4sigma_y_um)
    d4sigma_minor_um = min(d4sigma_x_um, d4sigma_y_um)
    return BeamPlaneResult(
        z_actual_mm=z_actual_mm,
        centroid_x_um=10.0,
        centroid_y_um=10.0,
        peak_x_um=10.0,
        peak_y_um=10.0,
        d4sigma_x_um=d4sigma_x_um,
        d4sigma_y_um=d4sigma_y_um,
        d4sigma_major_um=d4sigma_major_um,
        d4sigma_minor_um=d4sigma_minor_um,
        fwhm_x_um=d4sigma_x_um / 2.0,
        fwhm_y_um=d4sigma_y_um / 2.0,
        ellipticity=d4sigma_minor_um / d4sigma_major_um,
        azimuth_deg=0.0,
        peak_value=3000.0,
        saturation_pixels=0,
        edge_energy_percent=0.5,
        valid=valid,
        invalid_reason=invalid_reason,
        outlier=outlier,
        outlier_reason=outlier_reason,
    )


def test_saturated_image_is_invalid() -> None:
    image = np.zeros((32, 32), dtype=np.uint16)
    image[16, 16] = 4095

    quality = evaluate_frame_quality(
        image=image,
        bit_depth=12,
        edge_margin_px=4,
        edge_energy_limit_percent=5.0,
        saturation_threshold_percent=99.0,
    )

    assert not quality.valid
    assert quality.saturated
    assert quality.invalid_reason == ErrorCode.E_IMAGE_SATURATED.value


def test_edge_truncated_image_is_invalid() -> None:
    y_px, x_px = np.indices((64, 64), dtype=np.float64)
    image = 3000.0 * np.exp(-0.5 * (((x_px - 2.0) / 8.0) ** 2 + ((y_px - 32.0) / 5.0) ** 2))

    quality = evaluate_frame_quality(
        image=image,
        bit_depth=12,
        edge_margin_px=6,
        edge_energy_limit_percent=10.0,
        saturation_threshold_percent=99.0,
    )

    assert not quality.valid
    assert quality.invalid_reason == ErrorCode.E_IMAGE_EDGE_CLIPPED.value
    assert quality.edge_energy_percent > 10.0


def test_fit_zscan_recovers_known_full_angle_mrad_and_coefficients() -> None:
    full_angle_x_mrad = 3.0
    full_angle_y_mrad = 4.5
    waist_z_mm = 0.25
    waist_x_um = 12.0
    waist_y_um = 16.0
    points = []
    for z_actual_mm in [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]:
        d4sigma_x_um = math.sqrt(
            waist_x_um**2 + (full_angle_x_mrad * (z_actual_mm - waist_z_mm)) ** 2
        )
        d4sigma_y_um = math.sqrt(
            waist_y_um**2 + (full_angle_y_mrad * (z_actual_mm - waist_z_mm)) ** 2
        )
        points.append(_plane(z_actual_mm, d4sigma_x_um, d4sigma_y_um))

    result = fit_zscan(points)

    assert judge_result(result) == "PASS"
    assert result.valid_points_count == len(points)
    assert result.outlier_indices == []
    assert result.full_angle_x_mrad == pytest.approx(full_angle_x_mrad)
    assert result.half_angle_x_mrad == pytest.approx(full_angle_x_mrad / 2.0)
    assert result.full_angle_y_mrad == pytest.approx(full_angle_y_mrad)
    assert result.waist_z_x_mm == pytest.approx(waist_z_mm)
    assert result.waist_diameter_x_um == pytest.approx(waist_x_um)
    assert result.fit_r2_x == pytest.approx(1.0)
    assert result.fit_a_x_um2_per_mm2 == pytest.approx(full_angle_x_mrad**2)
    assert result.fit_b_x_um2_per_mm == pytest.approx(-2.0 * full_angle_x_mrad**2 * waist_z_mm)
    assert result.fit_c_x_um2 == pytest.approx(waist_x_um**2 + full_angle_x_mrad**2 * waist_z_mm**2)
    assert result.fit_a_y_um2_per_mm2 == pytest.approx(full_angle_y_mrad**2)
    assert result.fit_b_y_um2_per_mm == pytest.approx(-2.0 * full_angle_y_mrad**2 * waist_z_mm)
    assert result.fit_c_y_um2 == pytest.approx(waist_y_um**2 + full_angle_y_mrad**2 * waist_z_mm**2)


def test_fit_zscan_records_quality_filtered_points_as_outliers() -> None:
    points = [
        _plane(-1.0, math.sqrt(9.0 + 4.0), math.sqrt(16.0 + 9.0)),
        _plane(0.0, 3.0, 4.0),
        _plane(1.0, math.sqrt(9.0 + 4.0), math.sqrt(16.0 + 9.0)),
        _plane(
            2.0,
            999.0,
            999.0,
            valid=False,
            invalid_reason=ErrorCode.E_IMAGE_SATURATED.value,
        ),
    ]

    result = fit_zscan(points)

    assert result.valid_points_count == 3
    assert len(result.points) == 4
    assert result.outlier_indices == [3]
    assert result.full_angle_x_mrad == pytest.approx(2.0)
    assert result.full_angle_y_mrad == pytest.approx(3.0)


def test_fit_zscan_records_robust_fit_outliers() -> None:
    points = []
    for z_actual_mm in [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]:
        d4sigma_x_um = math.sqrt(10.0**2 + (2.0 * z_actual_mm) ** 2)
        d4sigma_y_um = math.sqrt(14.0**2 + (3.0 * z_actual_mm) ** 2)
        points.append(_plane(z_actual_mm, d4sigma_x_um, d4sigma_y_um))
    points[5] = _plane(2.0, 80.0, 90.0)

    result = fit_zscan(points)

    assert result.outlier_indices == [5]
    assert result.valid_points_count == 6
    assert result.full_angle_x_mrad == pytest.approx(2.0)
    assert result.full_angle_y_mrad == pytest.approx(3.0)


def test_fit_zscan_robust_diameter_fit_recovers_noisy_dual_axis_truth() -> None:
    rng = np.random.default_rng(20260820)
    z_values = np.asarray([-20.0, -12.0, -6.0, -2.0, 0.0, 2.0, 6.0, 12.0, 20.0])
    truth = {
        "waist_x_um": 18.0,
        "waist_y_um": 11.0,
        "waist_z_x_mm": 0.4,
        "waist_z_y_mm": -0.3,
        "full_angle_x_mrad": 3.2,
        "full_angle_y_mrad": 6.4,
    }
    points = []
    for z_actual_mm in z_values:
        diameter_x_um = math.sqrt(
            truth["waist_x_um"] ** 2
            + (
                truth["full_angle_x_mrad"]
                * (z_actual_mm - truth["waist_z_x_mm"])
            )
            ** 2
        )
        diameter_y_um = math.sqrt(
            truth["waist_y_um"] ** 2
            + (
                truth["full_angle_y_mrad"]
                * (z_actual_mm - truth["waist_z_y_mm"])
            )
            ** 2
        )
        point = _plane(
            float(z_actual_mm),
            diameter_x_um + float(rng.normal(0.0, 0.08)),
            diameter_y_um + float(rng.normal(0.0, 0.08)),
        )
        points.append(
            replace(
                point,
                width_uncertainty_x_um=0.1,
                width_uncertainty_y_um=0.1,
            )
        )
    points[7] = replace(
        points[7],
        d4sigma_x_um=points[7].d4sigma_x_um + 35.0,
        d4sigma_y_um=points[7].d4sigma_y_um + 50.0,
    )

    result = fit_zscan(points)

    assert result.outlier_indices == [7]
    assert result.valid_points_count == 8
    assert result.full_angle_x_mrad == pytest.approx(
        truth["full_angle_x_mrad"], rel=0.02
    )
    assert result.full_angle_y_mrad == pytest.approx(
        truth["full_angle_y_mrad"], rel=0.02
    )
    assert result.waist_diameter_x_um == pytest.approx(truth["waist_x_um"], rel=0.02)
    assert result.waist_diameter_y_um == pytest.approx(truth["waist_y_um"], rel=0.02)
    assert result.fit_diagnostics["fit_space"] == "diameter_um"
    assert result.fit_diagnostics["optimizer"] == "scipy_least_squares_soft_l1"
    assert result.fit_diagnostics["axis_semantics"] == "fixed_lab_x_y"
    assert result.fit_diagnostics["x"]["uncertainty_mode"] == "measured_per_point"
    assert len(result.fit_diagnostics["points"]) == len(points)
    assert result.fit_diagnostics["points"][7]["fit_outlier"] is True


def test_fit_zscan_with_too_few_valid_points_is_invalid() -> None:
    result = fit_zscan([_plane(0.0, 10.0, 12.0), _plane(1.0, 11.0, 13.0)])

    assert judge_result(result) == "INVALID"
    assert result.judgement == "invalid"
    assert result.invalid_reason == ErrorCode.E_ANALYSIS_ZSCAN_TOO_FEW_POINTS.value
    assert result.full_angle_x_mrad is None
