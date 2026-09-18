"""Robust two-axis Gaussian-caustic fitting for analyzed z-scan planes."""

from __future__ import annotations

import math
from collections.abc import Sequence
from itertools import combinations
from typing import Any, NamedTuple

import numpy as np
from lbqa_contracts.errors import ErrorCode
from lbqa_contracts.models import BeamPlaneResult, ZScanResult
from scipy.optimize import least_squares

_OUTLIER_RELATIVE_RESIDUAL_LIMIT = 0.15
_MIN_CONFIDENT_POINTS = 5
_MIN_CONFIDENT_RAYLEIGH_COVERAGE = 2.0


class _QuadraticInitial(NamedTuple):
    fit_a_um2_per_mm2: float
    fit_b_um2_per_mm: float
    fit_c_um2: float
    waist_z_mm: float
    waist_diameter_um: float
    slope_um_per_mm: float


class _AxisFit(NamedTuple):
    fit_a_um2_per_mm2: float
    fit_b_um2_per_mm: float
    fit_c_um2: float
    full_angle_mrad: float
    half_angle_mrad: float
    geometric_full_angle_mrad: float
    waist_z_mm: float
    waist_diameter_um: float
    rayleigh_range_mm: float
    fit_r2: float
    fit_rmse_um: float
    normalized_rmse: float
    reduced_chi_square: float | None
    max_normalized_residual: float
    uncertainty_mode: str
    predicted_diameter_um: np.ndarray
    residuals_um: np.ndarray
    normalized_residuals: np.ndarray
    parameter_ci95: dict[str, list[float] | None]


def fit_zscan(points: Sequence[BeamPlaneResult]) -> ZScanResult:
    """Fit independent X/Y caustics from quality-valid beam-width measurements."""

    all_points = list(points)
    candidate_indices = [
        index for index, point in enumerate(all_points) if _can_fit_point(point)
    ]
    quality_outlier_indices = [
        index for index, point in enumerate(all_points) if not _can_fit_point(point)
    ]
    if len(candidate_indices) < 3:
        return _invalid_zscan_result(
            all_points,
            valid_points_count=len(candidate_indices),
            invalid_reason=ErrorCode.E_ANALYSIS_ZSCAN_TOO_FEW_POINTS.value,
            outlier_indices=quality_outlier_indices,
        )

    candidate_points = [all_points[index] for index in candidate_indices]
    z_actual_mm = np.asarray([point.z_actual_mm for point in candidate_points], dtype=np.float64)
    diameter_x_um = np.asarray(
        [point.d4sigma_x_um for point in candidate_points], dtype=np.float64
    )
    diameter_y_um = np.asarray(
        [point.d4sigma_y_um for point in candidate_points], dtype=np.float64
    )
    uncertainty_x_um = _axis_uncertainties(candidate_points, "x")
    uncertainty_y_um = _axis_uncertainties(candidate_points, "y")

    try:
        fit_indices, robust_outlier_indices, fit_x, fit_y = _fit_axes_robust(
            candidate_indices=candidate_indices,
            z_actual_mm=z_actual_mm,
            diameter_x_um=diameter_x_um,
            diameter_y_um=diameter_y_um,
            uncertainty_x_um=uncertainty_x_um,
            uncertainty_y_um=uncertainty_y_um,
        )
    except (RuntimeError, ValueError):
        return _invalid_zscan_result(
            all_points,
            valid_points_count=len(candidate_indices),
            invalid_reason=ErrorCode.E_ANALYSIS_ZSCAN_FIT_FAILED.value,
            outlier_indices=quality_outlier_indices,
        )

    outlier_indices = sorted(set(quality_outlier_indices + robust_outlier_indices))
    return ZScanResult(
        points=all_points,
        full_angle_x_mrad=fit_x.full_angle_mrad,
        full_angle_y_mrad=fit_y.full_angle_mrad,
        half_angle_x_mrad=fit_x.half_angle_mrad,
        half_angle_y_mrad=fit_y.half_angle_mrad,
        waist_z_x_mm=fit_x.waist_z_mm,
        waist_z_y_mm=fit_y.waist_z_mm,
        waist_diameter_x_um=fit_x.waist_diameter_um,
        waist_diameter_y_um=fit_y.waist_diameter_um,
        fit_r2_x=fit_x.fit_r2,
        fit_r2_y=fit_y.fit_r2,
        valid_points_count=len(fit_indices),
        judgement="unknown",
        invalid_reason=None,
        fit_a_x_um2_per_mm2=fit_x.fit_a_um2_per_mm2,
        fit_b_x_um2_per_mm=fit_x.fit_b_um2_per_mm,
        fit_c_x_um2=fit_x.fit_c_um2,
        fit_a_y_um2_per_mm2=fit_y.fit_a_um2_per_mm2,
        fit_b_y_um2_per_mm=fit_y.fit_b_um2_per_mm,
        fit_c_y_um2=fit_y.fit_c_um2,
        outlier_indices=outlier_indices,
        fit_diagnostics=_fit_diagnostics(
            points=all_points,
            fit_indices=fit_indices,
            outlier_indices=outlier_indices,
            fit_x=fit_x,
            fit_y=fit_y,
        ),
    )


def _fit_axes_robust(
    *,
    candidate_indices: list[int],
    z_actual_mm: np.ndarray,
    diameter_x_um: np.ndarray,
    diameter_y_um: np.ndarray,
    uncertainty_x_um: np.ndarray | None,
    uncertainty_y_um: np.ndarray | None,
) -> tuple[list[int], list[int], _AxisFit, _AxisFit]:
    local_fit_indices = _select_robust_local_indices(
        z_actual_mm, diameter_x_um, diameter_y_um
    )
    if len(local_fit_indices) < 3:
        raise ValueError("robust z-scan fit left fewer than three points.")

    local_array = np.asarray(local_fit_indices, dtype=np.int64)
    fit_x = _fit_axis(
        z_actual_mm[local_array],
        diameter_x_um[local_array],
        None if uncertainty_x_um is None else uncertainty_x_um[local_array],
    )
    fit_y = _fit_axis(
        z_actual_mm[local_array],
        diameter_y_um[local_array],
        None if uncertainty_y_um is None else uncertainty_y_um[local_array],
    )
    fit_indices = [candidate_indices[index] for index in local_fit_indices]
    included = set(local_fit_indices)
    robust_outlier_indices = [
        candidate_indices[index]
        for index in range(len(candidate_indices))
        if index not in included
    ]
    return fit_indices, robust_outlier_indices, fit_x, fit_y


def _select_robust_local_indices(
    z_actual_mm: np.ndarray,
    diameter_x_um: np.ndarray,
    diameter_y_um: np.ndarray,
) -> list[int]:
    point_count = len(z_actual_mm)
    all_indices = list(range(point_count))
    if point_count <= 3:
        return all_indices

    best_inliers = all_indices
    best_key: tuple[int, float] | None = None
    for subset in combinations(all_indices, 3):
        subset_array = np.asarray(subset, dtype=np.int64)
        try:
            fit_x = _quadratic_initial(z_actual_mm[subset_array], diameter_x_um[subset_array])
            fit_y = _quadratic_initial(z_actual_mm[subset_array], diameter_y_um[subset_array])
        except ValueError:
            continue
        scores = _combined_relative_residual_scores(
            z_actual_mm=z_actual_mm,
            diameter_x_um=diameter_x_um,
            diameter_y_um=diameter_y_um,
            fit_x=fit_x,
            fit_y=fit_y,
        )
        inliers = [
            index
            for index, score in enumerate(scores)
            if float(score) <= _OUTLIER_RELATIVE_RESIDUAL_LIMIT
        ]
        if len(inliers) < 3:
            continue
        clipped_loss = float(
            np.sum(np.minimum(scores, _OUTLIER_RELATIVE_RESIDUAL_LIMIT) ** 2)
        )
        key = (len(inliers), -clipped_loss)
        if best_key is None or key > best_key:
            best_key = key
            best_inliers = inliers
    return best_inliers


def _fit_axis(
    z_actual_mm: np.ndarray,
    diameter_um: np.ndarray,
    diameter_uncertainty_um: np.ndarray | None,
) -> _AxisFit:
    initial = _quadratic_initial(z_actual_mm, diameter_um)
    z_reference_mm = float(np.median(z_actual_mm))
    centered_z_mm = z_actual_mm - z_reference_mm
    if diameter_uncertainty_um is None:
        residual_scale_um = max(float(np.median(diameter_um)) * 0.01, 1e-3)
        sigma_um = np.full_like(diameter_um, residual_scale_um)
        uncertainty_mode = "relative_scale_1_percent"
        measured_uncertainty = False
    else:
        sigma_um = np.asarray(diameter_uncertainty_um, dtype=np.float64)
        uncertainty_mode = "measured_per_point"
        measured_uncertainty = True

    parameters_initial = np.asarray(
        [
            math.log(initial.waist_diameter_um),
            initial.waist_z_mm - z_reference_mm,
            math.log(initial.slope_um_per_mm),
        ],
        dtype=np.float64,
    )
    span_mm = max(float(np.ptp(centered_z_mm)), 1e-6)
    lower = np.asarray([-np.inf, float(np.min(centered_z_mm)) - span_mm, -np.inf])
    upper = np.asarray([np.inf, float(np.max(centered_z_mm)) + span_mm, np.inf])

    def model(parameters: np.ndarray) -> np.ndarray:
        waist_diameter_um = math.exp(float(parameters[0]))
        waist_centered_mm = float(parameters[1])
        slope_um_per_mm = math.exp(float(parameters[2]))
        return np.sqrt(
            waist_diameter_um**2
            + slope_um_per_mm**2 * (centered_z_mm - waist_centered_mm) ** 2
        )

    def residual(parameters: np.ndarray) -> np.ndarray:
        return (model(parameters) - diameter_um) / sigma_um

    optimized = least_squares(
        residual,
        parameters_initial,
        bounds=(lower, upper),
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=5000,
    )
    if not optimized.success:
        raise RuntimeError(f"caustic fit failed: {optimized.message}")

    waist_diameter_um = math.exp(float(optimized.x[0]))
    waist_z_mm = z_reference_mm + float(optimized.x[1])
    slope_um_per_mm = math.exp(float(optimized.x[2]))
    if not all(
        math.isfinite(value) and value > 0.0
        for value in (waist_diameter_um, slope_um_per_mm)
    ):
        raise ValueError("caustic fit returned non-physical parameters.")

    predicted_um = model(optimized.x)
    residuals_um = diameter_um - predicted_um
    normalized_residuals = residuals_um / sigma_um
    fit_a = slope_um_per_mm**2
    fit_b = -2.0 * fit_a * waist_z_mm
    fit_c = waist_diameter_um**2 + fit_a * waist_z_mm**2
    rmse_um = float(np.sqrt(np.mean(residuals_um**2)))
    normalized_rmse = rmse_um / max(float(np.mean(diameter_um)), 1e-12)
    reduced_chi_square = None
    if measured_uncertainty and len(diameter_um) > 3:
        reduced_chi_square = float(np.sum(normalized_residuals**2) / (len(diameter_um) - 3))
    return _AxisFit(
        fit_a_um2_per_mm2=fit_a,
        fit_b_um2_per_mm=fit_b,
        fit_c_um2=fit_c,
        full_angle_mrad=slope_um_per_mm,
        half_angle_mrad=slope_um_per_mm / 2.0,
        geometric_full_angle_mrad=2000.0 * math.atan(slope_um_per_mm * 1e-3 / 2.0),
        waist_z_mm=waist_z_mm,
        waist_diameter_um=waist_diameter_um,
        rayleigh_range_mm=waist_diameter_um / slope_um_per_mm,
        fit_r2=_r2_score(diameter_um, predicted_um),
        fit_rmse_um=rmse_um,
        normalized_rmse=normalized_rmse,
        reduced_chi_square=reduced_chi_square,
        max_normalized_residual=float(np.max(np.abs(normalized_residuals))),
        uncertainty_mode=uncertainty_mode,
        predicted_diameter_um=predicted_um,
        residuals_um=residuals_um,
        normalized_residuals=normalized_residuals,
        parameter_ci95=_parameter_confidence_intervals(
            optimized.x,
            optimized.jac,
            optimized.fun,
            z_reference_mm=z_reference_mm,
        ),
    )


def _quadratic_initial(z_actual_mm: np.ndarray, diameter_um: np.ndarray) -> _QuadraticInitial:
    if len(z_actual_mm) < 3 or len(set(float(value) for value in z_actual_mm)) < 3:
        raise ValueError("z-scan fitting requires at least three distinct z positions.")
    z_reference_mm = float(np.median(z_actual_mm))
    centered_z_mm = z_actual_mm - z_reference_mm
    design_matrix = np.column_stack(
        [centered_z_mm**2, centered_z_mm, np.ones_like(centered_z_mm)]
    )
    coefficients, _, rank, _ = np.linalg.lstsq(
        design_matrix, diameter_um**2, rcond=None
    )
    if rank < 3:
        raise ValueError("z-scan quadratic initialization is rank deficient.")
    a_centered, b_centered, c_centered = (float(value) for value in coefficients)
    if a_centered <= 0.0:
        raise ValueError("z-scan quadratic coefficient a must be positive.")
    waist_centered_mm = -b_centered / (2.0 * a_centered)
    waist_squared_um2 = c_centered - b_centered**2 / (4.0 * a_centered)
    if waist_squared_um2 <= 0.0:
        raise ValueError("z-scan waist diameter squared must be positive.")
    waist_z_mm = z_reference_mm + waist_centered_mm
    fit_a = a_centered
    fit_b = -2.0 * fit_a * waist_z_mm
    fit_c = waist_squared_um2 + fit_a * waist_z_mm**2
    return _QuadraticInitial(
        fit_a_um2_per_mm2=fit_a,
        fit_b_um2_per_mm=fit_b,
        fit_c_um2=fit_c,
        waist_z_mm=waist_z_mm,
        waist_diameter_um=math.sqrt(waist_squared_um2),
        slope_um_per_mm=math.sqrt(a_centered),
    )


def _combined_relative_residual_scores(
    *,
    z_actual_mm: np.ndarray,
    diameter_x_um: np.ndarray,
    diameter_y_um: np.ndarray,
    fit_x: _QuadraticInitial,
    fit_y: _QuadraticInitial,
) -> np.ndarray:
    predicted_x = _predict_initial_diameter(z_actual_mm, fit_x)
    predicted_y = _predict_initial_diameter(z_actual_mm, fit_y)
    relative_x = np.abs(diameter_x_um - predicted_x) / np.maximum(diameter_x_um, 1e-12)
    relative_y = np.abs(diameter_y_um - predicted_y) / np.maximum(diameter_y_um, 1e-12)
    return np.maximum(relative_x, relative_y)


def _predict_initial_diameter(
    z_actual_mm: np.ndarray, fit: _QuadraticInitial
) -> np.ndarray:
    return np.sqrt(
        fit.waist_diameter_um**2
        + fit.slope_um_per_mm**2 * (z_actual_mm - fit.waist_z_mm) ** 2
    )


def _fit_diagnostics(
    *,
    points: list[BeamPlaneResult],
    fit_indices: list[int],
    outlier_indices: list[int],
    fit_x: _AxisFit,
    fit_y: _AxisFit,
) -> dict[str, Any]:
    included = set(fit_indices)
    outliers = set(outlier_indices)
    point_rows = []
    for index, point in enumerate(points):
        z_mm = float(point.z_actual_mm)
        predicted_x = _predict_axis_diameter(z_mm, fit_x)
        predicted_y = _predict_axis_diameter(z_mm, fit_y)
        measured_x = _finite_or_none(point.d4sigma_x_um)
        measured_y = _finite_or_none(point.d4sigma_y_um)
        residual_x = None if measured_x is None else measured_x - predicted_x
        residual_y = None if measured_y is None else measured_y - predicted_y
        point_rows.append(
            {
                "index": index,
                "z_actual_mm": z_mm,
                "fit_included": index in included,
                "fit_outlier": index in outliers,
                "predicted_x_um": predicted_x,
                "predicted_y_um": predicted_y,
                "residual_x_um": residual_x,
                "residual_y_um": residual_y,
                "normalized_residual_x": _point_normalized_residual(
                    residual_x, measured_x, point.width_uncertainty_x_um
                ),
                "normalized_residual_y": _point_normalized_residual(
                    residual_y, measured_y, point.width_uncertainty_y_um
                ),
            }
        )

    x_diagnostics = _axis_fit_diagnostics(points, fit_indices, fit_x, axis_name="x")
    y_diagnostics = _axis_fit_diagnostics(points, fit_indices, fit_y, axis_name="y")
    confidence_reasons = []
    for axis_name, axis_values in (("x", x_diagnostics), ("y", y_diagnostics)):
        if not bool(axis_values["waist_bracketed"]):
            confidence_reasons.append(f"waist_not_bracketed_{axis_name}")
        if float(axis_values["z_coverage_in_rayleigh_ranges"]) < _MIN_CONFIDENT_RAYLEIGH_COVERAGE:
            confidence_reasons.append(f"insufficient_rayleigh_coverage_{axis_name}")
    if len(fit_indices) < _MIN_CONFIDENT_POINTS:
        confidence_reasons.append("fewer_than_five_fit_points")
    return {
        "model": "D(z)=sqrt(D0^2+S^2*(z-z0)^2)",
        "fit_space": "diameter_um",
        "initializer": "centered_quadratic_D_squared",
        "optimizer": "scipy_least_squares_soft_l1",
        "axis_semantics": "fixed_lab_x_y",
        "paraxial_angle_identity": "1_um_per_mm_equals_1_mrad",
        "confidence_status": "ok" if not confidence_reasons else "low_confidence",
        "confidence_reasons": confidence_reasons,
        "x": x_diagnostics,
        "y": y_diagnostics,
        "points": point_rows,
    }


def _axis_fit_diagnostics(
    points: list[BeamPlaneResult],
    fit_indices: list[int],
    fit: _AxisFit,
    *,
    axis_name: str,
) -> dict[str, Any]:
    z_values = np.asarray([points[index].z_actual_mm for index in fit_indices], dtype=np.float64)
    offsets = z_values - fit.waist_z_mm
    rayleigh = max(fit.rayleigh_range_mm, 1e-12)
    far_mask = np.abs(offsets) >= 2.0 * rayleigh
    return {
        "axis": axis_name,
        "full_angle_mrad": fit.full_angle_mrad,
        "half_angle_mrad": fit.half_angle_mrad,
        "geometric_full_angle_mrad": fit.geometric_full_angle_mrad,
        "geometric_full_angle_deg": math.degrees(fit.geometric_full_angle_mrad / 1000.0),
        "waist_z_mm": fit.waist_z_mm,
        "waist_diameter_um": fit.waist_diameter_um,
        "rayleigh_range_mm": fit.rayleigh_range_mm,
        "fit_r2": fit.fit_r2,
        "fit_rmse_um": fit.fit_rmse_um,
        "normalized_rmse": fit.normalized_rmse,
        "reduced_chi_square": fit.reduced_chi_square,
        "max_normalized_residual": fit.max_normalized_residual,
        "uncertainty_mode": fit.uncertainty_mode,
        "parameter_ci95": fit.parameter_ci95,
        "waist_bracketed": bool(np.min(z_values) <= fit.waist_z_mm <= np.max(z_values)),
        "near_waist_points_count": int(np.count_nonzero(np.abs(offsets) <= rayleigh)),
        "far_field_left_points_count": int(np.count_nonzero(far_mask & (offsets < 0.0))),
        "far_field_right_points_count": int(np.count_nonzero(far_mask & (offsets > 0.0))),
        "z_coverage_in_rayleigh_ranges": float(np.max(np.abs(offsets)) / rayleigh),
    }


def _parameter_confidence_intervals(
    parameters: np.ndarray,
    jacobian: np.ndarray,
    residuals: np.ndarray,
    *,
    z_reference_mm: float,
) -> dict[str, list[float] | None]:
    empty: dict[str, list[float] | None] = {
        "waist_diameter_um": None,
        "waist_z_mm": None,
        "slope_um_per_mm": None,
    }
    degrees_of_freedom = len(residuals) - len(parameters)
    if degrees_of_freedom <= 0 or jacobian.shape[0] < jacobian.shape[1]:
        return empty
    try:
        covariance = np.linalg.pinv(jacobian.T @ jacobian)
        covariance *= float(np.sum(residuals**2) / degrees_of_freedom)
        standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    except np.linalg.LinAlgError:
        return empty
    if not np.all(np.isfinite(standard_errors)):
        return empty
    factor = 1.96
    log_diameter, centered_waist, log_slope = (float(value) for value in parameters)
    return {
        "waist_diameter_um": [
            math.exp(log_diameter - factor * float(standard_errors[0])),
            math.exp(log_diameter + factor * float(standard_errors[0])),
        ],
        "waist_z_mm": [
            z_reference_mm + centered_waist - factor * float(standard_errors[1]),
            z_reference_mm + centered_waist + factor * float(standard_errors[1]),
        ],
        "slope_um_per_mm": [
            math.exp(log_slope - factor * float(standard_errors[2])),
            math.exp(log_slope + factor * float(standard_errors[2])),
        ],
    }


def _axis_uncertainties(
    points: Sequence[BeamPlaneResult], axis_name: str
) -> np.ndarray | None:
    values = [getattr(point, f"width_uncertainty_{axis_name}_um", None) for point in points]
    if any(value is None for value in values):
        return None
    result = np.asarray(values, dtype=np.float64)
    if np.any(~np.isfinite(result)) or np.any(result <= 0.0):
        return None
    return result


def _point_normalized_residual(
    residual_um: float | None,
    measured_um: float | None,
    uncertainty_um: float | None,
) -> float | None:
    if residual_um is None or measured_um is None:
        return None
    scale = uncertainty_um
    if scale is None or not math.isfinite(scale) or scale <= 0.0:
        scale = max(abs(measured_um) * 0.01, 1e-3)
    return residual_um / scale


def _predict_axis_diameter(z_actual_mm: float, fit: _AxisFit) -> float:
    return math.sqrt(
        fit.waist_diameter_um**2
        + fit.full_angle_mrad**2 * (z_actual_mm - fit.waist_z_mm) ** 2
    )


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    residual_sum_squares = float(np.sum((y_true - y_pred) ** 2))
    total_sum_squares = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
    if total_sum_squares <= 0.0:
        return 1.0 if residual_sum_squares <= 1e-18 else 0.0
    return min(max(1.0 - residual_sum_squares / total_sum_squares, 0.0), 1.0)


def _can_fit_point(point: BeamPlaneResult) -> bool:
    return (
        point.valid
        and not point.outlier
        and _is_positive_finite(point.d4sigma_x_um)
        and _is_positive_finite(point.d4sigma_y_um)
    )


def _is_positive_finite(value: float) -> bool:
    return math.isfinite(value) and value > 0.0


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def _invalid_zscan_result(
    points: list[BeamPlaneResult],
    valid_points_count: int,
    invalid_reason: str,
    outlier_indices: list[int],
) -> ZScanResult:
    return ZScanResult(
        points=points,
        full_angle_x_mrad=None,
        full_angle_y_mrad=None,
        half_angle_x_mrad=None,
        half_angle_y_mrad=None,
        waist_z_x_mm=None,
        waist_z_y_mm=None,
        waist_diameter_x_um=None,
        waist_diameter_y_um=None,
        fit_r2_x=None,
        fit_r2_y=None,
        valid_points_count=valid_points_count,
        judgement="invalid",
        invalid_reason=invalid_reason,
        outlier_indices=sorted(set(outlier_indices)),
        fit_diagnostics={
            "model": "D(z)=sqrt(D0^2+S^2*(z-z0)^2)",
            "fit_space": "diameter_um",
            "confidence_status": "invalid",
            "confidence_reasons": [invalid_reason],
            "points": [],
        },
    )


__all__ = ["fit_zscan"]
