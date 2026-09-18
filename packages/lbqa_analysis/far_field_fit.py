"""Independent, unweighted diameter-versus-z line fits and advisory tail selection."""

from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Integral
from typing import Any

import numpy as np
from lbqa_contracts.errors import ErrorCode
from lbqa_contracts.models import BeamPlaneResult, ZScanResult


def fit_far_field(
    points: Sequence[BeamPlaneResult],
    *,
    selected_indices_x: Sequence[int] | None = None,
    selected_indices_y: Sequence[int] | None = None,
) -> ZScanResult:
    """Fit selected continuous ellipse X/Y diameters without residual rejection.

    None selects every quality-valid, finite, positive measurement for that axis;
    an empty selection selects none. Legacy ``point.outlier`` flags do not veto
    selections. Inputs already need continuous ellipse X/Y semantics: this
    function does not relabel laboratory widths or track ellipse identities.

    ``fit_indices`` and inclusion flags describe retained inputs even when an
    axis cannot be estimated. The result's count is the union of those inputs.
    Invalid index values invalidate only their axis. Residual scatter is not an
    angle uncertainty, and a successful line fit does not establish far-field
    coverage or locate a waist.
    """
    all_points = list(points)
    axes = {
        "x": _fit_selected_axis(all_points, "x", selected_indices_x),
        "y": _fit_selected_axis(all_points, "y", selected_indices_y),
    }
    included = {axis: set(fit["fit_indices"]) for axis, fit in axes.items()}
    quality_indices = [
        index
        for index, point in enumerate(all_points)
        if not all(_can_fit_axis(point, axis) for axis in axes)
    ]
    quality_set = set(quality_indices)
    rows = []
    for index, point in enumerate(all_points):
        row: dict[str, Any] = {
            "index": index,
            "z_actual_mm": _finite_or_none(point.z_actual_mm),
            "fit_included": index in included["x"] or index in included["y"],
            "fit_outlier": index in quality_set,
        }
        for axis, fit in axes.items():
            prediction = _predict(point.z_actual_mm, fit)
            measured = _finite_or_none(getattr(point, f"d4sigma_{axis}_um"))
            row.update(
                {
                    f"fit_included_{axis}": index in included[axis],
                    f"quality_invalid_{axis}": not _can_fit_axis(point, axis),
                    f"predicted_{axis}_um": prediction,
                    f"residual_{axis}_um": (
                        None
                        if measured is None or prediction is None
                        else _finite_or_none(measured - prediction)
                    ),
                }
            )
        rows.append(row)

    statuses = [fit["status"] for fit in axes.values()]
    invalid_reason = None
    if all(status == "invalid" for status in statuses):
        judgement = "invalid"
        invalid_reason = (
            ErrorCode.E_ANALYSIS_ZSCAN_TOO_FEW_POINTS.value
            if all("too_few_points" in fit["reasons"] for fit in axes.values())
            else ErrorCode.E_ANALYSIS_ZSCAN_FIT_FAILED.value
        )
    else:
        judgement = "unknown" if all(status == "ok" for status in statuses) else "warning"
    return ZScanResult(
        points=all_points,
        full_angle_x_mrad=axes["x"]["full_angle_mrad"],
        full_angle_y_mrad=axes["y"]["full_angle_mrad"],
        half_angle_x_mrad=axes["x"]["half_angle_mrad"],
        half_angle_y_mrad=axes["y"]["half_angle_mrad"],
        waist_z_x_mm=None,
        waist_z_y_mm=None,
        waist_diameter_x_um=None,
        waist_diameter_y_um=None,
        fit_r2_x=axes["x"]["fit_r2"],
        fit_r2_y=axes["y"]["fit_r2"],
        valid_points_count=len(included["x"] | included["y"]),
        judgement=judgement,
        invalid_reason=invalid_reason,
        outlier_indices=quality_indices,
        fit_diagnostics={
            "model": "far_field_linear",
            "axis_semantics": "continuous_ellipse_x_y",
            "fit_space": "diameter_um",
            "optimizer": "centered_ordinary_least_squares",
            "uncertainty_mode": "none",
            "note": (
                "A line fit does not establish far-field coverage or locate a waist. "
                "Residual standard deviation is diameter scatter, not angle uncertainty."
            ),
            **axes,
            "points": rows,
        },
    )


def _fit_selected_axis(
    points: list[BeamPlaneResult], axis: str, selection: Sequence[int] | None
) -> dict[str, Any]:
    requested = range(len(points)) if selection is None else selection
    indices = set()
    bad_indices = False
    for index in requested:
        if (
            isinstance(index, bool)
            or not isinstance(index, Integral)
            or not 0 <= index < len(points)
        ):
            bad_indices = True
        else:
            indices.add(int(index))
    fit_indices = [index for index in sorted(indices) if _can_fit_axis(points[index], axis)]
    fit: dict[str, Any] = {
        "fit_indices": fit_indices,
        "slope_um_per_mm": None,
        "intercept_um": None,
        "z_reference_mm": None,
        "intercept_at_reference_um": None,
        "full_angle_mrad": None,
        "half_angle_mrad": None,
        "fit_r2": None,
        "fit_rmse_um": None,
        "residual_std_um": None,
        "valid_points_count": len(fit_indices),
        "status": "invalid",
        "reasons": [],
    }
    reasons = fit["reasons"]
    if bad_indices:
        reasons.append("invalid_selection_indices")
    if len(fit_indices) < 2:
        reasons.append("too_few_points")
    elif len({points[index].z_actual_mm for index in fit_indices}) < 2:
        reasons.append("fewer_than_two_distinct_z")
    if reasons:
        return fit
    z_mm = np.asarray([points[index].z_actual_mm for index in fit_indices], dtype=float)
    widths_um = np.asarray(
        [getattr(points[index], f"d4sigma_{axis}_um") for index in fit_indices], dtype=float
    )
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            estimates = _linear_estimates(z_mm, widths_um)
    except (FloatingPointError, OverflowError, ValueError, np.linalg.LinAlgError):
        reasons.append("numerical_fit_failed")
        return fit
    fit.update(estimates)
    if fit["slope_um_per_mm"] == 0.0:
        reasons.append("zero_slope")
    elif fit["slope_um_per_mm"] < 0.0:
        reasons.append("negative_slope")
    if len(fit_indices) == 2:
        reasons.append("residual_std_unavailable_two_points")
    fit["status"] = "warning" if reasons else "ok"
    return fit


def _linear_estimates(z_mm: np.ndarray, widths_um: np.ndarray) -> dict[str, Any]:
    # Center both quantities before solving; scale z so a tiny physical span
    # does not become numerically rank deficient beside the constant column.
    z_reference_mm = float(z_mm[0] + np.mean(z_mm - z_mm[0]))
    centered_z_mm = z_mm - z_reference_mm
    z_scale_mm = float(np.max(np.abs(centered_z_mm)))
    scaled_z = centered_z_mm / z_scale_mm
    width_reference_um = float(widths_um[0] + np.mean(widths_um - widths_um[0]))
    centered_widths_um = widths_um - width_reference_um
    design = np.column_stack((scaled_z, np.ones_like(scaled_z)))
    coefficients, _, rank, _ = np.linalg.lstsq(design, centered_widths_um, rcond=None)
    if rank < 2:
        raise ValueError("Line fit is rank deficient.")
    slope = float(coefficients[0]) / z_scale_mm
    intercept_at_reference_um = width_reference_um + float(coefficients[1])
    intercept_um = intercept_at_reference_um - slope * z_reference_mm
    residuals_um = centered_widths_um - design @ coefficients
    sse = float(np.sum(residuals_um**2))
    sst = float(np.sum((centered_widths_um - np.mean(centered_widths_um)) ** 2))
    full_angle_mrad = 2000.0 * math.atan(abs(slope) / 2000.0)
    estimates = {
        "slope_um_per_mm": slope,
        "intercept_um": intercept_um,
        "z_reference_mm": z_reference_mm,
        "intercept_at_reference_um": intercept_at_reference_um,
        "full_angle_mrad": full_angle_mrad,
        "half_angle_mrad": full_angle_mrad / 2.0,
        "fit_r2": None if sst == 0.0 else min(1.0, max(0.0, 1.0 - sse / sst)),
        "fit_rmse_um": math.sqrt(sse / len(z_mm)),
        "residual_std_um": math.sqrt(sse / (len(z_mm) - 2)) if len(z_mm) > 2 else None,
    }
    if any(value is not None and not math.isfinite(value) for value in estimates.values()):
        raise ValueError("Line fit produced a non-finite estimate.")
    return estimates


def _can_fit_axis(point: BeamPlaneResult, axis: str) -> bool:
    width = getattr(point, f"d4sigma_{axis}_um")
    return point.valid and math.isfinite(point.z_actual_mm) and math.isfinite(width) and width > 0.0


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def _predict(z_mm: float, fit: dict[str, Any]) -> float | None:
    if fit["slope_um_per_mm"] is None or not math.isfinite(z_mm):
        return None
    return _finite_or_none(
        fit["intercept_at_reference_um"] + fit["slope_um_per_mm"] * (z_mm - fit["z_reference_mm"])
    )


def suggest_far_field_indices(points: Sequence[BeamPlaneResult]) -> dict[str, Any]:
    """Suggest, but never certify, increasing-z tails after observed minima.

    Use the final half of distinct post-minimum z positions, retaining at least
    three when available and every valid replicate at a selected position. No
    R2 maximization or residual rejection is performed. Two-position tails and
    inclusion of the observed minimum are explicitly labelled as fallbacks.
    ``common`` is the X/Y intersection for callers needing one default mask;
    this function never applies that mask or changes the points.
    """
    all_points = list(points)
    x, note_x = _suggest_axis(all_points, "x")
    y, note_y = _suggest_axis(all_points, "y")
    y_indices = set(y)
    common = [index for index in x if index in y_indices]
    common_note = "Common is the intersection of the axis suggestions."
    if len({all_points[index].z_actual_mm for index in common}) < 2:
        common_note += " Insufficient common distinct z positions for a line fit."
    elif len(common) == 2:
        common_note += " Only two common points; residual scatter cannot be estimated."
    return {
        "x": x,
        "y": y,
        "common": common,
        "note": (
            "Suggestion only; this never guarantees far-field coverage. "
            f"X: {note_x} Y: {note_y} {common_note}"
        ),
    }


def _suggest_axis(points: list[BeamPlaneResult], axis: str) -> tuple[list[int], str]:
    ordered = sorted(
        (index for index, point in enumerate(points) if _can_fit_axis(point, axis)),
        key=lambda index: (points[index].z_actual_mm, index),
    )
    if len({points[index].z_actual_mm for index in ordered}) < 2:
        return [], "Insufficient valid distinct z positions."
    minimum = min(ordered, key=lambda index: getattr(points[index], f"d4sigma_{axis}_um"))
    minimum_z = points[minimum].z_actual_mm
    candidates = [index for index in ordered if points[index].z_actual_mm > minimum_z]
    positions = sorted({points[index].z_actual_mm for index in candidates})
    notes = []
    if not positions:
        return [], "Insufficient post-observed-minimum data; the minimum is at the last z."
    if len(positions) == 1:
        candidates = [index for index in ordered if points[index].z_actual_mm >= minimum_z]
        positions = sorted({points[index].z_actual_mm for index in candidates})
        notes.append("Fallback includes the observed minimum; only one later z is available.")
    tail_position_count = min(len(positions), max(3, math.ceil(len(positions) / 2)))
    cutoff = positions[-tail_position_count]
    tail = [index for index in candidates if points[index].z_actual_mm >= cutoff]
    notes.append("Conservative increasing-z suffix; no R2 optimization or residual rejection.")
    if len(tail) == 2:
        notes.append("Only two points; residual scatter cannot be estimated.")
    if tail_position_count < 4:
        notes.append("Fallback: too few distinct z positions to assess slope stability.")
    else:
        tail_positions = positions[-tail_position_count:]
        midpoint = tail_positions[len(tail_positions) // 2]
        first = [index for index in tail if points[index].z_actual_mm < midpoint]
        last = [index for index in tail if points[index].z_actual_mm >= midpoint]
        slopes = [
            _fit_selected_axis(points, axis, subset)["slope_um_per_mm"] for subset in (first, last)
        ]
        if any(slope is None or slope <= 0.0 for slope in slopes):
            notes.append("Fallback: positive slope stability is not established.")
        elif abs(slopes[0] - slopes[1]) > 0.25 * max(slopes):
            notes.append("Fallback: half-tail slopes differ by more than 25%; unstable trend.")
        else:
            notes.append("Positive half-tail slopes agree within a 25% advisory tolerance.")
    tail_fit = _fit_selected_axis(points, axis, tail)
    slope = tail_fit["slope_um_per_mm"]
    if slope is None or slope <= 0.0:
        notes.append("Warning: selected tail has no estimable positive slope.")
    return tail, " ".join(notes)


__all__ = ["fit_far_field", "suggest_far_field_indices"]
