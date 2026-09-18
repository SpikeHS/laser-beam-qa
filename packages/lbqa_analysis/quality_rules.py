"""Frame-quality checks and final analysis judgement rules."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from lbqa_contracts.errors import ErrorCode
from lbqa_contracts.models import FrameQuality, ZScanResult

from lbqa_analysis.exceptions import raise_analysis_error
from lbqa_analysis.preprocess import as_float_image

PASS = "PASS"
FAIL = "FAIL"
INVALID = "INVALID"


def evaluate_frame_quality(
    image: np.ndarray[Any, Any],
    bit_depth: int,
    edge_margin_px: int,
    edge_energy_limit_percent: float,
    saturation_threshold_percent: float,
    low_signal_threshold_percent: float = 1.0,
) -> FrameQuality:
    """Evaluate saturation, low signal, and edge-clipping risk for one frame."""

    _validate_quality_limits(
        bit_depth=bit_depth,
        edge_margin_px=edge_margin_px,
        edge_energy_limit_percent=edge_energy_limit_percent,
        saturation_threshold_percent=saturation_threshold_percent,
        low_signal_threshold_percent=low_signal_threshold_percent,
    )
    image_float = as_float_image(image)
    max_count = float((1 << bit_depth) - 1)
    saturation_threshold_value = max_count * saturation_threshold_percent / 100.0
    saturation_pixels = int(np.count_nonzero(image_float >= saturation_threshold_value))
    peak_value = float(np.max(image_float))
    mean_value = float(np.mean(image_float))

    signal_image = image_float - float(np.min(image_float))
    np.maximum(signal_image, 0.0, out=signal_image)
    signal_peak = float(np.max(signal_image))
    low_signal = signal_peak <= max_count * low_signal_threshold_percent / 100.0
    edge_energy_percent = _edge_energy_percent(signal_image, edge_margin_px)

    invalid_reason = None
    if saturation_pixels > 0:
        invalid_reason = ErrorCode.E_IMAGE_SATURATED.value
    elif low_signal:
        invalid_reason = ErrorCode.E_IMAGE_LOW_SIGNAL.value
    elif edge_energy_percent > edge_energy_limit_percent:
        invalid_reason = ErrorCode.E_IMAGE_EDGE_CLIPPED.value

    return FrameQuality(
        saturated=saturation_pixels > 0,
        saturation_pixels=saturation_pixels,
        peak_value=peak_value,
        mean_value=mean_value,
        edge_energy_percent=edge_energy_percent,
        low_signal=low_signal,
        valid=invalid_reason is None,
        invalid_reason=invalid_reason,
    )


def judge_result(zscan_result: ZScanResult, limits: Mapping[str, Any] | None = None) -> str:
    """Return PASS, FAIL, or INVALID for a z-scan result and optional limits."""

    if zscan_result.invalid_reason is not None:
        return INVALID
    required_values = (
        zscan_result.full_angle_x_mrad,
        zscan_result.full_angle_y_mrad,
        zscan_result.half_angle_x_mrad,
        zscan_result.half_angle_y_mrad,
        zscan_result.fit_r2_x,
        zscan_result.fit_r2_y,
    )
    if any(value is None for value in required_values):
        return INVALID

    rule_limits = dict(limits or {})
    max_full_angle_x_mrad = _limit(rule_limits, "max_full_angle_x_mrad", "max_full_angle_mrad")
    max_full_angle_y_mrad = _limit(rule_limits, "max_full_angle_y_mrad", "max_full_angle_mrad")
    max_half_angle_x_mrad = _limit(rule_limits, "max_half_angle_x_mrad", "max_half_angle_mrad")
    max_half_angle_y_mrad = _limit(rule_limits, "max_half_angle_y_mrad", "max_half_angle_mrad")
    min_fit_r2_x = _limit(rule_limits, "min_fit_r2_x", "min_fit_r2")
    min_fit_r2_y = _limit(rule_limits, "min_fit_r2_y", "min_fit_r2")
    min_waist_diameter_x_um = _limit(
        rule_limits, "min_waist_diameter_x_um", "min_waist_diameter_um"
    )
    min_waist_diameter_y_um = _limit(
        rule_limits, "min_waist_diameter_y_um", "min_waist_diameter_um"
    )
    max_waist_diameter_x_um = _limit(
        rule_limits, "max_waist_diameter_x_um", "max_waist_diameter_um"
    )
    max_waist_diameter_y_um = _limit(
        rule_limits, "max_waist_diameter_y_um", "max_waist_diameter_um"
    )
    failures: list[bool] = [
        _exceeds(zscan_result.full_angle_x_mrad, max_full_angle_x_mrad),
        _exceeds(zscan_result.full_angle_y_mrad, max_full_angle_y_mrad),
        _exceeds(zscan_result.half_angle_x_mrad, max_half_angle_x_mrad),
        _exceeds(zscan_result.half_angle_y_mrad, max_half_angle_y_mrad),
        _below(zscan_result.fit_r2_x, min_fit_r2_x),
        _below(zscan_result.fit_r2_y, min_fit_r2_y),
        _below(zscan_result.waist_diameter_x_um, min_waist_diameter_x_um),
        _below(zscan_result.waist_diameter_y_um, min_waist_diameter_y_um),
        _exceeds(zscan_result.waist_diameter_x_um, max_waist_diameter_x_um),
        _exceeds(zscan_result.waist_diameter_y_um, max_waist_diameter_y_um),
    ]
    return FAIL if any(failures) else PASS


def _validate_quality_limits(
    bit_depth: int,
    edge_margin_px: int,
    edge_energy_limit_percent: float,
    saturation_threshold_percent: float,
    low_signal_threshold_percent: float,
) -> None:
    if bit_depth < 1 or bit_depth > 16:
        raise_analysis_error(ErrorCode.E_IMAGE_ROI_NOT_FOUND, "bit_depth must be 1..16.")
    if edge_margin_px < 0:
        raise_analysis_error(
            ErrorCode.E_IMAGE_ROI_NOT_FOUND,
            "edge_margin_px must be non-negative.",
        )
    for field_name, value in (
        ("edge_energy_limit_percent", edge_energy_limit_percent),
        ("saturation_threshold_percent", saturation_threshold_percent),
        ("low_signal_threshold_percent", low_signal_threshold_percent),
    ):
        if not 0.0 <= value <= 100.0:
            raise_analysis_error(
                ErrorCode.E_IMAGE_ROI_NOT_FOUND,
                f"{field_name} must be between 0 and 100.",
            )


def _edge_energy_percent(image: np.ndarray[Any, Any], edge_margin_px: int) -> float:
    total_energy = float(np.sum(image))
    if total_energy <= 0.0:
        return 100.0
    if edge_margin_px == 0:
        return 0.0

    height_px, width_px = image.shape
    if edge_margin_px * 2 >= min(height_px, width_px):
        return 100.0

    edge_mask = np.zeros(image.shape, dtype=bool)
    edge_mask[:edge_margin_px, :] = True
    edge_mask[-edge_margin_px:, :] = True
    edge_mask[:, :edge_margin_px] = True
    edge_mask[:, -edge_margin_px:] = True
    edge_energy = float(np.sum(image[edge_mask]))
    return 100.0 * edge_energy / total_energy


def _exceeds(value: float | None, limit: Any) -> bool:
    if value is None or limit is None:
        return False
    return value > float(limit)


def _below(value: float | None, limit: Any) -> bool:
    if value is None or limit is None:
        return False
    return value < float(limit)


def _limit(limits: Mapping[str, Any], axis_key: str, fallback_key: str) -> Any:
    return limits[axis_key] if axis_key in limits else limits.get(fallback_key)


__all__ = ["FAIL", "INVALID", "PASS", "evaluate_frame_quality", "judge_result"]
