"""Acquisition-side image validity checks."""

from __future__ import annotations

from typing import Any

import numpy as np
from lbqa_contracts.errors import ErrorCode, LBQAError
from lbqa_contracts.models import FrameQuality, ImageFrame


def evaluate_frame_validity(
    image: np.ndarray[Any, Any],
    *,
    bit_depth: int,
    expected_width_px: int | None = None,
    expected_height_px: int | None = None,
    min_width_px: int = 1,
    min_height_px: int = 1,
    saturation_threshold_percent: float = 99.0,
    low_signal_threshold_percent: float = 1.0,
    edge_margin_px: int = 4,
    edge_energy_limit_percent: float = 5.0,
    spot_edge_margin_px: int = 4,
    spot_threshold_percent: float = 5.0,
) -> FrameQuality:
    _validate_thresholds(
        bit_depth=bit_depth,
        saturation_threshold_percent=saturation_threshold_percent,
        low_signal_threshold_percent=low_signal_threshold_percent,
        edge_margin_px=edge_margin_px,
        edge_energy_limit_percent=edge_energy_limit_percent,
        spot_edge_margin_px=spot_edge_margin_px,
        spot_threshold_percent=spot_threshold_percent,
    )
    image_array = np.asarray(image)
    image_shape_reason = _image_shape_reason(
        image_array=image_array,
        expected_width_px=expected_width_px,
        expected_height_px=expected_height_px,
        min_width_px=min_width_px,
        min_height_px=min_height_px,
    )
    if image_shape_reason is not None:
        return FrameQuality(
            saturated=False,
            saturation_pixels=0,
            peak_value=0.0,
            mean_value=0.0,
            edge_energy_percent=100.0,
            low_signal=True,
            valid=False,
            invalid_reason=ErrorCode.E_IMAGE_ROI_NOT_FOUND.value,
        )

    image_float = image_array.astype(np.float64, copy=False)
    max_count = float((1 << bit_depth) - 1)
    saturation_threshold_value = max_count * saturation_threshold_percent / 100.0
    saturation_pixels = int(np.count_nonzero(image_float >= saturation_threshold_value))
    peak_value = float(np.max(image_float))
    mean_value = float(np.mean(image_float))

    signal = image_float - float(np.min(image_float))
    np.maximum(signal, 0.0, out=signal)
    signal_peak = float(np.max(signal))
    low_signal = signal_peak <= max_count * low_signal_threshold_percent / 100.0
    edge_energy_percent = _edge_energy_percent(signal, edge_margin_px)
    spot_near_edge = _spot_near_edge(
        signal,
        threshold_percent=spot_threshold_percent,
        margin_px=spot_edge_margin_px,
    )

    invalid_reason = None
    if saturation_pixels > 0:
        invalid_reason = ErrorCode.E_IMAGE_SATURATED.value
    elif low_signal:
        invalid_reason = ErrorCode.E_IMAGE_LOW_SIGNAL.value
    elif edge_energy_percent > edge_energy_limit_percent or spot_near_edge:
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


def validate_frame(frame: ImageFrame, *, bit_depth: int, **kwargs: Any) -> FrameQuality:
    return evaluate_frame_validity(frame.image, bit_depth=bit_depth, **kwargs)


def invalid_quality_like(quality: FrameQuality, invalid_reason: ErrorCode) -> FrameQuality:
    return FrameQuality(
        saturated=quality.saturated,
        saturation_pixels=quality.saturation_pixels,
        peak_value=quality.peak_value,
        mean_value=quality.mean_value,
        edge_energy_percent=quality.edge_energy_percent,
        low_signal=quality.low_signal,
        valid=False,
        invalid_reason=invalid_reason.value,
    )


def _image_shape_reason(
    *,
    image_array: np.ndarray[Any, Any],
    expected_width_px: int | None,
    expected_height_px: int | None,
    min_width_px: int,
    min_height_px: int,
) -> str | None:
    if image_array.ndim != 2:
        return "image must be 2D"
    height_px, width_px = image_array.shape
    if width_px < min_width_px or height_px < min_height_px:
        return "image is smaller than minimum dimensions"
    if expected_width_px is not None and width_px != expected_width_px:
        return "image width_px does not match expected_width_px"
    if expected_height_px is not None and height_px != expected_height_px:
        return "image height_px does not match expected_height_px"
    return None


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
    return 100.0 * float(np.sum(image[edge_mask])) / total_energy


def _spot_near_edge(
    image: np.ndarray[Any, Any],
    *,
    threshold_percent: float,
    margin_px: int,
) -> bool:
    peak_value = float(np.max(image))
    if peak_value <= 0.0 or margin_px <= 0:
        return False
    mask = image >= peak_value * threshold_percent / 100.0
    if not np.any(mask):
        return False
    y_indices, x_indices = np.nonzero(mask)
    height_px, width_px = image.shape
    return (
        int(np.min(x_indices)) < margin_px
        or int(np.min(y_indices)) < margin_px
        or int(np.max(x_indices)) >= width_px - margin_px
        or int(np.max(y_indices)) >= height_px - margin_px
    )


def _validate_thresholds(
    *,
    bit_depth: int,
    saturation_threshold_percent: float,
    low_signal_threshold_percent: float,
    edge_margin_px: int,
    edge_energy_limit_percent: float,
    spot_edge_margin_px: int,
    spot_threshold_percent: float,
) -> None:
    if bit_depth < 1 or bit_depth > 16:
        raise LBQAError(ErrorCode.E_IMAGE_ROI_NOT_FOUND, "bit_depth must be in 1..16.")
    for field_name, value in (
        ("saturation_threshold_percent", saturation_threshold_percent),
        ("low_signal_threshold_percent", low_signal_threshold_percent),
        ("edge_energy_limit_percent", edge_energy_limit_percent),
        ("spot_threshold_percent", spot_threshold_percent),
    ):
        if not 0.0 <= value <= 100.0:
            raise ValueError(f"{field_name} must be between 0 and 100.")
    if edge_margin_px < 0 or spot_edge_margin_px < 0:
        raise ValueError("edge margins must be non-negative.")


__all__ = ["evaluate_frame_validity", "invalid_quality_like", "validate_frame"]
