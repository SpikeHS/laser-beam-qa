"""Auto-exposure control for beam-profiler acquisition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from lbqa_contracts.errors import ErrorCode, LBQAError
from lbqa_contracts.ports import BeamProfilerPort


@dataclass(frozen=True, slots=True)
class ExposureLimits:
    min_exposure_us: float = 10.0
    max_exposure_us: float = 100_000.0

    def __post_init__(self) -> None:
        if self.min_exposure_us <= 0.0:
            raise ValueError("min_exposure_us must be positive.")
        if self.max_exposure_us < self.min_exposure_us:
            raise ValueError("max_exposure_us must be >= min_exposure_us.")


@dataclass(frozen=True, slots=True)
class AutoExposureResult:
    exposure_us: float
    peak_percent: float
    iterations_count: int
    locked: bool
    saturated: bool


def auto_exposure(
    profiler: BeamProfilerPort,
    target_peak_percent: float,
    max_peak_percent: float,
    *,
    limits: ExposureLimits | None = None,
    max_iterations_count: int = 12,
    tolerance_percent: float = 2.0,
) -> AutoExposureResult:
    _validate_auto_exposure_inputs(
        target_peak_percent=target_peak_percent,
        max_peak_percent=max_peak_percent,
        max_iterations_count=max_iterations_count,
        tolerance_percent=tolerance_percent,
    )
    exposure_limits = limits or ExposureLimits()
    bit_depth = profiler.get_camera_info().bit_depth
    exposure_us = _clamp(_status_exposure_us(profiler), exposure_limits)
    last_peak_percent = 0.0
    last_saturated = False

    for iteration_count in range(1, max_iterations_count + 1):
        profiler.set_exposure_us(exposure_us)
        frame = profiler.capture_single()
        last_peak_percent = _peak_percent(frame.image, bit_depth)
        last_saturated = last_peak_percent >= max_peak_percent

        if (
            not last_saturated
            and abs(last_peak_percent - target_peak_percent) <= tolerance_percent
        ):
            return AutoExposureResult(
                exposure_us=exposure_us,
                peak_percent=last_peak_percent,
                iterations_count=iteration_count,
                locked=True,
                saturated=False,
            )

        if iteration_count == max_iterations_count:
            return _locked_or_raise_at_limit(
                exposure_us=exposure_us,
                peak_percent=last_peak_percent,
                iterations_count=iteration_count,
                target_peak_percent=target_peak_percent,
                max_peak_percent=max_peak_percent,
                saturated=last_saturated,
            )

        next_exposure_us = _next_exposure_us(
            exposure_us=exposure_us,
            peak_percent=last_peak_percent,
            target_peak_percent=target_peak_percent,
            limits=exposure_limits,
        )
        if next_exposure_us == exposure_us:
            return _locked_or_raise_at_limit(
                exposure_us=exposure_us,
                peak_percent=last_peak_percent,
                iterations_count=iteration_count,
                target_peak_percent=target_peak_percent,
                max_peak_percent=max_peak_percent,
                saturated=last_saturated,
            )
        exposure_us = next_exposure_us

    raise AssertionError("unreachable auto exposure loop exit.")


def _next_exposure_us(
    *,
    exposure_us: float,
    peak_percent: float,
    target_peak_percent: float,
    limits: ExposureLimits,
) -> float:
    ratio = 2.0 if peak_percent <= 0.0 else target_peak_percent / peak_percent
    ratio = min(4.0, max(0.25, ratio))
    return _clamp(exposure_us * ratio, limits)


def _locked_or_raise_at_limit(
    *,
    exposure_us: float,
    peak_percent: float,
    iterations_count: int,
    target_peak_percent: float,
    max_peak_percent: float,
    saturated: bool,
) -> AutoExposureResult:
    if saturated or peak_percent > max_peak_percent:
        raise LBQAError(
            ErrorCode.E_IMAGE_SATURATED,
            "auto exposure could not avoid saturation within recipe exposure limits.",
        )
    if peak_percent < target_peak_percent:
        raise LBQAError(
            ErrorCode.E_IMAGE_LOW_SIGNAL,
            "auto exposure could not reach target_peak_percent within recipe limits.",
        )
    return AutoExposureResult(
        exposure_us=exposure_us,
        peak_percent=peak_percent,
        iterations_count=iterations_count,
        locked=True,
        saturated=False,
    )


def _peak_percent(image: np.ndarray[Any, Any], bit_depth: int) -> float:
    max_count = float((1 << bit_depth) - 1)
    return 100.0 * float(np.max(np.asarray(image))) / max_count


def _status_exposure_us(profiler: BeamProfilerPort) -> float:
    status = profiler.get_status()
    exposure_us = float(status.get("exposure_us", 0.0))
    if exposure_us <= 0.0:
        raise LBQAError(ErrorCode.E_DEVICE_CAMERA_TIMEOUT, "profiler status lacks exposure_us.")
    return exposure_us


def _clamp(exposure_us: float, limits: ExposureLimits) -> float:
    return min(limits.max_exposure_us, max(limits.min_exposure_us, float(exposure_us)))


def _validate_auto_exposure_inputs(
    *,
    target_peak_percent: float,
    max_peak_percent: float,
    max_iterations_count: int,
    tolerance_percent: float,
) -> None:
    if not 0.0 < target_peak_percent < max_peak_percent <= 100.0:
        raise ValueError("target_peak_percent must be in (0, max_peak_percent).")
    if max_iterations_count <= 0:
        raise ValueError("max_iterations_count must be positive.")
    if tolerance_percent < 0.0:
        raise ValueError("tolerance_percent must be non-negative.")


__all__ = ["AutoExposureResult", "ExposureLimits", "auto_exposure"]
