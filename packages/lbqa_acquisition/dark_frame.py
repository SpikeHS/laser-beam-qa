"""Dark-frame acquisition and matching."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from lbqa_contracts.errors import ErrorCode, LBQAError
from lbqa_contracts.models import ImageFrame
from lbqa_contracts.ports import BeamProfilerPort

from lbqa_acquisition.frame_capture import capture_average
from lbqa_acquisition.roi import RegionPx, current_aoi, frame_aoi


@dataclass(frozen=True, slots=True)
class DarkFrame:
    image: np.ndarray[Any, Any] = field(repr=False)
    timestamp_iso: str
    exposure_us: float
    gain: float
    aoi: RegionPx
    frame_count: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.exposure_us <= 0.0:
            raise ValueError("exposure_us must be positive.")
        if self.gain < 0.0:
            raise ValueError("gain must be non-negative.")
        if self.frame_count <= 0:
            raise ValueError("frame_count must be positive.")
        image = np.asarray(self.image)
        if image.ndim != 2 or image.shape != (self.aoi.height_px, self.aoi.width_px):
            raise ValueError("dark frame image shape must match AOI height_px/width_px.")


def acquire_dark_frame(profiler: BeamProfilerPort, frame_count: int) -> DarkFrame:
    averaged = capture_average(profiler, frame_count)
    aoi = current_aoi(profiler, averaged)
    metadata: dict[str, Any] = dict(averaged.metadata)
    metadata.update(aoi.metadata("aoi"))
    metadata["dark_frame"] = True
    return DarkFrame(
        image=np.asarray(averaged.image).astype(np.float64, copy=True),
        timestamp_iso=averaged.timestamp_iso,
        exposure_us=averaged.exposure_us,
        gain=averaged.gain,
        aoi=aoi,
        frame_count=frame_count,
        metadata=metadata,
    )


def validate_dark_frame_match(
    dark_frame: DarkFrame,
    frame: ImageFrame,
    *,
    frame_aoi_px: RegionPx | None = None,
    exposure_abs_tol_us: float = 1e-6,
    gain_abs_tol: float = 1e-9,
) -> None:
    candidate_aoi = frame_aoi_px or frame_aoi(frame)
    if not math.isclose(
        dark_frame.exposure_us,
        frame.exposure_us,
        rel_tol=0.0,
        abs_tol=exposure_abs_tol_us,
    ):
        _raise_dark_mismatch(
            "exposure_us mismatch: "
            f"dark={dark_frame.exposure_us}, frame={frame.exposure_us}."
        )
    if not math.isclose(dark_frame.gain, frame.gain, rel_tol=0.0, abs_tol=gain_abs_tol):
        _raise_dark_mismatch(f"gain mismatch: dark={dark_frame.gain}, frame={frame.gain}.")
    if dark_frame.aoi != candidate_aoi:
        _raise_dark_mismatch(f"AOI mismatch: dark={dark_frame.aoi}, frame={candidate_aoi}.")
    if np.asarray(dark_frame.image).shape != np.asarray(frame.image).shape:
        _raise_dark_mismatch(
            "image shape mismatch: "
            f"dark={np.asarray(dark_frame.image).shape}, frame={np.asarray(frame.image).shape}."
        )


def apply_dark_frame(
    frame: ImageFrame,
    dark_frame: DarkFrame,
    *,
    frame_aoi_px: RegionPx | None = None,
) -> ImageFrame:
    validate_dark_frame_match(dark_frame, frame, frame_aoi_px=frame_aoi_px)
    corrected = np.asarray(frame.image).astype(np.float64) - np.asarray(dark_frame.image)
    np.maximum(corrected, 0.0, out=corrected)
    metadata: dict[str, Any] = dict(frame.metadata)
    metadata.update(
        {
            "dark_subtracted": True,
            "dark_timestamp_iso": dark_frame.timestamp_iso,
            "dark_frame_count": dark_frame.frame_count,
        }
    )
    return ImageFrame(
        image=corrected,
        timestamp_iso=frame.timestamp_iso,
        exposure_us=frame.exposure_us,
        gain=frame.gain,
        width_px=frame.width_px,
        height_px=frame.height_px,
        z_actual_mm=frame.z_actual_mm,
        metadata=metadata,
    )


def _raise_dark_mismatch(message: str) -> None:
    raise LBQAError(ErrorCode.E_IMAGE_ROI_NOT_FOUND, f"dark frame mismatch: {message}")


__all__ = [
    "DarkFrame",
    "acquire_dark_frame",
    "apply_dark_frame",
    "validate_dark_frame_match",
]
