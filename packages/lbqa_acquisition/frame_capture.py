"""Multi-frame capture helpers."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import numpy as np
from lbqa_contracts.errors import ErrorCode, LBQAError
from lbqa_contracts.models import ImageFrame
from lbqa_contracts.ports import BeamProfilerPort

from lbqa_acquisition.roi import frame_aoi


def capture_frames(profiler: BeamProfilerPort, frame_count: int) -> list[ImageFrame]:
    if frame_count <= 0:
        raise ValueError("frame_count must be positive.")
    frames = [profiler.capture_single() for _ in range(frame_count)]
    for frame in frames:
        if not frame.timestamp_iso:
            raise LBQAError(ErrorCode.E_DEVICE_CAMERA_TIMEOUT, "captured frame has no timestamp.")
    return frames


def capture_average(profiler: BeamProfilerPort, frame_count: int) -> ImageFrame:
    if frame_count <= 0:
        raise ValueError("frame_count must be positive.")
    native_capture_average = getattr(profiler, "capture_average", None)
    if native_capture_average is not None:
        frame = native_capture_average(frame_count)
        if not frame.timestamp_iso:
            raise LBQAError(ErrorCode.E_DEVICE_CAMERA_TIMEOUT, "captured frame has no timestamp.")
        metadata: dict[str, Any] = dict(frame.metadata)
        metadata.setdefault("averaged", True)
        metadata.setdefault("averaged_frame_count", frame_count)
        return ImageFrame(
            image=frame.image,
            timestamp_iso=frame.timestamp_iso,
            exposure_us=frame.exposure_us,
            gain=frame.gain,
            width_px=frame.width_px,
            height_px=frame.height_px,
            z_actual_mm=frame.z_actual_mm,
            metadata=metadata,
        )

    frames = capture_frames(profiler, frame_count)
    _validate_consistent_frames(frames)
    averaged_image = np.mean(
        [np.asarray(frame.image).astype(np.float64, copy=False) for frame in frames],
        axis=0,
        dtype=np.float64,
    )
    first_frame = frames[0]
    last_frame = frames[-1]
    metadata: dict[str, Any] = dict(last_frame.metadata)
    metadata.update(
        {
            "averaged": True,
            "averaged_frame_count": frame_count,
            "averaged_first_timestamp_iso": first_frame.timestamp_iso,
            "averaged_last_timestamp_iso": last_frame.timestamp_iso,
        }
    )
    return ImageFrame(
        image=averaged_image,
        timestamp_iso=_timestamp_iso(),
        exposure_us=last_frame.exposure_us,
        gain=last_frame.gain,
        width_px=last_frame.width_px,
        height_px=last_frame.height_px,
        z_actual_mm=last_frame.z_actual_mm,
        metadata=metadata,
    )


def with_z_actual_mm(frame: ImageFrame, z_actual_mm: float) -> ImageFrame:
    metadata = dict(frame.metadata)
    metadata["z_actual_mm"] = float(z_actual_mm)
    return ImageFrame(
        image=frame.image,
        timestamp_iso=frame.timestamp_iso,
        exposure_us=frame.exposure_us,
        gain=frame.gain,
        width_px=frame.width_px,
        height_px=frame.height_px,
        z_actual_mm=float(z_actual_mm),
        metadata=metadata,
    )


def _validate_consistent_frames(frames: list[ImageFrame]) -> None:
    first = frames[0]
    first_aoi = frame_aoi(first)
    for frame in frames[1:]:
        if frame.image.shape != first.image.shape:
            _raise_inconsistent("image shape changed during averaging.")
        if frame.exposure_us != first.exposure_us:
            _raise_inconsistent("exposure_us changed during averaging.")
        if frame.gain != first.gain:
            _raise_inconsistent("gain changed during averaging.")
        if frame_aoi(frame) != first_aoi:
            _raise_inconsistent("AOI changed during averaging.")


def _raise_inconsistent(message: str) -> None:
    raise LBQAError(ErrorCode.E_IMAGE_ROI_NOT_FOUND, message)


def _timestamp_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")


__all__ = ["capture_average", "capture_frames", "with_z_actual_mm"]
