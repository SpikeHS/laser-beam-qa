"""ROI and AOI helpers for acquisition frames."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from lbqa_contracts.errors import ErrorCode, LBQAError
from lbqa_contracts.models import ImageFrame
from lbqa_contracts.ports import BeamProfilerPort


@dataclass(frozen=True, slots=True)
class RegionPx:
    x_px: int
    y_px: int
    width_px: int
    height_px: int

    def __post_init__(self) -> None:
        if self.x_px < 0 or self.y_px < 0:
            _raise_roi_error("region origin must be non-negative.")
        if self.width_px <= 0 or self.height_px <= 0:
            _raise_roi_error("region width_px and height_px must be positive.")

    @property
    def right_px(self) -> int:
        return self.x_px + self.width_px

    @property
    def bottom_px(self) -> int:
        return self.y_px + self.height_px

    def touches_bounds(self, width_px: int, height_px: int) -> bool:
        return (
            self.x_px == 0
            or self.y_px == 0
            or self.right_px == width_px
            or self.bottom_px == height_px
        )

    def metadata(self, prefix: str) -> dict[str, int]:
        return {
            f"{prefix}_x_px": self.x_px,
            f"{prefix}_y_px": self.y_px,
            f"{prefix}_width_px": self.width_px,
            f"{prefix}_height_px": self.height_px,
        }


@dataclass(frozen=True, slots=True)
class RoiResult:
    roi: RegionPx
    mode: Literal["auto", "fixed", "full"]
    source_width_px: int
    source_height_px: int
    touches_image_boundary: bool


def validate_region_bounds(region: RegionPx, width_px: int, height_px: int) -> None:
    if region.right_px > width_px or region.bottom_px > height_px:
        _raise_roi_error(
            "region exceeds image bounds: "
            f"region={region}, image_width_px={width_px}, image_height_px={height_px}."
        )


def configure_aoi(profiler: BeamProfilerPort, aoi: RegionPx | None) -> RegionPx:
    """Apply an explicit AOI only; absence preserves the device's current AOI."""
    if aoi is None:
        return current_aoi(profiler)
    info = profiler.get_camera_info()
    validate_region_bounds(aoi, width_px=info.width_px, height_px=info.height_px)
    profiler.set_aoi(aoi.x_px, aoi.y_px, aoi.width_px, aoi.height_px)
    return aoi


def frame_aoi(frame: ImageFrame) -> RegionPx:
    metadata = frame.metadata
    return RegionPx(
        x_px=int(metadata.get("aoi_x_px", 0)),
        y_px=int(metadata.get("aoi_y_px", 0)),
        width_px=int(metadata.get("aoi_width_px", frame.width_px)),
        height_px=int(metadata.get("aoi_height_px", frame.height_px)),
    )


def current_aoi(profiler: BeamProfilerPort, frame: ImageFrame | None = None) -> RegionPx:
    if frame is not None:
        fallback = frame_aoi(frame)
    else:
        info = profiler.get_camera_info()
        fallback = RegionPx(0, 0, info.width_px, info.height_px)
    status = profiler.get_status()
    # Optional hardware readbacks may explicitly be None, not just absent.
    values = {
        key: status[key] if status.get(key) is not None else value
        for key, value in fallback.metadata("aoi").items()
    }
    return RegionPx(
        x_px=int(values["aoi_x_px"]),
        y_px=int(values["aoi_y_px"]),
        width_px=int(values["aoi_width_px"]),
        height_px=int(values["aoi_height_px"]),
    )


def auto_roi(
    image: np.ndarray[Any, Any],
    *,
    threshold_percent: float = 5.0,
    padding_px: int = 8,
    min_width_px: int = 8,
    min_height_px: int = 8,
) -> RoiResult:
    image_float = _as_2d_float(image)
    height_px, width_px = image_float.shape
    _validate_roi_tuning(
        threshold_percent=threshold_percent,
        padding_px=padding_px,
        min_width_px=min_width_px,
        min_height_px=min_height_px,
    )

    signal = image_float - float(np.min(image_float))
    np.maximum(signal, 0.0, out=signal)
    peak_value = float(np.max(signal))
    if peak_value <= 0.0:
        _raise_roi_error("auto ROI could not find signal above background.")

    mask = signal >= peak_value * threshold_percent / 100.0
    if not np.any(mask):
        _raise_roi_error("auto ROI threshold produced an empty mask.")

    y_indices, x_indices = np.nonzero(mask)
    x_min_px = max(0, int(np.min(x_indices)) - padding_px)
    x_max_px = min(width_px - 1, int(np.max(x_indices)) + padding_px)
    y_min_px = max(0, int(np.min(y_indices)) - padding_px)
    y_max_px = min(height_px - 1, int(np.max(y_indices)) + padding_px)

    roi = _expand_to_minimum_size(
        RegionPx(
            x_px=x_min_px,
            y_px=y_min_px,
            width_px=x_max_px - x_min_px + 1,
            height_px=y_max_px - y_min_px + 1,
        ),
        min_width_px=min_width_px,
        min_height_px=min_height_px,
        source_width_px=width_px,
        source_height_px=height_px,
    )
    return RoiResult(
        roi=roi,
        mode="auto",
        source_width_px=width_px,
        source_height_px=height_px,
        touches_image_boundary=roi.touches_bounds(width_px, height_px),
    )


def select_roi(
    frame: ImageFrame,
    *,
    fixed_roi: RegionPx | None = None,
    auto_roi_enabled: bool = True,
    threshold_percent: float = 5.0,
    padding_px: int = 8,
    min_width_px: int = 8,
    min_height_px: int = 8,
) -> RoiResult:
    if fixed_roi is not None:
        validate_region_bounds(fixed_roi, width_px=frame.width_px, height_px=frame.height_px)
        return RoiResult(
            roi=fixed_roi,
            mode="fixed",
            source_width_px=frame.width_px,
            source_height_px=frame.height_px,
            touches_image_boundary=fixed_roi.touches_bounds(frame.width_px, frame.height_px),
        )
    if auto_roi_enabled:
        return auto_roi(
            frame.image,
            threshold_percent=threshold_percent,
            padding_px=padding_px,
            min_width_px=min_width_px,
            min_height_px=min_height_px,
        )
    roi = RegionPx(x_px=0, y_px=0, width_px=frame.width_px, height_px=frame.height_px)
    return RoiResult(
        roi=roi,
        mode="full",
        source_width_px=frame.width_px,
        source_height_px=frame.height_px,
        touches_image_boundary=False,
    )


def crop_frame_to_roi(frame: ImageFrame, roi_result: RoiResult) -> ImageFrame:
    roi = roi_result.roi
    validate_region_bounds(roi, width_px=frame.width_px, height_px=frame.height_px)
    image = np.asarray(frame.image)[roi.y_px : roi.bottom_px, roi.x_px : roi.right_px]
    metadata: dict[str, Any] = dict(frame.metadata)
    metadata.update(roi.metadata("roi"))
    metadata.update(
        {
            "roi_mode": roi_result.mode,
            "roi_source_width_px": roi_result.source_width_px,
            "roi_source_height_px": roi_result.source_height_px,
            "roi_touches_image_boundary": roi_result.touches_image_boundary,
        }
    )
    return ImageFrame(
        image=image.copy(),
        timestamp_iso=frame.timestamp_iso,
        exposure_us=frame.exposure_us,
        gain=frame.gain,
        width_px=roi.width_px,
        height_px=roi.height_px,
        z_actual_mm=frame.z_actual_mm,
        metadata=metadata,
    )


def _expand_to_minimum_size(
    roi: RegionPx,
    *,
    min_width_px: int,
    min_height_px: int,
    source_width_px: int,
    source_height_px: int,
) -> RegionPx:
    width_px = max(roi.width_px, min_width_px)
    height_px = max(roi.height_px, min_height_px)
    center_x_px = roi.x_px + roi.width_px // 2
    center_y_px = roi.y_px + roi.height_px // 2
    x_px = min(max(0, center_x_px - width_px // 2), max(0, source_width_px - width_px))
    y_px = min(max(0, center_y_px - height_px // 2), max(0, source_height_px - height_px))
    return RegionPx(x_px=x_px, y_px=y_px, width_px=width_px, height_px=height_px)


def _as_2d_float(image: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    image_array = np.asarray(image)
    if image_array.ndim != 2 or image_array.size == 0:
        _raise_roi_error("image must be a non-empty 2D array.")
    return image_array.astype(np.float64, copy=False)


def _validate_roi_tuning(
    *,
    threshold_percent: float,
    padding_px: int,
    min_width_px: int,
    min_height_px: int,
) -> None:
    if not 0.0 < threshold_percent <= 100.0:
        _raise_roi_error("threshold_percent must be in (0, 100].")
    if padding_px < 0:
        _raise_roi_error("padding_px must be non-negative.")
    if min_width_px <= 0 or min_height_px <= 0:
        _raise_roi_error("minimum ROI dimensions must be positive.")


def _raise_roi_error(message: str) -> None:
    raise LBQAError(ErrorCode.E_IMAGE_ROI_NOT_FOUND, message)


__all__ = [
    "RegionPx",
    "RoiResult",
    "auto_roi",
    "configure_aoi",
    "crop_frame_to_roi",
    "current_aoi",
    "frame_aoi",
    "select_roi",
    "validate_region_bounds",
]
