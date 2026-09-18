"""High-level acquisition service used by sequencer workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from lbqa_contracts.errors import ErrorCode
from lbqa_contracts.models import FrameQuality, ImageFrame
from lbqa_contracts.ports import BeamProfilerPort

from lbqa_acquisition.dark_frame import DarkFrame, acquire_dark_frame, apply_dark_frame
from lbqa_acquisition.exposure import AutoExposureResult, ExposureLimits, auto_exposure
from lbqa_acquisition.frame_capture import capture_average, with_z_actual_mm
from lbqa_acquisition.roi import RegionPx, configure_aoi, crop_frame_to_roi, current_aoi, select_roi
from lbqa_acquisition.validation import evaluate_frame_validity, invalid_quality_like


@dataclass(frozen=True, slots=True)
class AcquisitionSettings:
    dark_frame_count: int = 8
    average_frame_count: int = 3
    target_peak_percent: float = 70.0
    max_peak_percent: float = 90.0
    exposure_limits: ExposureLimits = field(default_factory=ExposureLimits)
    aoi: RegionPx | None = None
    fixed_roi: RegionPx | None = None
    auto_roi_enabled: bool = True
    roi_threshold_percent: float = 5.0
    roi_padding_px: int = 8
    roi_min_width_px: int = 8
    roi_min_height_px: int = 8
    saturation_threshold_percent: float = 99.0
    low_signal_threshold_percent: float = 1.0
    edge_margin_px: int = 4
    edge_energy_limit_percent: float = 5.0
    spot_edge_margin_px: int = 4
    spot_threshold_percent: float = 5.0

    def __post_init__(self) -> None:
        if self.dark_frame_count <= 0:
            raise ValueError("dark_frame_count must be positive.")
        if self.average_frame_count <= 0:
            raise ValueError("average_frame_count must be positive.")


@dataclass(frozen=True, slots=True)
class CorrectedFrame:
    image: np.ndarray[Any, Any] = field(repr=False)
    timestamp_iso: str
    exposure_us: float
    gain: float
    width_px: int
    height_px: int
    z_actual_mm: float
    quality: FrameQuality
    roi: RegionPx
    raw_frame: ImageFrame = field(repr=False)
    dark_frame: DarkFrame | None = field(default=None, repr=False)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_image_frame(self) -> ImageFrame:
        return ImageFrame(
            image=self.image,
            timestamp_iso=self.timestamp_iso,
            exposure_us=self.exposure_us,
            gain=self.gain,
            width_px=self.width_px,
            height_px=self.height_px,
            z_actual_mm=self.z_actual_mm,
            metadata=dict(self.metadata),
        )


class AcquisitionService:
    def __init__(
        self,
        profiler: BeamProfilerPort,
        settings: AcquisitionSettings | None = None,
    ) -> None:
        self._profiler = profiler
        self._settings = settings or AcquisitionSettings()
        self._configured_aoi: RegionPx | None = None
        self._dark_frame: DarkFrame | None = None
        self._exposure_result: AutoExposureResult | None = None

    @property
    def dark_frame(self) -> DarkFrame | None:
        return self._dark_frame

    @property
    def exposure_result(self) -> AutoExposureResult | None:
        return self._exposure_result

    def prepare_dark_frame(self, frame_count: int | None = None) -> DarkFrame:
        self._ensure_aoi()
        self._dark_frame = acquire_dark_frame(
            self._profiler,
            frame_count or self._settings.dark_frame_count,
        )
        return self._dark_frame

    def prepare_exposure(self) -> AutoExposureResult:
        self._ensure_aoi()
        self._exposure_result = auto_exposure(
            self._profiler,
            target_peak_percent=self._settings.target_peak_percent,
            max_peak_percent=self._settings.max_peak_percent,
            limits=self._settings.exposure_limits,
        )
        return self._exposure_result

    def capture_plane(self, z_actual_mm: float) -> CorrectedFrame:
        self._ensure_aoi()
        raw_frame = with_z_actual_mm(self._profiler.capture_single(), z_actual_mm)
        return self._finalize_frame(raw_frame)

    def capture_plane_average(
        self,
        z_actual_mm: float,
        frame_count: int | None = None,
    ) -> CorrectedFrame:
        self._ensure_aoi()
        averaged = capture_average(
            self._profiler,
            frame_count or self._settings.average_frame_count,
        )
        raw_frame = with_z_actual_mm(averaged, z_actual_mm)
        return self._finalize_frame(raw_frame)

    def _ensure_aoi(self) -> RegionPx:
        if self._configured_aoi is None:
            self._configured_aoi = configure_aoi(self._profiler, self._settings.aoi)
        return self._configured_aoi

    def _finalize_frame(self, raw_frame: ImageFrame) -> CorrectedFrame:
        corrected_frame = raw_frame
        if self._dark_frame is not None:
            corrected_frame = apply_dark_frame(
                raw_frame,
                self._dark_frame,
                frame_aoi_px=current_aoi(self._profiler, raw_frame),
            )

        roi_result = select_roi(
            corrected_frame,
            fixed_roi=self._settings.fixed_roi,
            auto_roi_enabled=self._settings.auto_roi_enabled,
            threshold_percent=self._settings.roi_threshold_percent,
            padding_px=self._settings.roi_padding_px,
            min_width_px=self._settings.roi_min_width_px,
            min_height_px=self._settings.roi_min_height_px,
        )
        analysis_frame = crop_frame_to_roi(corrected_frame, roi_result)
        bit_depth = self._profiler.get_camera_info().bit_depth
        quality = evaluate_frame_validity(
            analysis_frame.image,
            bit_depth=bit_depth,
            saturation_threshold_percent=self._settings.saturation_threshold_percent,
            low_signal_threshold_percent=self._settings.low_signal_threshold_percent,
            edge_margin_px=self._settings.edge_margin_px,
            edge_energy_limit_percent=self._settings.edge_energy_limit_percent,
            spot_edge_margin_px=self._settings.spot_edge_margin_px,
            spot_threshold_percent=self._settings.spot_threshold_percent,
        )
        if roi_result.mode == "auto" and roi_result.touches_image_boundary and quality.valid:
            quality = invalid_quality_like(quality, ErrorCode.E_IMAGE_EDGE_CLIPPED)

        metadata: dict[str, Any] = dict(analysis_frame.metadata)
        metadata["frame_quality_valid"] = quality.valid
        metadata["frame_quality_invalid_reason"] = quality.invalid_reason
        return CorrectedFrame(
            image=np.asarray(analysis_frame.image),
            timestamp_iso=analysis_frame.timestamp_iso,
            exposure_us=analysis_frame.exposure_us,
            gain=analysis_frame.gain,
            width_px=analysis_frame.width_px,
            height_px=analysis_frame.height_px,
            z_actual_mm=float(raw_frame.z_actual_mm),
            quality=quality,
            roi=roi_result.roi,
            raw_frame=raw_frame,
            dark_frame=self._dark_frame,
            metadata=metadata,
        )


__all__ = ["AcquisitionService", "AcquisitionSettings", "CorrectedFrame"]
