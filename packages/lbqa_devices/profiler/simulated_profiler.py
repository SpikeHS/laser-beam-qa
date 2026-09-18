"""Simulated Cinogy beam profiler implementing BeamProfilerPort."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np
from lbqa_contracts.errors import ErrorCode, LBQAError
from lbqa_contracts.models import CameraInfo, ImageFrame, OpticalCalibration
from lbqa_contracts.schemas import BeamFrame

from lbqa_devices.simulated.synthetic_beam import generate_z_dependent_beam_image


@dataclass(slots=True)
class SimulatedProfiler:
    width_px: int = 2048
    height_px: int = 2048
    camera_pixel_size_um: float = 5.5
    objective_magnification: float = 40.0
    bit_depth: int = 10
    model: str = "CMOS-1.001-Nano"
    serial_number: str = "SIM-CMOS1-001-NANO"
    exposure_us: float = 2000.0
    reference_exposure_us: float = 2000.0
    gain: float = 1.0
    waist_z_mm: float = 0.0
    waist_diameter_x_um: float = 18.0
    waist_diameter_y_um: float = 22.0
    full_angle_x_mrad: float = 8.0
    full_angle_y_mrad: float = 10.0
    beam_angle_deg: float = 0.0
    amplitude_count: float = 760.0
    background_count: float = 20.0
    noise_std_count: float = 0.0
    z_position_source: Callable[[], float] | None = None
    z_actual_mm: float = 0.0
    rng_seed: int | None = 0
    connected: bool = False
    background_calibration_applied: bool = False
    background_calibration_frame_span: int | None = None
    saturation: bool = False
    low_signal: bool = False
    edge_clipped: bool = False
    bad_pixel_count: int = 0
    centroid_offset_x_px: float = 0.0
    centroid_offset_y_px: float = 0.0
    _aoi_x_px: int = 0
    _aoi_y_px: int = 0
    _aoi_width_px: int = field(init=False)
    _aoi_height_px: int = field(init=False)
    _frame_id_count: int = 0
    _last_image: ImageFrame | None = field(default=None, init=False, repr=False)
    _last_error_code: str | None = None
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._validate_positive_int("width_px", self.width_px)
        self._validate_positive_int("height_px", self.height_px)
        self._validate_positive_float("camera_pixel_size_um", self.camera_pixel_size_um)
        self._validate_positive_float("objective_magnification", self.objective_magnification)
        self._validate_positive_int("bit_depth", self.bit_depth)
        self._validate_positive_float("exposure_us", self.exposure_us)
        self._validate_positive_float("reference_exposure_us", self.reference_exposure_us)
        self._aoi_width_px = self.width_px
        self._aoi_height_px = self.height_px
        self._rng = np.random.default_rng(self.rng_seed)

    @property
    def effective_pixel_x_um(self) -> float:
        return self.camera_pixel_size_um / self.objective_magnification

    @property
    def effective_pixel_y_um(self) -> float:
        return self.camera_pixel_size_um / self.objective_magnification

    def connect(self) -> None:
        self.connected = True
        self._last_error_code = None

    def disconnect(self) -> None:
        self.connected = False

    def get_camera_info(self) -> CameraInfo:
        return CameraInfo(
            model=self.model,
            width_px=self.width_px,
            height_px=self.height_px,
            pixel_size_um=self.camera_pixel_size_um,
            bit_depth=self.bit_depth,
            serial_number=self.serial_number,
        )

    def get_optical_calibration(self) -> OpticalCalibration:
        return OpticalCalibration(
            magnification=self.objective_magnification,
            camera_pixel_size_um=self.camera_pixel_size_um,
            effective_pixel_x_um=self.effective_pixel_x_um,
            effective_pixel_y_um=self.effective_pixel_y_um,
            field_of_view_x_um=self.width_px * self.effective_pixel_x_um,
            field_of_view_y_um=self.height_px * self.effective_pixel_y_um,
            working_distance_mm=0.5,
            calibration_id="simulated-40x",
        )

    def set_exposure_us(self, exposure_us: float) -> None:
        self._validate_positive_float("exposure_us", exposure_us)
        self.exposure_us = float(exposure_us)

    def set_gain(self, gain: float) -> None:
        if gain < 0.0:
            raise ValueError("gain must be non-negative.")
        self.gain = float(gain)

    def set_aoi(self, x_px: int, y_px: int, width_px: int, height_px: int) -> None:
        if width_px <= 0 or height_px <= 0:
            self._raise(
                ErrorCode.E_IMAGE_ROI_NOT_FOUND,
                "AOI width_px and height_px must be positive.",
            )
        if x_px < 0 or y_px < 0:
            self._raise(ErrorCode.E_IMAGE_ROI_NOT_FOUND, "AOI origin must be non-negative.")
        if x_px + width_px > self.width_px or y_px + height_px > self.height_px:
            self._raise(ErrorCode.E_IMAGE_ROI_NOT_FOUND, "AOI exceeds simulated sensor bounds.")
        self._aoi_x_px = int(x_px)
        self._aoi_y_px = int(y_px)
        self._aoi_width_px = int(width_px)
        self._aoi_height_px = int(height_px)

    def capture_single(self) -> ImageFrame:
        self._require_connected()
        self._frame_id_count += 1
        z_actual_mm = self._current_z_actual_mm()
        beam = generate_z_dependent_beam_image(
            width_px=self._aoi_width_px,
            height_px=self._aoi_height_px,
            z_actual_mm=z_actual_mm,
            effective_pixel_x_um=self.effective_pixel_x_um,
            effective_pixel_y_um=self.effective_pixel_y_um,
            waist_z_mm=self.waist_z_mm,
            waist_diameter_x_um=self.waist_diameter_x_um,
            waist_diameter_y_um=self.waist_diameter_y_um,
            full_angle_x_mrad=self.full_angle_x_mrad,
            full_angle_y_mrad=self.full_angle_y_mrad,
            bit_depth=self.bit_depth,
            amplitude_count=self._scaled_amplitude_count(),
            background_count=self.background_count,
            noise_std_count=self.noise_std_count,
            angle_deg=self.beam_angle_deg,
            centroid_x_px=(self.width_px - 1) / 2.0 - self._aoi_x_px,
            centroid_y_px=(self.height_px - 1) / 2.0 - self._aoi_y_px,
            centroid_offset_x_px=self.centroid_offset_x_px,
            centroid_offset_y_px=self.centroid_offset_y_px,
            saturation=self.saturation,
            low_signal=self.low_signal,
            edge_clipped=self.edge_clipped,
            bad_pixel_count=self.bad_pixel_count,
            rng=self._rng,
        )
        frame = ImageFrame(
            image=beam.image_counts,
            timestamp_iso=_timestamp_iso(),
            exposure_us=self.exposure_us,
            gain=self.gain,
            width_px=self._aoi_width_px,
            height_px=self._aoi_height_px,
            z_actual_mm=z_actual_mm,
            metadata={
                "mode": "simulated",
                "frame_id_count": self._frame_id_count,
                "model": self.model,
                "bit_depth": self.bit_depth,
                "aoi_x_px": self._aoi_x_px,
                "aoi_y_px": self._aoi_y_px,
                "effective_pixel_x_um": self.effective_pixel_x_um,
                "effective_pixel_y_um": self.effective_pixel_y_um,
                "d4sigma_x_um": beam.d4sigma_x_um,
                "d4sigma_y_um": beam.d4sigma_y_um,
                "sigma_x_px": beam.sigma_x_px,
                "sigma_y_px": beam.sigma_y_px,
                "centroid_x_px": beam.centroid_x_px,
                "centroid_y_px": beam.centroid_y_px,
                "full_angle_x_mrad": self.full_angle_x_mrad,
                "full_angle_y_mrad": self.full_angle_y_mrad,
                "peak_intensity_count": beam.peak_intensity_count,
                "background_intensity_count": beam.background_intensity_count,
                "anomaly_tags": beam.anomaly_tags,
            },
        )
        self._last_image = frame
        self._last_error_code = None
        return frame

    def capture_average(self, frame_count: int) -> ImageFrame:
        if frame_count <= 0:
            raise ValueError("frame_count must be positive.")
        frames = [self.capture_single() for _ in range(frame_count)]
        averaged_image = np.mean(
            [frame.image.astype(np.float32, copy=False) for frame in frames],
            axis=0,
            dtype=np.float32,
        )
        last_frame = frames[-1]
        metadata: dict[str, Any] = dict(last_frame.metadata)
        metadata["averaged"] = True
        metadata["averaged_frame_count"] = frame_count
        frame = ImageFrame(
            image=averaged_image,
            timestamp_iso=_timestamp_iso(),
            exposure_us=self.exposure_us,
            gain=self.gain,
            width_px=last_frame.width_px,
            height_px=last_frame.height_px,
            z_actual_mm=last_frame.z_actual_mm,
            metadata=metadata,
        )
        self._last_image = frame
        return frame

    def get_last_image(self) -> ImageFrame | None:
        return self._last_image

    def get_status(self) -> dict[str, object]:
        return {
            "connected": self.connected,
            "model": self.model,
            "width_px": self.width_px,
            "height_px": self.height_px,
            "aoi_x_px": self._aoi_x_px,
            "aoi_y_px": self._aoi_y_px,
            "aoi_width_px": self._aoi_width_px,
            "aoi_height_px": self._aoi_height_px,
            "exposure_us": self.exposure_us,
            "gain": self.gain,
            "z_actual_mm": self._current_z_actual_mm(),
            "effective_pixel_x_um": self.effective_pixel_x_um,
            "effective_pixel_y_um": self.effective_pixel_y_um,
            "saturation_enabled": self.saturation,
            "low_signal_enabled": self.low_signal,
            "edge_clipped_enabled": self.edge_clipped,
            "background_calibration_applied": self.background_calibration_applied,
            "background_calibration_frame_span": self.background_calibration_frame_span,
            "error_code": self._last_error_code,
        }

    def get_background_calibration_status(self) -> dict[str, object]:
        self._require_connected()
        return {
            "mode": "simulated",
            "background_state": 2 if self.background_calibration_applied else 0,
            "background_can_save": False,
            "background_calibration_applied": self.background_calibration_applied,
            "background_calibration_frame_span": self.background_calibration_frame_span,
        }

    def perform_background_calibration(
        self,
        *,
        frame_span: int = 16,
        apply_calibration: bool = True,
    ) -> dict[str, object]:
        if frame_span <= 0:
            raise ValueError("frame_span must be positive.")
        self._require_connected()
        before = self.get_background_calibration_status()
        self.background_calibration_frame_span = int(frame_span)
        self.background_calibration_applied = bool(apply_calibration)
        return {
            "frame_span": int(frame_span),
            "all_exposure": True,
            "apply_requested": bool(apply_calibration),
            "status_before": before,
            "status_after": self.get_background_calibration_status(),
        }

    def set_z_actual_mm(self, z_actual_mm: float) -> None:
        self.z_actual_mm = float(z_actual_mm)

    def set_faults(
        self,
        *,
        saturation: bool | None = None,
        low_signal: bool | None = None,
        edge_clipped: bool | None = None,
        bad_pixel_count: int | None = None,
    ) -> None:
        if saturation is not None:
            self.saturation = saturation
        if low_signal is not None:
            self.low_signal = low_signal
        if edge_clipped is not None:
            self.edge_clipped = edge_clipped
        if bad_pixel_count is not None:
            if bad_pixel_count < 0:
                raise ValueError("bad_pixel_count must be non-negative.")
            self.bad_pixel_count = bad_pixel_count

    def _current_z_actual_mm(self) -> float:
        if self.z_position_source is not None:
            return float(self.z_position_source())
        return self.z_actual_mm

    def _scaled_amplitude_count(self) -> float:
        exposure_scale = self.exposure_us / self.reference_exposure_us
        return self.amplitude_count * exposure_scale * self.gain

    def _require_connected(self) -> None:
        if not self.connected:
            self._raise(
                ErrorCode.E_DEVICE_RAYCI_CONNECT_FAILED,
                "simulated beam profiler is not connected.",
            )

    def _raise(self, error_code: ErrorCode, message: str) -> None:
        self._last_error_code = error_code.value
        raise LBQAError(error_code, message)

    @staticmethod
    def _validate_positive_int(name: str, value: int) -> None:
        if value <= 0:
            raise ValueError(f"{name} must be positive.")

    @staticmethod
    def _validate_positive_float(name: str, value: float) -> None:
        if value <= 0.0:
            raise ValueError(f"{name} must be positive.")


class SimulatedCamera(SimulatedProfiler):
    """Backward-compatible wrapper for the early tuple-frame simulation API."""

    def __init__(
        self,
        width_px: int = 128,
        height_px: int = 128,
        object_pixel_size_um_per_px: float = 0.1375,
        exposure_ms: float = 2.0,
        base_d4sigma_x_um: float = 8.0,
        base_d4sigma_y_um: float = 10.0,
        full_angle_x_mrad: float = 2.0,
        full_angle_y_mrad: float = 3.0,
        frame_id_count: int = 0,
        connected: bool = False,
    ) -> None:
        super().__init__(
            width_px=width_px,
            height_px=height_px,
            objective_magnification=5.5 / object_pixel_size_um_per_px,
            exposure_us=exposure_ms * 1000.0,
            reference_exposure_us=exposure_ms * 1000.0,
            waist_diameter_x_um=base_d4sigma_x_um,
            waist_diameter_y_um=base_d4sigma_y_um,
            full_angle_x_mrad=full_angle_x_mrad,
            full_angle_y_mrad=full_angle_y_mrad,
            bit_depth=12,
            amplitude_count=3800.0,
            background_count=0.0,
            noise_std_count=0.0,
            connected=connected,
        )
        self._frame_id_count = frame_id_count

    def capture_frame(self, *, z_actual_mm: float) -> BeamFrame:
        self.set_z_actual_mm(z_actual_mm)
        image_frame = self.capture_single()
        return BeamFrame(
            frame_id_count=int(image_frame.metadata["frame_id_count"]),
            width_px=image_frame.width_px,
            height_px=image_frame.height_px,
            object_pixel_size_um_per_px=self.effective_pixel_x_um,
            exposure_ms=image_frame.exposure_us / 1000.0,
            z_actual_mm=float(image_frame.z_actual_mm),
            pixels=tuple(
                tuple(float(value) for value in row) for row in image_frame.image.tolist()
            ),
        )


def _timestamp_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")
