"""Hardware and service ports for dependency inversion."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from lbqa_contracts.models import CameraInfo, ImageFrame, StageStatus


@runtime_checkable
class StagePort(Protocol):
    def connect(self) -> None:
        """Connect to the z-axis stage."""

    def disconnect(self) -> None:
        """Disconnect from the z-axis stage."""

    def enable(self) -> None:
        """Enable stage motion."""

    def disable(self) -> None:
        """Disable stage motion."""

    def home(self) -> None:
        """Home the z-axis stage."""

    def is_homed(self) -> bool:
        """Return whether the z-axis stage has been homed."""

    def move_abs_mm(self, position_mm: float) -> None:
        """Move to an absolute z position in mm."""

    def move_rel_mm(self, delta_mm: float) -> None:
        """Move by a relative z distance in mm."""

    def get_position_mm(self) -> float:
        """Return the current z position in mm."""

    def get_status(self) -> StageStatus:
        """Return normalized stage status."""

    def stop(self) -> None:
        """Stop current motion."""

    def emergency_stop(self) -> None:
        """Trigger emergency stop."""

    def set_soft_limits(self, min_mm: float, max_mm: float) -> None:
        """Set z-axis software limits in mm."""

    def wait_in_position(self, timeout_s: float, tolerance_um: float) -> bool:
        """Wait until the stage is in position within tolerance."""


@runtime_checkable
class BeamProfilerPort(Protocol):
    def connect(self) -> None:
        """Connect to the beam profiler or camera backend."""

    def disconnect(self) -> None:
        """Disconnect from the beam profiler or camera backend."""

    def get_camera_info(self) -> CameraInfo:
        """Return normalized camera information."""

    def set_exposure_us(self, exposure_us: float) -> None:
        """Set exposure time in us."""

    def set_gain(self, gain: float) -> None:
        """Set camera gain."""

    def set_aoi(self, x_px: int, y_px: int, width_px: int, height_px: int) -> None:
        """Set area of interest in sensor pixels."""

    def capture_single(self) -> ImageFrame:
        """Capture a single image frame."""

    def capture_average(self, frame_count: int) -> ImageFrame:
        """Capture and average multiple image frames."""

    def get_last_image(self) -> ImageFrame | None:
        """Return the last captured frame if available."""

    def get_status(self) -> dict[str, object]:
        """Return normalized beam profiler status."""


@runtime_checkable
class InterlockPort(Protocol):
    def is_safe(self) -> bool:
        """Return whether all safety interlocks are closed/safe."""

    def get_status(self) -> dict[str, object]:
        """Return normalized interlock status."""
