"""Legacy device protocols retained for compatibility.

MVP integrations should use `lbqa_contracts.ports` instead. Do not expand this
module for new public interfaces.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from lbqa_contracts.schemas import BeamFrame


@runtime_checkable
class CameraDevice(Protocol):
    """Camera or beam profiler capture boundary."""

    def connect(self) -> None:
        """Open the camera connection."""

    def disconnect(self) -> None:
        """Close the camera connection."""

    def capture_frame(self, *, z_actual_mm: float) -> BeamFrame:
        """Capture one calibrated frame at a known z position."""


@runtime_checkable
class ZAxisStage(Protocol):
    """Z-axis motion boundary."""

    def connect(self) -> None:
        """Open the stage connection."""

    def disconnect(self) -> None:
        """Close the stage connection."""

    def move_absolute_mm(self, z_target_mm: float) -> float:
        """Move to the requested absolute z position and return z_actual_mm."""

    def get_position_mm(self) -> float:
        """Return the latest measured z position."""
