"""Simulated z-axis stage implementing StagePort."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from lbqa_contracts.errors import ErrorCode, LBQAError
from lbqa_contracts.models import StageStatus


@dataclass(slots=True)
class SimulatedStage:
    z_min_mm: float = -1.0
    z_max_mm: float = 1.0
    z_actual_mm: float = 0.0
    home_position_mm: float = 0.0
    require_home_before_move: bool = True
    position_noise_um_std: float = 0.0
    motion_delay_s: float = 0.0
    velocity_mm_per_s: float | None = None
    move_timeout_s: float = 5.0
    rng_seed: int | None = 0
    connected: bool = False
    enabled: bool = False
    homed: bool = False
    moving: bool = False
    emergency_stopped: bool = False
    _last_target_mm: float = 0.0
    _last_error_code: str | None = None
    _force_next_move_timeout: bool = False
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.z_min_mm >= self.z_max_mm:
            raise ValueError("z_min_mm must be less than z_max_mm.")
        if not self.z_min_mm <= self.z_actual_mm <= self.z_max_mm:
            raise LBQAError(
                ErrorCode.E_DEVICE_STAGE_SOFT_LIMIT,
                "initial z_actual_mm is outside configured soft limits.",
            )
        self._last_target_mm = self.z_actual_mm
        self._rng = np.random.default_rng(self.rng_seed)

    def connect(self) -> None:
        self.connected = True
        self._last_error_code = None

    def disconnect(self) -> None:
        self.connected = False
        self.enabled = False
        self.moving = False

    def enable(self) -> None:
        self._require_connected()
        self._require_not_emergency_stopped()
        self.enabled = True
        self._last_error_code = None

    def disable(self) -> None:
        self._require_connected()
        self.enabled = False
        self.moving = False

    def home(self) -> None:
        self._require_connected()
        self._require_not_emergency_stopped()
        self._validate_soft_limits(self.home_position_mm)
        self._simulate_motion_delay(distance_mm=abs(self.z_actual_mm - self.home_position_mm))
        self.z_actual_mm = self.home_position_mm
        self._last_target_mm = self.home_position_mm
        self.homed = True
        self._last_error_code = None

    def is_homed(self) -> bool:
        return self.homed

    def move_abs_mm(self, position_mm: float) -> None:
        self._ensure_ready_for_motion()
        self._validate_soft_limits(position_mm)
        distance_mm = abs(position_mm - self.z_actual_mm)
        self._last_target_mm = position_mm
        self.moving = True
        try:
            self._simulate_motion_delay(distance_mm=distance_mm)
            self.z_actual_mm = float(position_mm)
            self._last_error_code = None
        finally:
            self.moving = False

    def move_rel_mm(self, delta_mm: float) -> None:
        self.move_abs_mm(self.z_actual_mm + delta_mm)

    def get_position_mm(self) -> float:
        self._require_connected()
        if self.position_noise_um_std <= 0.0:
            return self.z_actual_mm
        noise_mm = float(self._rng.normal(0.0, self.position_noise_um_std / 1000.0))
        return self.z_actual_mm + noise_mm

    def get_status(self) -> StageStatus:
        return StageStatus(
            connected=self.connected,
            homed=self.homed,
            enabled=self.enabled,
            moving=self.moving,
            position_mm=self.z_actual_mm,
            error_code=self._last_error_code,
        )

    def stop(self) -> None:
        self._require_connected()
        self.moving = False

    def emergency_stop(self) -> None:
        self.moving = False
        self.enabled = False
        self.emergency_stopped = True
        self._last_error_code = ErrorCode.E_SAFETY_COLLISION_RISK.value

    def clear_emergency_stop(self) -> None:
        self._require_connected()
        self.emergency_stopped = False
        self._last_error_code = None

    def reset(self) -> None:
        self.clear_emergency_stop()

    def set_soft_limits(self, min_mm: float, max_mm: float) -> None:
        if min_mm >= max_mm:
            raise ValueError("min_mm must be less than max_mm.")
        self.z_min_mm = float(min_mm)
        self.z_max_mm = float(max_mm)
        self._validate_soft_limits(self.z_actual_mm)

    def wait_in_position(self, timeout_s: float, tolerance_um: float) -> bool:
        self._require_connected()
        if timeout_s < 0.0:
            raise ValueError("timeout_s must be non-negative.")
        if tolerance_um < 0.0:
            raise ValueError("tolerance_um must be non-negative.")
        deadline_s = time.monotonic() + timeout_s
        tolerance_mm = tolerance_um / 1000.0
        while True:
            if abs(self.get_position_mm() - self._last_target_mm) <= tolerance_mm:
                return True
            if time.monotonic() >= deadline_s:
                return False
            time.sleep(min(0.005, max(0.0, deadline_s - time.monotonic())))

    def force_next_move_timeout(self) -> None:
        self._force_next_move_timeout = True

    def move_absolute_mm(self, z_target_mm: float) -> float:
        self.move_abs_mm(z_target_mm)
        return self.z_actual_mm

    def _ensure_ready_for_motion(self) -> None:
        self._require_connected()
        self._require_not_emergency_stopped()
        if not self.enabled:
            self._raise(ErrorCode.E_SAFETY_COLLISION_RISK, "stage is not enabled.")
        if self.require_home_before_move and not self.homed:
            self._raise(ErrorCode.E_DEVICE_STAGE_NOT_HOMED, "stage must be homed before motion.")

    def _require_connected(self) -> None:
        if not self.connected:
            self._raise(
                ErrorCode.E_DEVICE_STAGE_CONNECT_FAILED,
                "simulated stage is not connected.",
            )

    def _require_not_emergency_stopped(self) -> None:
        if self.emergency_stopped:
            self._raise(
                ErrorCode.E_SAFETY_COLLISION_RISK,
                "emergency stop is active; clear/reset before moving.",
            )

    def _validate_soft_limits(self, position_mm: float) -> None:
        if not self.z_min_mm <= position_mm <= self.z_max_mm:
            self._raise(
                ErrorCode.E_DEVICE_STAGE_SOFT_LIMIT,
                f"z position {position_mm} mm is outside [{self.z_min_mm}, {self.z_max_mm}] mm.",
            )

    def _simulate_motion_delay(self, *, distance_mm: float) -> None:
        delay_s = max(0.0, self.motion_delay_s)
        if self.velocity_mm_per_s is not None:
            if self.velocity_mm_per_s <= 0.0:
                raise ValueError("velocity_mm_per_s must be positive when configured.")
            delay_s += distance_mm / self.velocity_mm_per_s

        if self._force_next_move_timeout or delay_s > self.move_timeout_s:
            self._force_next_move_timeout = False
            self._raise(ErrorCode.E_DEVICE_STAGE_MOVE_TIMEOUT, "simulated stage move timed out.")

        if delay_s > 0.0:
            time.sleep(delay_s)

    def _raise(self, error_code: ErrorCode, message: str) -> None:
        self._last_error_code = error_code.value
        raise LBQAError(error_code, message)


class SimulatedZAxisStage(SimulatedStage):
    """Backward-compatible alias for the early repository API."""

    def connect(self) -> None:
        super().connect()
        self.enabled = True
        self.homed = True
