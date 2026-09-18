from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
from lbqa_contracts.errors import ErrorCode, LBQAError
from lbqa_contracts.models import BeamPlaneResult, CameraInfo, ImageFrame, StageStatus, ZScanResult
from lbqa_sequencer import (
    SequencerDependencies,
    SequencerService,
    SequencerState,
    ZScanRecipe,
    ZScanRunCommand,
)


def _plane(
    z_actual_mm: float,
    d4sigma_um: float = 20.0,
    *,
    valid: bool = True,
    invalid_reason: str | None = None,
) -> BeamPlaneResult:
    return BeamPlaneResult(
        z_actual_mm=z_actual_mm,
        centroid_x_um=10.0,
        centroid_y_um=10.0,
        peak_x_um=10.0,
        peak_y_um=10.0,
        d4sigma_x_um=d4sigma_um,
        d4sigma_y_um=d4sigma_um,
        d4sigma_major_um=d4sigma_um,
        d4sigma_minor_um=d4sigma_um,
        fwhm_x_um=d4sigma_um / 2.0,
        fwhm_y_um=d4sigma_um / 2.0,
        ellipticity=1.0,
        azimuth_deg=0.0,
        peak_value=500.0,
        saturation_pixels=0,
        edge_energy_percent=0.0,
        valid=valid,
        invalid_reason=invalid_reason,
    )


class FakeStage:
    def __init__(self, *, z_min_mm: float = -2.0, z_max_mm: float = 2.0) -> None:
        self.z_min_mm = z_min_mm
        self.z_max_mm = z_max_mm
        self.position_mm = 0.0
        self.connected = False
        self.enabled = False
        self.homed = False
        self.stopped = False

    def connect(self) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False

    def enable(self) -> None:
        self.enabled = True

    def disable(self) -> None:
        self.enabled = False

    def home(self) -> None:
        self.homed = True
        self.position_mm = 0.0

    def is_homed(self) -> bool:
        return self.homed

    def move_abs_mm(self, position_mm: float) -> None:
        if not self.z_min_mm <= position_mm <= self.z_max_mm:
            raise LBQAError(ErrorCode.E_DEVICE_STAGE_SOFT_LIMIT, "outside fake soft limits.")
        self.position_mm = float(position_mm)

    def move_rel_mm(self, delta_mm: float) -> None:
        self.move_abs_mm(self.position_mm + delta_mm)

    def get_position_mm(self) -> float:
        return self.position_mm

    def get_status(self) -> StageStatus:
        return StageStatus(
            connected=self.connected,
            homed=self.homed,
            enabled=self.enabled,
            moving=False,
            position_mm=self.position_mm,
        )

    def stop(self) -> None:
        self.stopped = True

    def emergency_stop(self) -> None:
        self.stopped = True
        self.enabled = False

    def set_soft_limits(self, min_mm: float, max_mm: float) -> None:
        self.z_min_mm = min_mm
        self.z_max_mm = max_mm

    def wait_in_position(self, timeout_s: float, tolerance_um: float) -> bool:
        return True


class FakeProfiler:
    def __init__(self) -> None:
        self.connected = False

    def connect(self) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False

    def get_camera_info(self) -> CameraInfo:
        return CameraInfo("fake", 8, 8, 5.5, 12)

    def set_exposure_us(self, exposure_us: float) -> None:
        self.exposure_us = exposure_us

    def set_gain(self, gain: float) -> None:
        self.gain = gain

    def set_aoi(self, x_px: int, y_px: int, width_px: int, height_px: int) -> None:
        self.aoi = (x_px, y_px, width_px, height_px)

    def capture_single(self) -> ImageFrame:
        return _frame(0.0)

    def capture_average(self, frame_count: int) -> ImageFrame:
        return _frame(0.0)

    def get_last_image(self) -> ImageFrame | None:
        return None

    def get_status(self) -> dict[str, object]:
        return {"connected": self.connected}


@dataclass(slots=True)
class FakeInterlock:
    safe: bool = True

    def is_safe(self) -> bool:
        return self.safe

    def get_status(self) -> dict[str, object]:
        return {
            "safe": self.safe,
            "reason": None if self.safe else "fake interlock open",
        }


class FakeAcquisition:
    def __init__(self, *, fail_capture: bool = False) -> None:
        self.fail_capture = fail_capture
        self.stopped = False

    def prepare_dark_frame(self, frame_count: int | None = None) -> None:
        self.dark_frame_count = frame_count

    def prepare_exposure(self) -> None:
        self.exposure_prepared = True

    def capture_plane_average(self, z_actual_mm: float, frame_count: int | None = None) -> Any:
        if self.fail_capture:
            raise LBQAError(ErrorCode.E_DEVICE_CAMERA_TIMEOUT, "fake camera timeout.")
        return _frame(z_actual_mm)

    def stop(self) -> None:
        self.stopped = True


class FakeAnalysis:
    def __init__(self, *, invalid_after_prescan: bool = False) -> None:
        self.invalid_after_prescan = invalid_after_prescan
        self.calls_count = 0

    def analyze_beam_plane(
        self,
        image: Any,
        calibration: Any,
        z_actual_mm: float,
        quality_limits: dict[str, Any],
    ) -> BeamPlaneResult:
        self.calls_count += 1
        diameter_um = math.sqrt(18.0**2 + (4.0 * z_actual_mm) ** 2)
        if self.invalid_after_prescan and self.calls_count > 4:
            return _plane(
                z_actual_mm,
                diameter_um,
                valid=False,
                invalid_reason=ErrorCode.E_IMAGE_LOW_SIGNAL.value,
            )
        return _plane(z_actual_mm, diameter_um)

    def fit_zscan(self, points: list[BeamPlaneResult]) -> ZScanResult:
        valid_points_count = sum(point.valid for point in points)
        if valid_points_count < 3:
            return ZScanResult(
                points=points,
                full_angle_x_mrad=None,
                full_angle_y_mrad=None,
                half_angle_x_mrad=None,
                half_angle_y_mrad=None,
                waist_z_x_mm=None,
                waist_z_y_mm=None,
                waist_diameter_x_um=None,
                waist_diameter_y_um=None,
                fit_r2_x=None,
                fit_r2_y=None,
                valid_points_count=valid_points_count,
                judgement="fail",
                invalid_reason=ErrorCode.E_ANALYSIS_ZSCAN_TOO_FEW_POINTS.value,
            )
        return ZScanResult(
            points=points,
            full_angle_x_mrad=4.0,
            full_angle_y_mrad=4.0,
            half_angle_x_mrad=2.0,
            half_angle_y_mrad=2.0,
            waist_z_x_mm=0.0,
            waist_z_y_mm=0.0,
            waist_diameter_x_um=18.0,
            waist_diameter_y_um=18.0,
            fit_r2_x=1.0,
            fit_r2_y=1.0,
            valid_points_count=valid_points_count,
            judgement="unknown",
        )

    def judge_result(self, zscan_result: ZScanResult, limits: dict[str, Any] | None = None) -> str:
        return "INVALID" if zscan_result.invalid_reason else "PASS"


class FakeDataStore:
    def __init__(self) -> None:
        self.plane_results: list[BeamPlaneResult] = []
        self.failures: list[dict[str, Any]] = []
        self.zscan_results: list[ZScanResult] = []

    def save_plane_result(self, **kwargs: Any) -> None:
        self.plane_results.append(kwargs["plane_result"])

    def save_failure_data(self, failure_payload: dict[str, Any]) -> None:
        self.failures.append(failure_payload)

    def save_zscan_result(self, **kwargs: Any) -> None:
        self.zscan_results.append(kwargs["zscan_result"])


def _frame(z_actual_mm: float) -> ImageFrame:
    return ImageFrame(
        image=np.ones((8, 8), dtype=np.float64),
        timestamp_iso="2026-06-18T00:00:00+00:00",
        exposure_us=1000.0,
        gain=1.0,
        width_px=8,
        height_px=8,
        z_actual_mm=z_actual_mm,
    )


def _recipe(**overrides: object) -> ZScanRecipe:
    values = {
        "recipe_id": "unit-sequence",
        "operator_mode": "simulated",
        "z_ref_mm": 0.0,
        "prescan_delta_um": 50.0,
        "min_points": 5,
        "max_points": 5,
        "max_scan_half_range_um": 500.0,
        "target_diameter_change_per_step_um": 2.0,
        "auto_exposure_enabled": False,
        "average_frame_count": 1,
        "field_of_view_margin_percent": 95.0,
    }
    values.update(overrides)
    return ZScanRecipe(**values)


def _command(recipe: ZScanRecipe) -> ZScanRunCommand:
    return ZScanRunCommand(
        run_id="run-unit",
        sample_id="sample-unit",
        recipe=recipe,
        calibration={"field_of_view_x_um": 100.0, "field_of_view_y_um": 100.0},
    )


def _service(
    *,
    stage: FakeStage | None = None,
    interlock: FakeInterlock | None = None,
    acquisition: FakeAcquisition | None = None,
    analysis: FakeAnalysis | None = None,
    data_store: FakeDataStore | None = None,
) -> tuple[SequencerService, FakeDataStore, FakeStage, FakeAcquisition]:
    stage = stage or FakeStage()
    acquisition = acquisition or FakeAcquisition()
    data_store = data_store or FakeDataStore()
    dependencies = SequencerDependencies(
        stage=stage,
        profiler=FakeProfiler(),
        interlock=interlock or FakeInterlock(),
        acquisition=acquisition,
        analysis=analysis or FakeAnalysis(),
        data_store=data_store,
    )
    return SequencerService(dependencies), data_store, stage, acquisition


def test_valid_sequence_completes_and_saves_result() -> None:
    service, data_store, _, _ = _service()

    result = service.run_zscan(_command(_recipe()))

    assert result.judgement == "pass"
    assert result.full_angle_x_mrad == pytest.approx(4.0)
    assert len(data_store.zscan_results) == 1
    assert SequencerState.COMPLETE in {event.state for event in service.events}


def test_soft_limit_exception_runs_recovery_and_preserves_failure_data() -> None:
    service, data_store, stage, _ = _service(stage=FakeStage(z_min_mm=-0.1, z_max_mm=0.1))

    with pytest.raises(LBQAError) as exc_info:
        service.run_zscan(_command(_recipe()))

    assert exc_info.value.error_code == ErrorCode.E_DEVICE_STAGE_SOFT_LIMIT.value
    assert stage.stopped is True
    assert data_store.failures
    assert data_store.failures[-1]["errors"] == [ErrorCode.E_DEVICE_STAGE_SOFT_LIMIT.value]


def test_camera_timeout_exception_runs_recovery() -> None:
    service, data_store, _, acquisition = _service(acquisition=FakeAcquisition(fail_capture=True))

    with pytest.raises(LBQAError) as exc_info:
        service.run_zscan(_command(_recipe()))

    assert exc_info.value.error_code == ErrorCode.E_DEVICE_CAMERA_TIMEOUT.value
    assert acquisition.stopped is True
    assert data_store.failures[-1]["errors"] == [ErrorCode.E_DEVICE_CAMERA_TIMEOUT.value]


def test_interlock_unsafe_stops_before_motion_and_saves_failure() -> None:
    service, data_store, stage, _ = _service(interlock=FakeInterlock(safe=False))

    with pytest.raises(LBQAError) as exc_info:
        service.run_zscan(_command(_recipe()))

    assert exc_info.value.error_code == ErrorCode.E_SAFETY_INTERLOCK_OPEN.value
    assert stage.position_mm == 0.0
    assert data_store.failures[-1]["errors"] == [ErrorCode.E_SAFETY_INTERLOCK_OPEN.value]


def test_invalid_points_too_few_returns_invalid_zscan_result() -> None:
    service, data_store, _, _ = _service(analysis=FakeAnalysis(invalid_after_prescan=True))

    result = service.run_zscan(_command(_recipe(invalid_point_policy="skip")))

    assert result.judgement == "fail"
    assert result.invalid_reason == ErrorCode.E_ANALYSIS_ZSCAN_TOO_FEW_POINTS.value
    assert result.valid_points_count < 3
    assert len(data_store.zscan_results) == 1
