import pytest
from lbqa_contracts.errors import ErrorCode, LBQAError
from lbqa_devices.stage.simulated_stage import SimulatedStage


def test_simulated_stage_home_move_and_get_position() -> None:
    stage = SimulatedStage(z_min_mm=-0.5, z_max_mm=0.5)

    stage.connect()
    stage.enable()
    stage.home()
    stage.move_abs_mm(0.2)

    assert stage.is_homed()
    assert stage.get_position_mm() == pytest.approx(0.2)
    status = stage.get_status()
    assert status.connected
    assert status.enabled
    assert status.homed
    assert status.position_mm == pytest.approx(0.2)


def test_simulated_stage_soft_limit_is_enforced() -> None:
    stage = SimulatedStage(z_min_mm=-0.1, z_max_mm=0.1)
    stage.connect()
    stage.enable()
    stage.home()

    with pytest.raises(LBQAError) as exc_info:
        stage.move_abs_mm(0.2)

    assert exc_info.value.error_code == ErrorCode.E_DEVICE_STAGE_SOFT_LIMIT.value


def test_simulated_stage_requires_home_before_absolute_move() -> None:
    stage = SimulatedStage()
    stage.connect()
    stage.enable()

    with pytest.raises(LBQAError) as exc_info:
        stage.move_abs_mm(0.1)

    assert exc_info.value.error_code == ErrorCode.E_DEVICE_STAGE_NOT_HOMED.value


def test_simulated_stage_emergency_stop_locks_motion_until_clear() -> None:
    stage = SimulatedStage()
    stage.connect()
    stage.enable()
    stage.home()
    stage.emergency_stop()

    with pytest.raises(LBQAError) as exc_info:
        stage.move_abs_mm(0.1)

    assert exc_info.value.error_code == ErrorCode.E_SAFETY_COLLISION_RISK.value
    stage.clear_emergency_stop()
    stage.enable()
    stage.move_abs_mm(0.1)
    assert stage.get_position_mm() == pytest.approx(0.1)
