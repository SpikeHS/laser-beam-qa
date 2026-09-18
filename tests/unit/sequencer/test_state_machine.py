import pytest
from lbqa_sequencer.state_machine import SequencerState, SequenceStateMachine


def test_state_machine_allows_automatic_zscan_flow() -> None:
    machine = SequenceStateMachine()
    for state in (
        SequencerState.INITIALIZE,
        SequencerState.CONNECT_DEVICES,
        SequencerState.SAFETY_CHECK,
        SequencerState.LOAD_RECIPE,
        SequencerState.HOME_STAGE,
        SequencerState.LOAD_CALIBRATION,
        SequencerState.DARK_FRAME,
        SequencerState.AUTO_ALIGN,
        SequencerState.AUTO_EXPOSURE,
        SequencerState.PRE_SCAN_Z,
        SequencerState.BUILD_ZSCAN_PLAN,
        SequencerState.ZSCAN_LOOP,
        SequencerState.ANALYZE_ZSCAN,
        SequencerState.JUDGE_RESULT,
        SequencerState.SAVE_DATA,
        SequencerState.GENERATE_REPORT,
        SequencerState.COMPLETE,
        SequencerState.IDLE,
    ):
        machine.transition_to(state)

    assert machine.state == SequencerState.IDLE
    assert len(machine.history) == 18


def test_state_machine_rejects_out_of_order_transition() -> None:
    machine = SequenceStateMachine()

    with pytest.raises(ValueError, match="invalid sequencer transition"):
        machine.transition_to(SequencerState.ZSCAN_LOOP)


def test_state_machine_error_recovery_returns_to_idle() -> None:
    machine = SequenceStateMachine()
    machine.transition_to(SequencerState.INITIALIZE)
    machine.transition_to(SequencerState.CONNECT_DEVICES)

    for state in (
        SequencerState.ERROR_HANDLING,
        SequencerState.STOP_STAGE,
        SequencerState.STOP_ACQUISITION,
        SequencerState.SAVE_FAILURE_DATA,
        SequencerState.REPORT_ERROR,
        SequencerState.IDLE,
    ):
        machine.transition_to(state)

    assert machine.state == SequencerState.IDLE
