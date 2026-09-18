"""State machine for automatic and semi-automatic test sequencing."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class SequencerState(StrEnum):
    IDLE = "Idle"
    INITIALIZE = "Initialize"
    CONNECT_DEVICES = "ConnectDevices"
    SAFETY_CHECK = "SafetyCheck"
    LOAD_RECIPE = "LoadRecipe"
    HOME_STAGE = "HomeStage"
    LOAD_CALIBRATION = "LoadCalibration"
    DARK_FRAME = "DarkFrame"
    MANUAL_ALIGN = "ManualAlign"
    AUTO_ALIGN = "AutoAlign"
    AUTO_EXPOSURE = "AutoExposure"
    PRE_SCAN_Z = "PreScanZ"
    BUILD_ZSCAN_PLAN = "BuildZScanPlan"
    ZSCAN_LOOP = "ZScanLoop"
    ANALYZE_ZSCAN = "AnalyzeZScan"
    JUDGE_RESULT = "JudgeResult"
    SAVE_DATA = "SaveData"
    GENERATE_REPORT = "GenerateReport"
    COMPLETE = "Complete"
    ERROR_HANDLING = "ErrorHandling"
    STOP_STAGE = "StopStage"
    STOP_ACQUISITION = "StopAcquisition"
    SAVE_FAILURE_DATA = "SaveFailureData"
    REPORT_ERROR = "ReportError"
    ABORT = "Abort"


NORMAL_FLOW: tuple[SequencerState, ...] = (
    SequencerState.IDLE,
    SequencerState.INITIALIZE,
    SequencerState.CONNECT_DEVICES,
    SequencerState.SAFETY_CHECK,
    SequencerState.LOAD_RECIPE,
    SequencerState.HOME_STAGE,
    SequencerState.LOAD_CALIBRATION,
    SequencerState.DARK_FRAME,
    SequencerState.AUTO_EXPOSURE,
    SequencerState.PRE_SCAN_Z,
    SequencerState.BUILD_ZSCAN_PLAN,
    SequencerState.ZSCAN_LOOP,
    SequencerState.ANALYZE_ZSCAN,
    SequencerState.JUDGE_RESULT,
    SequencerState.SAVE_DATA,
    SequencerState.GENERATE_REPORT,
    SequencerState.COMPLETE,
)


ERROR_FLOW: tuple[SequencerState, ...] = (
    SequencerState.ERROR_HANDLING,
    SequencerState.STOP_STAGE,
    SequencerState.STOP_ACQUISITION,
    SequencerState.SAVE_FAILURE_DATA,
    SequencerState.REPORT_ERROR,
)


_ALLOWED_TRANSITIONS: dict[SequencerState, set[SequencerState]] = {
    SequencerState.IDLE: {SequencerState.INITIALIZE},
    SequencerState.INITIALIZE: {SequencerState.CONNECT_DEVICES},
    SequencerState.CONNECT_DEVICES: {SequencerState.SAFETY_CHECK},
    SequencerState.SAFETY_CHECK: {SequencerState.LOAD_RECIPE},
    SequencerState.LOAD_RECIPE: {SequencerState.HOME_STAGE},
    SequencerState.HOME_STAGE: {SequencerState.LOAD_CALIBRATION},
    SequencerState.LOAD_CALIBRATION: {SequencerState.DARK_FRAME},
    SequencerState.DARK_FRAME: {SequencerState.MANUAL_ALIGN, SequencerState.AUTO_ALIGN},
    SequencerState.MANUAL_ALIGN: {SequencerState.AUTO_EXPOSURE},
    SequencerState.AUTO_ALIGN: {SequencerState.AUTO_EXPOSURE},
    SequencerState.AUTO_EXPOSURE: {SequencerState.PRE_SCAN_Z},
    SequencerState.PRE_SCAN_Z: {SequencerState.BUILD_ZSCAN_PLAN},
    SequencerState.BUILD_ZSCAN_PLAN: {SequencerState.ZSCAN_LOOP},
    SequencerState.ZSCAN_LOOP: {SequencerState.ANALYZE_ZSCAN},
    SequencerState.ANALYZE_ZSCAN: {SequencerState.JUDGE_RESULT},
    SequencerState.JUDGE_RESULT: {SequencerState.SAVE_DATA},
    SequencerState.SAVE_DATA: {SequencerState.GENERATE_REPORT},
    SequencerState.GENERATE_REPORT: {SequencerState.COMPLETE},
    SequencerState.COMPLETE: {SequencerState.IDLE},
    SequencerState.ERROR_HANDLING: {SequencerState.STOP_STAGE},
    SequencerState.STOP_STAGE: {SequencerState.STOP_ACQUISITION},
    SequencerState.STOP_ACQUISITION: {SequencerState.SAVE_FAILURE_DATA},
    SequencerState.SAVE_FAILURE_DATA: {SequencerState.REPORT_ERROR},
    SequencerState.REPORT_ERROR: {SequencerState.IDLE, SequencerState.ABORT},
    SequencerState.ABORT: {SequencerState.IDLE},
}

_ERROR_SOURCES = set(SequencerState) - set(ERROR_FLOW) - {SequencerState.ABORT}
for _state in _ERROR_SOURCES:
    _ALLOWED_TRANSITIONS.setdefault(_state, set()).add(SequencerState.ERROR_HANDLING)


@dataclass(frozen=True, slots=True)
class StateTransition:
    from_state: SequencerState
    to_state: SequencerState


class SequenceStateMachine:
    """Small deterministic state machine with explicit error recovery edges."""

    def __init__(self, initial_state: SequencerState = SequencerState.IDLE) -> None:
        self._state = initial_state
        self._history: list[StateTransition] = []

    @property
    def state(self) -> SequencerState:
        return self._state

    @property
    def history(self) -> tuple[StateTransition, ...]:
        return tuple(self._history)

    def can_transition_to(self, next_state: SequencerState) -> bool:
        return next_state in _ALLOWED_TRANSITIONS.get(self._state, set())

    def transition_to(self, next_state: SequencerState) -> StateTransition:
        if not self.can_transition_to(next_state):
            raise ValueError(f"invalid sequencer transition: {self._state} -> {next_state}")
        transition = StateTransition(from_state=self._state, to_state=next_state)
        self._state = next_state
        self._history.append(transition)
        return transition

    def reset_to_idle(self) -> None:
        self._state = SequencerState.IDLE


__all__ = [
    "ERROR_FLOW",
    "NORMAL_FLOW",
    "SequencerState",
    "SequenceStateMachine",
    "StateTransition",
]
