"""Error recovery flow for sequencer-controlled runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lbqa_contracts.errors import ErrorCode, LBQAError

from lbqa_sequencer.events import SequencerEvent, SequencerEventType
from lbqa_sequencer.state_machine import SequencerState, SequenceStateMachine


@dataclass(slots=True)
class ErrorRecoveryService:
    """Run the required error path and preserve partial run data."""

    state_machine: SequenceStateMachine
    stage: Any
    acquisition: Any
    data_store: Any | None = None
    report_service: Any | None = None
    events: list[SequencerEvent] | None = None

    def recover(self, context: Any, exc: BaseException) -> None:
        error_code = error_code_from_exception(exc)
        context.errors.append(error_code)
        context.failure_exception = exc
        context.aborted = error_code == ErrorCode.E_USER_ABORT.value

        self._transition(SequencerState.ERROR_HANDLING, "entered sequencer error recovery")
        self._transition(SequencerState.STOP_STAGE, "stopping stage after sequencer error")
        self._call_safely(self.stage, "stop")
        self._transition(
            SequencerState.STOP_ACQUISITION,
            "stopping acquisition after sequencer error",
        )
        self._call_safely(self.acquisition, "stop")
        self._transition(SequencerState.SAVE_FAILURE_DATA, "saving partial failure data")
        self._call_safely(
            self.data_store,
            "save_failure_data",
            failure_payload=context.failure_payload(),
        )
        self._transition(SequencerState.REPORT_ERROR, "reporting sequencer error")
        self._call_safely(
            self.report_service,
            "report_error",
            failure_payload=context.failure_payload(),
        )
        terminal_state = SequencerState.ABORT if context.aborted else SequencerState.IDLE
        self._transition(terminal_state, f"error recovery finished at {terminal_state.value}")

    def _transition(self, state: SequencerState, message: str) -> None:
        self.state_machine.transition_to(state)
        if self.events is not None:
            self.events.append(
                SequencerEvent(
                    event_type=SequencerEventType.RECOVERY_ACTION,
                    state=state,
                    message=message,
                )
            )

    @staticmethod
    def _call_safely(target: Any | None, method_name: str, **kwargs: Any) -> None:
        if target is None:
            return
        method = getattr(target, method_name, None)
        if method is None:
            return
        try:
            method(**kwargs)
        except TypeError:
            method()
        except Exception:
            return


def error_code_from_exception(exc: BaseException) -> str:
    if isinstance(exc, LBQAError):
        return exc.error_code
    return ErrorCode.E_ANALYSIS_ZSCAN_FIT_FAILED.value


__all__ = ["ErrorRecoveryService", "error_code_from_exception"]
