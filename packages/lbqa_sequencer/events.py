"""Sequencer event records for state, command, data, and error traceability."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from lbqa_sequencer.state_machine import SequencerState


class SequencerEventType(StrEnum):
    STATE_CHANGED = "state_changed"
    COMMAND_ACCEPTED = "command_accepted"
    PLANE_CAPTURED = "plane_captured"
    POINT_INVALID = "point_invalid"
    PLAN_BUILT = "plan_built"
    ERROR = "error"
    RECOVERY_ACTION = "recovery_action"
    COMPLETE = "complete"


@dataclass(frozen=True, slots=True)
class SequencerEvent:
    event_type: SequencerEventType
    state: SequencerState
    message: str
    timestamp_iso: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    error_code: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)


__all__ = ["SequencerEvent", "SequencerEventType"]
