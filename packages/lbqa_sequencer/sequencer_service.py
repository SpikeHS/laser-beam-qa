"""Public service facade for sequencer-controlled z-scan runs."""

from __future__ import annotations

from lbqa_contracts.models import ZScanResult

from lbqa_sequencer.commands import CommandQueue, ZScanRunCommand
from lbqa_sequencer.events import SequencerEvent
from lbqa_sequencer.zscan_sequence import SequencerDependencies, ZScanSequence


class SequencerService:
    """Thin facade used by CLI/UI layers without exposing workflow internals."""

    def __init__(self, dependencies: SequencerDependencies) -> None:
        self._dependencies = dependencies
        self._last_sequence: ZScanSequence | None = None

    @property
    def command_queue(self) -> CommandQueue:
        return self._dependencies.command_queue

    @property
    def events(self) -> tuple[SequencerEvent, ...]:
        if self._last_sequence is None:
            return ()
        return tuple(self._last_sequence.events)

    def run_zscan(self, command: ZScanRunCommand) -> ZScanResult:
        sequence = ZScanSequence(self._dependencies)
        self._last_sequence = sequence
        return sequence.run(command)

    def request_abort(self, reason: str | None = None) -> None:
        self.command_queue.request_abort(reason)

    def request_retry_point(self, reason: str | None = None) -> None:
        self.command_queue.request_retry_point(reason)

    def request_skip_point(self, reason: str | None = None) -> None:
        self.command_queue.request_skip_point(reason)


__all__ = ["SequencerService"]
