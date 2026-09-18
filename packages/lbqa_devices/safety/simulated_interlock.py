"""Simulated safety interlock implementing InterlockPort."""

from __future__ import annotations

from dataclasses import dataclass

from lbqa_contracts.errors import ErrorCode


@dataclass(slots=True)
class SimulatedInterlock:
    safe: bool = True
    reason: str | None = None

    def is_safe(self) -> bool:
        return self.safe

    def get_status(self) -> dict[str, object]:
        return {
            "safe": self.safe,
            "reason": self.reason,
            "error_code": None if self.safe else ErrorCode.E_SAFETY_INTERLOCK_OPEN.value,
        }

    def set_safe(self) -> None:
        self.safe = True
        self.reason = None

    def set_unsafe(self, reason: str = "simulated interlock open") -> None:
        self.safe = False
        self.reason = reason
