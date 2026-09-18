"""Analysis exceptions mapped to the project-wide error-code system."""

from __future__ import annotations

from lbqa_contracts.errors import ErrorCode, LBQAError


class AnalysisError(LBQAError):
    """Raised when a pure analysis operation cannot produce a valid result."""


def raise_analysis_error(error_code: ErrorCode, message: str) -> None:
    """Raise an analysis error with a stable project error code."""

    raise AnalysisError(error_code, message)


__all__ = ["AnalysisError", "raise_analysis_error"]
