"""Data-layer exceptions using the project error-code contract."""

from __future__ import annotations

from lbqa_contracts.errors import ErrorCode, LBQAError


class DataStoreError(LBQAError):
    """Raised when the data/report layer cannot persist a run artifact."""


class DataStoreOverwriteError(DataStoreError):
    """Raised when a write would overwrite raw or processed measurement data."""


def raise_data_error(error_code: ErrorCode | str, message: str) -> None:
    """Raise a project-coded data-layer error."""

    raise DataStoreError(error_code, message)


def raise_overwrite_error(path: object) -> None:
    """Raise the standard no-overwrite error for measurement artifacts."""

    raise DataStoreOverwriteError(
        ErrorCode.E_REPORT_GENERATION_FAILED,
        f"refusing to overwrite existing measurement artifact: {path}",
    )


__all__ = [
    "DataStoreError",
    "DataStoreOverwriteError",
    "raise_data_error",
    "raise_overwrite_error",
]
