"""Validation helpers shared by contract models and tests."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any

import numpy as np

from lbqa_contracts.errors import ERROR_CODE_VALUES
from lbqa_contracts.units import is_allowed_contract_field_name


def validate_positive(field_name: str, value: float | int) -> None:
    if value <= 0:
        raise ValueError(f"{field_name} must be positive.")


def validate_non_negative(field_name: str, value: float | int) -> None:
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative.")


def validate_optional_error_code(error_code: str | None) -> None:
    if error_code is not None and error_code not in ERROR_CODE_VALUES:
        raise ValueError(f"Unknown error_code: {error_code}")


def validate_error_codes(errors: list[str]) -> None:
    unknown_codes = [error_code for error_code in errors if error_code not in ERROR_CODE_VALUES]
    if unknown_codes:
        raise ValueError(f"Unknown error codes: {unknown_codes}")


def validate_image_shape(
    image: np.ndarray[Any, Any], *, width_px: int, height_px: int
) -> None:
    if image.ndim != 2:
        raise ValueError("image must be a 2D numpy.ndarray.")
    if image.shape != (height_px, width_px):
        raise ValueError(
            f"image shape {image.shape} does not match height_px/width_px "
            f"({height_px}, {width_px})."
        )


def invalid_unit_field_names(model_type: type[Any]) -> list[str]:
    """Return dataclass field names that do not follow the contract convention."""

    if not is_dataclass(model_type):
        raise TypeError(f"{model_type!r} is not a dataclass type.")
    return [
        field.name
        for field in fields(model_type)
        if not is_allowed_contract_field_name(field.name)
    ]
