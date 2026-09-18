"""Image preprocessing helpers for beam analysis."""

from __future__ import annotations

from typing import Any

import numpy as np
from lbqa_contracts.errors import ErrorCode
from numpy.typing import NDArray

from lbqa_analysis.exceptions import raise_analysis_error

ImageArray = NDArray[Any]


def as_float_image(image: ImageArray) -> NDArray[np.float64]:
    """Return a 2D image as ``float64`` without modifying the input array."""

    image_array = np.asarray(image)
    if image_array.ndim != 2:
        raise_analysis_error(ErrorCode.E_IMAGE_ROI_NOT_FOUND, "image must be a 2D array.")
    if image_array.size == 0:
        raise_analysis_error(ErrorCode.E_IMAGE_ROI_NOT_FOUND, "image must not be empty.")
    return image_array.astype(np.float64, copy=False)


def dark_subtract(raw: ImageArray, dark: ImageArray) -> NDArray[np.float64]:
    """Subtract a dark frame in floating point and clip negative values to zero."""

    raw_image = np.asarray(raw)
    dark_image = np.asarray(dark)
    if raw_image.shape != dark_image.shape:
        raise_analysis_error(
            ErrorCode.E_IMAGE_ROI_NOT_FOUND,
            f"raw shape {raw_image.shape} does not match dark shape {dark_image.shape}.",
        )
    corrected = raw_image.astype(np.float64) - dark_image.astype(np.float64)
    np.maximum(corrected, 0.0, out=corrected)
    return corrected


__all__ = ["as_float_image", "dark_subtract"]
