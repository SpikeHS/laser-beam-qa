"""D4sigma and second-moment beam metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
from lbqa_contracts.errors import ErrorCode

from lbqa_analysis.beam_metrics import calculate_centroid
from lbqa_analysis.exceptions import raise_analysis_error
from lbqa_analysis.preprocess import as_float_image


def calculate_d4sigma_2d(
    image: np.ndarray[Any, Any],
    effective_pixel_x_um: float,
    effective_pixel_y_um: float,
) -> dict[str, float]:
    """Calculate ISO-style second-moment diameters for a 2D beam image."""

    if effective_pixel_x_um <= 0.0 or effective_pixel_y_um <= 0.0:
        raise_analysis_error(
            ErrorCode.E_ANALYSIS_D4SIGMA_FAILED,
            "effective pixel sizes must be positive.",
        )

    image_float = as_float_image(image)
    total_intensity = float(np.sum(image_float))
    if total_intensity <= 0.0:
        raise_analysis_error(
            ErrorCode.E_IMAGE_LOW_SIGNAL,
            "D4sigma cannot be calculated because total image intensity is zero.",
        )

    centroid = calculate_centroid(image_float, effective_pixel_x_um, effective_pixel_y_um)
    y_px, x_px = np.indices(image_float.shape, dtype=np.float64)
    x_um = x_px * effective_pixel_x_um
    y_um = y_px * effective_pixel_y_um
    dx_um = x_um - centroid["centroid_x_um"]
    dy_um = y_um - centroid["centroid_y_um"]

    sigma_xx_um2 = float(np.sum(image_float * dx_um * dx_um) / total_intensity)
    sigma_yy_um2 = float(np.sum(image_float * dy_um * dy_um) / total_intensity)
    sigma_xy_um2 = float(np.sum(image_float * dx_um * dy_um) / total_intensity)

    covariance_um2 = np.array(
        [[sigma_xx_um2, sigma_xy_um2], [sigma_xy_um2, sigma_yy_um2]],
        dtype=np.float64,
    )
    eigenvalues_um2 = np.linalg.eigvalsh(covariance_um2)
    minor_lambda_um2 = float(max(eigenvalues_um2[0], 0.0))
    major_lambda_um2 = float(max(eigenvalues_um2[1], 0.0))

    d4sigma_major_um = 4.0 * float(np.sqrt(major_lambda_um2))
    d4sigma_minor_um = 4.0 * float(np.sqrt(minor_lambda_um2))
    d4sigma_x_um = 4.0 * float(np.sqrt(max(sigma_xx_um2, 0.0)))
    d4sigma_y_um = 4.0 * float(np.sqrt(max(sigma_yy_um2, 0.0)))
    if d4sigma_major_um <= 0.0:
        raise_analysis_error(
            ErrorCode.E_ANALYSIS_D4SIGMA_FAILED,
            "major D4sigma diameter must be positive.",
        )

    ellipticity = d4sigma_minor_um / d4sigma_major_um
    azimuth_deg = _major_axis_azimuth_deg(sigma_xx_um2, sigma_yy_um2, sigma_xy_um2)

    return {
        "centroid_x_um": centroid["centroid_x_um"],
        "centroid_y_um": centroid["centroid_y_um"],
        "sigma_xx_um2": sigma_xx_um2,
        "sigma_yy_um2": sigma_yy_um2,
        "sigma_xy_um2": sigma_xy_um2,
        "d4sigma_x_um": d4sigma_x_um,
        "d4sigma_y_um": d4sigma_y_um,
        "d4sigma_major_um": d4sigma_major_um,
        "d4sigma_minor_um": d4sigma_minor_um,
        "ellipticity": ellipticity,
        "azimuth_deg": azimuth_deg,
    }


def _major_axis_azimuth_deg(
    sigma_xx_um2: float,
    sigma_yy_um2: float,
    sigma_xy_um2: float,
) -> float:
    if abs(sigma_xy_um2) < 1e-15 and abs(sigma_xx_um2 - sigma_yy_um2) < 1e-15:
        return 0.0
    azimuth_rad = 0.5 * np.arctan2(2.0 * sigma_xy_um2, sigma_xx_um2 - sigma_yy_um2)
    azimuth_deg = float(np.degrees(azimuth_rad))
    if azimuth_deg >= 90.0:
        azimuth_deg -= 180.0
    if azimuth_deg < -90.0:
        azimuth_deg += 180.0
    return azimuth_deg


__all__ = ["calculate_d4sigma_2d"]
