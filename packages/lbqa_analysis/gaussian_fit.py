"""Gaussian beam-width estimates without SciPy.

RayCi's displayed beam width is configured as a Gaussian-fit metric.  The
project keeps the public plane-result contract fields named `d4sigma_*`; for a
Gaussian beam those fields are populated with the fitted 4-sigma, 1/e^2
diameter equivalent.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from lbqa_contracts.errors import ErrorCode

from lbqa_analysis.exceptions import raise_analysis_error
from lbqa_analysis.preprocess import as_float_image

_DEFAULT_CORE_THRESHOLD_PERCENT = 20.0
_DEFAULT_ROI_HALF_WIDTH_PX = 120
_BACKGROUND_PERCENTILE = 20.0
_MIN_FIT_PIXELS = 30


def estimate_gaussian_2d(
    image: np.ndarray[Any, Any],
    effective_pixel_x_um: float,
    effective_pixel_y_um: float,
    *,
    core_threshold_percent: float = _DEFAULT_CORE_THRESHOLD_PERCENT,
    roi_half_width_px: int = _DEFAULT_ROI_HALF_WIDTH_PX,
) -> dict[str, float]:
    """Fit an elliptical Gaussian core and return 4-sigma width equivalents.

    The fit is a weighted least-squares solve of a quadratic surface in
    `log(signal)`, restricted to a peak-centered ROI. This keeps broad
    background/side-lobe structure from dominating the beam-width result.
    """

    _validate_inputs(
        effective_pixel_x_um=effective_pixel_x_um,
        effective_pixel_y_um=effective_pixel_y_um,
        core_threshold_percent=core_threshold_percent,
        roi_half_width_px=roi_half_width_px,
    )
    image_float = as_float_image(image)
    signal, background_value = _background_subtracted_signal(image_float)
    peak_value = float(np.max(signal))
    if peak_value <= 0.0:
        raise_analysis_error(
            ErrorCode.E_IMAGE_LOW_SIGNAL,
            "Gaussian fit cannot be calculated because the signal peak is zero.",
        )

    peak_y_px, peak_x_px = np.unravel_index(np.argmax(signal), signal.shape)
    roi, x_offset_px, y_offset_px = _peak_centered_roi(
        signal,
        peak_x_px=int(peak_x_px),
        peak_y_px=int(peak_y_px),
        roi_half_width_px=int(roi_half_width_px),
    )
    fit = _fit_log_gaussian_roi(
        roi,
        threshold_value=peak_value * float(core_threshold_percent) / 100.0,
    )
    center_x_px = float(x_offset_px) + fit["centroid_x_px"]
    center_y_px = float(y_offset_px) + fit["centroid_y_px"]

    sigma_x_um = fit["sigma_x_px"] * effective_pixel_x_um
    sigma_y_um = fit["sigma_y_px"] * effective_pixel_y_um
    sigma_major_um = fit["sigma_major_px"] * _mean_pixel_size_um(
        effective_pixel_x_um,
        effective_pixel_y_um,
    )
    sigma_minor_um = fit["sigma_minor_px"] * _mean_pixel_size_um(
        effective_pixel_x_um,
        effective_pixel_y_um,
    )
    d4sigma_major_um = 4.0 * sigma_major_um
    d4sigma_minor_um = 4.0 * sigma_minor_um
    return {
        "amplitude_value": peak_value,
        "background_value": background_value,
        "centroid_x_um": center_x_px * effective_pixel_x_um,
        "centroid_y_um": center_y_px * effective_pixel_y_um,
        "centroid_x_px": center_x_px,
        "centroid_y_px": center_y_px,
        "sigma_x_um": sigma_x_um,
        "sigma_y_um": sigma_y_um,
        "sigma_major_um": sigma_major_um,
        "sigma_minor_um": sigma_minor_um,
        "d4sigma_x_um": 4.0 * sigma_x_um,
        "d4sigma_y_um": 4.0 * sigma_y_um,
        "d4sigma_major_um": d4sigma_major_um,
        "d4sigma_minor_um": d4sigma_minor_um,
        "ellipticity": d4sigma_minor_um / d4sigma_major_um,
        "azimuth_deg": fit["azimuth_deg"],
        "fit_pixels_count": float(fit["fit_pixels_count"]),
        "core_threshold_percent": float(core_threshold_percent),
    }


def _validate_inputs(
    *,
    effective_pixel_x_um: float,
    effective_pixel_y_um: float,
    core_threshold_percent: float,
    roi_half_width_px: int,
) -> None:
    if effective_pixel_x_um <= 0.0 or effective_pixel_y_um <= 0.0:
        raise_analysis_error(
            ErrorCode.E_ANALYSIS_D4SIGMA_FAILED,
            "effective pixel sizes must be positive.",
        )
    if not 0.0 < float(core_threshold_percent) < 100.0:
        raise_analysis_error(
            ErrorCode.E_ANALYSIS_D4SIGMA_FAILED,
            "Gaussian core_threshold_percent must be in (0, 100).",
        )
    if roi_half_width_px < 8:
        raise_analysis_error(
            ErrorCode.E_ANALYSIS_D4SIGMA_FAILED,
            "Gaussian roi_half_width_px must be at least 8.",
        )


def _background_subtracted_signal(
    image_float: np.ndarray[Any, Any],
) -> tuple[np.ndarray[Any, Any], float]:
    background_value = float(np.percentile(image_float, _BACKGROUND_PERCENTILE))
    signal = image_float.astype(np.float64, copy=True) - background_value
    np.maximum(signal, 0.0, out=signal)
    return signal, background_value


def _peak_centered_roi(
    signal: np.ndarray[Any, Any],
    *,
    peak_x_px: int,
    peak_y_px: int,
    roi_half_width_px: int,
) -> tuple[np.ndarray[Any, Any], int, int]:
    height_px, width_px = signal.shape
    x_min = max(0, int(peak_x_px) - int(roi_half_width_px))
    x_max = min(width_px, int(peak_x_px) + int(roi_half_width_px) + 1)
    y_min = max(0, int(peak_y_px) - int(roi_half_width_px))
    y_max = min(height_px, int(peak_y_px) + int(roi_half_width_px) + 1)
    return signal[y_min:y_max, x_min:x_max], x_min, y_min


def _fit_log_gaussian_roi(
    roi_signal: np.ndarray[Any, Any],
    *,
    threshold_value: float,
) -> dict[str, float]:
    mask = roi_signal >= threshold_value
    if int(np.count_nonzero(mask)) < _MIN_FIT_PIXELS:
        raise_analysis_error(
            ErrorCode.E_ANALYSIS_D4SIGMA_FAILED,
            "Gaussian fit has too few core pixels.",
        )

    y_px, x_px = np.indices(roi_signal.shape, dtype=np.float64)
    signal_values = roi_signal[mask].astype(np.float64, copy=False)
    weight_sum = float(np.sum(signal_values))
    if weight_sum <= 0.0:
        raise_analysis_error(
            ErrorCode.E_IMAGE_LOW_SIGNAL,
            "Gaussian fit cannot be calculated because core signal is zero.",
        )

    centroid_x_px = float(np.sum(signal_values * x_px[mask]) / weight_sum)
    centroid_y_px = float(np.sum(signal_values * y_px[mask]) / weight_sum)
    x_centered_px = x_px[mask] - centroid_x_px
    y_centered_px = y_px[mask] - centroid_y_px
    log_signal = np.log(np.maximum(signal_values, float(np.max(signal_values)) * 1e-9))
    weights = np.sqrt(signal_values / float(np.max(signal_values)))
    design = np.column_stack(
        [
            np.ones_like(x_centered_px),
            x_centered_px,
            y_centered_px,
            x_centered_px * x_centered_px,
            x_centered_px * y_centered_px,
            y_centered_px * y_centered_px,
        ]
    )
    coefficients, _, rank, _ = np.linalg.lstsq(
        design * weights[:, None],
        log_signal * weights,
        rcond=None,
    )
    if rank < 6:
        raise_analysis_error(
            ErrorCode.E_ANALYSIS_D4SIGMA_FAILED,
            "Gaussian fit quadratic solve is rank deficient.",
        )

    _, linear_x, linear_y, quad_xx, quad_xy, quad_yy = coefficients
    quadratic = np.array(
        [[quad_xx, quad_xy / 2.0], [quad_xy / 2.0, quad_yy]],
        dtype=np.float64,
    )
    eigenvalues = np.linalg.eigvalsh(quadratic)
    if not np.all(eigenvalues < 0.0):
        raise_analysis_error(
            ErrorCode.E_ANALYSIS_D4SIGMA_FAILED,
            "Gaussian fit quadratic surface is not negative definite.",
        )

    inverse_quadratic = np.linalg.inv(quadratic)
    covariance_px2 = -0.5 * inverse_quadratic
    covariance_eigenvalues, covariance_eigenvectors = np.linalg.eigh(covariance_px2)
    if not np.all(covariance_eigenvalues > 0.0):
        raise_analysis_error(
            ErrorCode.E_ANALYSIS_D4SIGMA_FAILED,
            "Gaussian fit covariance is not positive definite.",
        )

    center_correction_px = -0.5 * inverse_quadratic @ np.array([linear_x, linear_y])
    centroid_x_px += float(center_correction_px[0])
    centroid_y_px += float(center_correction_px[1])
    sigma_x_px = math.sqrt(max(float(covariance_px2[0, 0]), 0.0))
    sigma_y_px = math.sqrt(max(float(covariance_px2[1, 1]), 0.0))
    sigma_minor_px = math.sqrt(float(covariance_eigenvalues[0]))
    sigma_major_px = math.sqrt(float(covariance_eigenvalues[1]))
    major_vector = covariance_eigenvectors[:, 1]
    azimuth_deg = math.degrees(math.atan2(float(major_vector[1]), float(major_vector[0])))
    if azimuth_deg >= 90.0:
        azimuth_deg -= 180.0
    if azimuth_deg < -90.0:
        azimuth_deg += 180.0

    return {
        "centroid_x_px": centroid_x_px,
        "centroid_y_px": centroid_y_px,
        "sigma_x_px": sigma_x_px,
        "sigma_y_px": sigma_y_px,
        "sigma_major_px": sigma_major_px,
        "sigma_minor_px": sigma_minor_px,
        "azimuth_deg": float(azimuth_deg),
        "fit_pixels_count": float(np.count_nonzero(mask)),
    }


def _mean_pixel_size_um(effective_pixel_x_um: float, effective_pixel_y_um: float) -> float:
    return (float(effective_pixel_x_um) + float(effective_pixel_y_um)) / 2.0


__all__ = ["estimate_gaussian_2d"]
