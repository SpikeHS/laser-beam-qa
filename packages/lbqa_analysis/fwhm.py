"""Full-width-at-half-maximum beam metrics."""

from __future__ import annotations

from typing import Any

import numpy as np

from lbqa_analysis.beam_metrics import calculate_centroid
from lbqa_analysis.preprocess import as_float_image


def calculate_fwhm_xy(
    image: np.ndarray[Any, Any],
    effective_pixel_x_um: float,
    effective_pixel_y_um: float,
) -> dict[str, float | None]:
    """Calculate center-line FWHM in x/y using linear crossing interpolation."""

    image_float = as_float_image(image)
    centroid = calculate_centroid(image_float, effective_pixel_x_um, effective_pixel_y_um)
    centroid_x_px = centroid["centroid_x_um"] / effective_pixel_x_um
    centroid_y_px = centroid["centroid_y_um"] / effective_pixel_y_um
    center_column_px = int(round(centroid_x_px))
    center_row_px = int(round(centroid_y_px))
    center_column_px = int(np.clip(center_column_px, 0, image_float.shape[1] - 1))
    center_row_px = int(np.clip(center_row_px, 0, image_float.shape[0] - 1))

    fwhm_x_um = _profile_fwhm_um(image_float[center_row_px, :], effective_pixel_x_um)
    fwhm_y_um = _profile_fwhm_um(image_float[:, center_column_px], effective_pixel_y_um)
    return {"fwhm_x_um": fwhm_x_um, "fwhm_y_um": fwhm_y_um}


def _profile_fwhm_um(profile: np.ndarray[Any, Any], pixel_size_um: float) -> float | None:
    profile_float = np.asarray(profile, dtype=np.float64)
    baseline = float(np.min(profile_float))
    peak = float(np.max(profile_float))
    amplitude = peak - baseline
    if amplitude <= 0.0:
        return None

    half_level = baseline + 0.5 * amplitude
    above_indices = np.flatnonzero(profile_float >= half_level)
    if above_indices.size == 0:
        return None

    left_index = int(above_indices[0])
    right_index = int(above_indices[-1])
    if left_index == 0 or right_index == profile_float.size - 1:
        return None

    left_crossing_px = _interpolate_crossing_px(
        left_index - 1,
        profile_float[left_index - 1],
        left_index,
        profile_float[left_index],
        half_level,
    )
    right_crossing_px = _interpolate_crossing_px(
        right_index,
        profile_float[right_index],
        right_index + 1,
        profile_float[right_index + 1],
        half_level,
    )
    width_px = max(right_crossing_px - left_crossing_px, 0.0)
    return width_px * pixel_size_um


def _interpolate_crossing_px(
    x0_px: int,
    y0: float,
    x1_px: int,
    y1: float,
    target_y: float,
) -> float:
    if abs(y1 - y0) < 1e-15:
        return float(x0_px)
    fraction = (target_y - y0) / (y1 - y0)
    return float(x0_px) + float(fraction) * float(x1_px - x0_px)


__all__ = ["calculate_fwhm_xy"]
