"""Single-plane beam metric orchestration."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
from lbqa_contracts.errors import ErrorCode
from lbqa_contracts.models import BeamPlaneResult

from lbqa_analysis.exceptions import AnalysisError, raise_analysis_error
from lbqa_analysis.preprocess import as_float_image


def calculate_centroid(
    image: np.ndarray[Any, Any],
    effective_pixel_x_um: float,
    effective_pixel_y_um: float,
) -> dict[str, float]:
    """Calculate the intensity-weighted beam centroid in object-plane micrometers."""

    _validate_pixel_size(effective_pixel_x_um, effective_pixel_y_um)
    image_float = as_float_image(image)
    total_intensity = float(np.sum(image_float))
    if total_intensity <= 0.0:
        raise_analysis_error(
            ErrorCode.E_IMAGE_LOW_SIGNAL,
            "centroid cannot be calculated because total image intensity is zero.",
        )

    y_px, x_px = np.indices(image_float.shape, dtype=np.float64)
    centroid_x_um = float(np.sum(image_float * x_px) / total_intensity) * effective_pixel_x_um
    centroid_y_um = float(np.sum(image_float * y_px) / total_intensity) * effective_pixel_y_um
    return {"centroid_x_um": centroid_x_um, "centroid_y_um": centroid_y_um}


def analyze_beam_plane(
    image: np.ndarray[Any, Any],
    calibration: Any,
    z_actual_mm: float,
    quality_limits: Mapping[str, Any] | None = None,
) -> BeamPlaneResult:
    """Analyze one dark-corrected beam image into the shared plane-result contract."""

    limits = dict(quality_limits or {})
    effective_pixel_x_um, effective_pixel_y_um = _effective_pixel_um(calibration)

    from lbqa_analysis.d4sigma import calculate_d4sigma_2d
    from lbqa_analysis.fwhm import calculate_fwhm_xy
    from lbqa_analysis.gaussian_fit import estimate_gaussian_2d
    from lbqa_analysis.quality_rules import evaluate_frame_quality

    bit_depth = int(limits.get("bit_depth", 12))
    frame_quality = evaluate_frame_quality(
        image=image,
        bit_depth=bit_depth,
        edge_margin_px=int(limits.get("edge_margin_px", 4)),
        edge_energy_limit_percent=float(limits.get("edge_energy_limit_percent", 5.0)),
        saturation_threshold_percent=float(limits.get("saturation_threshold_percent", 99.0)),
        low_signal_threshold_percent=float(limits.get("low_signal_threshold_percent", 1.0)),
    )

    try:
        if _uses_gaussian_width(limits):
            gaussian = estimate_gaussian_2d(
                image,
                effective_pixel_x_um,
                effective_pixel_y_um,
                core_threshold_percent=float(limits.get("gaussian_core_threshold_percent", 20.0)),
                roi_half_width_px=int(limits.get("gaussian_roi_half_width_px", 120)),
            )
            centroid = {
                "centroid_x_um": gaussian["centroid_x_um"],
                "centroid_y_um": gaussian["centroid_y_um"],
            }
            d4sigma = {
                "d4sigma_x_um": gaussian["d4sigma_x_um"],
                "d4sigma_y_um": gaussian["d4sigma_y_um"],
                "d4sigma_major_um": gaussian["d4sigma_major_um"],
                "d4sigma_minor_um": gaussian["d4sigma_minor_um"],
                "ellipticity": gaussian["ellipticity"],
                "azimuth_deg": gaussian["azimuth_deg"],
            }
            fwhm = _gaussian_fwhm(gaussian)
        else:
            centroid = calculate_centroid(image, effective_pixel_x_um, effective_pixel_y_um)
            d4sigma = calculate_d4sigma_2d(image, effective_pixel_x_um, effective_pixel_y_um)
            fwhm = calculate_fwhm_xy(image, effective_pixel_x_um, effective_pixel_y_um)
    except AnalysisError:
        invalid_reason = (
            frame_quality.invalid_reason or ErrorCode.E_ANALYSIS_D4SIGMA_FAILED.value
        )
        return _invalid_plane_result(
            image=image,
            z_actual_mm=z_actual_mm,
            frame_peak_value=frame_quality.peak_value,
            saturation_pixels=frame_quality.saturation_pixels,
            edge_energy_percent=frame_quality.edge_energy_percent,
            invalid_reason=invalid_reason,
        )

    peak_x_um, peak_y_um = _peak_position_um(image, effective_pixel_x_um, effective_pixel_y_um)
    valid = frame_quality.valid
    invalid_reason = frame_quality.invalid_reason if not valid else None

    return BeamPlaneResult(
        z_actual_mm=float(z_actual_mm),
        centroid_x_um=centroid["centroid_x_um"],
        centroid_y_um=centroid["centroid_y_um"],
        peak_x_um=peak_x_um,
        peak_y_um=peak_y_um,
        d4sigma_x_um=d4sigma["d4sigma_x_um"],
        d4sigma_y_um=d4sigma["d4sigma_y_um"],
        d4sigma_major_um=d4sigma["d4sigma_major_um"],
        d4sigma_minor_um=d4sigma["d4sigma_minor_um"],
        fwhm_x_um=fwhm["fwhm_x_um"],
        fwhm_y_um=fwhm["fwhm_y_um"],
        ellipticity=d4sigma["ellipticity"],
        azimuth_deg=d4sigma["azimuth_deg"],
        peak_value=frame_quality.peak_value,
        saturation_pixels=frame_quality.saturation_pixels,
        edge_energy_percent=frame_quality.edge_energy_percent,
        valid=valid,
        invalid_reason=invalid_reason,
    )


def _validate_pixel_size(effective_pixel_x_um: float, effective_pixel_y_um: float) -> None:
    if effective_pixel_x_um <= 0.0 or effective_pixel_y_um <= 0.0:
        raise_analysis_error(
            ErrorCode.E_ANALYSIS_D4SIGMA_FAILED,
            "effective pixel sizes must be positive.",
        )


def _uses_gaussian_width(limits: Mapping[str, Any]) -> bool:
    width_method = str(
        limits.get(
            "width_method",
            limits.get("primary_width_method", ""),
        )
    )
    normalized = width_method.strip().lower().replace("-", "_").replace(" ", "_")
    return normalized in {"gaussian", "gaussian_fit", "gauss_fit", "rayci_gaussian_fit"}


def _gaussian_fwhm(gaussian: Mapping[str, float]) -> dict[str, float]:
    factor = 2.0 * math.sqrt(2.0 * math.log(2.0))
    return {
        "fwhm_x_um": factor * float(gaussian["sigma_x_um"]),
        "fwhm_y_um": factor * float(gaussian["sigma_y_um"]),
    }


def _effective_pixel_um(calibration: Any) -> tuple[float, float]:
    x_value = _read_field(calibration, "effective_pixel_x_um", None)
    y_value = _read_field(calibration, "effective_pixel_y_um", None)
    if x_value is not None and y_value is not None:
        return float(x_value), float(y_value)

    square_value = _read_field(calibration, "object_pixel_size_um_per_px", None)
    if square_value is not None:
        return float(square_value), float(square_value)

    raise_analysis_error(
        ErrorCode.E_ANALYSIS_D4SIGMA_FAILED,
        "calibration must provide effective_pixel_x_um/effective_pixel_y_um.",
    )


def _read_field(source: Any, field_name: str, default: Any) -> Any:
    if isinstance(source, Mapping):
        return source.get(field_name, default)
    return getattr(source, field_name, default)


def _peak_position_um(
    image: np.ndarray[Any, Any],
    effective_pixel_x_um: float,
    effective_pixel_y_um: float,
) -> tuple[float, float]:
    image_float = as_float_image(image)
    peak_y_px, peak_x_px = np.unravel_index(np.argmax(image_float), image_float.shape)
    return float(peak_x_px) * effective_pixel_x_um, float(peak_y_px) * effective_pixel_y_um


def _invalid_plane_result(
    image: np.ndarray[Any, Any],
    z_actual_mm: float,
    frame_peak_value: float,
    saturation_pixels: int,
    edge_energy_percent: float,
    invalid_reason: str,
) -> BeamPlaneResult:
    image_float = as_float_image(image)
    peak_value = float(np.max(image_float)) if image_float.size else frame_peak_value
    if not math.isfinite(frame_peak_value):
        frame_peak_value = peak_value

    return BeamPlaneResult(
        z_actual_mm=float(z_actual_mm),
        centroid_x_um=math.nan,
        centroid_y_um=math.nan,
        peak_x_um=None,
        peak_y_um=None,
        d4sigma_x_um=math.nan,
        d4sigma_y_um=math.nan,
        d4sigma_major_um=math.nan,
        d4sigma_minor_um=math.nan,
        fwhm_x_um=None,
        fwhm_y_um=None,
        ellipticity=math.nan,
        azimuth_deg=math.nan,
        peak_value=frame_peak_value,
        saturation_pixels=saturation_pixels,
        edge_energy_percent=edge_energy_percent,
        valid=False,
        invalid_reason=invalid_reason,
    )


__all__ = ["analyze_beam_plane", "calculate_centroid"]
