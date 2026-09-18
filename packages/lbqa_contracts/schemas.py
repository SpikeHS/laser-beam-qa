"""Legacy dataclass contracts retained for compatibility.

MVP integrations should use `lbqa_contracts.models` instead. Do not expand this
module for new public interfaces.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Calibration40x:
    camera_model: str
    camera_pixel_size_um: float
    objective_magnification: float
    object_pixel_size_um_per_px: float
    field_of_view_x_um: float
    field_of_view_y_um: float
    working_distance_mm: float


@dataclass(frozen=True, slots=True)
class BeamFrame:
    frame_id_count: int
    width_px: int
    height_px: int
    object_pixel_size_um_per_px: float
    exposure_ms: float
    z_actual_mm: float
    pixels: Sequence[Sequence[float]] = field(repr=False)


@dataclass(frozen=True, slots=True)
class BeamMetrics:
    frame_id_count: int
    z_actual_mm: float
    centroid_x_um: float
    centroid_y_um: float
    d4sigma_x_um: float
    d4sigma_y_um: float
    fwhm_x_um: float
    fwhm_y_um: float
    ellipticity_ratio: float
    ellipse_angle_deg: float
    peak_intensity_count: float
    background_intensity_count: float


@dataclass(frozen=True, slots=True)
class ZScanPlaneRecord:
    plane_index_count: int
    z_target_mm: float
    z_actual_mm: float
    raw_image_path: str
    processed_data_path: str
    centroid_x_um: float
    centroid_y_um: float
    d4sigma_x_um: float
    d4sigma_y_um: float
    fwhm_x_um: float
    fwhm_y_um: float
    ellipticity_ratio: float
    ellipse_angle_deg: float
    peak_intensity_count: float


@dataclass(frozen=True, slots=True)
class DivergenceFitResult:
    sample_count: int
    fit_a_x_um2_per_mm2: float
    fit_b_x_um2_per_mm: float
    fit_c_x_um2: float
    fit_a_y_um2_per_mm2: float
    fit_b_y_um2_per_mm: float
    fit_c_y_um2: float
    waist_z_x_mm: float
    waist_z_y_mm: float
    waist_d4sigma_x_um: float
    waist_d4sigma_y_um: float
    full_angle_x_mrad: float
    half_angle_x_mrad: float
    full_angle_y_mrad: float
    half_angle_y_mrad: float
