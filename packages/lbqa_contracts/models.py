"""Shared data models for laser beam quality testing."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from lbqa_contracts.validators import (
    validate_error_codes,
    validate_image_shape,
    validate_non_negative,
    validate_optional_error_code,
    validate_positive,
)

JUDGEMENTS = {"pass", "fail", "warning", "unknown", "invalid"}


@dataclass(frozen=True, slots=True)
class CameraInfo:
    model: str
    width_px: int
    height_px: int
    pixel_size_um: float
    bit_depth: int
    serial_number: str | None = None

    def __post_init__(self) -> None:
        validate_positive("width_px", self.width_px)
        validate_positive("height_px", self.height_px)
        validate_positive("pixel_size_um", self.pixel_size_um)
        validate_positive("bit_depth", self.bit_depth)


@dataclass(frozen=True, slots=True)
class OpticalCalibration:
    magnification: float
    camera_pixel_size_um: float
    effective_pixel_x_um: float
    effective_pixel_y_um: float
    field_of_view_x_um: float
    field_of_view_y_um: float
    working_distance_mm: float
    calibration_id: str | None = None
    calibration_date: str | None = None

    def __post_init__(self) -> None:
        validate_positive("magnification", self.magnification)
        validate_positive("camera_pixel_size_um", self.camera_pixel_size_um)
        validate_positive("effective_pixel_x_um", self.effective_pixel_x_um)
        validate_positive("effective_pixel_y_um", self.effective_pixel_y_um)
        validate_positive("field_of_view_x_um", self.field_of_view_x_um)
        validate_positive("field_of_view_y_um", self.field_of_view_y_um)
        validate_positive("working_distance_mm", self.working_distance_mm)


@dataclass(frozen=True, slots=True)
class StageStatus:
    connected: bool
    homed: bool
    enabled: bool
    moving: bool
    position_mm: float
    error_code: str | None = None

    def __post_init__(self) -> None:
        validate_optional_error_code(self.error_code)


@dataclass(frozen=True, slots=True)
class ImageFrame:
    image: np.ndarray[Any, Any] = field(repr=False)
    timestamp_iso: str
    exposure_us: float
    gain: float
    width_px: int
    height_px: int
    z_actual_mm: float | None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_positive("exposure_us", self.exposure_us)
        validate_positive("width_px", self.width_px)
        validate_positive("height_px", self.height_px)
        validate_image_shape(self.image, width_px=self.width_px, height_px=self.height_px)


@dataclass(frozen=True, slots=True)
class FrameQuality:
    saturated: bool
    saturation_pixels: int
    peak_value: float
    mean_value: float
    edge_energy_percent: float
    low_signal: bool
    valid: bool
    invalid_reason: str | None = None

    def __post_init__(self) -> None:
        validate_non_negative("saturation_pixels", self.saturation_pixels)
        validate_non_negative("peak_value", self.peak_value)
        validate_non_negative("mean_value", self.mean_value)
        validate_non_negative("edge_energy_percent", self.edge_energy_percent)
        if not 0.0 <= self.edge_energy_percent <= 100.0:
            raise ValueError("edge_energy_percent must be between 0 and 100.")
        if not self.valid and not self.invalid_reason:
            raise ValueError("invalid_reason is required when valid is False.")


@dataclass(frozen=True, slots=True)
class BeamPlaneResult:
    z_actual_mm: float
    centroid_x_um: float
    centroid_y_um: float
    peak_x_um: float | None
    peak_y_um: float | None
    d4sigma_x_um: float
    d4sigma_y_um: float
    d4sigma_major_um: float
    d4sigma_minor_um: float
    fwhm_x_um: float | None
    fwhm_y_um: float | None
    ellipticity: float
    azimuth_deg: float
    peak_value: float
    saturation_pixels: int
    edge_energy_percent: float
    valid: bool
    invalid_reason: str | None = None
    outlier: bool = False
    outlier_reason: str | None = None
    width_source: str | None = None
    width_uncertainty_x_um: float | None = None
    width_uncertainty_y_um: float | None = None
    native_width_x_um: float | None = None
    native_width_y_um: float | None = None
    native_width_scale_x_ratio: float | None = None
    native_width_scale_y_ratio: float | None = None
    rayci_gfi_ratio: float | None = None

    def __post_init__(self) -> None:
        validate_positive("d4sigma_x_um", self.d4sigma_x_um)
        validate_positive("d4sigma_y_um", self.d4sigma_y_um)
        validate_positive("d4sigma_major_um", self.d4sigma_major_um)
        validate_positive("d4sigma_minor_um", self.d4sigma_minor_um)
        if self.fwhm_x_um is not None:
            validate_positive("fwhm_x_um", self.fwhm_x_um)
        if self.fwhm_y_um is not None:
            validate_positive("fwhm_y_um", self.fwhm_y_um)
        validate_positive("ellipticity", self.ellipticity)
        validate_non_negative("peak_value", self.peak_value)
        validate_non_negative("saturation_pixels", self.saturation_pixels)
        validate_non_negative("edge_energy_percent", self.edge_energy_percent)
        if not 0.0 <= self.edge_energy_percent <= 100.0:
            raise ValueError("edge_energy_percent must be between 0 and 100.")
        if not self.valid and not self.invalid_reason:
            raise ValueError("invalid_reason is required when valid is False.")
        if self.outlier and not self.outlier_reason:
            raise ValueError("outlier_reason is required when outlier is True.")
        for field_name in (
            "width_uncertainty_x_um",
            "width_uncertainty_y_um",
            "native_width_x_um",
            "native_width_y_um",
            "native_width_scale_x_ratio",
            "native_width_scale_y_ratio",
        ):
            value = getattr(self, field_name)
            if value is not None:
                validate_positive(field_name, value)
        if self.rayci_gfi_ratio is not None and not math.isfinite(self.rayci_gfi_ratio):
            raise ValueError("rayci_gfi_ratio must be finite when provided.")


@dataclass(frozen=True, slots=True)
class ZScanResult:
    points: list[BeamPlaneResult]
    full_angle_x_mrad: float | None
    full_angle_y_mrad: float | None
    half_angle_x_mrad: float | None
    half_angle_y_mrad: float | None
    waist_z_x_mm: float | None
    waist_z_y_mm: float | None
    waist_diameter_x_um: float | None
    waist_diameter_y_um: float | None
    fit_r2_x: float | None
    fit_r2_y: float | None
    valid_points_count: int
    judgement: str
    invalid_reason: str | None = None
    fit_a_x_um2_per_mm2: float | None = None
    fit_b_x_um2_per_mm: float | None = None
    fit_c_x_um2: float | None = None
    fit_a_y_um2_per_mm2: float | None = None
    fit_b_y_um2_per_mm: float | None = None
    fit_c_y_um2: float | None = None
    outlier_indices: list[int] = field(default_factory=list)
    fit_diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_non_negative("valid_points_count", self.valid_points_count)
        if self.valid_points_count > len(self.points):
            raise ValueError("valid_points_count cannot exceed len(points).")
        for field_name in (
            "full_angle_x_mrad",
            "full_angle_y_mrad",
            "half_angle_x_mrad",
            "half_angle_y_mrad",
            "waist_diameter_x_um",
            "waist_diameter_y_um",
            "fit_a_x_um2_per_mm2",
            "fit_c_x_um2",
            "fit_a_y_um2_per_mm2",
            "fit_c_y_um2",
        ):
            value = getattr(self, field_name)
            if value is not None:
                validate_non_negative(field_name, value)
        for field_name in ("fit_r2_x", "fit_r2_y"):
            value = getattr(self, field_name)
            if value is not None and not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be between 0 and 1.")
        for outlier_index in self.outlier_indices:
            if outlier_index < 0 or outlier_index >= len(self.points):
                raise ValueError("outlier_indices must refer to existing points.")
        if self.judgement not in JUDGEMENTS:
            raise ValueError("judgement must be one of pass, fail, warning, unknown, invalid.")
        if self.judgement in {"fail", "invalid"} and not self.invalid_reason:
            raise ValueError("invalid_reason is required when judgement is fail or invalid.")


@dataclass(frozen=True, slots=True)
class Recipe:
    recipe_id: str
    operator_mode: str
    camera: dict[str, Any]
    optics: dict[str, Any]
    stage: dict[str, Any]
    z_scan: dict[str, Any]
    analysis: dict[str, Any]
    limits: dict[str, Any]

    def __post_init__(self) -> None:
        if not self.recipe_id:
            raise ValueError("recipe_id is required.")
        if self.operator_mode not in {"automatic", "semi_auto", "maintenance", "simulated"}:
            raise ValueError(
                "operator_mode must be one of automatic, semi_auto, maintenance, simulated."
            )


@dataclass(frozen=True, slots=True)
class RunResult:
    run_id: str
    sample_id: str
    timestamp_iso: str
    recipe_id: str
    calibration_id: str | None
    plane_result: BeamPlaneResult | None
    zscan_result: ZScanResult | None
    judgement: str
    errors: list[str]
    output_dir: str

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id is required.")
        if not self.sample_id:
            raise ValueError("sample_id is required.")
        if self.judgement not in JUDGEMENTS:
            raise ValueError("judgement must be one of pass, fail, warning, unknown, invalid.")
        validate_error_codes(self.errors)
