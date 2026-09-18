"""Synthetic beam image generation for simulated device adapters."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

NumberArray = NDArray[np.integer[Any]]
RngInput = np.random.Generator | int | None


@dataclass(frozen=True, slots=True)
class SyntheticBeamImage:
    image_counts: NumberArray
    d4sigma_x_um: float
    d4sigma_y_um: float
    sigma_x_px: float
    sigma_y_px: float
    centroid_x_px: float
    centroid_y_px: float
    peak_intensity_count: float
    background_intensity_count: float
    anomaly_tags: tuple[str, ...]


def generate_z_dependent_beam_image(
    *,
    width_px: int,
    height_px: int,
    z_actual_mm: float,
    effective_pixel_x_um: float,
    effective_pixel_y_um: float,
    waist_z_mm: float,
    waist_diameter_x_um: float,
    waist_diameter_y_um: float,
    full_angle_x_mrad: float,
    full_angle_y_mrad: float,
    bit_depth: int,
    amplitude_count: float,
    background_count: float,
    noise_std_count: float,
    angle_deg: float = 0.0,
    centroid_x_px: float | None = None,
    centroid_y_px: float | None = None,
    centroid_offset_x_px: float = 0.0,
    centroid_offset_y_px: float = 0.0,
    saturation: bool = False,
    low_signal: bool = False,
    low_signal_scale: float = 0.05,
    edge_clipped: bool = False,
    bad_pixel_count: int = 0,
    rng: RngInput = None,
) -> SyntheticBeamImage:
    """Generate one z-dependent elliptical Gaussian beam image.

    The D4sigma model follows D^2(z) = D0^2 + (theta * dz)^2. With theta in
    mrad and dz in mm, the product has units of um.
    """

    _validate_positive_int("width_px", width_px)
    _validate_positive_int("height_px", height_px)
    _validate_positive_float("effective_pixel_x_um", effective_pixel_x_um)
    _validate_positive_float("effective_pixel_y_um", effective_pixel_y_um)
    _validate_positive_float("waist_diameter_x_um", waist_diameter_x_um)
    _validate_positive_float("waist_diameter_y_um", waist_diameter_y_um)
    _validate_positive_int("bit_depth", bit_depth)
    max_count = max_count_for_bit_depth(bit_depth)

    d4sigma_x_um, d4sigma_y_um = z_scan_d4sigma_um(
        z_actual_mm=z_actual_mm,
        waist_z_mm=waist_z_mm,
        waist_diameter_x_um=waist_diameter_x_um,
        waist_diameter_y_um=waist_diameter_y_um,
        full_angle_x_mrad=full_angle_x_mrad,
        full_angle_y_mrad=full_angle_y_mrad,
    )
    sigma_x_px = max(d4sigma_x_um / (4.0 * effective_pixel_x_um), 1.0)
    sigma_y_px = max(d4sigma_y_um / (4.0 * effective_pixel_y_um), 1.0)

    base_centroid_x_px = (width_px - 1) / 2.0 if centroid_x_px is None else centroid_x_px
    base_centroid_y_px = (height_px - 1) / 2.0 if centroid_y_px is None else centroid_y_px
    beam_centroid_x_px = base_centroid_x_px + centroid_offset_x_px
    beam_centroid_y_px = base_centroid_y_px + centroid_offset_y_px
    if edge_clipped:
        beam_centroid_x_px = min(width_px - 1.0, max(0.0, 1.25 * sigma_x_px))
        beam_centroid_y_px = (height_px - 1) / 2.0

    signal_amplitude_count = max(0.0, amplitude_count)
    if low_signal:
        signal_amplitude_count *= max(0.0, low_signal_scale)

    generator = _coerce_rng(rng)
    image_float = _elliptical_gaussian_counts(
        width_px=width_px,
        height_px=height_px,
        centroid_x_px=beam_centroid_x_px,
        centroid_y_px=beam_centroid_y_px,
        sigma_x_px=sigma_x_px,
        sigma_y_px=sigma_y_px,
        angle_deg=angle_deg,
        amplitude_count=signal_amplitude_count,
        background_count=background_count,
    )

    if noise_std_count > 0.0:
        image_float += generator.normal(0.0, noise_std_count, size=image_float.shape)

    if saturation:
        peak_row_px, peak_col_px = np.unravel_index(np.argmax(image_float), image_float.shape)
        row_start_px = max(0, int(peak_row_px) - 1)
        row_stop_px = min(height_px, int(peak_row_px) + 2)
        col_start_px = max(0, int(peak_col_px) - 1)
        col_stop_px = min(width_px, int(peak_col_px) + 2)
        image_float[row_start_px:row_stop_px, col_start_px:col_stop_px] = max_count

    if bad_pixel_count > 0:
        flat_indices = generator.choice(
            width_px * height_px,
            size=min(bad_pixel_count, width_px * height_px),
            replace=False,
        )
        image_float.reshape(-1)[flat_indices] = max_count

    image_counts = np.rint(np.clip(image_float, 0.0, max_count)).astype(
        dtype_for_bit_depth(bit_depth),
        copy=False,
    )

    return SyntheticBeamImage(
        image_counts=image_counts,
        d4sigma_x_um=d4sigma_x_um,
        d4sigma_y_um=d4sigma_y_um,
        sigma_x_px=sigma_x_px,
        sigma_y_px=sigma_y_px,
        centroid_x_px=beam_centroid_x_px,
        centroid_y_px=beam_centroid_y_px,
        peak_intensity_count=float(image_counts.max()),
        background_intensity_count=float(background_count),
        anomaly_tags=_anomaly_tags(
            width_px=width_px,
            height_px=height_px,
            centroid_x_px=beam_centroid_x_px,
            centroid_y_px=beam_centroid_y_px,
            sigma_x_px=sigma_x_px,
            sigma_y_px=sigma_y_px,
            noise_std_count=noise_std_count,
            saturation=saturation,
            low_signal=low_signal,
            edge_clipped=edge_clipped,
            bad_pixel_count=bad_pixel_count,
        ),
    )


def z_scan_d4sigma_um(
    *,
    z_actual_mm: float,
    waist_z_mm: float,
    waist_diameter_x_um: float,
    waist_diameter_y_um: float,
    full_angle_x_mrad: float,
    full_angle_y_mrad: float,
) -> tuple[float, float]:
    delta_z_mm = z_actual_mm - waist_z_mm
    d4sigma_x_um = math.sqrt(waist_diameter_x_um**2 + (full_angle_x_mrad * delta_z_mm) ** 2)
    d4sigma_y_um = math.sqrt(waist_diameter_y_um**2 + (full_angle_y_mrad * delta_z_mm) ** 2)
    return d4sigma_x_um, d4sigma_y_um


def max_count_for_bit_depth(bit_depth: int) -> int:
    if bit_depth < 1 or bit_depth > 16:
        raise ValueError("bit_depth must be between 1 and 16.")
    return (1 << bit_depth) - 1


def dtype_for_bit_depth(bit_depth: int) -> type[np.uint8] | type[np.uint16]:
    return np.uint8 if max_count_for_bit_depth(bit_depth) <= np.iinfo(np.uint8).max else np.uint16


def _elliptical_gaussian_counts(
    *,
    width_px: int,
    height_px: int,
    centroid_x_px: float,
    centroid_y_px: float,
    sigma_x_px: float,
    sigma_y_px: float,
    angle_deg: float,
    amplitude_count: float,
    background_count: float,
) -> NDArray[np.float64]:
    y_px, x_px = np.indices((height_px, width_px), dtype=np.float64)
    x_shift_px = x_px - centroid_x_px
    y_shift_px = y_px - centroid_y_px
    angle_rad = math.radians(angle_deg)
    cos_angle = math.cos(angle_rad)
    sin_angle = math.sin(angle_rad)
    x_rot_px = cos_angle * x_shift_px + sin_angle * y_shift_px
    y_rot_px = -sin_angle * x_shift_px + cos_angle * y_shift_px
    gaussian = np.exp(-0.5 * ((x_rot_px / sigma_x_px) ** 2 + (y_rot_px / sigma_y_px) ** 2))
    return background_count + amplitude_count * gaussian


def _anomaly_tags(
    *,
    width_px: int,
    height_px: int,
    centroid_x_px: float,
    centroid_y_px: float,
    sigma_x_px: float,
    sigma_y_px: float,
    noise_std_count: float,
    saturation: bool,
    low_signal: bool,
    edge_clipped: bool,
    bad_pixel_count: int,
) -> tuple[str, ...]:
    tags: list[str] = []
    if saturation:
        tags.append("saturation")
    if low_signal:
        tags.append("low_signal")
    if noise_std_count > 0.0:
        tags.append("random_noise")
    if bad_pixel_count > 0:
        tags.append("bad_pixels")
    if edge_clipped or _near_image_boundary(
        width_px=width_px,
        height_px=height_px,
        centroid_x_px=centroid_x_px,
        centroid_y_px=centroid_y_px,
        sigma_x_px=sigma_x_px,
        sigma_y_px=sigma_y_px,
    ):
        tags.append("edge_clipped")
    return tuple(tags)


def _near_image_boundary(
    *,
    width_px: int,
    height_px: int,
    centroid_x_px: float,
    centroid_y_px: float,
    sigma_x_px: float,
    sigma_y_px: float,
) -> bool:
    return (
        centroid_x_px < 3.0 * sigma_x_px
        or centroid_x_px > width_px - 1 - 3.0 * sigma_x_px
        or centroid_y_px < 3.0 * sigma_y_px
        or centroid_y_px > height_px - 1 - 3.0 * sigma_y_px
    )


def _coerce_rng(rng: RngInput) -> np.random.Generator:
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def _validate_positive_int(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive.")


def _validate_positive_float(name: str, value: float) -> None:
    if value <= 0.0:
        raise ValueError(f"{name} must be positive.")
