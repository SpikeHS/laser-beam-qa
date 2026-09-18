"""Synthetic beam images for offline analysis and flow simulation.

The functions in this module are intentionally hardware-free. They generate
pixel-count images and truth metadata that can be used by analysis tests,
sequencer smoke tests, and report/data pipeline rehearsals.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

NumberArray = NDArray[np.integer[Any]]
RngInput = np.random.Generator | int | None


@dataclass(frozen=True, slots=True)
class SyntheticZScanFrame:
    """One generated z-scan plane plus its ground truth."""

    plane_index_count: int
    z_target_mm: float
    z_actual_mm: float
    centroid_x_px: float
    centroid_y_px: float
    sigma_x_px: float
    sigma_y_px: float
    d4sigma_x_um: float
    d4sigma_y_um: float
    fwhm_x_um: float
    fwhm_y_um: float
    ellipse_angle_deg: float
    peak_intensity_count: float
    background_intensity_count: float
    bit_depth: int
    anomaly_tags: tuple[str, ...]
    image_counts: NumberArray

    def to_truth_record(self, raw_image_path: str | None = None) -> dict[str, Any]:
        """Return JSON-serializable truth metadata for this plane."""

        record: dict[str, Any] = {
            "plane_index_count": self.plane_index_count,
            "z_target_mm": self.z_target_mm,
            "z_actual_mm": self.z_actual_mm,
            "centroid_x_px": self.centroid_x_px,
            "centroid_y_px": self.centroid_y_px,
            "sigma_x_px": self.sigma_x_px,
            "sigma_y_px": self.sigma_y_px,
            "d4sigma_x_um": self.d4sigma_x_um,
            "d4sigma_y_um": self.d4sigma_y_um,
            "fwhm_x_um": self.fwhm_x_um,
            "fwhm_y_um": self.fwhm_y_um,
            "ellipse_angle_deg": self.ellipse_angle_deg,
            "peak_intensity_count": self.peak_intensity_count,
            "background_intensity_count": self.background_intensity_count,
            "bit_depth": self.bit_depth,
            "anomaly_tags": list(self.anomaly_tags),
        }
        if raw_image_path is not None:
            record["raw_image_path"] = raw_image_path
        return record


def generate_gaussian_beam_image(
    width_px: int,
    height_px: int,
    centroid_x_px: float,
    centroid_y_px: float,
    sigma_x_px: float,
    sigma_y_px: float,
    angle_deg: float,
    amplitude: float,
    background: float,
    noise_std: float,
    bit_depth: int,
    *,
    saturation: bool = False,
    low_signal: bool = False,
    low_signal_scale: float | None = None,
    background_offset: float = 0.0,
    bad_pixel_count: int = 0,
    bad_pixel_value: float | None = None,
    dark_frame: bool = False,
    rng: RngInput = None,
) -> NumberArray:
    """Generate a rotated elliptical Gaussian beam image.

    Parameters match the synthetic-data agent brief. `amplitude`, `background`,
    `noise_std`, `background_offset`, and `bad_pixel_value` are camera counts.
    The returned array is clipped to the requested bit depth and uses uint8 for
    <=8-bit data, otherwise uint16.
    """

    _validate_image_inputs(width_px, height_px, sigma_x_px, sigma_y_px, bit_depth)
    max_count = max_count_for_bit_depth(bit_depth)

    y_px, x_px = np.indices((height_px, width_px), dtype=np.float64)
    x_shift_px = x_px - centroid_x_px
    y_shift_px = y_px - centroid_y_px

    angle_rad = math.radians(angle_deg)
    cos_angle = math.cos(angle_rad)
    sin_angle = math.sin(angle_rad)
    x_rot_px = cos_angle * x_shift_px + sin_angle * y_shift_px
    y_rot_px = -sin_angle * x_shift_px + cos_angle * y_shift_px

    gaussian = np.exp(
        -0.5 * ((x_rot_px / sigma_x_px) ** 2 + (y_rot_px / sigma_y_px) ** 2)
    )

    signal_counts = np.zeros_like(gaussian) if dark_frame else amplitude * gaussian
    scale = _low_signal_scale(low_signal, low_signal_scale)
    if scale is not None:
        signal_counts *= scale

    image_counts = background + background_offset + signal_counts
    if noise_std > 0:
        generator = _coerce_rng(rng)
        image_counts += generator.normal(0.0, noise_std, size=image_counts.shape)

    if saturation:
        peak_row_px, peak_column_px = np.unravel_index(np.argmax(gaussian), gaussian.shape)
        image_counts[peak_row_px, peak_column_px] = max_count

    if bad_pixel_count > 0:
        generator = _coerce_rng(rng)
        _apply_bad_pixels(image_counts, bad_pixel_count, bad_pixel_value, max_count, generator)

    clipped_counts = np.clip(image_counts, 0.0, max_count)
    return np.rint(clipped_counts).astype(dtype_for_bit_depth(bit_depth), copy=False)


def generate_elliptical_gaussian_beam_image(
    width_px: int,
    height_px: int,
    centroid_x_px: float,
    centroid_y_px: float,
    sigma_x_px: float,
    sigma_y_px: float,
    angle_deg: float,
    amplitude: float,
    background: float,
    noise_std: float,
    bit_depth: int,
    **kwargs: Any,
) -> NumberArray:
    """Alias with an explicit elliptical-beam name."""

    return generate_gaussian_beam_image(
        width_px=width_px,
        height_px=height_px,
        centroid_x_px=centroid_x_px,
        centroid_y_px=centroid_y_px,
        sigma_x_px=sigma_x_px,
        sigma_y_px=sigma_y_px,
        angle_deg=angle_deg,
        amplitude=amplitude,
        background=background,
        noise_std=noise_std,
        bit_depth=bit_depth,
        **kwargs,
    )


def z_scan_diameters_um(
    z_actual_mm: float,
    waist_z_mm: float,
    waist_diameter_x_um: float,
    waist_diameter_y_um: float,
    full_angle_x_mrad: float,
    full_angle_y_mrad: float,
) -> tuple[float, float]:
    """Return D4sigma diameters from D(z)^2 = D0^2 + (Theta * dz)^2."""

    delta_z_mm = z_actual_mm - waist_z_mm
    d4sigma_x_um = math.sqrt(waist_diameter_x_um**2 + (full_angle_x_mrad * delta_z_mm) ** 2)
    d4sigma_y_um = math.sqrt(waist_diameter_y_um**2 + (full_angle_y_mrad * delta_z_mm) ** 2)
    return d4sigma_x_um, d4sigma_y_um


def generate_z_scan_sequence(
    z_positions_mm: Sequence[float],
    waist_z_mm: float,
    waist_diameter_x_um: float,
    waist_diameter_y_um: float,
    full_angle_x_mrad: float,
    full_angle_y_mrad: float,
    effective_pixel_x_um: float,
    effective_pixel_y_um: float,
    *,
    width_px: int = 512,
    height_px: int = 512,
    centroid_x_px: float | None = None,
    centroid_y_px: float | None = None,
    angle_deg: float = 0.0,
    amplitude: float = 3200.0,
    background: float = 80.0,
    noise_std: float = 2.0,
    bit_depth: int = 12,
    rng: RngInput = None,
    anomalies_by_index: Mapping[int, Mapping[str, Any]] | None = None,
) -> list[SyntheticZScanFrame]:
    """Generate synthetic images and truth metadata for a full z scan."""

    _validate_pixel_size(effective_pixel_x_um, effective_pixel_y_um)
    if len(z_positions_mm) == 0:
        raise ValueError("z_positions_mm must contain at least one plane.")

    base_centroid_x_px = (width_px - 1) / 2.0 if centroid_x_px is None else centroid_x_px
    base_centroid_y_px = (height_px - 1) / 2.0 if centroid_y_px is None else centroid_y_px
    generator = _coerce_rng(rng)

    frames: list[SyntheticZScanFrame] = []
    for plane_index_count, z_actual_mm in enumerate(z_positions_mm):
        d4sigma_x_um, d4sigma_y_um = z_scan_diameters_um(
            z_actual_mm=z_actual_mm,
            waist_z_mm=waist_z_mm,
            waist_diameter_x_um=waist_diameter_x_um,
            waist_diameter_y_um=waist_diameter_y_um,
            full_angle_x_mrad=full_angle_x_mrad,
            full_angle_y_mrad=full_angle_y_mrad,
        )
        sigma_x_px = d4sigma_x_um / (4.0 * effective_pixel_x_um)
        sigma_y_px = d4sigma_y_um / (4.0 * effective_pixel_y_um)

        overrides = dict((anomalies_by_index or {}).get(plane_index_count, {}))
        plane_centroid_x_px = float(overrides.pop("centroid_x_px", base_centroid_x_px))
        plane_centroid_y_px = float(overrides.pop("centroid_y_px", base_centroid_y_px))
        plane_amplitude = float(overrides.pop("amplitude", amplitude))
        plane_background = float(overrides.pop("background", background))
        plane_noise_std = float(overrides.pop("noise_std", noise_std))
        plane_angle_deg = float(overrides.pop("angle_deg", angle_deg))

        image_counts = generate_gaussian_beam_image(
            width_px=width_px,
            height_px=height_px,
            centroid_x_px=plane_centroid_x_px,
            centroid_y_px=plane_centroid_y_px,
            sigma_x_px=sigma_x_px,
            sigma_y_px=sigma_y_px,
            angle_deg=plane_angle_deg,
            amplitude=plane_amplitude,
            background=plane_background,
            noise_std=plane_noise_std,
            bit_depth=bit_depth,
            rng=generator,
            **overrides,
        )

        frames.append(
            SyntheticZScanFrame(
                plane_index_count=plane_index_count,
                z_target_mm=float(z_actual_mm),
                z_actual_mm=float(z_actual_mm),
                centroid_x_px=plane_centroid_x_px,
                centroid_y_px=plane_centroid_y_px,
                sigma_x_px=sigma_x_px,
                sigma_y_px=sigma_y_px,
                d4sigma_x_um=d4sigma_x_um,
                d4sigma_y_um=d4sigma_y_um,
                fwhm_x_um=_fwhm_from_d4sigma_um(d4sigma_x_um),
                fwhm_y_um=_fwhm_from_d4sigma_um(d4sigma_y_um),
                ellipse_angle_deg=plane_angle_deg,
                peak_intensity_count=float(image_counts.max()),
                background_intensity_count=plane_background,
                bit_depth=bit_depth,
                anomaly_tags=_anomaly_tags(
                    width_px=width_px,
                    height_px=height_px,
                    centroid_x_px=plane_centroid_x_px,
                    centroid_y_px=plane_centroid_y_px,
                    sigma_x_px=sigma_x_px,
                    sigma_y_px=sigma_y_px,
                    noise_std=plane_noise_std,
                    overrides=overrides,
                ),
                image_counts=image_counts,
            )
        )

    return frames


def max_count_for_bit_depth(bit_depth: int) -> int:
    """Return the largest camera count representable at `bit_depth`."""

    if bit_depth < 1 or bit_depth > 16:
        raise ValueError("bit_depth must be between 1 and 16.")
    return (1 << bit_depth) - 1


def dtype_for_bit_depth(bit_depth: int) -> type[np.uint8] | type[np.uint16]:
    """Return a lossless integer dtype for the requested bit depth."""

    max_count_for_bit_depth(bit_depth)
    return np.uint8 if bit_depth <= 8 else np.uint16


def generate_2d_gaussian(*args: Any, **kwargs: Any) -> NumberArray:
    """Compatibility alias for brief wording."""

    return generate_gaussian_beam_image(*args, **kwargs)


def generate_elliptical_gaussian(*args: Any, **kwargs: Any) -> NumberArray:
    """Compatibility alias for brief wording."""

    return generate_elliptical_gaussian_beam_image(*args, **kwargs)


def _validate_image_inputs(
    width_px: int, height_px: int, sigma_x_px: float, sigma_y_px: float, bit_depth: int
) -> None:
    if width_px <= 0 or height_px <= 0:
        raise ValueError("width_px and height_px must be positive.")
    if sigma_x_px <= 0 or sigma_y_px <= 0:
        raise ValueError("sigma_x_px and sigma_y_px must be positive.")
    max_count_for_bit_depth(bit_depth)


def _validate_pixel_size(effective_pixel_x_um: float, effective_pixel_y_um: float) -> None:
    if effective_pixel_x_um <= 0 or effective_pixel_y_um <= 0:
        raise ValueError("effective pixel sizes must be positive.")


def _coerce_rng(rng: RngInput) -> np.random.Generator:
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def _low_signal_scale(low_signal: bool, low_signal_scale: float | None) -> float | None:
    if low_signal_scale is not None:
        if low_signal_scale < 0:
            raise ValueError("low_signal_scale must be non-negative.")
        return low_signal_scale
    return 0.05 if low_signal else None


def _apply_bad_pixels(
    image_counts: NDArray[np.float64],
    bad_pixel_count: int,
    bad_pixel_value: float | None,
    max_count: int,
    generator: np.random.Generator,
) -> None:
    if bad_pixel_count < 0:
        raise ValueError("bad_pixel_count must be non-negative.")
    height_px, width_px = image_counts.shape
    pixel_count = height_px * width_px
    selected_count = min(bad_pixel_count, pixel_count)
    flat_indices = generator.choice(pixel_count, size=selected_count, replace=False)
    flat_image = image_counts.reshape(-1)
    flat_image[flat_indices] = max_count if bad_pixel_value is None else bad_pixel_value


def _fwhm_from_d4sigma_um(d4sigma_um: float) -> float:
    sigma_um = d4sigma_um / 4.0
    return 2.0 * math.sqrt(2.0 * math.log(2.0)) * sigma_um


def _anomaly_tags(
    width_px: int,
    height_px: int,
    centroid_x_px: float,
    centroid_y_px: float,
    sigma_x_px: float,
    sigma_y_px: float,
    noise_std: float,
    overrides: Mapping[str, Any],
) -> tuple[str, ...]:
    tags: list[str] = []
    if bool(overrides.get("saturation", False)):
        tags.append("saturation")
    if bool(overrides.get("low_signal", False)) or "low_signal_scale" in overrides:
        tags.append("low_signal")
    if float(overrides.get("background_offset", 0.0)) != 0.0:
        tags.append("background_offset")
    if noise_std > 0.0:
        tags.append("random_noise")
    if int(overrides.get("bad_pixel_count", 0)) > 0:
        tags.append("bad_pixels")
    if bool(overrides.get("dark_frame", False)):
        tags.append("dark_frame")

    if (
        centroid_x_px < 0.0
        or centroid_x_px > width_px - 1
        or centroid_y_px < 0.0
        or centroid_y_px > height_px - 1
    ):
        tags.append("out_of_field")
    elif (
        centroid_x_px < 3.0 * sigma_x_px
        or centroid_x_px > width_px - 1 - 3.0 * sigma_x_px
        or centroid_y_px < 3.0 * sigma_y_px
        or centroid_y_px > height_px - 1 - 3.0 * sigma_y_px
    ):
        tags.append("near_boundary")

    return tuple(tags)


__all__ = [
    "SyntheticZScanFrame",
    "dtype_for_bit_depth",
    "generate_2d_gaussian",
    "generate_elliptical_gaussian",
    "generate_elliptical_gaussian_beam_image",
    "generate_gaussian_beam_image",
    "generate_z_scan_sequence",
    "max_count_for_bit_depth",
    "z_scan_diameters_um",
]
