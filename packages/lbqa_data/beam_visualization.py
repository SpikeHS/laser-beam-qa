"""Beam image rendering helpers for reports and operator preview."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True, slots=True)
class BeamDisplayOptions:
    threshold_percent: float = 50.0
    min_display_px: int = 360
    max_display_px: int = 640
    overlay_mode: str = "box"
    colormap: str = "thermal"
    zoom_factor: float = 1.0
    center_x_px: float | None = None
    center_y_px: float | None = None
    intensity_min: float | None = None
    intensity_max: float | None = None


@dataclass(frozen=True, slots=True)
class BeamCrop:
    image: np.ndarray[Any, Any]
    x_offset_px: int
    y_offset_px: int


@dataclass(frozen=True, slots=True)
class BeamAxes:
    centroid_x_px: float
    centroid_y_px: float
    azimuth_deg: float
    major_sigma_px: float
    minor_sigma_px: float
    threshold_percent: float
    sigma_environment: float = 2.0
    x_axis_azimuth_deg: float | None = None
    x_axis_sigma_px: float | None = None
    y_axis_sigma_px: float | None = None
    axis_assignment: str = "major_to_x"
    orientation_status: str = "raw"
    source: str = "local_moments"
    y_axis_azimuth_deg: float | None = None


def beam_axes_from_measurement(
    *,
    centroid_x_um: float,
    centroid_y_um: float,
    major_diameter_um: float,
    minor_diameter_um: float,
    major_azimuth_deg: float,
    x_diameter_um: float,
    y_diameter_um: float,
    x_azimuth_deg: float,
    pixel_x_um: float,
    pixel_y_um: float,
    threshold_percent: float = 50.0,
    source: str = "measurement",
) -> BeamAxes:
    """Transform the measured ellipse and tracked axes into source pixel space."""
    values = (
        centroid_x_um,
        centroid_y_um,
        major_diameter_um,
        minor_diameter_um,
        major_azimuth_deg,
        x_diameter_um,
        y_diameter_um,
        x_azimuth_deg,
        pixel_x_um,
        pixel_y_um,
    )
    if (
        not all(math.isfinite(v) for v in values)
        or min(
            pixel_x_um,
            pixel_y_um,
            minor_diameter_um,
            major_diameter_um,
            x_diameter_um,
            y_diameter_um,
        )
        <= 0
    ):
        raise ValueError("Finite measurements and positive diameters/pixel sizes are required.")
    angle = math.radians(major_azimuth_deg)
    c, s = math.cos(angle), math.sin(angle)
    rotation = np.array([[c, -s], [s, c]])
    inverse_pixel = np.diag([1 / pixel_x_um, 1 / pixel_y_um])
    transform = inverse_pixel @ rotation
    covariance = (
        transform
        @ np.diag([(major_diameter_um / 4) ** 2, (minor_diameter_um / 4) ** 2])
        @ transform.T
    )
    eigenvalues, vectors = np.linalg.eigh(covariance)
    directions, sigmas = [], []
    for direction_deg, diameter in (
        (x_azimuth_deg, x_diameter_um),
        (x_azimuth_deg + 90, y_diameter_um),
    ):
        radians = math.radians(direction_deg)
        direction = (math.cos(radians) / pixel_x_um, math.sin(radians) / pixel_y_um)
        directions.append(math.degrees(math.atan2(direction[1], direction[0])))
        sigmas.append(diameter / 4 * math.hypot(*direction))
    return BeamAxes(
        centroid_x_px=centroid_x_um / pixel_x_um,
        centroid_y_px=centroid_y_um / pixel_y_um,
        azimuth_deg=math.degrees(math.atan2(vectors[1, 1], vectors[0, 1])),
        major_sigma_px=math.sqrt(max(0, eigenvalues[1])),
        minor_sigma_px=math.sqrt(max(0, eigenvalues[0])),
        threshold_percent=threshold_percent,
        x_axis_azimuth_deg=directions[0],
        y_axis_azimuth_deg=directions[1],
        x_axis_sigma_px=sigmas[0],
        y_axis_sigma_px=sigmas[1],
        source=source,
        orientation_status="measurement",
    )


def save_beam_spot_png(
    image: np.ndarray[Any, Any],
    output_path: Path,
    *,
    title: str | None = None,
    options: BeamDisplayOptions | None = None,
    rotation_deg: float | None = None,
    axes: BeamAxes | None = None,
) -> None:
    """Save a beam PNG using manual spatial zoom and a RayCi-like heat map."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    render_beam_spot_image(
        image,
        title=title,
        options=options,
        rotation_deg=rotation_deg,
        axes=axes,
    ).save(output_path, format="PNG")


def render_beam_spot_image(
    image: np.ndarray[Any, Any],
    *,
    title: str | None = None,
    options: BeamDisplayOptions | None = None,
    rotation_deg: float | None = None,
    axes: BeamAxes | None = None,
) -> Image.Image:
    opts = _validated_options(options or BeamDisplayOptions())
    crop = crop_beam_region(image, options=opts)
    scaled = _scale_to_uint8(
        crop.image,
        intensity_min=opts.intensity_min,
        intensity_max=opts.intensity_max,
    )
    display = Image.fromarray(_apply_colormap(scaled, opts.colormap), mode="RGB")
    display, scale = _resize_for_display(display, opts)
    if opts.overlay_mode == "box":
        if axes is not None:
            _draw_beam_axes(ImageDraw.Draw(display), axes, crop=crop, scale=scale)
        else:
            bbox = _beam_range_bbox(crop.image, threshold_percent=opts.threshold_percent)
            if bbox is not None:
                _draw_threshold_box(ImageDraw.Draw(display), bbox, scale=scale)
    if rotation_deg is not None and math.isfinite(float(rotation_deg)):
        display = _rotate_for_report(display, float(rotation_deg))
    if title:
        _draw_title(display, title)
    return display


def crop_beam_region(
    image: np.ndarray[Any, Any],
    *,
    options: BeamDisplayOptions | None = None,
) -> BeamCrop:
    """Return a manual zoom crop, centered on explicit x/y or image center."""

    opts = _validated_options(options or BeamDisplayOptions())
    frame = _as_2d_float(image)
    height_px, width_px = frame.shape
    zoom_factor = max(1.0, float(opts.zoom_factor))
    if zoom_factor <= 1.0:
        return BeamCrop(image=frame, x_offset_px=0, y_offset_px=0)

    crop_width_px = max(8, min(width_px, int(round(width_px / zoom_factor))))
    crop_height_px = max(8, min(height_px, int(round(height_px / zoom_factor))))
    center_x = float(opts.center_x_px) if opts.center_x_px is not None else (width_px - 1) / 2.0
    center_y = float(opts.center_y_px) if opts.center_y_px is not None else (height_px - 1) / 2.0
    x_min = _clamped_start(center_x, crop_width_px, width_px)
    y_min = _clamped_start(center_y, crop_height_px, height_px)
    return BeamCrop(
        image=frame[y_min : y_min + crop_height_px, x_min : x_min + crop_width_px],
        x_offset_px=x_min,
        y_offset_px=y_min,
    )


def estimate_beam_axes(
    image: np.ndarray[Any, Any],
    *,
    threshold_percent: float = 50.0,
) -> BeamAxes | None:
    """Estimate beam principal-axis direction for operator status and reports."""

    if not 0.0 <= float(threshold_percent) <= 100.0:
        raise ValueError("threshold_percent must be between 0 and 100.")

    frame = _as_2d_float(image)
    signal = frame - float(np.min(frame))
    np.maximum(signal, 0.0, out=signal)
    peak = float(np.max(signal)) if signal.size else 0.0
    if peak <= 0.0:
        return None

    if threshold_percent > 0.0:
        weights = np.where(signal >= peak * float(threshold_percent) / 100.0, signal, 0.0)
    else:
        weights = signal
    total = float(np.sum(weights))
    if total <= 0.0:
        return None

    y_px, x_px = np.indices(frame.shape, dtype=np.float64)
    centroid_x_px = float(np.sum(weights * x_px) / total)
    centroid_y_px = float(np.sum(weights * y_px) / total)
    dx_px = x_px - centroid_x_px
    dy_px = y_px - centroid_y_px
    sigma_xx_px2 = float(np.sum(weights * dx_px * dx_px) / total)
    sigma_yy_px2 = float(np.sum(weights * dy_px * dy_px) / total)
    sigma_xy_px2 = float(np.sum(weights * dx_px * dy_px) / total)

    covariance_px2 = np.array(
        [[sigma_xx_px2, sigma_xy_px2], [sigma_xy_px2, sigma_yy_px2]],
        dtype=np.float64,
    )
    eigenvalues_px2 = np.linalg.eigvalsh(covariance_px2)
    minor_sigma_px = math.sqrt(max(float(eigenvalues_px2[0]), 0.0))
    major_sigma_px = math.sqrt(max(float(eigenvalues_px2[1]), 0.0))
    if major_sigma_px <= 0.0:
        return None
    return BeamAxes(
        centroid_x_px=centroid_x_px,
        centroid_y_px=centroid_y_px,
        azimuth_deg=_major_axis_azimuth_deg(sigma_xx_px2, sigma_yy_px2, sigma_xy_px2),
        major_sigma_px=major_sigma_px,
        minor_sigma_px=minor_sigma_px,
        threshold_percent=float(threshold_percent),
    )


def beam_axes_from_rayci_fit(
    payload: Mapping[str, Any],
    *,
    threshold_percent: float = 50.0,
) -> BeamAxes | None:
    """Convert RayCi Gaussian-fit metadata into display-space beam axes."""

    if not bool(payload.get("available")) or not bool(payload.get("valid")):
        return None
    try:
        return BeamAxes(
            centroid_x_px=float(payload["centroid_x_px"]),
            centroid_y_px=float(payload["centroid_y_px"]),
            azimuth_deg=_normalize_axis_angle_deg(float(payload["azimuth_deg"])),
            major_sigma_px=float(payload["major_sigma_px"]),
            minor_sigma_px=float(payload["minor_sigma_px"]),
            threshold_percent=float(threshold_percent),
            sigma_environment=2.0,
            source=str(payload.get("source") or "rayci_gaussian_fit"),
        )
    except (KeyError, TypeError, ValueError):
        return None


def stabilize_beam_axes(
    current: BeamAxes,
    previous: BeamAxes | None,
    *,
    circularity_freeze_ratio: float = 0.98,
    max_step_deg: float = 90.0,
) -> BeamAxes:
    """Keep physical X/Y labels continuous when major/minor ordering changes."""

    if not 0.0 < circularity_freeze_ratio <= 1.0:
        raise ValueError("circularity_freeze_ratio must be between 0 and 1.")
    if not 0.0 < max_step_deg <= 90.0:
        raise ValueError("max_step_deg must be between 0 and 90.")

    major_angle = _normalize_axis_angle_deg(float(current.azimuth_deg))
    previous_x = previous.x_axis_azimuth_deg if previous is not None else 0.0
    if previous_x is None and previous is not None:
        previous_x = previous.azimuth_deg
    previous_x = _normalize_axis_angle_deg(float(previous_x))
    minor_angle = _normalize_axis_angle_deg(major_angle + 90.0)
    major_distance = _axis_angle_distance_deg(major_angle, previous_x)
    minor_distance = _axis_angle_distance_deg(minor_angle, previous_x)
    if major_distance <= minor_distance:
        candidate_angle = major_angle
        x_sigma_px = float(current.major_sigma_px)
        y_sigma_px = float(current.minor_sigma_px)
        assignment = "major_to_x"
        candidate_distance = major_distance
    else:
        candidate_angle = minor_angle
        x_sigma_px = float(current.minor_sigma_px)
        y_sigma_px = float(current.major_sigma_px)
        assignment = "minor_to_x"
        candidate_distance = minor_distance

    ratio = float(current.minor_sigma_px) / max(float(current.major_sigma_px), 1e-12)
    if ratio >= circularity_freeze_ratio and previous is not None:
        tracked_angle = previous_x
        status = "held_near_circular"
    elif candidate_distance > max_step_deg:
        tracked_angle = previous_x
        status = "held_large_jump"
    else:
        tracked_angle = candidate_angle
        status = "initialized" if previous is None else "tracked"
    if status.startswith("held"):
        delta = math.radians(tracked_angle - major_angle)
        cosine, sine = math.cos(delta), math.sin(delta)
        x_sigma_px = math.hypot(current.major_sigma_px * cosine, current.minor_sigma_px * sine)
        y_sigma_px = math.hypot(current.major_sigma_px * sine, current.minor_sigma_px * cosine)
    return replace(
        current,
        x_axis_azimuth_deg=_normalize_axis_angle_deg(tracked_angle),
        x_axis_sigma_px=x_sigma_px,
        y_axis_sigma_px=y_sigma_px,
        axis_assignment=assignment,
        orientation_status=status,
    )


def _validated_options(options: BeamDisplayOptions) -> BeamDisplayOptions:
    if not 0.0 <= float(options.threshold_percent) <= 100.0:
        raise ValueError("threshold_percent must be between 0 and 100.")
    if options.min_display_px <= 0 or options.max_display_px <= 0:
        raise ValueError("display sizes must be positive.")
    min_display_px = min(int(options.min_display_px), int(options.max_display_px))
    if options.zoom_factor < 1.0:
        raise ValueError("zoom_factor must be >= 1.")
    overlay_mode = options.overlay_mode.lower()
    if overlay_mode not in {"none", "box"}:
        raise ValueError("overlay_mode must be none or box.")
    colormap = options.colormap.lower()
    if colormap not in {"thermal", "gray"}:
        raise ValueError("colormap must be thermal or gray.")
    return BeamDisplayOptions(
        threshold_percent=float(options.threshold_percent),
        min_display_px=min_display_px,
        max_display_px=int(options.max_display_px),
        overlay_mode=overlay_mode,
        colormap=colormap,
        zoom_factor=float(options.zoom_factor),
        center_x_px=options.center_x_px,
        center_y_px=options.center_y_px,
        intensity_min=options.intensity_min,
        intensity_max=options.intensity_max,
    )


def _as_2d_float(image: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    frame = np.asarray(image, dtype=np.float32)
    if frame.ndim != 2:
        frame = np.squeeze(frame)
    if frame.ndim != 2:
        raise ValueError(f"beam image must be 2D, got shape {frame.shape}.")
    return np.where(np.isfinite(frame), frame, 0.0)


def _clamped_start(center_px: float, crop_size_px: int, full_size_px: int) -> int:
    start = int(round(center_px - crop_size_px / 2.0))
    return min(max(0, start), max(0, full_size_px - crop_size_px))


def _scale_to_uint8(
    image: np.ndarray[Any, Any],
    *,
    intensity_min: float | None = None,
    intensity_max: float | None = None,
) -> np.ndarray[Any, Any]:
    frame = _as_2d_float(image)
    min_value = float(np.min(frame)) if intensity_min is None else float(intensity_min)
    max_value = float(np.max(frame)) if intensity_max is None else float(intensity_max)
    if max_value <= min_value:
        return np.zeros(frame.shape, dtype=np.uint8)
    scaled = np.clip((frame - min_value) / (max_value - min_value), 0.0, 1.0)
    return (scaled * 255.0).astype(np.uint8)


def _apply_colormap(gray: np.ndarray[Any, Any], colormap: str) -> np.ndarray[Any, Any]:
    if colormap == "gray":
        return np.stack([gray, gray, gray], axis=-1)

    stops = np.asarray(
        [
            [0, 0, 0],
            [16, 0, 96],
            [0, 84, 255],
            [0, 210, 210],
            [70, 220, 70],
            [255, 230, 0],
            [255, 80, 0],
            [255, 255, 255],
        ],
        dtype=np.float32,
    )
    positions = np.asarray([0, 32, 72, 112, 150, 190, 230, 255], dtype=np.float32)
    values = gray.astype(np.float32, copy=False)
    channels = [np.interp(values, positions, stops[:, index]) for index in range(3)]
    return np.stack(channels, axis=-1).astype(np.uint8)


def _resize_for_display(
    image: Image.Image,
    options: BeamDisplayOptions,
) -> tuple[Image.Image, float]:
    width_px, height_px = image.size
    longest = max(width_px, height_px)
    if longest <= 0:
        return image, 1.0
    if longest < options.min_display_px:
        scale = options.min_display_px / float(longest)
    elif longest > options.max_display_px:
        scale = options.max_display_px / float(longest)
    else:
        scale = 1.0
    if abs(scale - 1.0) <= 1e-9:
        return image, 1.0
    return (
        image.resize(
            (max(1, int(width_px * scale)), max(1, int(height_px * scale))),
            Image.Resampling.NEAREST,
        ),
        scale,
    )


def _rotate_for_report(image: Image.Image, rotation_deg: float) -> Image.Image:
    if abs(rotation_deg) <= 1e-9:
        return image
    return image.rotate(
        rotation_deg,
        resample=Image.Resampling.NEAREST,
        expand=True,
        fillcolor=(0, 0, 0),
    )


def _beam_range_bbox(
    image: np.ndarray[Any, Any],
    *,
    threshold_percent: float,
) -> tuple[int, int, int, int] | None:
    if threshold_percent <= 0.0:
        return None
    frame = _as_2d_float(image)
    signal = frame - float(np.min(frame))
    np.maximum(signal, 0.0, out=signal)
    peak = float(np.max(signal)) if signal.size else 0.0
    if peak <= 0.0:
        return None
    mask = signal >= peak * float(threshold_percent) / 100.0
    if not np.any(mask):
        return None
    y_indices, x_indices = np.nonzero(mask)
    return (
        int(np.min(x_indices)),
        int(np.min(y_indices)),
        int(np.max(x_indices)),
        int(np.max(y_indices)),
    )


def _major_axis_azimuth_deg(
    sigma_xx_px2: float,
    sigma_yy_px2: float,
    sigma_xy_px2: float,
) -> float:
    if abs(sigma_xy_px2) < 1e-15 and abs(sigma_xx_px2 - sigma_yy_px2) < 1e-15:
        return 0.0
    azimuth_rad = 0.5 * math.atan2(2.0 * sigma_xy_px2, sigma_xx_px2 - sigma_yy_px2)
    azimuth_deg = math.degrees(azimuth_rad)
    if azimuth_deg >= 90.0:
        azimuth_deg -= 180.0
    if azimuth_deg < -90.0:
        azimuth_deg += 180.0
    return float(azimuth_deg)


def _draw_threshold_box(
    draw: ImageDraw.ImageDraw,
    bbox: tuple[int, int, int, int],
    *,
    scale: float,
) -> None:
    x_min, y_min, x_max, y_max = (float(value) * scale for value in bbox)
    draw.rectangle(
        [(x_min, y_min), (x_max, y_max)],
        outline=(255, 240, 80),
        width=2,
    )


def _draw_beam_axes(
    draw: ImageDraw.ImageDraw,
    axes: BeamAxes,
    *,
    crop: BeamCrop,
    scale: float,
) -> None:
    image_height_px, image_width_px = crop.image.shape
    center_x = (float(axes.centroid_x_px) - float(crop.x_offset_px)) * scale
    center_y = (float(axes.centroid_y_px) - float(crop.y_offset_px)) * scale
    width_px = float(image_width_px) * scale
    height_px = float(image_height_px) * scale
    if not (-10.0 <= center_x <= width_px + 10.0 and -10.0 <= center_y <= height_px + 10.0):
        return

    angle_rad = math.radians(float(axes.azimuth_deg))
    major = (math.cos(angle_rad), math.sin(angle_rad))
    minor = (-math.sin(angle_rad), math.cos(angle_rad))
    _draw_sigma_ellipse(
        draw,
        center=(center_x, center_y),
        major_direction=major,
        minor_direction=minor,
        major_radius=max(1.0, axes.major_sigma_px * axes.sigma_environment * scale),
        minor_radius=max(1.0, axes.minor_sigma_px * axes.sigma_environment * scale),
        color=(80, 255, 170),
    )
    x_angle_deg = (
        float(axes.x_axis_azimuth_deg)
        if axes.x_axis_azimuth_deg is not None
        else float(axes.azimuth_deg)
    )
    x_angle_rad = math.radians(x_angle_deg)
    x_direction = (math.cos(x_angle_rad), math.sin(x_angle_rad))
    y_angle = math.radians(
        axes.y_axis_azimuth_deg if axes.y_axis_azimuth_deg is not None else x_angle_deg + 90
    )
    y_direction = (math.cos(y_angle), math.sin(y_angle))
    x_sigma_px = (
        float(axes.x_axis_sigma_px)
        if axes.x_axis_sigma_px is not None
        else float(axes.major_sigma_px)
    )
    y_sigma_px = (
        float(axes.y_axis_sigma_px)
        if axes.y_axis_sigma_px is not None
        else float(axes.minor_sigma_px)
    )
    x_half_length = _axis_half_length_px(
        x_sigma_px,
        scale=scale,
        fallback=min(width_px, height_px) * 0.35,
    )
    y_half_length = _axis_half_length_px(
        y_sigma_px,
        scale=scale,
        fallback=min(width_px, height_px) * 0.25,
    )
    _draw_axis_line(
        draw,
        center=(center_x, center_y),
        direction=x_direction,
        half_length=x_half_length,
        color=(80, 220, 255),
        label="X",
    )
    _draw_axis_line(
        draw,
        center=(center_x, center_y),
        direction=y_direction,
        half_length=y_half_length,
        color=(255, 80, 255),
        label="Y",
    )


def _draw_sigma_ellipse(
    draw: ImageDraw.ImageDraw,
    *,
    center: tuple[float, float],
    major_direction: tuple[float, float],
    minor_direction: tuple[float, float],
    major_radius: float,
    minor_radius: float,
    color: tuple[int, int, int],
) -> None:
    center_x, center_y = center
    major_x, major_y = major_direction
    minor_x, minor_y = minor_direction
    points: list[tuple[float, float]] = []
    for index in range(80):
        theta = 2.0 * math.pi * float(index) / 80.0
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        points.append(
            (
                center_x + major_x * major_radius * cos_theta + minor_x * minor_radius * sin_theta,
                center_y + major_y * major_radius * cos_theta + minor_y * minor_radius * sin_theta,
            )
        )
    if len(points) >= 2:
        draw.line(points + [points[0]], fill=color, width=2)


def _axis_half_length_px(sigma_px: float, *, scale: float, fallback: float) -> float:
    if math.isfinite(float(sigma_px)) and sigma_px > 0.0:
        return max(16.0, float(sigma_px) * 2.3 * scale)
    return max(16.0, fallback)


def _draw_axis_line(
    draw: ImageDraw.ImageDraw,
    *,
    center: tuple[float, float],
    direction: tuple[float, float],
    half_length: float,
    color: tuple[int, int, int],
    label: str,
) -> None:
    center_x, center_y = center
    dx, dy = direction
    start = (center_x - dx * half_length, center_y - dy * half_length)
    end = (center_x + dx * half_length, center_y + dy * half_length)
    draw.line([start, end], fill=color, width=2)
    label_position = (end[0] + dx * 4.0, end[1] + dy * 4.0)
    draw.text(label_position, label, fill=color, font=ImageFont.load_default())


def _draw_title(image: Image.Image, title: str) -> None:
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    text_bbox = draw.textbbox((0, 0), title, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    pad = 5
    draw.rectangle(
        [(0, 0), (text_width + pad * 2, text_height + pad * 2)],
        fill=(0, 0, 0),
    )
    draw.text((pad, pad), title, fill=(255, 255, 255), font=font)


def _axis_angle_distance_deg(first: float, second: float) -> float:
    return abs((float(first) - float(second) + 90.0) % 180.0 - 90.0)


def _normalize_axis_angle_deg(value: float) -> float:
    return float((float(value) + 90.0) % 180.0 - 90.0)


__all__ = [
    "BeamAxes",
    "BeamCrop",
    "BeamDisplayOptions",
    "beam_axes_from_rayci_fit",
    "crop_beam_region",
    "estimate_beam_axes",
    "render_beam_spot_image",
    "save_beam_spot_png",
    "stabilize_beam_axes",
]
