"""Continuous ellipse-axis identity without filtering the measured diameters."""

from __future__ import annotations

import math
from typing import Any

_NEAR_CIRCLE_RELATIVE_GAP = 0.02


class StableEllipseAxes:
    """Track unoriented ellipse axes through rotations and major/minor crossings.

    Diameters may use any one consistent unit. The first X axis is whichever
    principal axis is closer to horizontal (major wins an exact tie). Later
    updates choose the closest direction modulo 180 degrees. As with any
    nearest-direction tracker, rotations of 45 degrees or more between
    observations can be ambiguous; no physical rotation rate is inferred.

    When ``1 - minor / major <= 0.02``, orientation is uncertain. Keep the
    previous direction, if available, and project the *current* covariance
    onto it. No diameter or image smoothing is applied. Use a new instance
    for an independent sequence.
    """

    def __init__(
        self, *, reference_angle_x_deg: float | None = None, x_is_major: bool = True
    ) -> None:
        if reference_angle_x_deg is not None and not math.isfinite(reference_angle_x_deg):
            raise ValueError("Reference axis angle must be finite.")
        self._angle_x_deg = (
            None if reference_angle_x_deg is None else _normalize_angle(reference_angle_x_deg)
        )
        self._x_is_major = bool(x_is_major)

    def update(
        self, major_diameter: float, minor_diameter: float, azimuth_deg: float
    ) -> dict[str, Any]:
        """Return tracked widths/directions plus the unmodified raw axes.

        Angles are represented in [-90, 90); they describe lines, not vectors.
        Invalid/non-positive diameters or major < minor raise ValueError and
        leave the tracking state unchanged.
        """
        major = float(major_diameter)
        minor = float(minor_diameter)
        azimuth = float(azimuth_deg)
        if not all(math.isfinite(value) for value in (major, minor, azimuth)):
            raise ValueError("Ellipse diameters and azimuth must be finite.")
        if minor <= 0.0 or major < minor:
            raise ValueError("Ellipse diameters must satisfy major >= minor > 0.")

        major_angle = _normalize_angle(azimuth)
        minor_angle = _normalize_angle(major_angle + 90.0)
        near_circle = minor / major >= 1.0 - _NEAR_CIRCLE_RELATIVE_GAP
        reference = 0.0 if self._angle_x_deg is None else self._angle_x_deg
        major_distance = _angle_distance(major_angle, reference)
        minor_distance = _angle_distance(minor_angle, reference)
        tied = math.isclose(major_distance, minor_distance, rel_tol=0.0, abs_tol=1e-12)
        x_is_major = self._x_is_major if tied else major_distance < minor_distance
        angle_x = major_angle if x_is_major else minor_angle
        if near_circle and self._angle_x_deg is not None:
            angle_x = self._angle_x_deg

        if near_circle:
            # Widths are a common multiple of covariance standard deviations,
            # so that multiplier cancels in the rotated squared-width formula.
            delta = math.radians(_normalize_angle(angle_x - major_angle))
            cosine, sine = math.cos(delta), math.sin(delta)
            width_x = math.hypot(major * cosine, minor * sine)
            width_y = math.hypot(major * sine, minor * cosine)
        else:
            width_x, width_y = (major, minor) if x_is_major else (minor, major)
            self._x_is_major = x_is_major
        if self._angle_x_deg is None:
            self._x_is_major = x_is_major
        self._angle_x_deg = angle_x
        return {
            "width_x": width_x,
            "width_y": width_y,
            "angle_x_deg": angle_x,
            "angle_y_deg": _normalize_angle(angle_x + 90.0),
            "orientation_uncertain": near_circle or tied,
            "raw_major_diameter": major,
            "raw_minor_diameter": minor,
            "raw_azimuth_deg": azimuth,
            "width_mode": "held_direction_projection" if near_circle else "principal_axes",
            "near_circle_relative_gap_threshold": _NEAR_CIRCLE_RELATIVE_GAP,
        }


def _normalize_angle(angle_deg: float) -> float:
    return (angle_deg + 90.0) % 180.0 - 90.0


def _angle_distance(first_deg: float, second_deg: float) -> float:
    return abs(_normalize_angle(first_deg - second_deg))


__all__ = ["StableEllipseAxes"]
