"""Paraxial Gaussian-beam predictions derived from a fitted waist diameter."""

from __future__ import annotations

import math
from typing import Any


def predict_gaussian_divergence(
    *,
    waist_diameter_x_um: float | None,
    waist_diameter_y_um: float | None,
    wavelength_nm: float | None,
    m2: float = 1.0,
) -> dict[str, Any]:
    """Predict far-field divergence using theta = M2 * lambda / (pi * w0)."""

    payload: dict[str, Any] = {
        "model": "paraxial_gaussian_from_fitted_waist",
        "wavelength_nm": wavelength_nm,
        "m2": m2,
        "waist_definition": "Gaussian 1/e^2 intensity diameter",
        "angle_convention": "full_angle_and_half_angle",
        "predicted_full_angle_x_mrad": None,
        "predicted_full_angle_y_mrad": None,
        "predicted_half_angle_x_mrad": None,
        "predicted_half_angle_y_mrad": None,
        "predicted_rayleigh_range_x_mm": None,
        "predicted_rayleigh_range_y_mm": None,
        "fraunhofer_ideal_full_angle_x_mrad": None,
        "fraunhofer_ideal_full_angle_y_mrad": None,
        "fraunhofer_ideal_half_angle_x_mrad": None,
        "fraunhofer_ideal_half_angle_y_mrad": None,
        "fraunhofer_ideal_full_angle_x_deg": None,
        "fraunhofer_ideal_full_angle_y_deg": None,
        "fraunhofer_valid": False,
        "fraunhofer_invalid_reason": None,
        "valid": False,
        "invalid_reason": None,
    }
    invalid_reason = _prediction_input_error(
        waist_diameter_x_um=waist_diameter_x_um,
        waist_diameter_y_um=waist_diameter_y_um,
        wavelength_nm=wavelength_nm,
        m2=m2,
    )
    if invalid_reason is not None:
        payload["invalid_reason"] = invalid_reason
        return payload

    assert wavelength_nm is not None
    assert waist_diameter_x_um is not None
    assert waist_diameter_y_um is not None
    full_x = _full_angle_mrad(wavelength_nm, waist_diameter_x_um, m2)
    full_y = _full_angle_mrad(wavelength_nm, waist_diameter_y_um, m2)
    fraunhofer_x = _ideal_fraunhofer_full_angle(wavelength_nm, waist_diameter_x_um)
    fraunhofer_y = _ideal_fraunhofer_full_angle(wavelength_nm, waist_diameter_y_um)
    fraunhofer_valid = fraunhofer_x is not None and fraunhofer_y is not None
    payload.update(
        {
            "predicted_full_angle_x_mrad": full_x,
            "predicted_full_angle_y_mrad": full_y,
            "predicted_half_angle_x_mrad": full_x / 2.0,
            "predicted_half_angle_y_mrad": full_y / 2.0,
            "predicted_rayleigh_range_x_mm": _rayleigh_range_mm(
                wavelength_nm,
                waist_diameter_x_um,
                m2,
            ),
            "predicted_rayleigh_range_y_mm": _rayleigh_range_mm(
                wavelength_nm,
                waist_diameter_y_um,
                m2,
            ),
            "fraunhofer_ideal_full_angle_x_mrad": _angle_mrad(fraunhofer_x),
            "fraunhofer_ideal_full_angle_y_mrad": _angle_mrad(fraunhofer_y),
            "fraunhofer_ideal_half_angle_x_mrad": _half_angle_mrad(fraunhofer_x),
            "fraunhofer_ideal_half_angle_y_mrad": _half_angle_mrad(fraunhofer_y),
            "fraunhofer_ideal_full_angle_x_deg": _angle_deg(fraunhofer_x),
            "fraunhofer_ideal_full_angle_y_deg": _angle_deg(fraunhofer_y),
            "fraunhofer_valid": fraunhofer_valid,
            "fraunhofer_invalid_reason": (
                None if fraunhofer_valid else "fraunhofer_arcsin_argument_exceeds_one"
            ),
            "valid": True,
        }
    )
    return payload


def calculate_measured_m2(
    *,
    waist_diameter_x_um: float | None,
    waist_diameter_y_um: float | None,
    full_angle_x_mrad: float | None,
    full_angle_y_mrad: float | None,
    wavelength_nm: float | None,
) -> dict[str, Any]:
    """Return paraxial effective M2 values from measured waist and caustic slope."""

    payload: dict[str, Any] = {
        "definition": "pi*D0*Theta_full/(4*lambda)",
        "wavelength_nm": wavelength_nm,
        "m2_x": None,
        "m2_y": None,
        "valid": False,
        "invalid_reason": None,
    }
    values = (
        waist_diameter_x_um,
        waist_diameter_y_um,
        full_angle_x_mrad,
        full_angle_y_mrad,
        wavelength_nm,
    )
    if any(value is None for value in values):
        payload["invalid_reason"] = "measured_waist_or_divergence_not_available"
        return payload
    numbers = tuple(float(value) for value in values if value is not None)
    if any(not math.isfinite(value) or value <= 0.0 for value in numbers):
        payload["invalid_reason"] = "measured_m2_inputs_must_be_positive"
        return payload
    assert wavelength_nm is not None
    assert waist_diameter_x_um is not None
    assert waist_diameter_y_um is not None
    assert full_angle_x_mrad is not None
    assert full_angle_y_mrad is not None
    payload.update(
        {
            "m2_x": _measured_m2_axis(
                waist_diameter_x_um,
                full_angle_x_mrad,
                wavelength_nm,
            ),
            "m2_y": _measured_m2_axis(
                waist_diameter_y_um,
                full_angle_y_mrad,
                wavelength_nm,
            ),
            "valid": True,
        }
    )
    return payload


def _prediction_input_error(
    *,
    waist_diameter_x_um: float | None,
    waist_diameter_y_um: float | None,
    wavelength_nm: float | None,
    m2: float,
) -> str | None:
    if wavelength_nm is None:
        return "wavelength_nm_not_provided"
    if not math.isfinite(float(wavelength_nm)) or wavelength_nm <= 0.0:
        return "wavelength_nm_must_be_positive"
    if not math.isfinite(float(m2)) or m2 < 1.0:
        return "m2_must_be_at_least_1"
    for axis_name, diameter_um in (
        ("x", waist_diameter_x_um),
        ("y", waist_diameter_y_um),
    ):
        if diameter_um is None:
            return f"waist_diameter_{axis_name}_um_not_available"
        if not math.isfinite(float(diameter_um)) or diameter_um <= 0.0:
            return f"waist_diameter_{axis_name}_um_must_be_positive"
    return None


def _full_angle_mrad(wavelength_nm: float, waist_diameter_um: float, m2: float) -> float:
    return 4.0 * float(m2) * float(wavelength_nm) / (math.pi * float(waist_diameter_um))


def _rayleigh_range_mm(wavelength_nm: float, waist_diameter_um: float, m2: float) -> float:
    waist_radius_um = float(waist_diameter_um) / 2.0
    wavelength_um = float(wavelength_nm) / 1000.0
    return math.pi * waist_radius_um**2 / (float(m2) * wavelength_um) / 1000.0


def _ideal_fraunhofer_full_angle(
    wavelength_nm: float,
    waist_diameter_um: float,
) -> float | None:
    argument = 2.0 * (float(wavelength_nm) / 1000.0) / (
        math.pi * float(waist_diameter_um)
    )
    if not 0.0 <= argument <= 1.0:
        return None
    return 2.0 * math.asin(argument)


def _angle_mrad(full_angle_rad: float | None) -> float | None:
    return None if full_angle_rad is None else full_angle_rad * 1000.0


def _half_angle_mrad(full_angle_rad: float | None) -> float | None:
    return None if full_angle_rad is None else full_angle_rad * 500.0


def _angle_deg(full_angle_rad: float | None) -> float | None:
    return None if full_angle_rad is None else math.degrees(full_angle_rad)


def _measured_m2_axis(
    waist_diameter_um: float,
    full_angle_mrad: float,
    wavelength_nm: float,
) -> float:
    return (
        math.pi
        * float(waist_diameter_um)
        * float(full_angle_mrad)
        / (4.0 * float(wavelength_nm))
    )


__all__ = ["calculate_measured_m2", "predict_gaussian_divergence"]
