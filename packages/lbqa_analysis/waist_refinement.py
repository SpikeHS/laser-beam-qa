"""Pure helpers for diagnostic waist refinement and astigmatic beam summaries."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from scipy.optimize import minimize_scalar


def diameter_at_z(
    *,
    z_mm: float,
    waist_z_mm: float,
    waist_diameter_um: float,
    full_angle_mrad: float,
) -> float:
    """Evaluate the fitted Gaussian-caustic diameter at one z position."""

    return math.sqrt(
        float(waist_diameter_um) ** 2
        + (float(full_angle_mrad) * (float(z_mm) - float(waist_z_mm))) ** 2
    )


def minimum_area_from_zscan(
    zscan_result: Any,
    *,
    plane_results: Sequence[Any] = (),
) -> dict[str, Any]:
    """Find the fitted z plane minimizing the X/Y ellipse area.

    X and Y waists remain independent. This helper minimizes their fitted
    diameter product and does not replace either axis-specific waist.
    """

    parameters = _fit_parameters(zscan_result)
    if parameters is None:
        return {
            "valid": False,
            "source": "main_zscan_fit",
            "invalid_reason": "fitted_axis_parameters_not_available",
        }
    z_x_mm, diameter_x_um, angle_x_mrad, z_y_mm, diameter_y_um, angle_y_mrad = parameters
    lower_mm = min(z_x_mm, z_y_mm)
    upper_mm = max(z_x_mm, z_y_mm)

    def area_um2(z_mm: float) -> float:
        width_x_um = diameter_at_z(
            z_mm=z_mm,
            waist_z_mm=z_x_mm,
            waist_diameter_um=diameter_x_um,
            full_angle_mrad=angle_x_mrad,
        )
        width_y_um = diameter_at_z(
            z_mm=z_mm,
            waist_z_mm=z_y_mm,
            waist_diameter_um=diameter_y_um,
            full_angle_mrad=angle_y_mrad,
        )
        return math.pi * width_x_um * width_y_um / 4.0

    if math.isclose(lower_mm, upper_mm, rel_tol=0.0, abs_tol=1e-15):
        z_area_mm = lower_mm
    else:
        optimization = minimize_scalar(
            area_um2,
            bounds=(lower_mm, upper_mm),
            method="bounded",
            options={"xatol": 1e-12},
        )
        if not optimization.success or not math.isfinite(float(optimization.x)):
            return {
                "valid": False,
                "source": "main_zscan_fit",
                "invalid_reason": "minimum_area_optimization_failed",
            }
        z_area_mm = float(optimization.x)

    width_x_um = diameter_at_z(
        z_mm=z_area_mm,
        waist_z_mm=z_x_mm,
        waist_diameter_um=diameter_x_um,
        full_angle_mrad=angle_x_mrad,
    )
    width_y_um = diameter_at_z(
        z_mm=z_area_mm,
        waist_z_mm=z_y_mm,
        waist_diameter_um=diameter_y_um,
        full_angle_mrad=angle_y_mrad,
    )
    nearest_index = _nearest_plane_index(plane_results, z_area_mm)
    return {
        "valid": True,
        "source": "main_zscan_fit",
        "z_mm": z_area_mm,
        "diameter_x_um": width_x_um,
        "diameter_y_um": width_y_um,
        "area_um2": math.pi * width_x_um * width_y_um / 4.0,
        "nearest_plane_index_count": nearest_index,
        "position_trusted_for_fit": True,
    }


def summarize_waist_refinement(
    records: Sequence[Mapping[str, Any]],
    *,
    requested_step_um: float,
    requested_move_count: int,
    windows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize diagnostic fine-scan observations without fitting divergence."""

    valid_records = [record for record in records if _point_valid(record.get("point"))]
    base = {
        "enabled": True,
        "performed": bool(records),
        "purpose": "waist_location_diagnostic_only",
        "position_policy": "excluded_from_main_divergence_fit",
        "requested_step_um": float(requested_step_um),
        "requested_move_count": int(requested_move_count),
        "windows": [dict(window) for window in windows],
        "points_count": len(records),
        "valid_points_count": len(valid_records),
    }
    if not valid_records:
        return {
            **base,
            "valid": False,
            "invalid_reason": "no_valid_waist_refinement_points",
            "x_minimum": {},
            "y_minimum": {},
            "observed_area_minimum": {},
        }

    x_record = min(valid_records, key=lambda value: _point_width(value, "x"))
    y_record = min(valid_records, key=lambda value: _point_width(value, "y"))
    area_record = min(valid_records, key=_record_area_um2)
    growth = waist_growth_status(records)
    return {
        **base,
        "valid": True,
        "invalid_reason": None,
        "x_minimum": {**_minimum_payload(x_record, axis_name="x"), **growth["x"]},
        "y_minimum": {**_minimum_payload(y_record, axis_name="y"), **growth["y"]},
        "observed_area_minimum": _area_payload(area_record),
        "both_confirmed": growth["both_confirmed"],
    }


def waist_growth_status(
    records: Sequence[Mapping[str, Any]], *, required_increases: int = 3
) -> dict[str, Any]:
    """Confirm each observed minimum only after a left flank and three rises.

    This is a search stopping rule, not a confidence interval for the true waist.
    Invalid frames interrupt a rising sequence; a new global low revokes a
    previous confirmation, even if the other axis is still being searched.
    """

    if required_increases < 1:
        raise ValueError("required_increases must be positive.")
    result: dict[str, Any] = {}
    for axis in ("x", "y"):
        minimum = math.inf
        previous: float | None = None
        rises = 0
        confirmed = False
        has_left_flank = False
        minimum_index: int | None = None
        prior_valid: list[float] = []
        for index, record in enumerate(records):
            if not _point_valid(record.get("point")):
                previous = None
                rises = 0
                continue
            width = _point_width(record, axis)
            if width < minimum:
                minimum = width
                minimum_index = index
                has_left_flank = any(value > width for value in prior_valid)
                rises = 0
                confirmed = False
            elif previous is not None and width > previous:
                rises += 1
                if has_left_flank and rises >= required_increases:
                    confirmed = True
            else:
                rises = 0
            prior_valid.append(width)
            previous = width
        result[axis] = {
            "confirmed": confirmed,
            "growth_count": rises,
            "required_increases": required_increases,
            "left_flank_observed": has_left_flank,
            "minimum_record_index": minimum_index,
            "confirmation_status": "trend_confirmed" if confirmed else "candidate_only",
        }
    result["both_confirmed"] = result["x"]["confirmed"] and result["y"]["confirmed"]
    return result


def _fit_parameters(zscan_result: Any) -> tuple[float, float, float, float, float, float] | None:
    names = (
        "waist_z_x_mm",
        "waist_diameter_x_um",
        "full_angle_x_mrad",
        "waist_z_y_mm",
        "waist_diameter_y_um",
        "full_angle_y_mrad",
    )
    values = [_optional_finite(_read_field(zscan_result, name, None)) for name in names]
    if any(value is None for value in values):
        return None
    result = tuple(float(value) for value in values if value is not None)
    if result[1] <= 0.0 or result[2] < 0.0 or result[4] <= 0.0 or result[5] < 0.0:
        return None
    return result  # type: ignore[return-value]


def _minimum_payload(record: Mapping[str, Any], *, axis_name: str) -> dict[str, Any]:
    point = record["point"]
    other_axis = "y" if axis_name == "x" else "x"
    return {
        "axis": axis_name.upper(),
        "diameter_um": _point_width(record, axis_name),
        "paired_diameter_um": _point_width(record, other_axis),
        "beam_area_um2": _record_area_um2(record),
        "z_command_mm": _optional_finite(record.get("z_command_mm")),
        "z_actual_mm": _optional_finite(_read_field(point, "z_actual_mm", None)),
        "plane_index_count": int(record["plane_index_count"]),
        "refinement_index_count": int(record["refinement_index_count"]),
        "window_index_count": int(record["window_index_count"]),
        "position_trusted_for_fit": False,
        "source": "waist_refinement_observed_minimum",
    }


def _area_payload(record: Mapping[str, Any]) -> dict[str, Any]:
    point = record["point"]
    return {
        "diameter_x_um": _point_width(record, "x"),
        "diameter_y_um": _point_width(record, "y"),
        "area_um2": _record_area_um2(record),
        "z_command_mm": _optional_finite(record.get("z_command_mm")),
        "z_actual_mm": _optional_finite(_read_field(point, "z_actual_mm", None)),
        "plane_index_count": int(record["plane_index_count"]),
        "refinement_index_count": int(record["refinement_index_count"]),
        "window_index_count": int(record["window_index_count"]),
        "position_trusted_for_fit": False,
        "source": "waist_refinement_observed_area_minimum",
    }


def _record_area_um2(record: Mapping[str, Any]) -> float:
    return math.pi * _point_width(record, "x") * _point_width(record, "y") / 4.0


def _point_width(record: Mapping[str, Any], axis_name: str) -> float:
    return float(_read_field(record["point"], f"d4sigma_{axis_name}_um", math.inf))


def _point_valid(point: Any) -> bool:
    if not bool(_read_field(point, "valid", False)):
        return False
    values = (
        _optional_finite(_read_field(point, "d4sigma_x_um", None)),
        _optional_finite(_read_field(point, "d4sigma_y_um", None)),
    )
    return all(value is not None and value > 0.0 for value in values)


def _nearest_plane_index(plane_results: Sequence[Any], z_mm: float) -> int | None:
    candidates = []
    for index, point in enumerate(plane_results):
        point_z_mm = _optional_finite(_read_field(point, "z_actual_mm", None))
        if point_z_mm is not None:
            candidates.append((abs(point_z_mm - z_mm), index))
    return None if not candidates else min(candidates)[1]


def _optional_finite(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _read_field(source: Any, field_name: str, default: Any) -> Any:
    if isinstance(source, Mapping):
        return source.get(field_name, default)
    return getattr(source, field_name, default)


__all__ = ["diameter_at_z", "minimum_area_from_zscan", "summarize_waist_refinement"]
