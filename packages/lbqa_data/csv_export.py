"""CSV export helpers for z-scan plane results."""

from __future__ import annotations

import csv
from collections.abc import Mapping
from pathlib import Path
from typing import Any

Z_SCAN_TABLE_FIELDS = [
    "run_id",
    "sample_id",
    "z_cmd_mm",
    "z_actual_mm",
    "scan_phase",
    "z_feedback_delta_um",
    "position_trusted_for_fit",
    "waist_refinement_included",
    "exposure_us",
    "frame_count",
    "centroid_x_um",
    "centroid_y_um",
    "peak_x_um",
    "peak_y_um",
    "d4sigma_x_um",
    "d4sigma_y_um",
    "d4sigma_major_um",
    "d4sigma_minor_um",
    "fwhm_x_um",
    "fwhm_y_um",
    "azimuth_deg",
    "ellipticity",
    "beam_area_um2",
    "width_source",
    "width_uncertainty_x_um",
    "width_uncertainty_y_um",
    "native_width_x_um",
    "native_width_y_um",
    "native_width_scale_x_ratio",
    "native_width_scale_y_ratio",
    "rayci_gfi_ratio",
    "peak_value",
    "saturation_pixels",
    "edge_energy_percent",
    "valid",
    "invalid_reason",
    "outlier",
    "outlier_reason",
    "fit_included",
    "fit_predicted_x_um",
    "fit_predicted_y_um",
    "fit_residual_x_um",
    "fit_residual_y_um",
    "fit_normalized_residual_x",
    "fit_normalized_residual_y",
    "fit_included_x",
    "fit_included_y",
    "axis_x_azimuth_deg",
    "axis_orientation_uncertain",
    "native_major_azimuth_deg",
]


def build_zscan_csv_row(
    *,
    run_id: str,
    sample_id: str,
    plane_result: Any,
    z_cmd_mm: float | None = None,
    exposure_us: float | None = None,
    frame_count: int | None = None,
    scan_phase: str | None = None,
    z_feedback_delta_um: float | None = None,
    position_trusted_for_fit: bool = True,
    waist_refinement_included: bool = False,
    axis_x_azimuth_deg: float | None = None,
    axis_orientation_uncertain: bool | None = None,
    native_major_azimuth_deg: float | None = None,
) -> dict[str, Any]:
    """Build one unit-bearing CSV row from a beam-plane result."""

    z_actual_mm = _read_field(plane_result, "z_actual_mm", None)
    invalid_reason = _read_field(plane_result, "invalid_reason", None)
    outlier_reason = _read_field(plane_result, "outlier_reason", None)
    diameter_x_um = _read_field(plane_result, "d4sigma_x_um", None)
    diameter_y_um = _read_field(plane_result, "d4sigma_y_um", None)
    return {
        "run_id": run_id,
        "sample_id": sample_id,
        "z_cmd_mm": _blank_if_none(z_actual_mm if z_cmd_mm is None else z_cmd_mm),
        "z_actual_mm": _blank_if_none(z_actual_mm),
        "scan_phase": "" if scan_phase is None else str(scan_phase),
        "z_feedback_delta_um": _blank_if_none(z_feedback_delta_um),
        "position_trusted_for_fit": bool(position_trusted_for_fit),
        "waist_refinement_included": bool(waist_refinement_included),
        "exposure_us": _blank_if_none(exposure_us),
        "frame_count": _blank_if_none(frame_count),
        "centroid_x_um": _blank_if_none(_read_field(plane_result, "centroid_x_um", None)),
        "centroid_y_um": _blank_if_none(_read_field(plane_result, "centroid_y_um", None)),
        "peak_x_um": _blank_if_none(_read_field(plane_result, "peak_x_um", None)),
        "peak_y_um": _blank_if_none(_read_field(plane_result, "peak_y_um", None)),
        "d4sigma_x_um": _blank_if_none(diameter_x_um),
        "d4sigma_y_um": _blank_if_none(diameter_y_um),
        "d4sigma_major_um": _blank_if_none(_read_field(plane_result, "d4sigma_major_um", None)),
        "d4sigma_minor_um": _blank_if_none(_read_field(plane_result, "d4sigma_minor_um", None)),
        "fwhm_x_um": _blank_if_none(_read_field(plane_result, "fwhm_x_um", None)),
        "fwhm_y_um": _blank_if_none(_read_field(plane_result, "fwhm_y_um", None)),
        "azimuth_deg": _blank_if_none(_read_field(plane_result, "azimuth_deg", None)),
        "ellipticity": _blank_if_none(_read_field(plane_result, "ellipticity", None)),
        "beam_area_um2": _beam_area_um2(diameter_x_um, diameter_y_um),
        "width_source": _blank_if_none(_read_field(plane_result, "width_source", None)),
        "width_uncertainty_x_um": _blank_if_none(
            _read_field(plane_result, "width_uncertainty_x_um", None)
        ),
        "width_uncertainty_y_um": _blank_if_none(
            _read_field(plane_result, "width_uncertainty_y_um", None)
        ),
        "native_width_x_um": _blank_if_none(
            _read_field(plane_result, "native_width_x_um", None)
        ),
        "native_width_y_um": _blank_if_none(
            _read_field(plane_result, "native_width_y_um", None)
        ),
        "native_width_scale_x_ratio": _blank_if_none(
            _read_field(plane_result, "native_width_scale_x_ratio", None)
        ),
        "native_width_scale_y_ratio": _blank_if_none(
            _read_field(plane_result, "native_width_scale_y_ratio", None)
        ),
        "rayci_gfi_ratio": _blank_if_none(
            _read_field(plane_result, "rayci_gfi_ratio", None)
        ),
        "peak_value": _blank_if_none(_read_field(plane_result, "peak_value", None)),
        "saturation_pixels": _blank_if_none(
            _read_field(plane_result, "saturation_pixels", None)
        ),
        "edge_energy_percent": _blank_if_none(
            _read_field(plane_result, "edge_energy_percent", None)
        ),
        "valid": bool(_read_field(plane_result, "valid", False)),
        "invalid_reason": "" if invalid_reason is None else str(invalid_reason),
        "outlier": bool(_read_field(plane_result, "outlier", False)),
        "outlier_reason": "" if outlier_reason is None else str(outlier_reason),
        "fit_included": False,
        "fit_included_x": False,
        "fit_included_y": False,
        "fit_predicted_x_um": "",
        "fit_predicted_y_um": "",
        "fit_residual_x_um": "",
        "fit_residual_y_um": "",
        "fit_normalized_residual_x": "",
        "fit_normalized_residual_y": "",
        "axis_x_azimuth_deg": _blank_if_none(axis_x_azimuth_deg),
        "axis_orientation_uncertain": _blank_if_none(axis_orientation_uncertain),
        "native_major_azimuth_deg": _blank_if_none(native_major_azimuth_deg),
    }


def append_zscan_csv_row(csv_path: str | Path, row: Mapping[str, Any]) -> None:
    """Append a z-scan row, writing the canonical header when needed."""

    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=Z_SCAN_TABLE_FIELDS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in Z_SCAN_TABLE_FIELDS})


def write_zscan_csv(csv_path: str | Path, rows: list[Mapping[str, Any]]) -> None:
    """Write a complete z-scan table using the canonical field order."""

    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=Z_SCAN_TABLE_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in Z_SCAN_TABLE_FIELDS})


def _read_field(source: Any, field_name: str, default: Any) -> Any:
    if isinstance(source, Mapping):
        return source.get(field_name, default)
    return getattr(source, field_name, default)


def _blank_if_none(value: Any) -> Any:
    return "" if value is None else value


def _beam_area_um2(diameter_x_um: Any, diameter_y_um: Any) -> float | str:
    if diameter_x_um is None or diameter_y_um is None:
        return ""
    return 3.141592653589793 * float(diameter_x_um) * float(diameter_y_um) / 4.0


__all__ = [
    "Z_SCAN_TABLE_FIELDS",
    "append_zscan_csv_row",
    "build_zscan_csv_row",
    "write_zscan_csv",
]
