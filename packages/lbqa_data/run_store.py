"""Run-level persistence for z-scan beam quality tests."""

from __future__ import annotations

import csv
import json
import math
import re
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from lbqa_analysis.gaussian_beam import calculate_measured_m2, predict_gaussian_divergence
from lbqa_analysis.waist_refinement import minimum_area_from_zscan
from lbqa_contracts.errors import ErrorCode

from lbqa_data.csv_export import append_zscan_csv_row, build_zscan_csv_row, write_zscan_csv
from lbqa_data.data_exceptions import DataStoreError
from lbqa_data.image_store import ImageStore
from lbqa_data.report_generator import ReportGenerator

try:
    import yaml
except ModuleNotFoundError:
    yaml = None


@dataclass(frozen=True, slots=True)
class RunRecord:
    """Location and identity of one persisted measurement run."""

    run_id: str
    sample_id: str
    run_dir: Path
    created_at_iso: str
    recipe_id: str | None


class RunStore:
    """Create a traceable run folder and persist all data/report artifacts."""

    def __init__(
        self,
        root_output_dir: str | Path,
        *,
        report_generator: ReportGenerator | None = None,
        timestamp_factory: Any | None = None,
    ) -> None:
        self.root_output_dir = Path(root_output_dir)
        self.root_output_dir.mkdir(parents=True, exist_ok=True)
        self.report_generator = report_generator or ReportGenerator()
        self.timestamp_factory = timestamp_factory or _now
        self.current_run: RunRecord | None = None
        self.image_store: ImageStore | None = None
        self._recipe: Any | None = None
        self._calibration: Any | None = None
        self._system_snapshot: Any | None = None
        self._plane_results: list[Any] = []
        self._plane_metadata: list[dict[str, Any]] = []
        self._zscan_result: Any | None = None
        self._errors: list[dict[str, Any]] = []
        self._fit_review: dict[str, Any] | None = None
        self._waist_refinement: dict[str, Any] | None = None
        self._finalized_at_iso: str | None = None

    def create_run(self, sample_id: str, recipe: Any) -> RunRecord:
        """Create the canonical run directory layout for a sample and recipe."""

        if not sample_id:
            raise DataStoreError(ErrorCode.E_REPORT_GENERATION_FAILED, "sample_id is required.")
        timestamp = self.timestamp_factory()
        timestamp_label = timestamp.strftime("%Y%m%d_%H%M%S")
        safe_sample_id = _safe_filename_token(sample_id)
        run_dir = _unique_run_dir(self.root_output_dir, f"Run_{timestamp_label}_{safe_sample_id}")
        record = RunRecord(
            run_id=run_dir.name,
            sample_id=sample_id,
            run_dir=run_dir,
            created_at_iso=timestamp.isoformat(),
            recipe_id=_read_field(recipe, "recipe_id", None),
        )
        self.current_run = record
        self.image_store = ImageStore(run_dir)
        self._recipe = recipe
        self._calibration = None
        self._system_snapshot = None
        self._plane_results = []
        self._plane_metadata = []
        self._zscan_result = None
        self._errors = []
        self._fit_review = None
        self._waist_refinement = None
        self._finalized_at_iso = None
        self.save_recipe_snapshot(recipe)
        return record

    def save_recipe_snapshot(self, recipe: Any | None = None) -> Path:
        """Write `recipe_snapshot.yaml` for the current run."""

        if recipe is not None:
            self._recipe = recipe
        return self._write_yaml_snapshot("recipe_snapshot.yaml", self._recipe)

    def save_calibration_snapshot(self, calibration: Any) -> Path:
        """Write `calibration_snapshot.yaml` for the current run."""

        self._calibration = calibration
        return self._write_yaml_snapshot("calibration_snapshot.yaml", calibration)

    def save_system_snapshot(self, system_snapshot: Any) -> Path:
        """Write `system_snapshot.yaml` for the current run."""

        self._system_snapshot = system_snapshot
        return self._write_yaml_snapshot("system_snapshot.yaml", system_snapshot)

    def save_plane_result(
        self,
        *,
        plane_result: Any,
        z_cmd_mm: float | None = None,
        z_target_mm: float | None = None,
        exposure_us: float | None = None,
        frame_count: int | None = None,
        plane_index_count: int | None = None,
        phase: str | None = None,
        run_id: str | None = None,
        sample_id: str | None = None,
        z_feedback_delta_um: float | None = None,
        position_trusted_for_fit: bool = True,
        waist_refinement_included: bool = False,
        axis_x_azimuth_deg: float | None = None,
        axis_orientation_uncertain: bool | None = None,
        native_major_azimuth_deg: float | None = None,
    ) -> Path:
        """Append one analyzed plane to `z_scan_table.csv`."""

        _ = (plane_index_count, run_id)
        record = self._require_run()
        z_command_mm = z_cmd_mm if z_cmd_mm is not None else z_target_mm
        row = build_zscan_csv_row(
            run_id=record.run_id,
            sample_id=sample_id or record.sample_id,
            plane_result=plane_result,
            z_cmd_mm=z_command_mm,
            exposure_us=exposure_us,
            frame_count=frame_count,
            scan_phase=phase,
            z_feedback_delta_um=z_feedback_delta_um,
            position_trusted_for_fit=position_trusted_for_fit,
            waist_refinement_included=waist_refinement_included,
            axis_x_azimuth_deg=axis_x_azimuth_deg,
            axis_orientation_uncertain=axis_orientation_uncertain,
            native_major_azimuth_deg=native_major_azimuth_deg,
        )
        csv_path = record.run_dir / "z_scan_table.csv"
        append_zscan_csv_row(csv_path, row)
        self._plane_results.append(plane_result)
        self._plane_metadata.append(dict(row))
        self._write_result_summary()
        return csv_path

    def save_zscan_result(
        self,
        *,
        zscan_result: Any,
        run_id: str | None = None,
        sample_id: str | None = None,
    ) -> Path:
        """Persist the final z-scan result into `result_summary.json`."""

        _ = (run_id, sample_id)
        self._require_run()
        self._zscan_result = zscan_result
        points = list(_read_field(zscan_result, "points", []) or [])
        existing_rows = _match_result_points_to_plane_rows(points, self._plane_results)
        for point, row_index in zip(points, existing_rows, strict=True):
            if row_index is None:
                self.save_plane_result(plane_result=point)
        self._sync_zscan_csv_outliers(zscan_result)
        return self._write_result_summary()

    def save_fit_review(self, review: Mapping[str, Any]) -> Path:
        """Keep the operator's fit selection in the consolidated result summary."""

        self._fit_review = dict(_to_plain_data(review))
        return self._write_result_summary()

    def preview_summary(self, zscan_result: Any | None = None) -> dict[str, Any]:
        """Build the final-summary payload without persisting or changing run state."""

        return self._build_summary(zscan_result=zscan_result)

    def save_waist_refinement(self, refinement: Mapping[str, Any]) -> Path:
        """Persist diagnostic waist-search details without changing contracts."""

        self._waist_refinement = dict(_to_plain_data(refinement))
        return self._write_result_summary()

    def save_failure(
        self,
        *,
        error_code: ErrorCode | str,
        message: str,
        details: Mapping[str, Any] | None = None,
    ) -> Path:
        """Persist a failure while preserving all partial artifacts."""

        record = self._require_run()
        payload = {
            "error_code": str(error_code),
            "message": message,
            "details": _to_plain_data(details or {}),
            "timestamp_iso": self.timestamp_factory().isoformat(),
        }
        self._errors.append(payload)
        error_log = record.run_dir / "logs" / "error.log"
        error_log.parent.mkdir(parents=True, exist_ok=True)
        with error_log.open("a", encoding="utf-8") as file:
            file.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        return self._write_result_summary()

    def save_failure_data(self, failure_payload: Mapping[str, Any]) -> Path:
        """Compatibility method called by sequencer error recovery."""

        errors = list(failure_payload.get("errors") or [])
        if not errors:
            errors = [ErrorCode.E_REPORT_GENERATION_FAILED.value]
        path = self._require_run().run_dir / "result_summary.json"
        for error_code in errors:
            path = self.save_failure(
                error_code=error_code,
                message=str(failure_payload.get("exception") or "sequencer failure"),
                details=failure_payload,
            )
        return path

    def finalize_run(
        self,
        *,
        render_html: bool = True,
        render_pdf: bool = False,
    ) -> dict[str, Any]:
        """Write the final summary and report artifacts for the current run."""

        record = self._require_run()
        self._finalized_at_iso = self.timestamp_factory().isoformat()
        summary = self._write_result_summary()
        summary_data = _read_json(summary)
        self._write_result_summary_csv(summary_data)
        try:
            if render_html:
                report_html = self.report_generator.generate_html(
                    run_dir=record.run_dir,
                    summary=summary_data,
                    zscan_result=self._zscan_result,
                    plane_results=self._plane_results,
                )
                if render_pdf:
                    self.report_generator.generate_pdf(html_path=report_html)
        finally:
            if self.image_store is not None:
                self.image_store.cleanup_working_files()
            _remove_empty_output_directories(record.run_dir)
        summary = self._write_result_summary()
        summary_data = _read_json(summary)
        self._write_result_summary_csv(summary_data)
        summary = self._write_result_summary()
        summary_data = _read_json(summary)
        return summary_data

    def discard_run(self) -> Path:
        """Delete the active unconfirmed run and clear all in-memory state."""

        record = self._require_run()
        run_dir = record.run_dir.resolve()
        root_dir = self.root_output_dir.resolve()
        if run_dir.parent != root_dir or not run_dir.name.startswith("Run_"):
            raise DataStoreError(
                ErrorCode.E_REPORT_GENERATION_FAILED,
                f"refusing to discard unexpected run directory: {run_dir}",
            )
        if run_dir.exists():
            shutil.rmtree(run_dir)
        self.current_run = None
        self.image_store = None
        self._recipe = None
        self._calibration = None
        self._system_snapshot = None
        self._plane_results = []
        self._plane_metadata = []
        self._zscan_result = None
        self._errors = []
        self._fit_review = None
        self._waist_refinement = None
        self._finalized_at_iso = None
        return run_dir

    def report_error(self, failure_payload: Mapping[str, Any]) -> None:
        """Compatibility hook if a RunStore is also used as a report service."""

        if not self._errors:
            self.save_failure_data(failure_payload)
        self.finalize_run(render_html=True, render_pdf=False)

    def _write_yaml_snapshot(self, filename: str, payload: Any) -> Path:
        if payload is None:
            raise DataStoreError(
                ErrorCode.E_REPORT_GENERATION_FAILED,
                f"{filename} payload is required.",
            )
        record = self._require_run()
        path = record.run_dir / filename
        plain_payload = _to_plain_data(payload)
        if yaml is None:
            text = json.dumps(plain_payload, ensure_ascii=False, indent=2, sort_keys=False)
        else:
            text = yaml.safe_dump(plain_payload, sort_keys=False, allow_unicode=True)
        path.write_text(
            text,
            encoding="utf-8",
        )
        return path

    def _write_result_summary(self) -> Path:
        record = self._require_run()
        path = record.run_dir / "result_summary.json"
        summary = self._build_summary()
        path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return path

    def _write_result_summary_csv(self, summary: Mapping[str, Any]) -> Path:
        record = self._require_run()
        path = record.run_dir / "result_summary.csv"
        rows = list(_flatten_summary_rows(summary))
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=("metric", "value", "value_type", "unit"))
            writer.writeheader()
            writer.writerows(rows)
        return path

    def _build_summary(self, zscan_result: Any | None = None) -> dict[str, Any]:
        record = self._require_run()
        if zscan_result is None:
            zscan_result = self._zscan_result
        points = _summary_points(self._plane_results, zscan_result)
        refinement = self._waist_refinement or {}
        diagnostics = _read_field(zscan_result, "fit_diagnostics", {}) or {}
        scan_error = diagnostics.get("scan_error") or refinement.get("scan_error")
        incomplete = bool(
            scan_error or self._errors or refinement.get("scan_status") == "incomplete"
            or diagnostics.get("scan_status") == "incomplete"
        )
        final_judgement = _final_judgement(zscan_result, self._errors, self._recipe)
        recipe_id = _read_field(self._recipe, "recipe_id", record.recipe_id)
        morphology_summary = _morphology_summary(points)
        divergence_summary = _divergence_summary(zscan_result)
        linear = divergence_summary.get("model") == "far_field_linear"
        if linear and incomplete:
            final_judgement = "UNKNOWN"
        observed_minima = _observed_minima(points, self._plane_metadata, self._waist_refinement)
        minimum_area = (
            _observed_minimum_area(points, self._plane_metadata)
            if linear
            else minimum_area_from_zscan(zscan_result, plane_results=points)
        )
        prediction_config = _read_field(self._recipe, "gaussian_prediction", {})
        if not isinstance(prediction_config, Mapping):
            prediction_config = {}
        highlight_diameters = _highlight_waist_diameters(
            divergence_summary,
            self._waist_refinement,
        )
        if linear:
            highlight_diameters = {
                f"{axis}_{field}": observed_minima[axis].get(key)
                for axis in ("x", "y")
                for field, key in (("um", "minimum_diameter_um"), ("source", "minimum_source"))
            }
        gaussian_prediction = predict_gaussian_divergence(
            waist_diameter_x_um=highlight_diameters["x_um"],
            waist_diameter_y_um=highlight_diameters["y_um"],
            wavelength_nm=_optional_number(prediction_config.get("wavelength_nm")),
            m2=_optional_number(prediction_config.get("m2")) or 1.0,
        )
        gaussian_prediction["waist_source_x"] = highlight_diameters["x_source"]
        gaussian_prediction["waist_source_y"] = highlight_diameters["y_source"]
        if linear:
            gaussian_prediction["model"] = "paraxial_gaussian_from_observed_minimum"
            for axis in ("x", "y"):
                gaussian_prediction[f"input_diameter_{axis}_um"] = observed_minima[axis][
                    "minimum_diameter_um"
                ]
                gaussian_prediction[f"waist_confirmed_{axis}"] = observed_minima[axis][
                    "waist_confirmed"
                ]
                gaussian_prediction[f"waist_status_{axis}"] = observed_minima[axis][
                    "minimum_status"
                ]
            gaussian_prediction["basis"] = "independent_observed_minima"
        measured_m2 = calculate_measured_m2(
            waist_diameter_x_um=divergence_summary.get("waist_diameter_x_um"),
            waist_diameter_y_um=divergence_summary.get("waist_diameter_y_um"),
            full_angle_x_mrad=divergence_summary.get("full_angle_x_mrad"),
            full_angle_y_mrad=divergence_summary.get("full_angle_y_mrad"),
            wavelength_nm=_optional_number(prediction_config.get("wavelength_nm")),
        )
        key_results = _key_results(
            divergence_summary=divergence_summary,
            waist_refinement=self._waist_refinement,
            minimum_area=minimum_area,
            gaussian_prediction=gaussian_prediction,
            plane_results=points,
        )
        if linear:
            for axis in ("x", "y"):
                key_results[axis].update(observed_minima[axis])
        return {
            "run_id": record.run_id,
            "run_dir": str(record.run_dir),
            "created_at_iso": record.created_at_iso,
            "finalized_at_iso": self._finalized_at_iso,
            "sample_id": record.sample_id,
            "recipe_id": recipe_id,
            "judgement": final_judgement,
            "measurement_status": (
                "incomplete" if incomplete else "complete" if zscan_result is not None
                else "in_progress"
            ),
            "scan_status": (
                "incomplete" if incomplete else "complete" if zscan_result is not None
                else "in_progress"
            ),
            "scan_error": _to_plain_data(scan_error),
            "acceptance_applied": diagnostics.get("acceptance_applied") is True and (
                not linear or _read_field(_read_field(self._recipe, "acceptance", {}),
                                         "enabled", False) is True
            ),
            "full_angle_x_mrad": divergence_summary.get("full_angle_x_mrad"),
            "full_angle_y_mrad": divergence_summary.get("full_angle_y_mrad"),
            "half_angle_x_mrad": divergence_summary.get("half_angle_x_mrad"),
            "half_angle_y_mrad": divergence_summary.get("half_angle_y_mrad"),
            "waist_z_x_mm": divergence_summary.get("waist_z_x_mm"),
            "waist_z_y_mm": divergence_summary.get("waist_z_y_mm"),
            "waist_diameter_x_um": divergence_summary.get("waist_diameter_x_um"),
            "waist_diameter_y_um": divergence_summary.get("waist_diameter_y_um"),
            "fit_r2_x": divergence_summary.get("fit_r2_x"),
            "fit_r2_y": divergence_summary.get("fit_r2_y"),
            "valid_points_count": (
                divergence_summary.get("valid_points_count") if zscan_result is not None
                else morphology_summary.get("valid_points_count")
            ),
            "predicted_full_angle_x_mrad": gaussian_prediction.get(
                "predicted_full_angle_x_mrad"
            ),
            "predicted_full_angle_y_mrad": gaussian_prediction.get(
                "predicted_full_angle_y_mrad"
            ),
            "predicted_half_angle_x_mrad": gaussian_prediction.get(
                "predicted_half_angle_x_mrad"
            ),
            "predicted_half_angle_y_mrad": gaussian_prediction.get(
                "predicted_half_angle_y_mrad"
            ),
            "fraunhofer_ideal_full_angle_x_mrad": gaussian_prediction.get(
                "fraunhofer_ideal_full_angle_x_mrad"
            ),
            "fraunhofer_ideal_full_angle_y_mrad": gaussian_prediction.get(
                "fraunhofer_ideal_full_angle_y_mrad"
            ),
            "sample_info": {"sample_id": record.sample_id},
            "recipe": {
                "recipe_id": recipe_id,
            },
            "calibration": {
                "calibration_id": _read_field(self._calibration, "calibration_id", None),
            },
            "system": _system_summary(self._system_snapshot),
            "final_judgement": final_judgement,
            "morphology_summary": morphology_summary,
            "divergence_result": divergence_summary,
            "gaussian_prediction": gaussian_prediction,
            "measured_beam_quality": measured_m2,
            "key_results": key_results,
            "minimum_area": minimum_area,
            "waist_refinement": dict(self._waist_refinement or {}),
            "fit_review": dict(self._fit_review or {}),
            "error_list": list(self._errors),
            "output_files": _output_files(record.run_dir),
        }

    def _require_run(self) -> RunRecord:
        if self.current_run is None:
            raise DataStoreError(
                ErrorCode.E_REPORT_GENERATION_FAILED,
                "create_run must be called before saving run artifacts.",
            )
        return self.current_run

    def _sync_zscan_csv_outliers(self, zscan_result: Any) -> None:
        record = self._require_run()
        csv_path = record.run_dir / "z_scan_table.csv"
        if not csv_path.exists():
            return
        with csv_path.open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        points = list(_read_field(zscan_result, "points", []) or [])
        row_indices = _match_result_points_to_plane_rows(points, self._plane_results)
        outlier_indices = set(_read_field(zscan_result, "outlier_indices", []) or [])
        fit_diagnostics = _read_field(zscan_result, "fit_diagnostics", {})
        diagnostic_points = {}
        if isinstance(fit_diagnostics, Mapping):
            for value in fit_diagnostics.get("points", []) or []:
                if isinstance(value, Mapping) and "index" in value:
                    diagnostic_points[int(value["index"])] = value
        linear = _read_field(fit_diagnostics, "model", None) == "far_field_linear"
        # A review can replace the previous selection, including a different point subset.
        for row in rows:
            for field in row:
                if field.startswith("fit_"):
                    row[field] = False if field.startswith("fit_included") else ""
            row["fit_included_x"] = False
            row["fit_included_y"] = False
        for point_index, row_index in enumerate(row_indices):
            if row_index is None or row_index >= len(rows):
                continue
            row = rows[row_index]
            point = points[point_index]
            is_outlier = (
                bool(_read_field(point, "outlier", False))
                or point_index in outlier_indices
            )
            reason = _read_field(point, "outlier_reason", None)
            if reason is None and point_index in outlier_indices:
                reason = (
                    _read_field(point, "invalid_reason", None)
                    if not bool(_read_field(point, "valid", False))
                    else "robust_fit_rejected"
                )
            row["outlier"] = is_outlier
            row["outlier_reason"] = "" if reason is None else str(reason)
            diagnostic = diagnostic_points.get(point_index, {})
            row["fit_included"] = bool(diagnostic.get("fit_included", False))
            for axis in ("x", "y"):
                axis_fit = _read_field(fit_diagnostics, axis, {})
                default = (
                    point_index in _read_field(axis_fit, "fit_indices", [])
                    if linear else row["fit_included"]
                )
                row[f"fit_included_{axis}"] = bool(
                    diagnostic.get(f"fit_included_{axis}", default)
                )
            if linear:
                row["fit_included"] = row["fit_included_x"] or row["fit_included_y"]
            row["fit_predicted_x_um"] = _blank_if_none(
                diagnostic.get("predicted_x_um")
            )
            row["fit_predicted_y_um"] = _blank_if_none(
                diagnostic.get("predicted_y_um")
            )
            row["fit_residual_x_um"] = _blank_if_none(diagnostic.get("residual_x_um"))
            row["fit_residual_y_um"] = _blank_if_none(diagnostic.get("residual_y_um"))
            row["fit_normalized_residual_x"] = _blank_if_none(
                diagnostic.get("normalized_residual_x")
            )
            row["fit_normalized_residual_y"] = _blank_if_none(
                diagnostic.get("normalized_residual_y")
            )
        write_zscan_csv(csv_path, rows)


def _now() -> datetime:
    return datetime.now().astimezone()


def _unique_run_dir(root_output_dir: Path, base_name: str) -> Path:
    candidate = root_output_dir / base_name
    suffix = 1
    while candidate.exists():
        candidate = root_output_dir / f"{base_name}_{suffix:03d}"
        suffix += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def _safe_filename_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return token or "sample"


def _to_plain_data(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _to_plain_data(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _to_plain_data(item) for key, item in value.items()}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, np.generic):
        return _to_plain_data(value.item())
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_to_plain_data(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _read_field(source: Any, field_name: str, default: Any) -> Any:
    if source is None:
        return default
    if isinstance(source, Mapping):
        return source.get(field_name, default)
    return getattr(source, field_name, default)


def _match_result_points_to_plane_rows(
    result_points: Sequence[Any],
    plane_results: Sequence[Any],
) -> list[int | None]:
    """Map fit-local points to persisted rows, preferring the latest duplicate."""

    available = set(range(len(plane_results)))
    matched: list[int | None] = []
    for result_point in result_points:
        identity_match = next(
            (
                index
                for index in reversed(range(len(plane_results)))
                if index in available and plane_results[index] is result_point
            ),
            None,
        )
        match = identity_match
        if match is None:
            match = next(
                (
                    index
                    for index in reversed(range(len(plane_results)))
                    if index in available
                    and _same_plane_measurement(result_point, plane_results[index])
                ),
                None,
            )
        matched.append(match)
        if match is not None:
            available.remove(match)
    return matched


def _same_plane_measurement(left: Any, right: Any) -> bool:
    for field_name in (
        "z_actual_mm",
        "centroid_x_um",
        "centroid_y_um",
        "d4sigma_x_um",
        "d4sigma_y_um",
    ):
        left_value = _optional_number(_read_field(left, field_name, None))
        right_value = _optional_number(_read_field(right, field_name, None))
        if left_value is None or right_value is None:
            if left_value != right_value:
                return False
        elif not math.isclose(left_value, right_value, rel_tol=1e-12, abs_tol=1e-12):
            return False
    return True


def _optional_number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _summary_points(plane_results: Sequence[Any], zscan_result: Any) -> list[Any]:
    points = list(plane_results)
    fit_points = list(_read_field(zscan_result, "points", []) or [])
    matches = _match_result_points_to_plane_rows(fit_points, points)
    points.extend(point for point, match in zip(fit_points, matches, strict=True) if match is None)
    return points


def _observed_minima(
    points: Sequence[Any],
    metadata: Sequence[Mapping[str, Any]],
    waist_refinement: Mapping[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    refinement = waist_refinement or {}
    result = {}
    for axis in ("x", "y"):
        candidates = []
        for index, point in enumerate(points):
            diameter = _optional_number(_read_field(point, f"d4sigma_{axis}_um", None))
            z_mm = _optional_number(_read_field(point, "z_actual_mm", None))
            if (
                bool(_read_field(point, "valid", False)) and diameter is not None
                and diameter > 0 and z_mm is not None
            ):
                candidates.append((diameter, index, z_mm))
        minimum = min(candidates) if candidates else (None, None, None)
        diameter, index, z_mm = minimum
        record = metadata[index] if index is not None and index < len(metadata) else {}
        claimed = _read_field(refinement, f"{axis}_minimum", {}) or {}
        confirmed_flag = _read_field(claimed, "confirmed", False) is True
        claimed_z = _optional_number(_read_field(claimed, "z_actual_mm", None))
        claimed_diameter = _optional_number(_read_field(claimed, "diameter_um", None))
        claimed_index = _optional_int(_read_field(claimed, "plane_index_count", None))
        confirmed = (
            confirmed_flag and diameter is not None and z_mm is not None
            and claimed_diameter is not None and claimed_z is not None
            and math.isclose(diameter, claimed_diameter, rel_tol=1e-9, abs_tol=1e-9)
            and math.isclose(z_mm, claimed_z, rel_tol=0.0, abs_tol=1e-9)
            and (claimed_index is None or claimed_index == index)
        )
        z_values = [value[2] for value in candidates]
        at_boundary = z_mm in (min(z_values), max(z_values)) if z_values else None
        confirmed = confirmed and not at_boundary
        result[axis] = {
            "minimum_diameter_um": diameter,
            "minimum_z_mm": z_mm,
            "minimum_source": "all_captured_observed_minimum",
            "image_plane_index_count": index,
            "minimum_z_position_trusted_for_fit": bool(
                record.get("position_trusted_for_fit", True)
            ),
            "minimum_scan_phase": record.get("scan_phase"),
            "waist_confirmed": confirmed,
            "minimum_status": (
                "confirmed" if confirmed else "candidate_estimate" if diameter is not None
                else "unavailable"
            ),
            "minimum_at_boundary": at_boundary,
        }
    return result


def _observed_minimum_area(
    points: Sequence[Any], metadata: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    candidates = []
    for index, point in enumerate(points):
        x_um = _optional_number(_read_field(point, "d4sigma_x_um", None))
        y_um = _optional_number(_read_field(point, "d4sigma_y_um", None))
        z_mm = _optional_number(_read_field(point, "z_actual_mm", None))
        if (
            bool(_read_field(point, "valid", False)) and x_um is not None and y_um is not None
            and z_mm is not None and x_um > 0 and y_um > 0
        ):
            candidates.append((math.pi * x_um * y_um / 4.0, index, z_mm, x_um, y_um))
    if not candidates:
        return {"valid": False, "source": "all_captured_observed_minimum",
                "invalid_reason": "no_valid_observed_area"}
    area, index, z_mm, x_um, y_um = min(candidates)
    record = metadata[index] if index < len(metadata) else {}
    return {
        "valid": True, "source": "all_captured_observed_minimum",
        "z_mm": z_mm, "diameter_x_um": x_um, "diameter_y_um": y_um,
        "area_um2": area, "nearest_plane_index_count": index,
        "position_trusted_for_fit": bool(record.get("position_trusted_for_fit", True)),
    }


def _morphology_summary(points: Sequence[Any]) -> dict[str, Any]:
    valid_points = [point for point in points if bool(_read_field(point, "valid", False))]
    if not valid_points:
        return {
            "valid_points_count": 0,
            "invalid_points_count": len(points),
        }
    return {
        "valid_points_count": len(valid_points),
        "invalid_points_count": len(points) - len(valid_points),
        "centroid_x_mean_um": _mean(valid_points, "centroid_x_um"),
        "centroid_y_mean_um": _mean(valid_points, "centroid_y_um"),
        "d4sigma_x_mean_um": _mean(valid_points, "d4sigma_x_um"),
        "d4sigma_y_mean_um": _mean(valid_points, "d4sigma_y_um"),
        "d4sigma_major_max_um": _max(valid_points, "d4sigma_major_um"),
        "d4sigma_minor_min_um": _min(valid_points, "d4sigma_minor_um"),
        "ellipticity_mean": _mean(valid_points, "ellipticity"),
    }


def _divergence_summary(zscan_result: Any | None) -> dict[str, Any]:
    if zscan_result is None:
        return {}
    result = {
        "full_angle_x_mrad": _read_field(zscan_result, "full_angle_x_mrad", None),
        "full_angle_y_mrad": _read_field(zscan_result, "full_angle_y_mrad", None),
        "half_angle_x_mrad": _read_field(zscan_result, "half_angle_x_mrad", None),
        "half_angle_y_mrad": _read_field(zscan_result, "half_angle_y_mrad", None),
        "waist_z_x_mm": _read_field(zscan_result, "waist_z_x_mm", None),
        "waist_z_y_mm": _read_field(zscan_result, "waist_z_y_mm", None),
        "waist_diameter_x_um": _read_field(zscan_result, "waist_diameter_x_um", None),
        "waist_diameter_y_um": _read_field(zscan_result, "waist_diameter_y_um", None),
        "fit_r2_x": _read_field(zscan_result, "fit_r2_x", None),
        "fit_r2_y": _read_field(zscan_result, "fit_r2_y", None),
        "valid_points_count": _read_field(zscan_result, "valid_points_count", None),
        "invalid_reason": _read_field(zscan_result, "invalid_reason", None),
        "fit_diagnostics": _to_plain_data(
            _read_field(zscan_result, "fit_diagnostics", {})
        ),
    }
    diagnostics = result["fit_diagnostics"] or {}
    result["model"] = diagnostics.get("model")
    for axis in ("x", "y"):
        axis_fit = diagnostics.get(axis, {})
        for field in (
            "slope_um_per_mm", "intercept_um", "z_reference_mm", "intercept_at_reference_um",
            "residual_std_um", "fit_rmse_um", "valid_points_count", "status", "reasons",
        ):
            result[f"{axis}_{field}"] = axis_fit.get(field)
        if result["model"] == "far_field_linear":
            result[f"waist_z_{axis}_mm"] = None
            result[f"waist_diameter_{axis}_um"] = None
    return result


def _highlight_waist_diameters(
    divergence_summary: Mapping[str, Any],
    waist_refinement: Mapping[str, Any] | None,
) -> dict[str, Any]:
    refinement = waist_refinement if isinstance(waist_refinement, Mapping) else {}
    x_minimum = refinement.get("x_minimum")
    y_minimum = refinement.get("y_minimum")
    x_mapping = x_minimum if isinstance(x_minimum, Mapping) else {}
    y_mapping = y_minimum if isinstance(y_minimum, Mapping) else {}
    refined = bool(refinement.get("valid"))
    x_refined_um = _optional_number(x_mapping.get("diameter_um")) if refined else None
    y_refined_um = _optional_number(y_mapping.get("diameter_um")) if refined else None
    return {
        "x_um": x_refined_um or divergence_summary.get("waist_diameter_x_um"),
        "y_um": y_refined_um or divergence_summary.get("waist_diameter_y_um"),
        "x_source": (
            "waist_refinement_observed_minimum"
            if x_refined_um is not None
            else "main_zscan_fitted_waist"
        ),
        "y_source": (
            "waist_refinement_observed_minimum"
            if y_refined_um is not None
            else "main_zscan_fitted_waist"
        ),
    }


def _key_results(
    *,
    divergence_summary: Mapping[str, Any],
    waist_refinement: Mapping[str, Any] | None,
    minimum_area: Mapping[str, Any],
    gaussian_prediction: Mapping[str, Any],
    plane_results: Sequence[Any],
) -> dict[str, Any]:
    refinement = waist_refinement if isinstance(waist_refinement, Mapping) else {}
    return {
        "x": _axis_key_result(
            axis_name="x",
            divergence_summary=divergence_summary,
            refinement=refinement,
            gaussian_prediction=gaussian_prediction,
            plane_results=plane_results,
        ),
        "y": _axis_key_result(
            axis_name="y",
            divergence_summary=divergence_summary,
            refinement=refinement,
            gaussian_prediction=gaussian_prediction,
            plane_results=plane_results,
        ),
        "minimum_area": dict(minimum_area),
    }


def _axis_key_result(
    *,
    axis_name: str,
    divergence_summary: Mapping[str, Any],
    refinement: Mapping[str, Any],
    gaussian_prediction: Mapping[str, Any],
    plane_results: Sequence[Any],
) -> dict[str, Any]:
    refinement_value = refinement.get(f"{axis_name}_minimum")
    refined = refinement_value if isinstance(refinement_value, Mapping) else {}
    refined_diameter_um = (
        _optional_number(refined.get("diameter_um"))
        if bool(refinement.get("valid"))
        else None
    )
    fitted_z_mm = _optional_number(divergence_summary.get(f"waist_z_{axis_name}_mm"))
    fitted_diameter_um = _optional_number(
        divergence_summary.get(f"waist_diameter_{axis_name}_um")
    )
    minimum_z_mm = _optional_number(refined.get("z_actual_mm"))
    plane_index = _optional_int(refined.get("plane_index_count"))
    if refined_diameter_um is None:
        minimum_z_mm = fitted_z_mm
        plane_index = _nearest_plane_index(plane_results, fitted_z_mm)
    return {
        "axis": axis_name.upper(),
        "minimum_diameter_um": refined_diameter_um or fitted_diameter_um,
        "minimum_z_mm": minimum_z_mm,
        "minimum_source": (
            "waist_refinement_observed_minimum"
            if refined_diameter_um is not None
            else "main_zscan_fitted_waist"
        ),
        "minimum_z_position_trusted_for_fit": refined_diameter_um is None,
        "image_plane_index_count": plane_index,
        "measured_full_angle_mrad": divergence_summary.get(
            f"full_angle_{axis_name}_mrad"
        ),
        "measured_half_angle_mrad": divergence_summary.get(
            f"half_angle_{axis_name}_mrad"
        ),
        "fraunhofer_ideal_full_angle_mrad": gaussian_prediction.get(
            f"fraunhofer_ideal_full_angle_{axis_name}_mrad"
        ),
        "fraunhofer_ideal_half_angle_mrad": gaussian_prediction.get(
            f"fraunhofer_ideal_half_angle_{axis_name}_mrad"
        ),
        "fit_r2": divergence_summary.get(f"fit_r2_{axis_name}"),
        "residual_std_um": divergence_summary.get(f"{axis_name}_residual_std_um"),
        "fit_rmse_um": divergence_summary.get(f"{axis_name}_fit_rmse_um"),
        "valid_points_count": divergence_summary.get(f"{axis_name}_valid_points_count"),
        "fit_status": divergence_summary.get(f"{axis_name}_status"),
        "fit_reasons": divergence_summary.get(f"{axis_name}_reasons"),
    }


def _nearest_plane_index(points: Sequence[Any], z_mm: float | None) -> int | None:
    if z_mm is None:
        return None
    candidates = []
    for index, point in enumerate(points):
        point_z_mm = _optional_number(_read_field(point, "z_actual_mm", None))
        if point_z_mm is not None:
            candidates.append((abs(point_z_mm - z_mm), index))
    return None if not candidates else min(candidates)[1]


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _final_judgement(
    zscan_result: Any | None, errors: Sequence[Mapping[str, Any]], recipe: Any = None,
) -> str:
    diagnostics = _read_field(zscan_result, "fit_diagnostics", {}) or {}
    if _read_field(diagnostics, "model", None) == "far_field_linear":
        acceptance = _read_field(recipe, "acceptance", {})
        if (
            _read_field(diagnostics, "acceptance_applied", False) is not True or errors
            or _read_field(acceptance, "enabled", False) is not True
        ):
            return "UNKNOWN"
        judgement = str(_read_field(zscan_result, "judgement", "unknown")).lower()
        return judgement.upper() if judgement in {"pass", "fail"} else "UNKNOWN"
    if errors:
        return "FAIL"
    judgement = _read_field(zscan_result, "judgement", None)
    if judgement is None:
        return "UNKNOWN"
    if str(judgement).lower() == "pass":
        return "PASS"
    if str(judgement).lower() in {"fail", "invalid"}:
        return "FAIL"
    if str(judgement).lower() == "warning":
        return "WARNING"
    return "UNKNOWN"


def _system_summary(system_snapshot: Any) -> dict[str, Any]:
    if system_snapshot is None:
        return {}
    data = _to_plain_data(system_snapshot)
    if isinstance(data, Mapping):
        return dict(data)
    return {"value": data}


def _output_files(run_dir: Path) -> dict[str, Any]:
    def relative(path: Path) -> str:
        return path.relative_to(run_dir).as_posix()

    files: dict[str, Any] = {}
    for key, filename in (
        ("recipe_snapshot", "recipe_snapshot.yaml"),
        ("calibration_snapshot", "calibration_snapshot.yaml"),
        ("system_snapshot", "system_snapshot.yaml"),
        ("result_summary", "result_summary.json"),
        ("result_summary_csv", "result_summary.csv"),
        ("z_scan_table", "z_scan_table.csv"),
        ("report_html", "report.html"),
        ("report_pdf", "report.pdf"),
    ):
        path = run_dir / filename
        if path.exists():
            files[key] = relative(path)
    images_dir = run_dir / "images"
    processed_dir = run_dir / "processed"
    logs_dir = run_dir / "logs"
    report_assets_dir = run_dir / "report_assets"
    for key, directory in (
        ("images", images_dir),
        ("processed", processed_dir),
        ("report_assets", report_assets_dir),
        ("logs", logs_dir),
    ):
        paths = [relative(path) for path in sorted(directory.glob("*")) if path.is_file()]
        if paths:
            files[key] = paths
    return files


def _flatten_summary_rows(
    value: Any,
    *,
    prefix: str = "",
) -> Sequence[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(_flatten_summary_rows(item, prefix=path))
        return rows
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        for index, item in enumerate(value):
            rows.extend(_flatten_summary_rows(item, prefix=f"{prefix}[{index}]"))
        return rows
    rows.append(
        {
            "metric": prefix,
            "value": "" if value is None else value,
            "value_type": "null" if value is None else type(value).__name__,
            "unit": _metric_unit(prefix),
        }
    )
    return rows


def _metric_unit(metric: str) -> str:
    normalized = metric.lower()
    for suffix, unit in (
        ("_um_per_mm", "um/mm"),
        ("_um2_per_mm2", "um^2/mm^2"),
        ("_um2_per_mm", "um^2/mm"),
        ("_um2", "um^2"),
        ("_mrad", "mrad"),
        ("_deg", "deg"),
        ("_mm", "mm"),
        ("_um", "um"),
        ("_nm", "nm"),
        ("_us", "us"),
        ("_percent", "percent"),
    ):
        if normalized.endswith(suffix):
            return unit
    return ""


def _blank_if_none(value: Any) -> Any:
    return "" if value is None else value


def _remove_empty_output_directories(run_dir: Path) -> None:
    for dirname in ("images", "processed", "logs"):
        path = run_dir / dirname
        if path.is_dir() and not any(path.iterdir()):
            path.rmdir()


def _mean(points: Sequence[Any], field_name: str) -> float | None:
    values = _number_values(points, field_name)
    if not values:
        return None
    return float(sum(values) / len(values))


def _min(points: Sequence[Any], field_name: str) -> float | None:
    values = _number_values(points, field_name)
    return None if not values else float(min(values))


def _max(points: Sequence[Any], field_name: str) -> float | None:
    values = _number_values(points, field_name)
    return None if not values else float(max(values))


def _number_values(points: Sequence[Any], field_name: str) -> list[float]:
    values = []
    for point in points:
        value = _read_field(point, field_name, None)
        if value is None:
            continue
        number = float(value)
        if math.isfinite(number):
            values.append(number)
    return values


def _read_json(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


__all__ = ["RunRecord", "RunStore"]
