from __future__ import annotations

import csv
import json
from dataclasses import replace
from datetime import UTC, datetime

import numpy as np
import pytest
from lbqa_contracts.errors import ErrorCode
from lbqa_contracts.models import BeamPlaneResult, ZScanResult
from lbqa_data import Z_SCAN_TABLE_FIELDS, RunStore


def fixed_timestamp() -> datetime:
    return datetime(2026, 6, 19, 10, 11, 12, tzinfo=UTC)


def make_plane(
    z_actual_mm: float = 0.0,
    *,
    valid: bool = True,
    invalid_reason: str | None = None,
) -> BeamPlaneResult:
    return BeamPlaneResult(
        z_actual_mm=z_actual_mm,
        centroid_x_um=1.0,
        centroid_y_um=2.0,
        peak_x_um=1.0,
        peak_y_um=2.0,
        d4sigma_x_um=8.0 + abs(z_actual_mm),
        d4sigma_y_um=10.0 + abs(z_actual_mm),
        d4sigma_major_um=10.0 + abs(z_actual_mm),
        d4sigma_minor_um=8.0 + abs(z_actual_mm),
        fwhm_x_um=4.0,
        fwhm_y_um=5.0,
        ellipticity=1.25,
        azimuth_deg=3.0,
        peak_value=3000.0,
        saturation_pixels=0,
        edge_energy_percent=0.5,
        valid=valid,
        invalid_reason=invalid_reason,
    )


def make_zscan_result(points: list[BeamPlaneResult]) -> ZScanResult:
    return ZScanResult(
        points=points,
        full_angle_x_mrad=3.0,
        full_angle_y_mrad=4.0,
        half_angle_x_mrad=1.5,
        half_angle_y_mrad=2.0,
        waist_z_x_mm=0.0,
        waist_z_y_mm=0.1,
        waist_diameter_x_um=8.0,
        waist_diameter_y_um=10.0,
        fit_r2_x=0.99,
        fit_r2_y=0.98,
        valid_points_count=sum(1 for point in points if point.valid),
        judgement="pass",
        invalid_reason=None,
    )


def test_create_run_generates_expected_directory_layout(tmp_path) -> None:
    store = RunStore(tmp_path, timestamp_factory=fixed_timestamp)

    record = store.create_run("SAMPLE-001", {"recipe_id": "recipe-40x"})

    assert record.run_id == "Run_20260619_101112_SAMPLE-001"
    assert not (record.run_dir / "images").exists()
    assert not (record.run_dir / "processed").exists()
    assert not (record.run_dir / "logs").exists()
    assert (record.run_dir / "recipe_snapshot.yaml").read_text(encoding="utf-8")


def test_save_plane_result_appends_canonical_csv_row(tmp_path) -> None:
    store = RunStore(tmp_path, timestamp_factory=fixed_timestamp)
    record = store.create_run("sample-a", {"recipe_id": "recipe-40x"})

    csv_path = store.save_plane_result(
        plane_result=make_plane(0.123),
        z_cmd_mm=0.12,
        exposure_us=2000.0,
        frame_count=3,
        phase="custom",
    )

    rows = list(csv.DictReader(csv_path.open(newline="", encoding="utf-8")))
    assert rows[0].keys() == set(Z_SCAN_TABLE_FIELDS)
    assert rows[0]["run_id"] == record.run_id
    assert rows[0]["sample_id"] == "sample-a"
    assert rows[0]["z_cmd_mm"] == "0.12"
    assert rows[0]["z_actual_mm"] == "0.123"
    assert rows[0]["scan_phase"] == "custom"
    assert rows[0]["position_trusted_for_fit"] == "True"
    assert rows[0]["waist_refinement_included"] == "False"
    assert float(rows[0]["beam_area_um2"]) > 0.0
    assert rows[0]["exposure_us"] == "2000.0"
    assert rows[0]["frame_count"] == "3"
    assert rows[0]["valid"] == "True"
    assert rows[0]["outlier"] == "False"
    assert rows[0]["outlier_reason"] == ""


def test_result_summary_json_is_readable_after_zscan_result(tmp_path) -> None:
    store = RunStore(tmp_path, timestamp_factory=fixed_timestamp)
    record = store.create_run("sample-a", {"recipe_id": "recipe-40x"})
    store.save_calibration_snapshot({"calibration_id": "cal-40x"})
    store.save_system_snapshot({"system_name": "laser-beam-qa", "mode": "simulated"})
    points = [make_plane(-0.1), make_plane(0.0), make_plane(0.1)]
    for point in points:
        store.save_plane_result(plane_result=point, z_cmd_mm=point.z_actual_mm)

    store.save_zscan_result(zscan_result=make_zscan_result(points))
    summary = store.finalize_run(render_html=False)

    summary_path = record.run_dir / "result_summary.json"
    loaded = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary == loaded
    assert loaded["sample_info"]["sample_id"] == "sample-a"
    assert loaded["recipe"]["recipe_id"] == "recipe-40x"
    assert loaded["calibration"]["calibration_id"] == "cal-40x"
    assert loaded["final_judgement"] == "PASS"
    assert loaded["sample_id"] == "sample-a"
    assert loaded["recipe_id"] == "recipe-40x"
    assert loaded["judgement"] == "PASS"
    assert loaded["full_angle_x_mrad"] == 3.0
    assert loaded["full_angle_y_mrad"] == 4.0
    assert loaded["half_angle_x_mrad"] == 1.5
    assert loaded["half_angle_y_mrad"] == 2.0
    assert loaded["fit_r2_x"] == 0.99
    assert loaded["fit_r2_y"] == 0.98
    assert loaded["valid_points_count"] == 3
    assert loaded["divergence_result"]["full_angle_x_mrad"] == 3.0
    assert loaded["morphology_summary"]["valid_points_count"] == 3


def test_result_summary_includes_separate_gaussian_waist_prediction(tmp_path) -> None:
    store = RunStore(tmp_path, timestamp_factory=fixed_timestamp)
    store.create_run(
        "sample-a",
        {
            "recipe_id": "recipe-40x",
            "gaussian_prediction": {"wavelength_nm": 1064.0, "m2": 1.2},
        },
    )
    points = [make_plane(-0.1), make_plane(0.0), make_plane(0.1)]
    store.save_zscan_result(zscan_result=make_zscan_result(points))

    summary = store.finalize_run(render_html=False)

    prediction = summary["gaussian_prediction"]
    assert prediction["valid"] is True
    assert prediction["wavelength_nm"] == 1064.0
    assert prediction["m2"] == 1.2
    assert prediction["predicted_full_angle_x_mrad"] == pytest.approx(
        4.0 * 1.2 * 1064.0 / (np.pi * 8.0)
    )
    assert summary["predicted_full_angle_x_mrad"] == prediction[
        "predicted_full_angle_x_mrad"
    ]
    assert summary["fraunhofer_ideal_full_angle_x_mrad"] == prediction[
        "fraunhofer_ideal_full_angle_x_mrad"
    ]


def test_report_html_can_be_generated(tmp_path) -> None:
    store = RunStore(tmp_path, timestamp_factory=fixed_timestamp)
    record = store.create_run("sample-a", {"recipe_id": "recipe-40x"})
    assert store.image_store is not None
    points = [make_plane(-0.1), make_plane(0.0), make_plane(0.1)]
    for index, point in enumerate(points):
        image = np.eye(16, dtype=np.float32) * float(index + 1)
        store.image_store.save_report_source_npy(
            image,
            frame_index_count=index,
        )
        store.save_plane_result(plane_result=point, z_cmd_mm=point.z_actual_mm)
    store.save_zscan_result(zscan_result=make_zscan_result(points))

    store.finalize_run(render_html=True)

    html = (record.run_dir / "report.html").read_text(encoding="utf-8")
    assert "激光模斑扫描报告" in html
    assert "实测全角" in html
    assert "夫琅禾费预测全角" in html
    assert "附录：全部扫描点模斑图" in html
    assert (record.run_dir / "report_assets" / "divergence_fit.png").exists()
    assert (record.run_dir / "report_assets" / "beam_propagation.png").exists()
    assert len(list((record.run_dir / "report_assets").glob("spot_*.png"))) == 3
    assert not list((record.run_dir / "report_assets").glob("surface_*.png"))
    assert (record.run_dir / "result_summary.csv").stat().st_size > 0
    assert not (record.run_dir / ".working").exists()
    assert not list(record.run_dir.rglob("*.npy"))
    assert not list(record.run_dir.rglob("*.tif"))
    assert not list(record.run_dir.rglob("*.tiff"))


def test_fit_review_and_outlier_selection_are_persisted(tmp_path) -> None:
    store = RunStore(tmp_path, timestamp_factory=fixed_timestamp)
    record = store.create_run("sample-review", {"recipe_id": "recipe-40x"})
    points = [make_plane(-0.1), make_plane(0.0), make_plane(0.1)]
    for point in points:
        store.save_plane_result(plane_result=point, z_cmd_mm=point.z_actual_mm)

    reviewed_points = [
        points[0],
        points[1],
        replace(
            points[2],
            outlier=True,
            outlier_reason="operator_excluded_from_fit",
        ),
    ]
    reviewed_result = make_zscan_result(reviewed_points)
    reviewed_result = replace(reviewed_result, outlier_indices=[2])
    store.save_zscan_result(zscan_result=reviewed_result)
    store.save_fit_review(
        {
            "outcome": "operator_confirmed",
            "selected_indices": [0, 1],
            "operator_excluded_indices": [2],
        }
    )

    rows = list(
        csv.DictReader((record.run_dir / "z_scan_table.csv").open(encoding="utf-8"))
    )
    summary = json.loads((record.run_dir / "result_summary.json").read_text("utf-8"))
    assert rows[2]["outlier"] == "True"
    assert rows[2]["outlier_reason"] == "operator_excluded_from_fit"
    assert summary["fit_review"]["selected_indices"] == [0, 1]
    assert not (record.run_dir / "fit_review.json").exists()


def test_fit_columns_align_to_final_scan_after_prescan_rows(tmp_path) -> None:
    store = RunStore(tmp_path, timestamp_factory=fixed_timestamp)
    record = store.create_run("sample-phases", {"recipe_id": "recipe-40x"})
    prescan_points = [make_plane(-0.05), make_plane(0.0), make_plane(0.05)]
    final_points = [
        make_plane(-0.3),
        make_plane(-0.15),
        make_plane(0.0),
        make_plane(0.15),
        make_plane(0.3),
    ]
    for point in [*prescan_points, *final_points]:
        store.save_plane_result(plane_result=point, z_cmd_mm=point.z_actual_mm)
    diagnostics = {
        "points": [
            {"index": index, "fit_included": True}
            for index in range(len(final_points))
        ]
    }
    result = replace(
        make_zscan_result(final_points),
        valid_points_count=len(final_points),
        fit_diagnostics=diagnostics,
    )

    store.save_zscan_result(zscan_result=result)

    rows = list(
        csv.DictReader((record.run_dir / "z_scan_table.csv").open(encoding="utf-8"))
    )
    assert [row["fit_included"] for row in rows[:3]] == ["False"] * 3
    assert [row["fit_included"] for row in rows[3:]] == ["True"] * 5


def test_discard_run_removes_the_entire_temporary_run(tmp_path) -> None:
    store = RunStore(tmp_path, timestamp_factory=fixed_timestamp)
    record = store.create_run("sample-cancelled", {"recipe_id": "recipe-40x"})
    assert store.image_store is not None
    store.image_store.save_report_source_npy(np.eye(8), frame_index_count=0)

    store.discard_run()

    assert not record.run_dir.exists()
    assert not list(tmp_path.glob("Run_*"))


def test_failure_run_saves_error_summary_and_report(tmp_path) -> None:
    store = RunStore(tmp_path, timestamp_factory=fixed_timestamp)
    record = store.create_run("sample-a", {"recipe_id": "recipe-40x"})

    store.save_failure(error_code=ErrorCode.E_USER_ABORT, message="operator aborted")
    summary = store.finalize_run(render_html=True)

    assert summary["final_judgement"] == "FAIL"
    assert summary["error_list"][0]["error_code"] == ErrorCode.E_USER_ABORT.value
    assert (record.run_dir / "logs" / "error.log").read_text(encoding="utf-8")
    assert (record.run_dir / "report.html").exists()
