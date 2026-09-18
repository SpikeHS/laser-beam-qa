from __future__ import annotations

import csv
import json
import math
from dataclasses import replace

import numpy as np
import pytest
from lbqa_analysis.far_field_fit import fit_far_field
from lbqa_contracts.models import BeamPlaneResult
from lbqa_data import RunStore
from lbqa_data import report_generator as report
from lbqa_data.beam_visualization import BeamDisplayOptions
from matplotlib.figure import Figure


def plane(z: float, x: float, y: float, **changes) -> BeamPlaneResult:
    return replace(BeamPlaneResult(
        z_actual_mm=z, centroid_x_um=32.0, centroid_y_um=32.0,
        peak_x_um=32.0, peak_y_um=32.0, d4sigma_x_um=x, d4sigma_y_um=y,
        d4sigma_major_um=max(x, y), d4sigma_minor_um=min(x, y),
        fwhm_x_um=x / 2, fwhm_y_um=y / 2, ellipticity=min(x, y) / max(x, y),
        azimuth_deg=12.0, peak_value=1.0, saturation_pixels=0,
        edge_energy_percent=0.0, valid=True, invalid_reason=None,
        width_source="gaussian_fit_continuous_axes",
    ), **changes)


@pytest.fixture
def points():
    return [plane(float(z), 10.0 + 2 * z, 20.0 + 3 * z) for z in range(5)]


def test_preview_and_final_share_observed_minima_and_leave_no_state_changes(tmp_path, points):
    store = RunStore(tmp_path)
    run = store.create_run("linear", {
        "recipe_id": "operator", "gaussian_prediction": {"wavelength_nm": 1064.0},
    })
    fine = [plane(0.5, 7.0, 18.0), plane(0.6, 9.0, 6.0)]
    invalid = plane(0.7, 1.0, 1.0, valid=False, invalid_reason="synthetic_invalid")
    for index, point in enumerate([*points, *fine, invalid]):
        store.save_plane_result(
            plane_result=point, phase="fine" if index >= len(points) else "main",
            position_trusted_for_fit=index < len(points),
            waist_refinement_included=index >= len(points),
        )
    store.save_waist_refinement({
        "valid": True,
        "x_minimum": {"diameter_um": 7.0, "z_actual_mm": 0.5, "plane_index_count": 5,
                      "confirmed": True, "confirmation_status": "trend_confirmed"},
        "y_minimum": {"diameter_um": 8.0, "z_actual_mm": 0.6, "plane_index_count": 6,
                      "confirmed": True},
        "termination_reason": "max_steps", "actual_move_count": 30, "max_steps": 30,
        "start_z_mm": 0.0,
    })
    result = fit_far_field(points, selected_indices_x=[1, 2, 3], selected_indices_y=[2, 3, 4])
    before = {path: path.read_bytes() for path in run.run_dir.rglob("*") if path.is_file()}
    preview = store.preview_summary(result)
    assert store._zscan_result is None
    assert len(store._plane_results) == 8
    assert before == {path: path.read_bytes() for path in run.run_dir.rglob("*") if path.is_file()}
    key = preview["key_results"]
    assert key["x"]["minimum_diameter_um"] == 7.0
    assert key["x"]["minimum_z_mm"] == 0.5
    assert key["x"]["waist_confirmed"] is True
    assert key["y"]["minimum_diameter_um"] == 6.0
    assert key["y"]["minimum_z_mm"] == 0.6
    assert key["y"]["waist_confirmed"] is False
    assert key["y"]["minimum_status"] == "candidate_estimate"
    assert key["x"]["minimum_z_position_trusted_for_fit"] is False
    assert key["minimum_area"]["diameter_x_um"] == 9.0
    assert key["minimum_area"]["diameter_y_um"] == 6.0
    assert key["minimum_area"]["area_um2"] == pytest.approx(math.pi * 9 * 6 / 4)
    assert key["minimum_area"]["nearest_plane_index_count"] == 6
    prediction = preview["gaussian_prediction"]
    assert prediction["input_diameter_x_um"] == 7.0
    assert prediction["input_diameter_y_um"] == 6.0
    assert prediction["model"] == "paraxial_gaussian_from_observed_minimum"
    assert prediction["predicted_full_angle_x_mrad"] == pytest.approx(4 * 1064 / (math.pi * 7))
    assert prediction["predicted_full_angle_y_mrad"] == pytest.approx(4 * 1064 / (math.pi * 6))
    assert prediction["waist_confirmed_x"] is True
    assert prediction["waist_confirmed_y"] is False
    assert preview["waist_diameter_x_um"] is None
    assert preview["waist_z_y_mm"] is None
    store.save_zscan_result(zscan_result=result)
    final = store.finalize_run(render_html=False)
    for name in ("key_results", "gaussian_prediction", "minimum_area", "divergence_result"):
        assert final[name] == preview[name] == store.preview_summary()[name]
    assert final["final_judgement"] == "UNKNOWN"
    assert final["measurement_status"] == "complete"
    with (run.run_dir / "result_summary.csv").open(encoding="utf-8", newline="") as file:
        rows = {row["metric"]: row for row in csv.DictReader(file)}
    assert rows["key_results.x.minimum_diameter_um"]["value"] == "7.0"
    assert rows["divergence_result.fit_diagnostics.x.slope_um_per_mm"]["unit"] == "um/mm"
    assert rows["key_results.x.residual_std_um"]["unit"] == "um"


def test_preview_merges_missing_result_points_without_persisting(tmp_path, points):
    store = RunStore(tmp_path)
    run = store.create_run("linear", {"recipe_id": "operator"})
    store.save_plane_result(plane_result=points[0])
    result = fit_far_field(points)
    preview = store.preview_summary(result)
    assert preview["morphology_summary"]["valid_points_count"] == 5
    assert len(store._plane_results) == 1
    store.save_zscan_result(zscan_result=result)
    assert store.preview_summary()["key_results"] == preview["key_results"]
    with (run.run_dir / "z_scan_table.csv").open(encoding="utf-8", newline="") as file:
        assert len(list(csv.DictReader(file))) == 5


def test_boundary_and_stale_refinement_are_not_promoted_to_waists(tmp_path, points):
    store = RunStore(tmp_path)
    store.create_run("linear", {"recipe_id": "operator"})
    store.save_waist_refinement({
        "valid": True, "both_confirmed": True,
        "x_minimum": {"confirmed": True, "diameter_um": 12.0, "z_actual_mm": 1.0},
    })
    summary = store.preview_summary(fit_far_field(points, selected_indices_x=[2, 3, 4]))
    assert summary["key_results"]["x"]["minimum_diameter_um"] == 10.0
    assert summary["key_results"]["x"]["minimum_at_boundary"] is True
    assert summary["key_results"]["x"]["waist_confirmed"] is False
    assert summary["key_results"]["y"]["minimum_status"] == "candidate_estimate"
    store.save_waist_refinement({
        "x_minimum": {"confirmed": True, "diameter_um": 10.0, "z_actual_mm": 0.0},
    })
    assert store.preview_summary(fit_far_field(points))["key_results"]["x"][
        "waist_confirmed"
    ] is False


def test_two_point_fit_preserves_angles_and_null_scatter(tmp_path, points):
    store = RunStore(tmp_path)
    run = store.create_run("two", {"recipe_id": "operator"})
    result = fit_far_field(points[:2])
    store.save_zscan_result(zscan_result=result)
    summary = store.finalize_run(render_html=False)
    for axis in ("x", "y"):
        key = summary["key_results"][axis]
        assert key["measured_full_angle_mrad"] is not None
        assert key["residual_std_um"] is None
        assert key["valid_points_count"] == 2
        assert key["minimum_status"] == "candidate_estimate"
    with (run.run_dir / "result_summary.csv").open(encoding="utf-8", newline="") as file:
        rows = {row["metric"]: row for row in csv.DictReader(file)}
    assert rows["key_results.x.residual_std_um"]["value"] == ""
    assert rows["key_results.x.residual_std_um"]["value_type"] == "null"


@pytest.mark.parametrize("enabled,applied,judgement,expected", [
    (False, False, "pass", "UNKNOWN"), (False, True, "pass", "UNKNOWN"),
    (True, False, "pass", "UNKNOWN"), (True, True, "pass", "PASS"),
    (True, True, "fail", "FAIL"), (True, True, "invalid", "UNKNOWN"),
])
def test_acceptance_requires_explicit_recipe_and_applied_flag(
    tmp_path, points, enabled, applied, judgement, expected,
):
    store = RunStore(tmp_path)
    store.create_run("linear", {"recipe_id": "operator", "acceptance": {"enabled": enabled}})
    result = fit_far_field(points)
    result = replace(
        result, judgement=judgement,
        invalid_reason="synthetic_spec_failure" if judgement in {"fail", "invalid"} else None,
        fit_diagnostics={**result.fit_diagnostics, "acceptance_applied": applied},
    )
    assert store.preview_summary(result)["final_judgement"] == expected


def test_partial_refinement_keeps_diagnostics_and_never_passes(tmp_path, points):
    store = RunStore(tmp_path)
    store.create_run("linear", {"recipe_id": "operator", "acceptance": {"enabled": True}})
    result = fit_far_field(points)
    result = replace(result, judgement="pass", fit_diagnostics={
        **result.fit_diagnostics, "acceptance_applied": True,
    })
    error = {"code": "capture_failed", "message": "synthetic failure"}
    store.save_waist_refinement({
        "scan_status": "incomplete", "scan_error": error,
        "termination_reason": "capture_failed", "actual_move_count": 3, "max_steps": 30,
    })
    summary = store.preview_summary(result)
    assert summary["scan_status"] == "incomplete"
    assert summary["scan_error"] == error
    assert summary["final_judgement"] == "UNKNOWN"
    assert summary["divergence_result"]["fit_diagnostics"]["x"]["fit_indices"] == list(range(5))


def test_partial_main_scan_status_comes_from_fit_diagnostics(tmp_path, points):
    store = RunStore(tmp_path)
    store.create_run("linear", {"recipe_id": "operator", "acceptance": {"enabled": True}})
    result = fit_far_field(points)
    error = {"message": "injected capture failure"}
    result = replace(result, judgement="pass", fit_diagnostics={
        **result.fit_diagnostics, "scan_status": "incomplete", "scan_error": error,
        "acceptance_applied": True,
    })
    summary = store.preview_summary(result)
    assert summary["scan_status"] == "incomplete"
    assert summary["measurement_status"] == "incomplete"
    assert summary["scan_error"] == error
    assert summary["final_judgement"] == "UNKNOWN"


def test_csv_masks_orientation_and_all_measurements_survive_refits(tmp_path, points):
    store = RunStore(tmp_path)
    run = store.create_run("linear", {"recipe_id": "operator"})
    for point in points:
        store.save_plane_result(
            plane_result=point, axis_x_azimuth_deg=12.0, axis_orientation_uncertain=False,
            native_major_azimuth_deg=102.0,
        )
    store.save_plane_result(plane_result=plane(0.5, 6.0, 7.0), phase="fine")
    result = fit_far_field(points, selected_indices_x=[1, 2], selected_indices_y=[3, 4])
    store.save_zscan_result(zscan_result=result)
    path = run.run_dir / "z_scan_table.csv"
    with path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 6
    assert [row["fit_included_x"] for row in rows] == ["False", "True", "True"] + ["False"] * 3
    assert [row["fit_included_y"] for row in rows] == ["False"] * 3 + ["True", "True", "False"]
    assert [row["fit_included"] for row in rows] == ["False"] + ["True"] * 4 + ["False"]
    assert float(rows[0]["fit_predicted_x_um"]) == pytest.approx(10.0)
    assert float(rows[0]["fit_residual_y_um"]) == pytest.approx(0.0, abs=1e-12)
    assert rows[0]["d4sigma_x_um"] == "10.0"
    assert rows[0]["axis_x_azimuth_deg"] == "12.0"
    assert rows[0]["axis_orientation_uncertain"] == "False"
    assert rows[0]["native_major_azimuth_deg"] == "102.0"
    store.save_zscan_result(zscan_result=fit_far_field(points[1:3], selected_indices_y=[]))
    with path.open(encoding="utf-8", newline="") as file:
        updated = list(csv.DictReader(file))
    assert all(row["fit_included_y"] == "False" for row in updated)
    assert updated[4]["fit_predicted_x_um"] == ""
    assert updated[4]["d4sigma_x_um"] == rows[4]["d4sigma_x_um"]


def test_linear_plot_uses_independent_ranges_and_no_quadratic_waist(points, monkeypatch):
    points[1] = replace(points[1], outlier=True, outlier_reason="old_automatic_flag")
    result = fit_far_field(points, selected_indices_x=[1, 2], selected_indices_y=[3, 4])
    result = replace(result, waist_z_x_mm=0.0, waist_diameter_x_um=1.0)
    figure = Figure(figsize=(7, 4))
    monkeypatch.setattr(report, "Figure", lambda **kwargs: figure)
    image = report._zscan_plot_image(points, result)
    assert image is not None
    axis = figure.axes[0]
    assert len(axis.lines) == 2
    for line, expected_z, slope in zip(axis.lines, [(1.0, 2.0), (3.0, 4.0)], [2, 3], strict=True):
        z, diameter = line.get_data()
        assert (min(z), max(z)) == expected_z
        assert np.allclose(np.diff(diameter) / np.diff(z), slope)
    excluded = [item for item in axis.collections if "excluded" in item.get_label()]
    assert len(excluded) == 2
    assert all(np.allclose(item.get_edgecolor()[0][:3], [107/255, 114/255, 128/255])
               for item in excluded)
    assert report._fit_inlier_mask(points, result, axis_name="x").tolist() == [
        False, True, True, False, False,
    ]
    assert all(value is None for fit in report._fit_coefficients(result).values()
               for value in fit.values())


def test_linear_envelopes_stop_at_each_axis_selection(points, monkeypatch):
    result = fit_far_field(points, selected_indices_x=[1, 2], selected_indices_y=[3, 4])
    figure = Figure(figsize=(8, 8))
    monkeypatch.setattr(report, "Figure", lambda **kwargs: figure)
    assert report.render_beam_propagation_image(points, result) is not None
    assert figure.axes[0].lines[0].get_xdata().tolist() == [1.0, 2.0]
    assert figure.axes[0].lines[1].get_xdata().tolist() == [3.0, 4.0]
    for axis, expected in zip(figure.axes[1:], [(1, 2), (3, 4)], strict=True):
        curves = [line for line in axis.lines if len(line.get_xdata()) == 160]
        assert len(curves) == 3
        assert all((min(line.get_xdata()), max(line.get_xdata())) == expected for line in curves)


def test_report_continuous_axes_never_reassigns_larger_y_to_x():
    point = plane(0.0, 8.0, 20.0, azimuth_deg=17.0)
    summary = {"system": {"effective_pixel_x_um": 0.5, "effective_pixel_y_um": 0.5}}
    axes = report._axes_from_plane_result(point, summary, BeamDisplayOptions())
    assert axes.x_axis_azimuth_deg == pytest.approx(17.0)
    assert axes.x_axis_sigma_px == 4.0
    assert axes.y_axis_sigma_px == 10.0
    assert axes.azimuth_deg == 107.0
    assert axes.axis_assignment == "minor_to_x"
    assert report._report_rotation_deg(point, axes=axes) == pytest.approx(-17.0)


@pytest.mark.parametrize("uncertain", [True, "True"])
def test_held_report_axes_keep_raw_ellipse_separate_from_tracked_markers(uncertain):
    delta = math.radians(12.0 - 55.0)
    diameter_x = math.hypot(20.0 * math.cos(delta), 19.8 * math.sin(delta))
    diameter_y = math.hypot(20.0 * math.sin(delta), 19.8 * math.cos(delta))
    point = plane(
        0.0, diameter_x, diameter_y, azimuth_deg=12.0,
        d4sigma_major_um=20.0, d4sigma_minor_um=19.8,
    )
    summary = {"system": {"effective_pixel_x_um": 0.5, "effective_pixel_y_um": 0.5}}
    axes = report._axes_for_report(
        np.eye(16), BeamDisplayOptions(), point=point, summary=summary,
        metadata={"axis_x_azimuth_deg": "12.0", "native_major_azimuth_deg": "55.0",
                  "axis_orientation_uncertain": uncertain},
    )
    assert axes.azimuth_deg == pytest.approx(55.0)
    assert axes.major_sigma_px == pytest.approx(10.0)
    assert axes.minor_sigma_px == pytest.approx(9.9)
    assert axes.x_axis_azimuth_deg == pytest.approx(12.0)
    assert axes.x_axis_sigma_px == pytest.approx(diameter_x / 2.0)
    assert axes.y_axis_sigma_px == pytest.approx(diameter_y / 2.0)
    assert axes.orientation_status == "held_near_circular"
    assert report._report_rotation_deg(point, axes=axes) == pytest.approx(-12.0)


def test_report_native_zero_angle_is_not_replaced_by_tracked_angle():
    point = plane(0.0, 8.0, 20.0, azimuth_deg=85.0)
    summary = {"system": {"effective_pixel_x_um": 1.0, "effective_pixel_y_um": 1.0}}
    axes = report._axes_from_plane_result(point, summary, BeamDisplayOptions(), metadata={
        "axis_x_azimuth_deg": "85.0", "native_major_azimuth_deg": "0.0",
        "axis_orientation_uncertain": "False",
    })
    assert axes.azimuth_deg == 0.0
    assert axes.x_axis_azimuth_deg == 85.0
    assert axes.orientation_status == "continuous"


def test_uncertain_axes_without_native_angle_do_not_invent_raw_ellipse():
    point = plane(0.0, 19.9, 19.8)
    summary = {"system": {"effective_pixel_x_um": 1.0, "effective_pixel_y_um": 1.0}}
    assert report._axes_from_plane_result(point, summary, BeamDisplayOptions(), metadata={
        "axis_orientation_uncertain": "True", "native_major_azimuth_deg": "",
    }) is None


def test_report_delegates_anisotropic_raw_ellipse_and_marker_conversion(monkeypatch):
    point = plane(
        0.0, 12.0, 10.0, azimuth_deg=30.0,
        d4sigma_major_um=16.0, d4sigma_minor_um=8.0,
    )
    summary = {"system": {"effective_pixel_x_um": 0.5, "effective_pixel_y_um": 1.0}}
    original = report.beam_axes_from_measurement
    received = []

    def convert(**values):
        received.append(values)
        return original(**values)

    monkeypatch.setattr(report, "beam_axes_from_measurement", convert)
    axes = report._axes_from_plane_result(point, summary, BeamDisplayOptions(), metadata={
        "axis_x_azimuth_deg": "30.0", "native_major_azimuth_deg": "45.0",
        "axis_orientation_uncertain": "True",
    })
    assert len(received) == 1
    assert received[0]["major_diameter_um"] == 16.0
    assert received[0]["minor_diameter_um"] == 8.0
    assert received[0]["major_azimuth_deg"] == 45.0
    assert received[0]["x_azimuth_deg"] == 30.0
    assert received[0]["x_diameter_um"] == 12.0
    assert received[0]["y_diameter_um"] == 10.0
    assert axes.y_axis_azimuth_deg != pytest.approx(axes.x_axis_azimuth_deg + 90)
    assert report._report_rotation_deg(point, axes=axes) == pytest.approx(
        -math.degrees(math.atan2(0.5, math.sqrt(3)))
    )


def test_missing_source_image_does_not_shift_key_plane_identity(tmp_path, points):
    store = RunStore(tmp_path)
    run = store.create_run("images", {"recipe_id": "operator"})
    for index in (0, 2):
        store.image_store.save_report_source_npy(np.eye(16), frame_index_count=index)
    entries = report._scan_image_entries(run.run_dir, points)
    assert [entry["plane_index_count"] for entry in entries] == [0, 2]
    assert [entry["point"] for entry in entries] == [points[0], points[2]]
    selected = report._key_spot_images([
        {"src": "spot_000.png", "plane_index_count": 0},
        {"src": "spot_002.png", "plane_index_count": 2},
    ], {"x": {"image_plane_index_count": 2}, "y": {"image_plane_index_count": 1}})
    assert len(selected) == 1
    assert selected[0]["src"] == "spot_002.png"


def test_failed_preview_does_not_replace_saved_result(tmp_path, points, monkeypatch):
    store = RunStore(tmp_path)
    store.create_run("linear", {"recipe_id": "operator"})
    result = fit_far_field(points)
    store.save_zscan_result(zscan_result=result)
    original = json.dumps(store.preview_summary(), sort_keys=True)
    with monkeypatch.context() as patch:
        patch.setattr("lbqa_data.run_store.predict_gaussian_divergence",
                      lambda **kwargs: (_ for _ in ()).throw(ValueError("synthetic failure")))
        with pytest.raises(ValueError, match="synthetic failure"):
            store.preview_summary(fit_far_field(points[:2]))
    assert store._zscan_result is result
    assert json.dumps(store.preview_summary(), sort_keys=True) == original
