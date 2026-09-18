import json
import math
from dataclasses import replace

import numpy as np
import pytest
from lbqa_analysis.ellipse_axes import StableEllipseAxes
from lbqa_analysis.far_field_fit import fit_far_field, suggest_far_field_indices
from lbqa_contracts.errors import ErrorCode
from lbqa_contracts.models import BeamPlaneResult, ZScanResult


def _plane(z: float, x: float, y: float) -> BeamPlaneResult:
    return BeamPlaneResult(
        z_actual_mm=z,
        centroid_x_um=0.0,
        centroid_y_um=0.0,
        peak_x_um=None,
        peak_y_um=None,
        d4sigma_x_um=x,
        d4sigma_y_um=y,
        d4sigma_major_um=max(x, y),
        d4sigma_minor_um=min(x, y),
        fwhm_x_um=None,
        fwhm_y_um=None,
        ellipticity=min(x, y) / max(x, y),
        azimuth_deg=0.0,
        peak_value=1000.0,
        saturation_pixels=0,
        edge_energy_percent=0.0,
        valid=True,
    )


@pytest.mark.parametrize("slope", [3.5, 2000.0])
def test_linear_truth_exact_geometric_angles_and_no_waist(slope: float) -> None:
    points = [_plane(z, 20.0 + slope * z, 40.0 + 7.0 * z) for z in (2.0, 3.0, 4.0, 5.0)]

    result = fit_far_field(points)

    assert isinstance(result, ZScanResult)
    assert result.full_angle_x_mrad == pytest.approx(2000.0 * math.atan(slope / 2000.0))
    assert result.half_angle_x_mrad == pytest.approx(result.full_angle_x_mrad / 2.0)
    assert result.full_angle_y_mrad == pytest.approx(2000.0 * math.atan(7.0 / 2000.0))
    assert result.half_angle_y_mrad == pytest.approx(result.full_angle_y_mrad / 2.0)
    assert result.judgement == "unknown"
    assert result.valid_points_count == 4
    assert result.points == points
    assert all(
        original is retained for original, retained in zip(points, result.points, strict=True)
    )
    for axis in ("x", "y"):
        assert getattr(result, f"waist_z_{axis}_mm") is None
        assert getattr(result, f"waist_diameter_{axis}_um") is None
        assert getattr(result, f"fit_a_{axis}_um2_per_mm2") is None
        assert getattr(result, f"fit_b_{axis}_um2_per_mm") is None
        assert getattr(result, f"fit_c_{axis}_um2") is None
        assert getattr(result, f"fit_r2_{axis}") == pytest.approx(1.0)
    diagnostics = result.fit_diagnostics
    assert diagnostics["model"] == "far_field_linear"
    assert diagnostics["axis_semantics"] == "continuous_ellipse_x_y"
    assert diagnostics["uncertainty_mode"] == "none"
    x = diagnostics["x"]
    assert x["slope_um_per_mm"] == pytest.approx(slope)
    assert x["intercept_um"] == pytest.approx(20.0)
    assert x["z_reference_mm"] == pytest.approx(3.5)
    assert x["intercept_at_reference_um"] == pytest.approx(20.0 + 3.5 * slope)
    assert x["status"] == "ok"
    assert x["reasons"] == []
    assert x["fit_indices"] == [0, 1, 2, 3]
    for index, row in enumerate(diagnostics["points"]):
        assert row["index"] == index
        assert row["z_actual_mm"] == points[index].z_actual_mm
        assert row["fit_included_x"] and row["fit_included_y"] and row["fit_included"]
        assert row["predicted_x_um"] == pytest.approx(points[index].d4sigma_x_um)
        assert row["residual_x_um"] == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("count,noise", [(2, [0, 0]), (3, [1, -2, 1]), (4, [1, -1, -1, 1])])
def test_residual_standard_deviation_uses_n_minus_two(count: int, noise: list[int]) -> None:
    points = [_plane(z, 10.0 + 2.0 * z + noise[z], 20.0 + 3.0 * z) for z in range(count)]

    result = fit_far_field(points)

    x = result.fit_diagnostics["x"]
    sse = sum(value**2 for value in noise)
    assert x["slope_um_per_mm"] == pytest.approx(2.0)
    assert x["fit_rmse_um"] == pytest.approx(math.sqrt(sse / count), abs=1e-12)
    if count == 2:
        assert x["residual_std_um"] is None
        assert "residual_std_unavailable_two_points" in x["reasons"]
        assert result.judgement == "warning"
    else:
        assert x["residual_std_um"] == pytest.approx(math.sqrt(sse / (count - 2)))
    assert "angle_uncertainty_mrad" not in x
    assert "parameter_ci95" not in x
    assert "normalized_residual_x" not in result.fit_diagnostics["points"][0]


def test_centered_fit_handles_large_z_origin_and_small_span() -> None:
    origin = 1e12
    points = [
        _plane(origin + index / 1024.0, 100.0 + index * 2.0, 50.0 + index) for index in range(6)
    ]

    result = fit_far_field(points)

    x = result.fit_diagnostics["x"]
    assert x["slope_um_per_mm"] == pytest.approx(2048.0)
    assert x["fit_r2"] == pytest.approx(1.0)
    assert x["fit_rmse_um"] == pytest.approx(0.0, abs=1e-12)
    for point, row in zip(points, result.fit_diagnostics["points"], strict=True):
        assert row["predicted_x_um"] == pytest.approx(point.d4sigma_x_um)


def test_continuous_ellipse_widths_fit_across_major_minor_crossing() -> None:
    tracker = StableEllipseAxes()
    points = []
    for z in range(9):
        x, y = 10.0 + 2.0 * z, 16.0 + 0.5 * z
        tracked = tracker.update(max(x, y), min(x, y), 0.0 if x >= y else 90.0)
        points.append(_plane(z, tracked["width_x"], tracked["width_y"]))

    result = fit_far_field(points)

    assert result.fit_diagnostics["x"]["slope_um_per_mm"] == pytest.approx(2.0)
    assert result.fit_diagnostics["y"]["slope_um_per_mm"] == pytest.approx(0.5)
    assert result.valid_points_count == 9
    assert result.outlier_indices == []


def test_manual_axis_masks_are_independent_and_exclusions_are_not_outliers() -> None:
    points = [_plane(z, 10.0 + 2.0 * z, 20.0 + 4.0 * z) for z in range(6)]
    points[3] = replace(points[3], d4sigma_x_um=900.0)
    points[0] = replace(points[0], d4sigma_y_um=900.0)

    result = fit_far_field(points, selected_indices_x=[2, 0, 1, 1], selected_indices_y=[2, 3, 4])

    assert result.fit_diagnostics["x"]["fit_indices"] == [0, 1, 2]
    assert result.fit_diagnostics["y"]["fit_indices"] == [2, 3, 4]
    assert result.fit_diagnostics["x"]["slope_um_per_mm"] == pytest.approx(2.0)
    assert result.fit_diagnostics["y"]["slope_um_per_mm"] == pytest.approx(4.0)
    assert result.valid_points_count == 5
    assert result.outlier_indices == []
    rows = result.fit_diagnostics["points"]
    assert len(rows) == len(points)
    assert [row["fit_included_x"] for row in rows] == [True, True, True, False, False, False]
    assert [row["fit_included_y"] for row in rows] == [False, False, True, True, True, False]
    assert [row["fit_included"] for row in rows] == [True, True, True, True, True, False]
    assert all(not row["fit_outlier"] for row in rows)
    assert rows[3]["residual_x_um"] == pytest.approx(884.0)
    assert rows[5]["predicted_x_um"] == pytest.approx(20.0)


def test_manual_large_residual_and_legacy_outlier_flag_are_retained() -> None:
    points = [_plane(z, 10.0 + 2.0 * z, 30.0 + 3.0 * z) for z in range(7)]
    points[5] = replace(
        points[5], d4sigma_x_um=150.0, outlier=True, outlier_reason="legacy_residual_rejection"
    )
    snapshot = list(points)

    result = fit_far_field(points, selected_indices_x=list(range(7)))

    expected = np.linalg.lstsq(
        np.column_stack((np.arange(7), np.ones(7))),
        np.asarray([point.d4sigma_x_um for point in points]),
        rcond=None,
    )[0]
    assert result.fit_diagnostics["x"]["fit_indices"] == list(range(7))
    assert result.fit_diagnostics["x"]["slope_um_per_mm"] == pytest.approx(expected[0])
    assert result.fit_diagnostics["x"]["intercept_um"] == pytest.approx(expected[1])
    assert result.fit_diagnostics["x"]["fit_r2"] < 0.5
    assert result.fit_diagnostics["points"][5]["fit_included_x"]
    assert not result.fit_diagnostics["points"][5]["fit_outlier"]
    assert result.outlier_indices == []
    assert points == snapshot


def test_ordinary_least_squares_does_not_invent_or_apply_errorbars() -> None:
    points = [_plane(z, x, 20 + z) for z, x in enumerate([10.0, 20.0, 12.0, 13.0])]
    with_uncertainties = [
        replace(point, width_uncertainty_x_um=0.1 if index == 1 else 100.0)
        for index, point in enumerate(points)
    ]

    assert (
        fit_far_field(points).fit_diagnostics == fit_far_field(with_uncertainties).fit_diagnostics
    )


@pytest.mark.parametrize("selection", [[], [0], [0, 1]])
def test_failed_x_axis_does_not_discard_y_estimate(selection: list[int]) -> None:
    points = [_plane(z, 10 + z, 20 + 2 * z) for z in [0, 0, 1, 2]]

    result = fit_far_field(points, selected_indices_x=selection)

    assert result.full_angle_x_mrad is None
    assert result.full_angle_y_mrad == pytest.approx(2000.0 * math.atan(2.0 / 2000.0))
    assert result.judgement == "warning"
    assert result.fit_diagnostics["x"]["status"] == "invalid"
    assert result.fit_diagnostics["y"]["status"] == "ok"
    assert result.valid_points_count == 4
    assert all(row["predicted_x_um"] is None for row in result.fit_diagnostics["points"])


def test_repeated_positions_are_retained_when_two_distinct_z_exist() -> None:
    points = [
        _plane(z, 10.0 + 2.0 * z + noise, 20 + z) for z, noise in [(0, 1), (0, -1), (1, 1), (1, -1)]
    ]

    result = fit_far_field(points)

    x = result.fit_diagnostics["x"]
    assert x["fit_indices"] == [0, 1, 2, 3]
    assert x["slope_um_per_mm"] == pytest.approx(2.0)
    assert x["residual_std_um"] == pytest.approx(math.sqrt(2.0))
    assert x["valid_points_count"] == 4


def test_all_duplicate_z_are_invalid_with_aligned_diagnostics() -> None:
    result = fit_far_field([_plane(4.0, 10.0, 20.0), _plane(4.0, 12.0, 22.0)])

    assert result.judgement == "invalid"
    assert result.invalid_reason == ErrorCode.E_ANALYSIS_ZSCAN_FIT_FAILED.value
    assert result.full_angle_x_mrad is None
    assert result.fit_diagnostics["x"]["reasons"] == ["fewer_than_two_distinct_z"]
    assert len(result.fit_diagnostics["points"]) == 2
    assert result.outlier_indices == []


@pytest.mark.parametrize("bad", [math.nan, math.inf])
def test_nonfinite_axis_width_does_not_filter_other_axis(bad: float) -> None:
    points = [_plane(z, 10 + z, 20 + 2 * z) for z in range(3)]
    points[1] = replace(points[1], d4sigma_x_um=bad)

    result = fit_far_field(points)

    assert result.fit_diagnostics["x"]["fit_indices"] == [0, 2]
    assert result.fit_diagnostics["y"]["fit_indices"] == [0, 1, 2]
    assert result.outlier_indices == [1]
    row = result.fit_diagnostics["points"][1]
    assert row["fit_included_y"] and row["fit_included"]
    assert not row["fit_included_x"]
    assert row["residual_x_um"] is None
    assert row["residual_y_um"] == pytest.approx(0.0, abs=1e-12)
    json.dumps(result.fit_diagnostics, allow_nan=False)


def test_entire_nonfinite_axis_still_allows_other_axis_fit() -> None:
    points = [replace(_plane(z, 10 + z, 20 + z), d4sigma_x_um=math.nan) for z in range(3)]

    result = fit_far_field(points)

    assert result.fit_diagnostics["x"]["fit_indices"] == []
    assert result.full_angle_x_mrad is None
    assert result.full_angle_y_mrad is not None
    assert result.judgement == "warning"
    assert result.valid_points_count == 3


def test_numerical_failure_does_not_propagate_to_the_other_axis() -> None:
    points = [_plane(z, width, 20 + z) for z, width in enumerate([1e308, 1e308, 1e307])]

    result = fit_far_field(points)

    assert result.fit_diagnostics["x"]["reasons"] == ["numerical_fit_failed"]
    assert result.fit_diagnostics["x"]["fit_indices"] == [0, 1, 2]
    assert result.full_angle_x_mrad is None
    assert result.full_angle_y_mrad is not None
    assert result.judgement == "warning"
    assert result.outlier_indices == []
    json.dumps(result.fit_diagnostics, allow_nan=False)


@pytest.mark.parametrize("bad_z", [math.nan, math.inf, -math.inf])
def test_nonfinite_z_is_excluded_from_both_axes(bad_z: float) -> None:
    points = [_plane(z, 10 + z, 20 + z) for z in range(3)]
    points[1] = replace(points[1], z_actual_mm=bad_z)

    result = fit_far_field(points)

    assert result.fit_diagnostics["x"]["fit_indices"] == [0, 2]
    assert result.fit_diagnostics["y"]["fit_indices"] == [0, 2]
    row = result.fit_diagnostics["points"][1]
    assert row["z_actual_mm"] is None
    assert row["predicted_x_um"] is None and row["predicted_y_um"] is None
    assert not row["fit_included"]
    json.dumps(result.fit_diagnostics, allow_nan=False)


def test_quality_invalid_points_cannot_be_manually_included() -> None:
    points = [_plane(z, 10 + z, 20 + z) for z in range(3)]
    points[1] = replace(points[1], valid=False, invalid_reason=ErrorCode.E_IMAGE_SATURATED.value)

    result = fit_far_field(points, selected_indices_x=[0, 1, 2])

    assert result.fit_diagnostics["x"]["fit_indices"] == [0, 2]
    assert result.fit_diagnostics["y"]["fit_indices"] == [0, 2]
    assert result.outlier_indices == [1]
    assert result.valid_points_count == 2


@pytest.mark.parametrize("slope,reason", [(0.0, "zero_slope"), (-3.0, "negative_slope")])
def test_nonpositive_slopes_are_flagged_but_estimates_retained(slope: float, reason: str) -> None:
    points = [_plane(z, 30.0 + slope * z, 40.0 + 2.0 * z) for z in range(4)]

    result = fit_far_field(points)

    x = result.fit_diagnostics["x"]
    assert result.judgement == "warning"
    assert x["slope_um_per_mm"] == pytest.approx(slope)
    assert result.full_angle_x_mrad == pytest.approx(2000.0 * math.atan(abs(slope) / 2000.0))
    assert reason in x["reasons"]
    assert x["status"] == "warning"
    assert x["residual_std_um"] == pytest.approx(0.0, abs=1e-12)
    if slope == 0.0:
        assert result.fit_r2_x is None


@pytest.mark.parametrize("count", [0, 1])
def test_insufficient_inputs_are_diagnostic_not_exceptions(count: int) -> None:
    result = fit_far_field([_plane(0, 10, 20)] * count)

    assert result.judgement == "invalid"
    assert result.invalid_reason == ErrorCode.E_ANALYSIS_ZSCAN_TOO_FEW_POINTS.value
    assert len(result.fit_diagnostics["points"]) == count
    assert result.valid_points_count == count


@pytest.mark.parametrize("bad_index", [-1, 3, 1.5, True, "1"])
def test_invalid_index_invalidates_only_its_axis(bad_index: object) -> None:
    result = fit_far_field(
        [_plane(z, 10 + z, 20 + z) for z in range(3)],
        selected_indices_x=[0, 1, bad_index],
    )

    assert result.full_angle_x_mrad is None
    assert result.full_angle_y_mrad is not None
    assert result.fit_diagnostics["x"]["reasons"] == ["invalid_selection_indices"]
    assert result.judgement == "warning"


def test_numpy_integer_indices_are_supported() -> None:
    result = fit_far_field(
        [_plane(z, 10 + z, 20 + z) for z in range(3)],
        selected_indices_x=np.asarray([0, 2]),
    )

    assert result.fit_diagnostics["x"]["fit_indices"] == [0, 2]
    assert result.full_angle_x_mrad is not None


def test_suggestion_uses_post_minimum_tail_and_common_intersection_without_applying_it() -> None:
    points = [_plane(z, 10 + 2 * z, 20 + abs(z - 6)) for z in range(10)]

    suggested = suggest_far_field_indices(points)

    assert suggested["x"] == [5, 6, 7, 8, 9]
    assert suggested["y"] == [7, 8, 9]
    assert suggested["common"] == [7, 8, 9]
    assert "never guarantees far-field" in suggested["note"]
    assert "half-tail slopes agree" in suggested["note"]
    assert fit_far_field(points).fit_diagnostics["x"]["fit_indices"] == list(range(10))


def test_suggestions_are_sorted_by_z_with_original_indices_and_valid_replicates() -> None:
    points = [_plane(z, 10 + z, 20 + z) for z in [6, 2, 4, 1, 0, 3, 5, 6]]
    points[2] = replace(points[2], d4sigma_x_um=math.nan)
    points[5] = replace(points[5], valid=False, invalid_reason=ErrorCode.E_IMAGE_LOW_SIGNAL.value)

    suggested = suggest_far_field_indices(points)

    for axis in ("x", "y", "common"):
        selected = suggested[axis]
        positions = [points[index].z_actual_mm for index in selected]
        assert positions == sorted(positions)
        assert all(position > 0 for position in positions)
        assert 5 not in selected
        assert 0 in selected and 7 in selected
    assert 2 not in suggested["x"]
    assert set(suggested["common"]) == set(suggested["x"]) & set(suggested["y"])


@pytest.mark.parametrize("count", [2, 3, 4, 5, 10])
def test_suggestions_do_not_always_shrink_to_two_points(count: int) -> None:
    result = suggest_far_field_indices([_plane(z, 10 + z, 20 + z) for z in range(count)])

    assert len(result["x"]) >= min(3, max(2, count - 1))
    if count == 2:
        assert result["common"] == [0, 1]
        assert "includes the observed minimum" in result["note"]
        assert "Only two points" in result["note"]


def test_unstable_tail_is_reported_not_trimmed_for_r2() -> None:
    result = suggest_far_field_indices([_plane(z, 10 + z**2, 20 + z) for z in range(12)])

    assert result["x"] == list(range(6, 12))
    assert "unstable trend" in result["note"]
    assert "Fallback" in result["note"]


@pytest.mark.parametrize("positions", [[], [0], [0, 0, 0]])
def test_insufficient_suggestion_explains_failure(positions: list[int]) -> None:
    result = suggest_far_field_indices([_plane(z, 10, 20) for z in positions])

    assert result["x"] == [] and result["y"] == [] and result["common"] == []
    assert "Insufficient" in result["note"]


def test_negative_trend_has_no_post_minimum_suggestion() -> None:
    result = suggest_far_field_indices([_plane(z, 20 - z, 30 + z) for z in range(6)])

    assert result["x"] == [] and result["common"] == []
    assert len(result["y"]) >= 3
    assert "minimum is at the last z" in result["note"]
    assert "Insufficient common" in result["note"]


def test_flat_suggestion_is_explicitly_not_a_positive_slope_claim() -> None:
    result = suggest_far_field_indices([_plane(z, 10, 20) for z in range(10)])

    assert len(result["common"]) > 2
    assert "no estimable positive slope" in result["note"]
