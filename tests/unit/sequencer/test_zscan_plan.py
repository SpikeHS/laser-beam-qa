import pytest
from lbqa_contracts.errors import ErrorCode, LBQAError
from lbqa_sequencer import ZScanRecipe
from lbqa_sequencer.zscan_plan import (
    ZScanPreScanSample,
    build_zscan_plan,
    estimate_prescan_result,
)


def _recipe(**overrides: object) -> ZScanRecipe:
    values = {
        "recipe_id": "unit-plan",
        "z_ref_mm": 0.0,
        "prescan_delta_um": 50.0,
        "min_points": 5,
        "max_points": 9,
        "max_scan_half_range_um": 500.0,
        "target_diameter_change_per_step_um": 1.0,
        "auto_exposure_enabled": False,
    }
    values.update(overrides)
    return ZScanRecipe(**values)


def test_build_zscan_plan_uses_prescan_slope_for_step_count() -> None:
    prescan = estimate_prescan_result(
        [
            ZScanPreScanSample(-0.05, 18.0, 20.0, True),
            ZScanPreScanSample(0.0, 18.0, 20.5, True),
            ZScanPreScanSample(0.05, 18.0, 21.0, True),
        ]
    )

    plan = build_zscan_plan(
        _recipe(target_diameter_change_per_step_um=2.0),
        prescan,
        field_of_view_x_um=100.0,
        field_of_view_y_um=100.0,
    )

    assert plan.z_positions_mm[0] == pytest.approx(-0.5)
    assert plan.z_positions_mm[-1] == pytest.approx(0.5)
    assert plan.scan_direction == "ascending"
    assert len(plan.z_positions_mm) >= 5
    assert plan.step_um > 0.0


def test_build_zscan_plan_shrinks_range_when_fov_risk_is_high() -> None:
    prescan = estimate_prescan_result(
        [
            ZScanPreScanSample(-0.05, 70.0, 70.0, True),
            ZScanPreScanSample(0.0, 71.0, 71.0, True),
            ZScanPreScanSample(0.05, 72.0, 72.0, True),
        ]
    )

    plan = build_zscan_plan(
        _recipe(max_scan_half_range_um=1000.0),
        prescan,
        field_of_view_x_um=100.0,
        field_of_view_y_um=100.0,
    )

    assert plan.scan_half_range_um < 1000.0
    assert "range_shrunk_for_field_of_view" in plan.risk_flags


def test_build_zscan_plan_errors_when_beam_already_exceeds_fov_margin() -> None:
    prescan = estimate_prescan_result(
        [
            ZScanPreScanSample(-0.05, 90.0, 90.0, True),
            ZScanPreScanSample(0.05, 91.0, 91.0, True),
        ]
    )

    with pytest.raises(LBQAError) as exc_info:
        build_zscan_plan(
            _recipe(),
            prescan,
            field_of_view_x_um=100.0,
            field_of_view_y_um=100.0,
        )

    assert exc_info.value.error_code == ErrorCode.E_SAFETY_COLLISION_RISK.value
