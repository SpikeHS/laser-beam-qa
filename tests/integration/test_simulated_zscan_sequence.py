from __future__ import annotations

from typing import Any

import pytest
from lbqa_acquisition import AcquisitionService, AcquisitionSettings
from lbqa_analysis import analyze_beam_plane, fit_zscan, judge_result
from lbqa_contracts.models import BeamPlaneResult, OpticalCalibration, ZScanResult
from lbqa_devices.profiler.simulated_profiler import SimulatedProfiler
from lbqa_devices.safety.simulated_interlock import SimulatedInterlock
from lbqa_devices.stage.simulated_stage import SimulatedStage
from lbqa_sequencer import SequencerDependencies, SequencerService, ZScanRecipe, ZScanRunCommand


class AnalysisAdapter:
    def analyze_beam_plane(
        self,
        image: Any,
        calibration: OpticalCalibration,
        z_actual_mm: float,
        quality_limits: dict[str, Any],
    ) -> BeamPlaneResult:
        return analyze_beam_plane(image, calibration, z_actual_mm, quality_limits)

    def fit_zscan(self, points: list[BeamPlaneResult]) -> ZScanResult:
        return fit_zscan(points)

    def judge_result(self, zscan_result: ZScanResult, limits: dict[str, Any] | None = None) -> str:
        return judge_result(zscan_result, limits)


class FakeDataStore:
    def __init__(self) -> None:
        self.plane_results: list[BeamPlaneResult] = []
        self.zscan_results: list[ZScanResult] = []
        self.failure_payloads: list[dict[str, Any]] = []

    def save_plane_result(self, **kwargs: Any) -> None:
        self.plane_results.append(kwargs["plane_result"])

    def save_zscan_result(self, **kwargs: Any) -> None:
        self.zscan_results.append(kwargs["zscan_result"])

    def save_failure_data(self, failure_payload: dict[str, Any]) -> None:
        self.failure_payloads.append(failure_payload)


def test_simulated_zscan_sequence_recovers_synthetic_full_angle() -> None:
    stage = SimulatedStage(z_min_mm=-1.5, z_max_mm=1.5)
    profiler = SimulatedProfiler(
        width_px=384,
        height_px=384,
        bit_depth=12,
        amplitude_count=3000.0,
        background_count=0.0,
        noise_std_count=0.0,
        waist_diameter_x_um=18.0,
        waist_diameter_y_um=22.0,
        full_angle_x_mrad=8.0,
        full_angle_y_mrad=10.0,
        z_position_source=stage.get_position_mm,
    )
    acquisition = AcquisitionService(
        profiler,
        AcquisitionSettings(
            average_frame_count=1,
            auto_roi_enabled=False,
            edge_energy_limit_percent=20.0,
            saturation_threshold_percent=99.5,
        ),
    )
    data_store = FakeDataStore()
    recipe = ZScanRecipe(
        recipe_id="integration-simulated-zscan",
        operator_mode="simulated",
        z_ref_mm=0.0,
        prescan_delta_um=100.0,
        min_points=7,
        max_points=9,
        max_scan_half_range_um=1000.0,
        target_diameter_change_per_step_um=2.0,
        auto_exposure_enabled=False,
        average_frame_count=1,
        field_of_view_margin_percent=90.0,
        quality_limits={
            "bit_depth": 12,
            "edge_margin_px": 4,
            "edge_energy_limit_percent": 20.0,
            "saturation_threshold_percent": 99.5,
        },
    )
    command = ZScanRunCommand(
        run_id="integration-run",
        sample_id="sample-sim",
        recipe=recipe,
        calibration=profiler.get_optical_calibration(),
    )
    service = SequencerService(
        SequencerDependencies(
            stage=stage,
            profiler=profiler,
            interlock=SimulatedInterlock(),
            acquisition=acquisition,
            analysis=AnalysisAdapter(),
            data_store=data_store,
        )
    )

    result = service.run_zscan(command)

    assert result.judgement == "pass"
    assert result.valid_points_count >= recipe.min_points
    assert result.full_angle_x_mrad == pytest.approx(8.0, rel=0.10)
    assert result.full_angle_y_mrad == pytest.approx(10.0, rel=0.10)
    assert result.half_angle_x_mrad == pytest.approx(result.full_angle_x_mrad / 2.0)
    assert data_store.zscan_results[-1] is result
    assert not data_store.failure_payloads
