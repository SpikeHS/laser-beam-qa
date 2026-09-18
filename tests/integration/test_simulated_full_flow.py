from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from lbqa_acquisition import AcquisitionService, AcquisitionSettings
from lbqa_analysis import analyze_beam_plane, fit_zscan, judge_result
from lbqa_contracts.models import BeamPlaneResult, OpticalCalibration, ZScanResult
from lbqa_data import RunStore
from lbqa_devices.profiler.simulated_profiler import SimulatedProfiler
from lbqa_devices.safety.simulated_interlock import SimulatedInterlock
from lbqa_devices.stage.simulated_stage import SimulatedStage
from lbqa_sequencer import SequencerDependencies, SequencerService, ZScanRecipe, ZScanRunCommand


def fixed_timestamp() -> datetime:
    return datetime(2026, 6, 22, 9, 30, 0, tzinfo=UTC)


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


class PersistingAcquisition:
    """Test adapter that stages only temporary report sources."""

    def __init__(self, service: AcquisitionService, store: RunStore) -> None:
        self._service = service
        self._store = store
        self._frame_index_count = 0

    def prepare_dark_frame(self, frame_count: int | None = None) -> Any:
        return self._service.prepare_dark_frame(frame_count)

    def prepare_exposure(self) -> Any:
        return self._service.prepare_exposure()

    def capture_plane_average(self, z_actual_mm: float, frame_count: int | None = None) -> Any:
        corrected_frame = self._service.capture_plane_average(z_actual_mm, frame_count)
        if self._store.image_store is not None:
            frame_index_count = self._frame_index_count
            self._store.image_store.save_report_source_npy(
                corrected_frame.image,
                frame_index_count=frame_index_count,
            )
            self._frame_index_count += 1
        return corrected_frame


def test_simulated_full_flow_generates_report_and_matches_synthetic_truth(tmp_path: Path) -> None:
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
    recipe = ZScanRecipe(
        recipe_id="integration-simulated-full-flow",
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
        soft_limit_min_mm=-1.5,
        soft_limit_max_mm=1.5,
        quality_limits={
            "bit_depth": 12,
            "edge_margin_px": 4,
            "edge_energy_limit_percent": 20.0,
            "saturation_threshold_percent": 99.5,
            "low_signal_threshold_percent": 1.0,
        },
        judgement_limits={
            "min_fit_r2": 0.98,
            "max_full_angle_x_mrad": 20.0,
            "max_full_angle_y_mrad": 20.0,
        },
    )
    store = RunStore(tmp_path, timestamp_factory=fixed_timestamp)
    record = store.create_run("sample-sim-full-flow", recipe)
    store.save_calibration_snapshot(profiler.get_optical_calibration())
    store.save_system_snapshot(
        {
            "system_name": "laser-beam-qa",
            "validation_mode": "simulated_full_flow",
            "stage": "SimulatedStage",
            "profiler": "SimulatedProfiler",
        }
    )
    service = SequencerService(
        SequencerDependencies(
            stage=stage,
            profiler=profiler,
            interlock=SimulatedInterlock(),
            acquisition=PersistingAcquisition(acquisition, store),
            analysis=AnalysisAdapter(),
            data_store=store,
        )
    )

    result = service.run_zscan(
        ZScanRunCommand(
            run_id=record.run_id,
            sample_id=record.sample_id,
            recipe=recipe,
            calibration=profiler.get_optical_calibration(),
        )
    )
    summary = store.finalize_run(render_html=True)

    rows = list(csv.DictReader((record.run_dir / "z_scan_table.csv").open(encoding="utf-8")))
    loaded_summary = json.loads((record.run_dir / "result_summary.json").read_text("utf-8"))
    report_html = (record.run_dir / "report.html").read_text("utf-8")

    assert result.judgement == "pass"
    assert result.valid_points_count >= recipe.min_points
    assert result.full_angle_x_mrad == pytest.approx(8.0, rel=0.10)
    assert result.full_angle_y_mrad == pytest.approx(10.0, rel=0.10)
    assert result.half_angle_x_mrad == pytest.approx(result.full_angle_x_mrad / 2.0)
    assert result.half_angle_y_mrad == pytest.approx(result.full_angle_y_mrad / 2.0)
    assert loaded_summary == summary
    assert summary["final_judgement"] == "PASS"
    assert summary["divergence_result"]["full_angle_x_mrad"] == pytest.approx(
        result.full_angle_x_mrad
    )
    assert len(rows) >= result.valid_points_count
    assert "images" not in summary["output_files"]
    assert "processed" not in summary["output_files"]
    assert len(summary["output_files"]["report_assets"]) >= result.valid_points_count
    assert summary["output_files"]["report_html"] == "report.html"
    assert "sample-sim-full-flow" in report_html
    assert "D(z)=sqrt(D0^2+Theta^2(z-z0)^2)" in report_html
    assert "附录：全部扫描点模斑图" in report_html
    assert not list(record.run_dir.rglob("*.npy"))
    assert not list(record.run_dir.rglob("*.tif"))
    assert not list(record.run_dir.rglob("*.tiff"))
