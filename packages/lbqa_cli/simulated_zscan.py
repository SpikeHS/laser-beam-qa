"""Run a hardware-free simulated z-scan from the command line."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:
    yaml = None

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGES_DIR = REPO_ROOT / "packages"
if str(PACKAGES_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGES_DIR))

from lbqa_acquisition import AcquisitionService, AcquisitionSettings  # noqa: E402
from lbqa_analysis import analyze_beam_plane, fit_zscan, judge_result  # noqa: E402
from lbqa_contracts.models import BeamPlaneResult, OpticalCalibration, ZScanResult  # noqa: E402
from lbqa_data import RunStore  # noqa: E402
from lbqa_devices.profiler.simulated_profiler import SimulatedProfiler  # noqa: E402
from lbqa_devices.safety.simulated_interlock import SimulatedInterlock  # noqa: E402
from lbqa_devices.stage.simulated_stage import SimulatedStage  # noqa: E402
from lbqa_sequencer import (  # noqa: E402
    SequencerDependencies,
    SequencerService,
    ZScanRecipe,
    ZScanRunCommand,
)

SIMULATED_FRAME_WIDTH_PX = 384
SIMULATED_FRAME_HEIGHT_PX = 384
SIMULATED_CAMERA_PIXEL_SIZE_UM = 5.5
SIMULATED_OBJECTIVE_MAGNIFICATION = 40.0
SIMULATED_EFFECTIVE_PIXEL_UM = SIMULATED_CAMERA_PIXEL_SIZE_UM / SIMULATED_OBJECTIVE_MAGNIFICATION
SIMULATED_FULL_ANGLE_X_MRAD = 8.0
SIMULATED_FULL_ANGLE_Y_MRAD = 10.0


class AnalysisAdapter:
    """Thin adapter exposing pure analysis functions to the sequencer."""

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
    """Thin adapter that persists frame artifacts without widening contracts."""

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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a simulated LBQA z-scan.")
    parser.add_argument("--recipe", required=True, type=Path, help="Path to a z-scan recipe YAML.")
    parser.add_argument(
        "--calibration",
        required=True,
        type=Path,
        help="Path to an optical calibration YAML.",
    )
    parser.add_argument("--sample-id", required=True, help="Sample identifier for this run.")
    parser.add_argument("--out", required=True, type=Path, help="Root output directory.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = run_simulated_zscan(
        recipe_path=args.recipe,
        calibration_path=args.calibration,
        sample_id=args.sample_id,
        output_dir=args.out,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def run_simulated_zscan(
    *,
    recipe_path: Path,
    calibration_path: Path,
    sample_id: str,
    output_dir: Path,
) -> dict[str, Any]:
    recipe_mapping = _load_yaml_mapping(recipe_path)
    calibration_mapping = _load_yaml_mapping(calibration_path)
    recipe = ZScanRecipe.from_mapping(recipe_mapping)
    calibration = _build_simulated_cropped_calibration(calibration_mapping)

    stage = SimulatedStage(z_min_mm=-1.5, z_max_mm=1.5)
    profiler = SimulatedProfiler(
        width_px=SIMULATED_FRAME_WIDTH_PX,
        height_px=SIMULATED_FRAME_HEIGHT_PX,
        camera_pixel_size_um=SIMULATED_CAMERA_PIXEL_SIZE_UM,
        objective_magnification=SIMULATED_OBJECTIVE_MAGNIFICATION,
        bit_depth=12,
        amplitude_count=3000.0,
        background_count=0.0,
        noise_std_count=0.0,
        waist_diameter_x_um=18.0,
        waist_diameter_y_um=22.0,
        full_angle_x_mrad=SIMULATED_FULL_ANGLE_X_MRAD,
        full_angle_y_mrad=SIMULATED_FULL_ANGLE_Y_MRAD,
        z_position_source=stage.get_position_mm,
    )
    acquisition = AcquisitionService(
        profiler,
        AcquisitionSettings(
            average_frame_count=recipe.average_frame_count or 1,
            auto_roi_enabled=False,
            edge_energy_limit_percent=20.0,
            saturation_threshold_percent=99.5,
        ),
    )
    store = RunStore(output_dir)
    record = store.create_run(sample_id, recipe_mapping)
    store.save_calibration_snapshot(calibration)
    store.save_system_snapshot(
        {
            "system_name": "laser-beam-qa",
            "mode": "simulated_mvp",
            "frame_mode": "simulated_cropped_frame",
            "stage": "SimulatedStage",
            "profiler": "SimulatedProfiler",
            "source_recipe_path": str(recipe_path),
            "source_calibration_path": str(calibration_path),
            "width_px": SIMULATED_FRAME_WIDTH_PX,
            "height_px": SIMULATED_FRAME_HEIGHT_PX,
            "effective_pixel_x_um": SIMULATED_EFFECTIVE_PIXEL_UM,
            "effective_pixel_y_um": SIMULATED_EFFECTIVE_PIXEL_UM,
            "field_of_view_x_um": calibration.field_of_view_x_um,
            "field_of_view_y_um": calibration.field_of_view_y_um,
            "synthetic_truth": {
                "full_angle_x_mrad": SIMULATED_FULL_ANGLE_X_MRAD,
                "full_angle_y_mrad": SIMULATED_FULL_ANGLE_Y_MRAD,
            },
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
            calibration=calibration,
        )
    )
    summary = store.finalize_run(render_html=True, render_pdf=False)
    return {
        "run_id": record.run_id,
        "run_dir": str(record.run_dir),
        "sample_id": record.sample_id,
        "judgement": result.judgement,
        "full_angle_x_mrad": result.full_angle_x_mrad,
        "full_angle_y_mrad": result.full_angle_y_mrad,
        "output_files": summary.get("output_files", {}),
    }


def _load_yaml_mapping(path: Path) -> Mapping[str, Any]:
    text = path.read_text(encoding="utf-8")
    loaded = _load_simple_yaml_mapping(text) if yaml is None else yaml.safe_load(text)
    if not isinstance(loaded, Mapping):
        raise ValueError(f"{path} must contain a YAML mapping.")
    return loaded


def _load_simple_yaml_mapping(text: str) -> dict[str, Any]:
    """Small fallback parser for the checked-in example YAML files."""

    tokens = [
        (len(line) - len(line.lstrip(" ")), line.strip())
        for line in text.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any] | list[Any]]] = [(-1, root)]

    for index, (indent_count, stripped_line) in enumerate(tokens):
        while indent_count <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]

        if stripped_line.startswith("- "):
            if not isinstance(parent, list):
                raise ValueError("fallback YAML parser found a list item outside a list.")
            parent.append(_parse_scalar(stripped_line[2:].strip()))
            continue

        key, separator, raw_value = stripped_line.partition(":")
        if separator == "":
            raise ValueError(f"fallback YAML parser expected a key/value line: {stripped_line}")
        if not isinstance(parent, dict):
            raise ValueError("fallback YAML parser found a mapping entry inside a list.")

        value = raw_value.strip()
        if value:
            parent[key] = _parse_scalar(value)
            continue

        next_token = tokens[index + 1] if index + 1 < len(tokens) else None
        starts_nested_list = (
            next_token is not None
            and next_token[0] > indent_count
            and next_token[1].startswith("- ")
        )
        next_container: dict[str, Any] | list[Any] = [] if starts_nested_list else {}
        parent[key] = next_container
        stack.append((indent_count, next_container))

    return root


def _parse_scalar(value: str) -> str | bool | int | float | None:
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    if value.startswith("'") and value.endswith("'"):
        return value[1:-1]
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    if value.lower() in {"null", "none", "~"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _build_simulated_cropped_calibration(calibration: Mapping[str, Any]) -> OpticalCalibration:
    camera = _mapping(calibration.get("camera"))
    objective = _mapping(calibration.get("objective"))
    field_of_view_x_um = SIMULATED_FRAME_WIDTH_PX * SIMULATED_EFFECTIVE_PIXEL_UM
    field_of_view_y_um = SIMULATED_FRAME_HEIGHT_PX * SIMULATED_EFFECTIVE_PIXEL_UM
    return OpticalCalibration(
        magnification=float(
            objective.get("objective_magnification", SIMULATED_OBJECTIVE_MAGNIFICATION)
        ),
        camera_pixel_size_um=float(
            camera.get("camera_pixel_size_um", SIMULATED_CAMERA_PIXEL_SIZE_UM)
        ),
        effective_pixel_x_um=SIMULATED_EFFECTIVE_PIXEL_UM,
        effective_pixel_y_um=SIMULATED_EFFECTIVE_PIXEL_UM,
        field_of_view_x_um=field_of_view_x_um,
        field_of_view_y_um=field_of_view_y_um,
        working_distance_mm=float(objective.get("working_distance_mm", 0.6)),
        calibration_id=str(calibration.get("calibration_id", "simulated-cropped-40x")),
    )


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


if __name__ == "__main__":
    raise SystemExit(main())
