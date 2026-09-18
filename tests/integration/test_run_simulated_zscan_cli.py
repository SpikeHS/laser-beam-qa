from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml
from lbqa_sequencer import ZScanRecipe

REQUIRED_CSV_COLUMNS = {
    "z_actual_mm",
    "centroid_x_um",
    "centroid_y_um",
    "d4sigma_x_um",
    "d4sigma_y_um",
    "d4sigma_major_um",
    "d4sigma_minor_um",
    "ellipticity",
    "azimuth_deg",
    "valid",
    "invalid_reason",
    "fit_included",
    "fit_predicted_x_um",
    "fit_predicted_y_um",
    "fit_residual_x_um",
    "fit_residual_y_um",
    "fit_normalized_residual_x",
    "fit_normalized_residual_y",
}

REQUIRED_SUMMARY_FIELDS = {
    "sample_id",
    "recipe_id",
    "judgement",
    "full_angle_x_mrad",
    "full_angle_y_mrad",
    "half_angle_x_mrad",
    "half_angle_y_mrad",
    "waist_z_x_mm",
    "waist_z_y_mm",
    "waist_diameter_x_um",
    "waist_diameter_y_um",
    "fit_r2_x",
    "fit_r2_y",
    "valid_points_count",
}


def test_run_simulated_zscan_cli_generates_traceable_artifacts(tmp_path: Path) -> None:
    repo_root = Path(__file__).parents[2]
    recipe_path = Path("configs/recipe_40x_zscan.example.yaml")
    calibration_path = Path("configs/calibration_40x.example.yaml")
    recipe_mapping = yaml.safe_load((repo_root / recipe_path).read_text(encoding="utf-8"))
    recipe = ZScanRecipe.from_mapping(recipe_mapping)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "apps.cli.run_simulated_zscan",
            "--recipe",
            str(recipe_path),
            "--calibration",
            str(calibration_path),
            "--sample-id",
            "SIM001",
            "--out",
            str(tmp_path),
        ],
        cwd=repo_root,
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    run_dir = Path(payload["run_dir"])
    assert run_dir.is_dir()

    recipe_snapshot = run_dir / "recipe_snapshot.yaml"
    calibration_snapshot = run_dir / "calibration_snapshot.yaml"
    csv_path = run_dir / "z_scan_table.csv"
    summary_path = run_dir / "result_summary.json"
    summary_csv_path = run_dir / "result_summary.csv"
    report_path = run_dir / "report.html"
    for artifact_path in (
        recipe_snapshot,
        calibration_snapshot,
        csv_path,
        summary_path,
        summary_csv_path,
        report_path,
    ):
        assert artifact_path.exists()

    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    assert rows
    assert REQUIRED_CSV_COLUMNS.issubset(rows[0].keys())
    valid_rows = [row for row in rows if _truthy(row["valid"])]
    assert len(valid_rows) >= recipe.min_points
    included_rows = [row for row in rows if _truthy(row["fit_included"])]
    assert len(included_rows) >= recipe.min_points
    for row in included_rows:
        assert row["fit_predicted_x_um"]
        assert row["fit_predicted_y_um"]
        assert row["fit_residual_x_um"]
        assert row["fit_residual_y_um"]

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    divergence = summary["divergence_result"]
    synthetic_truth = summary["system"]["synthetic_truth"]
    assert REQUIRED_SUMMARY_FIELDS.issubset(summary.keys())
    assert summary["sample_id"] == "SIM001"
    assert summary["recipe_id"] == recipe.recipe_id
    assert summary["sample_info"]["sample_id"] == "SIM001"
    assert _normalized_judgement(summary) == "PASS"

    full_angle_x_mrad = _number(summary["full_angle_x_mrad"])
    full_angle_y_mrad = _number(summary["full_angle_y_mrad"])
    half_angle_x_mrad = _number(summary["half_angle_x_mrad"])
    half_angle_y_mrad = _number(summary["half_angle_y_mrad"])
    fit_r2_x = _number(summary["fit_r2_x"])
    fit_r2_y = _number(summary["fit_r2_y"])
    valid_points_count = int(summary["valid_points_count"])

    assert full_angle_x_mrad > 0.0
    assert full_angle_y_mrad > 0.0
    assert half_angle_x_mrad == pytest.approx(full_angle_x_mrad / 2.0)
    assert half_angle_y_mrad == pytest.approx(full_angle_y_mrad / 2.0)
    assert 0.95 <= fit_r2_x <= 1.0
    assert 0.95 <= fit_r2_y <= 1.0
    assert valid_points_count >= recipe.min_points
    assert divergence["full_angle_x_mrad"] == summary["full_angle_x_mrad"]
    assert divergence["full_angle_y_mrad"] == summary["full_angle_y_mrad"]
    assert full_angle_x_mrad == pytest.approx(
        synthetic_truth["full_angle_x_mrad"], rel=0.10
    )
    assert full_angle_y_mrad == pytest.approx(
        synthetic_truth["full_angle_y_mrad"], rel=0.10
    )
    assert report_path.stat().st_size > 0
    summary_rows = list(csv.DictReader(summary_csv_path.open(encoding="utf-8")))
    summary_metrics = {row["metric"]: row for row in summary_rows}
    assert summary_metrics["full_angle_x_mrad"]["unit"] == "mrad"
    assert summary_metrics["divergence_result.fit_diagnostics.x.fit_rmse_um"]["unit"] == "um"
    assert "measured_beam_quality.m2_x" in summary_metrics
    assert summary_metrics["measured_beam_quality.invalid_reason"]["value"]
    assert not (run_dir / ".working").exists()
    assert not (run_dir / "images").exists()
    assert not (run_dir / "processed").exists()
    assert not list(run_dir.rglob("*.npy"))
    assert not list(run_dir.rglob("*.tif"))
    assert not list(run_dir.rglob("*.tiff"))


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _number(value: Any) -> float:
    assert value is not None
    return float(value)


def _normalized_judgement(summary: dict[str, Any]) -> str:
    value = summary.get("final_judgement", summary.get("judgement"))
    assert value is not None
    return str(value).upper()
