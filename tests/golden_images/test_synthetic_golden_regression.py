from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from lbqa_analysis import analyze_beam_plane, fit_zscan
from lbqa_contracts.models import OpticalCalibration
from PIL import Image

FIXTURE_DIR = Path(__file__).parent / "synthetic_zscan_40x"


def test_synthetic_golden_images_regress_d4sigma_and_divergence() -> None:
    expected = json.loads((FIXTURE_DIR / "expected_results.json").read_text(encoding="utf-8"))
    calibration = OpticalCalibration(**expected["calibration"])
    tolerances = expected["tolerances"]
    plane_results = []

    for plane in expected["planes"]:
        image = np.asarray(Image.open(FIXTURE_DIR / plane["image_path"]))
        result = analyze_beam_plane(
            image=image,
            calibration=calibration,
            z_actual_mm=float(plane["z_actual_mm"]),
            quality_limits=expected["quality_limits"],
        )
        plane_results.append(result)

        assert result.valid, result.invalid_reason
        assert result.d4sigma_x_um == pytest.approx(
            plane["expected_d4sigma_x_um"],
            abs=tolerances["d4sigma_abs_um"],
        )
        assert result.d4sigma_y_um == pytest.approx(
            plane["expected_d4sigma_y_um"],
            abs=tolerances["d4sigma_abs_um"],
        )
        assert result.centroid_x_um == pytest.approx(
            plane["expected_centroid_x_um"],
            abs=tolerances["centroid_abs_um"],
        )
        assert result.centroid_y_um == pytest.approx(
            plane["expected_centroid_y_um"],
            abs=tolerances["centroid_abs_um"],
        )

    zscan = fit_zscan(plane_results)
    zscan_expected = expected["expected_zscan"]
    synthetic_truth = expected["synthetic_truth"]

    assert zscan.valid_points_count == zscan_expected["valid_points_count"]
    assert zscan.full_angle_x_mrad == pytest.approx(
        zscan_expected["full_angle_x_mrad"],
        abs=tolerances["full_angle_abs_mrad"],
    )
    assert zscan.full_angle_y_mrad == pytest.approx(
        zscan_expected["full_angle_y_mrad"],
        abs=tolerances["full_angle_abs_mrad"],
    )
    assert zscan.full_angle_x_mrad == pytest.approx(
        synthetic_truth["full_angle_x_mrad"],
        abs=tolerances["full_angle_abs_mrad"],
    )
    assert zscan.full_angle_y_mrad == pytest.approx(
        synthetic_truth["full_angle_y_mrad"],
        abs=tolerances["full_angle_abs_mrad"],
    )
    assert zscan.half_angle_x_mrad == pytest.approx(zscan.full_angle_x_mrad / 2.0)
    assert zscan.half_angle_y_mrad == pytest.approx(zscan.full_angle_y_mrad / 2.0)
    assert zscan.fit_r2_x is not None and zscan.fit_r2_x >= tolerances["fit_r2_min"]
    assert zscan.fit_r2_y is not None and zscan.fit_r2_y >= tolerances["fit_r2_min"]
