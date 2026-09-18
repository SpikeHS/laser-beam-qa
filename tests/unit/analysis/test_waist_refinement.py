from __future__ import annotations

import math

import pytest
from lbqa_analysis.waist_refinement import (
    minimum_area_from_zscan,
    summarize_waist_refinement,
)
from lbqa_contracts.models import BeamPlaneResult, ZScanResult


def test_minimum_area_keeps_independent_x_and_y_waists() -> None:
    points = [
        _plane(-0.004, 10.1, 20.3),
        _plane(0.000, 10.0, 20.1),
        _plane(0.005, 10.2, 20.0),
    ]
    result = ZScanResult(
        points=points,
        full_angle_x_mrad=4.0,
        full_angle_y_mrad=6.0,
        half_angle_x_mrad=2.0,
        half_angle_y_mrad=3.0,
        waist_z_x_mm=-0.002,
        waist_z_y_mm=0.003,
        waist_diameter_x_um=10.0,
        waist_diameter_y_um=20.0,
        fit_r2_x=0.99,
        fit_r2_y=0.98,
        valid_points_count=3,
        judgement="pass",
    )

    minimum = minimum_area_from_zscan(result, plane_results=points)

    assert minimum["valid"] is True
    assert -0.002 <= minimum["z_mm"] <= 0.003
    assert minimum["diameter_x_um"] >= 10.0
    assert minimum["diameter_y_um"] >= 20.0
    assert minimum["area_um2"] == pytest.approx(
        math.pi * minimum["diameter_x_um"] * minimum["diameter_y_um"] / 4.0
    )
    assert minimum["nearest_plane_index_count"] in {0, 1, 2}


def test_refinement_reports_separate_observed_axis_minima() -> None:
    points = [
        _plane(-0.001, 9.8, 20.5),
        _plane(0.000, 10.0, 20.1),
        _plane(0.001, 10.4, 19.7),
    ]
    records = [
        {
            "point": point,
            "plane_index_count": index + 6,
            "refinement_index_count": index,
            "window_index_count": 0,
            "z_command_mm": point.z_actual_mm,
        }
        for index, point in enumerate(points)
    ]

    summary = summarize_waist_refinement(
        records,
        requested_step_um=0.5,
        requested_move_count=2,
        windows=[{"phase": "waist_refine_xy"}],
    )

    assert summary["valid"] is True
    assert summary["x_minimum"]["plane_index_count"] == 6
    assert summary["x_minimum"]["diameter_um"] == pytest.approx(9.8)
    assert summary["y_minimum"]["plane_index_count"] == 8
    assert summary["y_minimum"]["diameter_um"] == pytest.approx(19.7)
    assert summary["x_minimum"]["position_trusted_for_fit"] is False
    assert summary["position_policy"] == "excluded_from_main_divergence_fit"


def _plane(z_mm: float, width_x_um: float, width_y_um: float) -> BeamPlaneResult:
    return BeamPlaneResult(
        z_actual_mm=z_mm,
        centroid_x_um=0.0,
        centroid_y_um=0.0,
        peak_x_um=0.0,
        peak_y_um=0.0,
        d4sigma_x_um=width_x_um,
        d4sigma_y_um=width_y_um,
        d4sigma_major_um=max(width_x_um, width_y_um),
        d4sigma_minor_um=min(width_x_um, width_y_um),
        fwhm_x_um=None,
        fwhm_y_um=None,
        ellipticity=max(width_x_um, width_y_um) / min(width_x_um, width_y_um),
        azimuth_deg=0.0,
        peak_value=1000.0,
        saturation_pixels=0,
        edge_energy_percent=0.0,
        valid=True,
    )
