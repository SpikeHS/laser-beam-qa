import numpy as np
from lbqa_contracts.errors import ErrorCode
from lbqa_contracts.models import (
    BeamPlaneResult,
    CameraInfo,
    FrameQuality,
    ImageFrame,
    OpticalCalibration,
    Recipe,
    RunResult,
    StageStatus,
    ZScanResult,
)


def make_plane_result() -> BeamPlaneResult:
    return BeamPlaneResult(
        z_actual_mm=0.0,
        centroid_x_um=1.0,
        centroid_y_um=2.0,
        peak_x_um=1.1,
        peak_y_um=2.1,
        d4sigma_x_um=8.0,
        d4sigma_y_um=10.0,
        d4sigma_major_um=10.0,
        d4sigma_minor_um=8.0,
        fwhm_x_um=4.0,
        fwhm_y_um=5.0,
        ellipticity=1.25,
        azimuth_deg=3.0,
        peak_value=3000.0,
        saturation_pixels=0,
        edge_energy_percent=0.5,
        valid=True,
        invalid_reason=None,
    )


def test_camera_info_constructs() -> None:
    model = CameraInfo(
        model="CMOS-1.001-Nano",
        width_px=2048,
        height_px=2048,
        pixel_size_um=5.5,
        bit_depth=12,
        serial_number=None,
    )

    assert model.pixel_size_um == 5.5


def test_optical_calibration_constructs() -> None:
    model = OpticalCalibration(
        magnification=40.0,
        camera_pixel_size_um=5.5,
        effective_pixel_x_um=0.1375,
        effective_pixel_y_um=0.1375,
        field_of_view_x_um=283.0,
        field_of_view_y_um=283.0,
        working_distance_mm=0.6,
        calibration_id="cal-40x",
        calibration_date=None,
    )

    assert model.working_distance_mm == 0.6


def test_stage_status_constructs() -> None:
    model = StageStatus(
        connected=True,
        homed=True,
        enabled=True,
        moving=False,
        position_mm=0.0,
        error_code=None,
    )

    assert model.homed


def test_image_frame_constructs() -> None:
    image = np.zeros((4, 6), dtype=np.uint16)

    model = ImageFrame(
        image=image,
        timestamp_iso="2026-06-17T17:00:00+08:00",
        exposure_us=2000.0,
        gain=0.0,
        width_px=6,
        height_px=4,
        z_actual_mm=0.1,
        metadata={"mode": "simulated"},
    )

    assert model.image.shape == (4, 6)


def test_frame_quality_constructs() -> None:
    model = FrameQuality(
        saturated=False,
        saturation_pixels=0,
        peak_value=3000.0,
        mean_value=100.0,
        edge_energy_percent=1.0,
        low_signal=False,
        valid=True,
        invalid_reason=None,
    )

    assert model.valid


def test_beam_plane_result_constructs() -> None:
    model = make_plane_result()

    assert model.d4sigma_major_um == 10.0


def test_z_scan_result_constructs() -> None:
    plane = make_plane_result()

    model = ZScanResult(
        points=[plane],
        full_angle_x_mrad=2.0,
        full_angle_y_mrad=3.0,
        half_angle_x_mrad=1.0,
        half_angle_y_mrad=1.5,
        waist_z_x_mm=0.0,
        waist_z_y_mm=0.0,
        waist_diameter_x_um=8.0,
        waist_diameter_y_um=10.0,
        fit_r2_x=0.99,
        fit_r2_y=0.98,
        valid_points_count=1,
        judgement="pass",
        invalid_reason=None,
    )

    assert model.half_angle_y_mrad == 1.5


def test_recipe_constructs() -> None:
    model = Recipe(
        recipe_id="recipe-40x-zscan",
        operator_mode="simulated",
        camera={"exposure_us": 2000.0},
        optics={"magnification": 40.0},
        stage={"z_min_mm": -0.5, "z_max_mm": 0.5},
        z_scan={"z_targets_mm": [-0.1, 0.0, 0.1]},
        analysis={"primary_width_method": "D4Sigma"},
        limits={"max_edge_energy_percent": 5.0},
    )

    assert model.operator_mode == "simulated"


def test_run_result_constructs() -> None:
    plane = make_plane_result()
    zscan = ZScanResult(
        points=[plane],
        full_angle_x_mrad=2.0,
        full_angle_y_mrad=3.0,
        half_angle_x_mrad=1.0,
        half_angle_y_mrad=1.5,
        waist_z_x_mm=0.0,
        waist_z_y_mm=0.0,
        waist_diameter_x_um=8.0,
        waist_diameter_y_um=10.0,
        fit_r2_x=0.99,
        fit_r2_y=0.98,
        valid_points_count=1,
        judgement="pass",
        invalid_reason=None,
    )

    model = RunResult(
        run_id="run-001",
        sample_id="sample-001",
        timestamp_iso="2026-06-17T17:00:00+08:00",
        recipe_id="recipe-40x-zscan",
        calibration_id="cal-40x",
        plane_result=plane,
        zscan_result=zscan,
        judgement="warning",
        errors=[ErrorCode.E_IMAGE_EDGE_CLIPPED.value],
        output_dir="runs/run-001",
    )

    assert model.errors == [ErrorCode.E_IMAGE_EDGE_CLIPPED.value]



def test_z_scan_result_supports_invalid_fit_coefficients_and_outliers() -> None:
    plane = BeamPlaneResult(
        z_actual_mm=0.2,
        centroid_x_um=1.0,
        centroid_y_um=2.0,
        peak_x_um=None,
        peak_y_um=None,
        d4sigma_x_um=8.5,
        d4sigma_y_um=10.5,
        d4sigma_major_um=10.5,
        d4sigma_minor_um=8.5,
        fwhm_x_um=None,
        fwhm_y_um=None,
        ellipticity=1.24,
        azimuth_deg=4.0,
        peak_value=2800.0,
        saturation_pixels=0,
        edge_energy_percent=0.6,
        valid=True,
        invalid_reason=None,
        outlier=True,
        outlier_reason="robust_fit_rejected",
    )

    model = ZScanResult(
        points=[plane],
        full_angle_x_mrad=None,
        full_angle_y_mrad=None,
        half_angle_x_mrad=None,
        half_angle_y_mrad=None,
        waist_z_x_mm=None,
        waist_z_y_mm=None,
        waist_diameter_x_um=None,
        waist_diameter_y_um=None,
        fit_r2_x=None,
        fit_r2_y=None,
        valid_points_count=0,
        judgement="invalid",
        invalid_reason="too_few_valid_points_after_outlier_rejection",
        fit_a_x_um2_per_mm2=4.0,
        fit_b_x_um2_per_mm=-0.2,
        fit_c_x_um2=64.0,
        fit_a_y_um2_per_mm2=9.0,
        fit_b_y_um2_per_mm=0.3,
        fit_c_y_um2=100.0,
        outlier_indices=[0],
    )

    assert model.judgement == "invalid"
    assert model.fit_a_x_um2_per_mm2 == 4.0
    assert model.outlier_indices == [0]
    assert model.points[0].outlier
