import numpy as np
from lbqa_contracts.models import ImageFrame
from lbqa_devices.profiler.simulated_profiler import SimulatedProfiler


def test_simulated_profiler_capture_single_returns_image_frame() -> None:
    profiler = SimulatedProfiler(width_px=64, height_px=48)
    profiler.connect()
    profiler.set_z_actual_mm(0.0)

    frame = profiler.capture_single()

    assert isinstance(frame, ImageFrame)
    assert frame.image.shape == (48, 64)
    assert frame.width_px == 64
    assert frame.height_px == 48
    assert frame.z_actual_mm == 0.0
    assert frame.metadata["effective_pixel_x_um"] == 0.1375
    assert frame.image.max() <= 1023


def test_simulated_profiler_default_camera_info_matches_cmos_nano_40x() -> None:
    profiler = SimulatedProfiler()

    info = profiler.get_camera_info()
    calibration = profiler.get_optical_calibration()

    assert info.model == "CMOS-1.001-Nano"
    assert info.width_px == 2048
    assert info.height_px == 2048
    assert info.pixel_size_um == 5.5
    assert info.bit_depth == 10
    assert calibration.magnification == 40.0
    assert calibration.effective_pixel_x_um == 0.1375


def test_simulated_profiler_beam_diameter_changes_with_z_position() -> None:
    profiler = SimulatedProfiler(width_px=128, height_px=128, noise_std_count=0.0)
    profiler.connect()

    profiler.set_z_actual_mm(0.0)
    waist_frame = profiler.capture_single()
    profiler.set_z_actual_mm(5.0)
    far_frame = profiler.capture_single()

    assert far_frame.metadata["d4sigma_x_um"] > waist_frame.metadata["d4sigma_x_um"]
    assert far_frame.metadata["d4sigma_y_um"] > waist_frame.metadata["d4sigma_y_um"]
    assert _second_moment_px(far_frame.image) > _second_moment_px(waist_frame.image)


def test_simulated_profiler_capture_average_returns_average_frame() -> None:
    profiler = SimulatedProfiler(width_px=32, height_px=32, noise_std_count=1.0)
    profiler.connect()

    frame = profiler.capture_average(frame_count=3)

    assert frame.image.shape == (32, 32)
    assert frame.image.dtype == np.float32
    assert frame.metadata["averaged"] is True
    assert frame.metadata["averaged_frame_count"] == 3


def _second_moment_px(image: np.ndarray) -> float:
    weights = image.astype(float) - float(image.min())
    y_px, x_px = np.indices(image.shape)
    total = weights.sum()
    centroid_x_px = float((weights * x_px).sum() / total)
    centroid_y_px = float((weights * y_px).sum() / total)
    return float(
        (weights * ((x_px - centroid_x_px) ** 2 + (y_px - centroid_y_px) ** 2)).sum() / total
    )
