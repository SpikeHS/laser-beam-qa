import numpy as np
import pytest
from lbqa_acquisition import (
    AcquisitionService,
    AcquisitionSettings,
    ExposureLimits,
    acquire_dark_frame,
    apply_dark_frame,
    auto_exposure,
    capture_average,
    evaluate_frame_validity,
)
from lbqa_acquisition.roi import RegionPx, configure_aoi, current_aoi
from lbqa_contracts.errors import ErrorCode, LBQAError
from lbqa_contracts.models import ImageFrame
from lbqa_devices.profiler.simulated_profiler import SimulatedProfiler


def test_acquisition_without_aoi_preserves_existing_simulator_settings(monkeypatch) -> None:
    profiler = SimulatedProfiler(width_px=64, height_px=64, noise_std_count=0.0)
    profiler.connect()
    profiler.set_aoi(10, 12, 32, 30)
    before = profiler.get_status()

    def no_write(*args):
        raise AssertionError("unspecified settings must not cause a device write")

    for name in ("set_aoi", "set_exposure_us", "set_gain"):
        monkeypatch.setattr(SimulatedProfiler, name, no_write)
    service = AcquisitionService(profiler, AcquisitionSettings(aoi=None, auto_roi_enabled=False))
    frame = service.capture_plane(0.0)
    assert frame.raw_frame.image.shape == (30, 32)
    assert profiler.get_status() == before
    assert current_aoi(profiler) == RegionPx(10, 12, 32, 30)


def test_unspecified_aoi_handles_unknown_optional_hardware_status(monkeypatch) -> None:
    profiler = SimulatedProfiler(width_px=64, height_px=48, noise_std_count=0.0)
    profiler.connect()
    monkeypatch.setattr(SimulatedProfiler, "get_status", lambda self: {
        "aoi_x_px": None, "aoi_y_px": None, "aoi_width_px": None, "aoi_height_px": None,
    })
    monkeypatch.setattr(SimulatedProfiler, "set_aoi", lambda *args: pytest.fail("AOI write"))
    assert configure_aoi(profiler, None) == RegionPx(0, 0, 64, 48)
    raw = profiler.capture_single()
    assert current_aoi(profiler, raw) == RegionPx(0, 0, 64, 48)


def test_explicit_aoi_still_writes_and_validates_bounds() -> None:
    profiler = SimulatedProfiler(width_px=64, height_px=48, noise_std_count=0.0)
    profiler.connect()
    requested = RegionPx(10, 12, 32, 30)
    assert configure_aoi(profiler, requested) == requested
    assert profiler.capture_single().image.shape == (30, 32)
    with pytest.raises(LBQAError, match="exceeds image bounds"):
        configure_aoi(profiler, RegionPx(100, 100, 2, 2))


def test_dark_frame_matches_same_exposure_gain_and_aoi() -> None:
    profiler = SimulatedProfiler(width_px=48, height_px=40, noise_std_count=0.0)
    profiler.connect()

    dark_frame = acquire_dark_frame(profiler, frame_count=3)
    raw_frame = profiler.capture_single()
    corrected = apply_dark_frame(raw_frame, dark_frame)

    assert dark_frame.exposure_us == raw_frame.exposure_us
    assert dark_frame.gain == raw_frame.gain
    assert dark_frame.aoi.width_px == raw_frame.width_px
    assert corrected.image.dtype == np.float64
    assert corrected.metadata["dark_subtracted"] is True


def test_dark_frame_rejects_exposure_mismatch() -> None:
    profiler = SimulatedProfiler(width_px=48, height_px=40, noise_std_count=0.0)
    profiler.connect()

    dark_frame = acquire_dark_frame(profiler, frame_count=2)
    profiler.set_exposure_us(dark_frame.exposure_us * 1.5)
    raw_frame = profiler.capture_single()

    with pytest.raises(LBQAError, match="dark frame mismatch: exposure_us"):
        apply_dark_frame(raw_frame, dark_frame)


def test_auto_exposure_converges_with_simulated_profiler() -> None:
    profiler = SimulatedProfiler(
        width_px=64,
        height_px=64,
        exposure_us=250.0,
        reference_exposure_us=2000.0,
        noise_std_count=0.0,
    )
    profiler.connect()

    result = auto_exposure(
        profiler,
        target_peak_percent=65.0,
        max_peak_percent=85.0,
        limits=ExposureLimits(min_exposure_us=50.0, max_exposure_us=10_000.0),
        tolerance_percent=3.0,
    )
    frame = profiler.capture_single()
    peak_percent = 100.0 * float(np.max(frame.image)) / float((1 << profiler.bit_depth) - 1)

    assert result.locked is True
    assert result.exposure_us == pytest.approx(profiler.exposure_us)
    assert peak_percent <= 85.0
    assert peak_percent == pytest.approx(65.0, abs=3.5)


def test_capture_average_uses_float_to_avoid_uint_overflow() -> None:
    profiler = SimulatedProfiler(
        width_px=32,
        height_px=32,
        noise_std_count=1.0,
        rng_seed=42,
    )
    profiler.connect()

    averaged = capture_average(profiler, frame_count=4)

    assert averaged.image.dtype in {np.dtype(np.float32), np.dtype(np.float64)}
    assert averaged.image.shape == (32, 32)
    assert averaged.metadata["averaged"] is True
    assert averaged.metadata["averaged_frame_count"] == 4


def test_capture_average_prefers_profiler_native_average() -> None:
    class NativeAverageProfiler:
        def __init__(self) -> None:
            self.native_calls: list[int] = []

        def capture_single(self) -> ImageFrame:
            raise AssertionError("capture_single should not be used for native averaging.")

        def capture_average(self, frame_count: int) -> ImageFrame:
            self.native_calls.append(frame_count)
            return ImageFrame(
                image=np.ones((8, 8), dtype=np.float32),
                timestamp_iso="2026-07-09T00:00:00+00:00",
                exposure_us=100.0,
                gain=1.0,
                width_px=8,
                height_px=8,
                z_actual_mm=0.0,
                metadata={"rayci_recorded_frame_count": frame_count},
            )

    profiler = NativeAverageProfiler()

    averaged = capture_average(profiler, frame_count=16)  # type: ignore[arg-type]

    assert profiler.native_calls == [16]
    assert averaged.metadata["averaged"] is True
    assert averaged.metadata["averaged_frame_count"] == 16
    assert averaged.metadata["rayci_recorded_frame_count"] == 16


def test_saturation_detection_marks_frame_invalid() -> None:
    image = np.zeros((32, 32), dtype=np.uint16)
    image[16, 16] = 1023

    quality = evaluate_frame_validity(
        image,
        bit_depth=10,
        saturation_threshold_percent=99.0,
    )

    assert quality.valid is False
    assert quality.saturated is True
    assert quality.invalid_reason == ErrorCode.E_IMAGE_SATURATED.value


def test_boundary_detection_marks_near_edge_spot_invalid() -> None:
    y_px, x_px = np.indices((64, 64), dtype=np.float64)
    image = 600.0 * np.exp(-0.5 * (((x_px - 2.0) / 5.0) ** 2 + ((y_px - 32.0) / 6.0) ** 2))

    quality = evaluate_frame_validity(
        image,
        bit_depth=10,
        edge_margin_px=4,
        edge_energy_limit_percent=10.0,
        spot_edge_margin_px=6,
        spot_threshold_percent=10.0,
    )

    assert quality.valid is False
    assert quality.invalid_reason == ErrorCode.E_IMAGE_EDGE_CLIPPED.value


def test_acquisition_service_returns_corrected_frame_with_roi_metadata() -> None:
    profiler = SimulatedProfiler(width_px=96, height_px=96, noise_std_count=0.0)
    profiler.connect()
    settings = AcquisitionSettings(
        auto_roi_enabled=True,
        roi_padding_px=10,
        edge_energy_limit_percent=20.0,
        spot_edge_margin_px=2,
    )
    service = AcquisitionService(profiler, settings)

    result = service.capture_plane_average(z_actual_mm=0.125, frame_count=2)

    assert result.z_actual_mm == 0.125
    assert result.width_px <= 96
    assert result.height_px <= 96
    assert result.metadata["roi_mode"] == "auto"
    assert "roi_x_px" in result.metadata
