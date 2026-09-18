import numpy as np
from lbqa_contracts.models import CameraInfo, ImageFrame, StageStatus
from lbqa_contracts.ports import BeamProfilerPort, InterlockPort, StagePort


class FakeStage:
    def __init__(self) -> None:
        self.position_mm = 0.0
        self.homed = False

    def connect(self) -> None:
        pass

    def disconnect(self) -> None:
        pass

    def enable(self) -> None:
        pass

    def disable(self) -> None:
        pass

    def home(self) -> None:
        self.homed = True

    def is_homed(self) -> bool:
        return self.homed

    def move_abs_mm(self, position_mm: float) -> None:
        self.position_mm = position_mm

    def move_rel_mm(self, delta_mm: float) -> None:
        self.position_mm += delta_mm

    def get_position_mm(self) -> float:
        return self.position_mm

    def get_status(self) -> StageStatus:
        return StageStatus(True, self.homed, True, False, self.position_mm)

    def stop(self) -> None:
        pass

    def emergency_stop(self) -> None:
        pass

    def set_soft_limits(self, min_mm: float, max_mm: float) -> None:
        pass

    def wait_in_position(self, timeout_s: float, tolerance_um: float) -> bool:
        return True


class FakeBeamProfiler:
    def __init__(self) -> None:
        self.last_image: ImageFrame | None = None

    def connect(self) -> None:
        pass

    def disconnect(self) -> None:
        pass

    def get_camera_info(self) -> CameraInfo:
        return CameraInfo("CMOS-1.001-Nano", 4, 4, 5.5, 12)

    def set_exposure_us(self, exposure_us: float) -> None:
        pass

    def set_gain(self, gain: float) -> None:
        pass

    def set_aoi(self, x_px: int, y_px: int, width_px: int, height_px: int) -> None:
        pass

    def capture_single(self) -> ImageFrame:
        self.last_image = ImageFrame(
            image=np.zeros((4, 4), dtype=np.uint16),
            timestamp_iso="2026-06-17T17:00:00+08:00",
            exposure_us=2000.0,
            gain=0.0,
            width_px=4,
            height_px=4,
            z_actual_mm=None,
            metadata={},
        )
        return self.last_image

    def capture_average(self, frame_count: int) -> ImageFrame:
        return self.capture_single()

    def get_last_image(self) -> ImageFrame | None:
        return self.last_image

    def get_status(self) -> dict[str, object]:
        return {"connected": True}


class FakeInterlock:
    def is_safe(self) -> bool:
        return True

    def get_status(self) -> dict[str, object]:
        return {"safe": True}


def test_stage_protocol_can_be_implemented_by_simulated_class() -> None:
    assert isinstance(FakeStage(), StagePort)


def test_beam_profiler_protocol_can_be_implemented_by_simulated_class() -> None:
    assert isinstance(FakeBeamProfiler(), BeamProfilerPort)


def test_interlock_protocol_can_be_implemented_by_simulated_class() -> None:
    assert isinstance(FakeInterlock(), InterlockPort)
