import lbqa_contracts


def test_package_root_exports_only_current_mvp_contract_surface() -> None:
    public_names = set(lbqa_contracts.__all__)

    expected_names = {
        "BeamPlaneResult",
        "BeamProfilerPort",
        "CameraInfo",
        "ErrorCode",
        "FrameQuality",
        "ImageFrame",
        "InterlockPort",
        "JUDGEMENTS",
        "LBQAError",
        "OpticalCalibration",
        "Recipe",
        "RunResult",
        "StagePort",
        "StageStatus",
        "ZScanResult",
    }
    legacy_names = {
        "BeamFrame",
        "BeamMetrics",
        "Calibration40x",
        "CameraDevice",
        "DivergenceFitResult",
        "ZAxisStage",
        "ZScanPlaneRecord",
    }

    assert public_names == expected_names
    assert public_names.isdisjoint(legacy_names)
