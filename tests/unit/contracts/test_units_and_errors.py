from lbqa_contracts.errors import ERROR_CODE_VALUES, ErrorCode, LBQAError
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
from lbqa_contracts.validators import invalid_unit_field_names


def test_contract_model_field_names_follow_unit_rules() -> None:
    model_types = [
        CameraInfo,
        OpticalCalibration,
        StageStatus,
        ImageFrame,
        FrameQuality,
        BeamPlaneResult,
        ZScanResult,
        Recipe,
        RunResult,
    ]

    invalid_names = {
        model_type.__name__: invalid_unit_field_names(model_type)
        for model_type in model_types
        if invalid_unit_field_names(model_type)
    }

    assert invalid_names == {}


def test_error_codes_are_stable_strings() -> None:
    assert ErrorCode.E_DEVICE_CAMERA_TIMEOUT.value in ERROR_CODE_VALUES
    assert ErrorCode.E_ANALYSIS_ZSCAN_FIT_FAILED.value == "E_ANALYSIS_ZSCAN_FIT_FAILED"


def test_lbqa_error_carries_error_code() -> None:
    error = LBQAError(ErrorCode.E_USER_ABORT, "operator stopped run")

    assert error.error_code == ErrorCode.E_USER_ABORT
    assert "E_USER_ABORT" in str(error)
