"""Acquisition workflows for calibrated frame capture."""

from lbqa_acquisition.acquisition_service import (
    AcquisitionService,
    AcquisitionSettings,
    CorrectedFrame,
)
from lbqa_acquisition.dark_frame import (
    DarkFrame,
    acquire_dark_frame,
    apply_dark_frame,
    validate_dark_frame_match,
)
from lbqa_acquisition.exposure import AutoExposureResult, ExposureLimits, auto_exposure
from lbqa_acquisition.frame_capture import capture_average, capture_frames
from lbqa_acquisition.roi import RegionPx, RoiResult, auto_roi, crop_frame_to_roi
from lbqa_acquisition.validation import evaluate_frame_validity

__all__ = [
    "AcquisitionService",
    "AcquisitionSettings",
    "AutoExposureResult",
    "CorrectedFrame",
    "DarkFrame",
    "ExposureLimits",
    "RegionPx",
    "RoiResult",
    "acquire_dark_frame",
    "apply_dark_frame",
    "auto_exposure",
    "auto_roi",
    "capture_average",
    "capture_frames",
    "crop_frame_to_roi",
    "evaluate_frame_validity",
    "validate_dark_frame_match",
]
