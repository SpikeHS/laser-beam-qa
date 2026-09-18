"""Image and processed-array persistence for one test run."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np
from lbqa_contracts.errors import ErrorCode
from PIL import Image

from lbqa_data.data_exceptions import raise_data_error, raise_overwrite_error


class ImageStore:
    """Save raw TIFF images and processed NPY arrays without overwriting data."""

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.images_dir = self.run_dir / "images"
        self.processed_dir = self.run_dir / "processed"
        self.working_dir = self.run_dir / ".working"

    def save_raw_tiff(
        self,
        image: Any,
        *,
        z_actual_mm: float | None = None,
        frame_index_count: int | None = None,
    ) -> Path:
        """Save a raw frame as TIFF with z position or frame index in the filename."""

        stem = _measurement_stem(z_actual_mm=z_actual_mm, frame_index_count=frame_index_count)
        path = self.images_dir / f"{stem}_raw.tiff"
        _ensure_new_file(path)
        Image.fromarray(np.asarray(image)).save(path)
        return path

    def save_corrected_npy(
        self,
        corrected_image: Any,
        *,
        z_actual_mm: float | None = None,
        frame_index_count: int | None = None,
    ) -> Path:
        """Save a processed/corrected plane as an NPY array."""

        stem = _measurement_stem(z_actual_mm=z_actual_mm, frame_index_count=frame_index_count)
        path = self.processed_dir / f"{stem}_corrected.npy"
        _ensure_new_file(path)
        with path.open("xb") as file:
            np.save(file, np.asarray(corrected_image), allow_pickle=False)
        return path

    def save_dark_frame(self, dark_frame: Any, *, frame_index_count: int | None = None) -> Path:
        """Save the dark frame TIFF; repeated dark-frame saves must use an index."""

        filename = (
            "dark_frame.tiff"
            if frame_index_count is None
            else f"dark_frame_{frame_index_count:06d}.tiff"
        )
        path = self.images_dir / filename
        _ensure_new_file(path)
        Image.fromarray(np.asarray(dark_frame)).save(path)
        return path

    def save_report_source_npy(
        self,
        corrected_image: Any,
        *,
        frame_index_count: int,
    ) -> Path:
        """Save a temporary grayscale source used only while rendering the report."""

        stem = _measurement_stem(frame_index_count=frame_index_count)
        path = self.working_dir / f"{stem}_corrected.npy"
        _ensure_new_file(path)
        with path.open("xb") as file:
            np.save(file, np.asarray(corrected_image), allow_pickle=False)
        return path

    def cleanup_working_files(self) -> None:
        """Remove temporary grayscale arrays after report rendering or cancellation."""

        if self.working_dir.exists():
            shutil.rmtree(self.working_dir)


def _measurement_stem(
    *, z_actual_mm: float | None = None, frame_index_count: int | None = None
) -> str:
    if z_actual_mm is not None:
        return f"z_{_format_z_um_token(z_actual_mm)}"
    if frame_index_count is not None:
        if frame_index_count < 0:
            raise_data_error(
                ErrorCode.E_REPORT_GENERATION_FAILED,
                "frame_index_count must be non-negative.",
            )
        return f"frame_{frame_index_count:06d}"
    raise_data_error(
        ErrorCode.E_REPORT_GENERATION_FAILED,
        "image filenames must include z_actual_mm or frame_index_count.",
    )


def _format_z_um_token(z_actual_mm: float) -> str:
    z_um = int(round(z_actual_mm * 1000.0))
    if z_um < 0:
        return f"m{abs(z_um):06d}um"
    return f"{z_um:06d}um"


def _ensure_new_file(path: Path) -> None:
    if path.exists():
        raise_overwrite_error(path)
    path.parent.mkdir(parents=True, exist_ok=True)


__all__ = ["ImageStore"]
