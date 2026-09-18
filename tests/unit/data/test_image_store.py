from __future__ import annotations

import numpy as np
import pytest
from lbqa_data import DataStoreOverwriteError, ImageStore


def test_image_store_saves_raw_tiff_and_refuses_overwrite(tmp_path) -> None:
    store = ImageStore(tmp_path / "run")
    image = np.arange(16, dtype=np.uint16).reshape(4, 4)

    path = store.save_raw_tiff(image, z_actual_mm=0.0)

    assert path.name == "z_000000um_raw.tiff"
    assert path.exists()
    with pytest.raises(DataStoreOverwriteError):
        store.save_raw_tiff(image, z_actual_mm=0.0)


def test_image_store_saves_corrected_npy_with_frame_index(tmp_path) -> None:
    store = ImageStore(tmp_path / "run")
    corrected = np.arange(9, dtype=np.float32).reshape(3, 3)

    path = store.save_corrected_npy(corrected, frame_index_count=4)

    assert path.name == "frame_000004_corrected.npy"
    assert np.load(path).tolist() == corrected.tolist()


def test_image_store_saves_single_dark_frame(tmp_path) -> None:
    store = ImageStore(tmp_path / "run")
    dark = np.zeros((4, 4), dtype=np.uint16)

    path = store.save_dark_frame(dark)

    assert path.name == "dark_frame.tiff"
    with pytest.raises(DataStoreOverwriteError):
        store.save_dark_frame(dark)
