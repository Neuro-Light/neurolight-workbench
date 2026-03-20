from pathlib import Path

import numpy as np
from PIL import Image

from utils.file_handler import ImageStackHandler


def _save_image(path: Path, array: np.ndarray) -> None:
    Image.fromarray(array).save(path)


def test_load_image_stack_collects_tifs(tmp_path):
    handler = ImageStackHandler()
    tif1 = tmp_path / "frame1.TIF"
    tif2 = tmp_path / "frame2.tiff"
    other = tmp_path / "ignore.png"
    _save_image(tif1, np.zeros((4, 4), dtype=np.uint8))
    _save_image(tif2, np.ones((4, 4), dtype=np.uint8))
    Image.fromarray(np.ones((4, 4), dtype=np.uint8)).save(other)

    files = handler.load_image_stack(tmp_path)

    assert files == [str(tif1), str(tif2)]
    assert handler.get_image_count() == 2


def test_validate_tif_files_requires_tif_extension():
    handler = ImageStackHandler()
    assert handler.validate_tif_files(["a.tif", "b.TIFF"]) is True
    assert handler.validate_tif_files(["a.tif", "b.png"]) is False


def test_get_all_frames_as_array_handles_grayscale_and_color(tmp_path):
    handler = ImageStackHandler()
    gray = tmp_path / "gray.tif"
    color = tmp_path / "color.tif"
    gray_pixels = np.array([[0, 50], [100, 150]], dtype=np.uint8)
    color_pixels = np.array(
        [
            [[10, 20, 30], [40, 50, 60]],
            [[70, 80, 90], [100, 110, 120]],
        ],
        dtype=np.uint8,
    )
    _save_image(gray, gray_pixels)
    _save_image(color, color_pixels)
    handler.files = [str(gray), str(color)]

    frames = handler.get_all_frames_as_array()

    assert frames.shape == (2, 2, 2)
    assert np.array_equal(frames[0], gray_pixels)
    expected_color_gray = color_pixels.mean(axis=2)
    assert np.allclose(frames[1], expected_color_gray)
