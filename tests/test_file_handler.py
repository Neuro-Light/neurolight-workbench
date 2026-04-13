"""Tests for ``ImageStackHandler`` (load paths, TIFF IO, experiment association)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile

from core.experiment_manager import Experiment
from utils.file_handler import ImageStackHandler, _extract_valid_time


def test_load_stack_from_list_filters_non_tiff(tmp_path: Path) -> None:
    a = tmp_path / "a.tif"
    b = tmp_path / "b.png"
    a.write_bytes(b"")
    b.write_bytes(b"")
    h = ImageStackHandler()
    h.load_image_stack([str(a), str(b)])
    assert h.files == [str(a)]


def test_load_stack_from_dir_sorted_case_insensitive_suffix(tmp_path: Path) -> None:
    (tmp_path / "z.TIF").write_bytes(b"")
    (tmp_path / "a.tiff").write_bytes(b"")
    (tmp_path / "note.txt").write_text("x", encoding="utf-8")
    h = ImageStackHandler()
    h.load_image_stack(str(tmp_path))
    assert [Path(p).name.lower() for p in h.files] == ["a.tiff", "z.tif"]


def test_validate_tif_files() -> None:
    h = ImageStackHandler()
    assert h.validate_tif_files(["/x/a.tif", "b.TIFF"]) is True
    assert h.validate_tif_files(["/x/a.tif", "b.jpg"]) is False


def test_get_image_at_index_reads_tiff(tmp_path: Path) -> None:
    path = tmp_path / "f.tif"
    arr = np.arange(12, dtype=np.uint16).reshape(3, 4)
    tifffile.imwrite(path, arr)
    h = ImageStackHandler()
    h.files = [str(path)]
    got = h.get_image_at_index(0)
    np.testing.assert_array_equal(got, arr)


def test_get_image_at_index_out_of_range() -> None:
    h = ImageStackHandler()
    h.files = ["/dev/null"]
    with pytest.raises(IndexError, match="out of range"):
        h.get_image_at_index(1)


def test_get_all_frames_as_array_empty() -> None:
    h = ImageStackHandler()
    assert h.get_all_frames_as_array() is None


def test_get_all_frames_as_array_stack(tmp_path: Path) -> None:
    p1 = tmp_path / "1.tif"
    p2 = tmp_path / "2.tif"
    tifffile.imwrite(p1, np.ones((2, 2), dtype=np.uint8))
    tifffile.imwrite(p2, np.zeros((2, 2), dtype=np.uint8))
    h = ImageStackHandler()
    h.load_image_stack([str(p2), str(p1)])
    stack = h.get_all_frames_as_array()
    assert stack is not None
    assert stack.shape == (2, 2, 2)


def test_associate_with_experiment_updates_metadata() -> None:
    h = ImageStackHandler()
    h.files = ["/data/stack/frame.tif"]
    exp = Experiment(name="E")
    h.associate_with_experiment(exp)
    assert exp.image_count == 1
    assert exp.image_stack_path == "/data/stack"
    assert exp.image_stack_files == ["/data/stack/frame.tif"]


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("2024:01:02 09:10:11", "09:10:11"),
        ("09:10", "09:10"),
        (b"2024:01:02 21:59:00", "21:59:00"),
        ("invalid", None),
        ("2024:01:02", None),
        ("99:99:99", None),
        (None, None),
    ],
)
def test_extract_valid_time_normalizes_and_validates(raw, expected) -> None:
    assert _extract_valid_time(raw) == expected
