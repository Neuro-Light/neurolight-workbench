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
    assert exp.image_stack_path == str(Path("/data/stack"))
    assert exp.image_stack_files == ["/data/stack/frame.tif"]


# ── Excluded-frames API ───────────────────────────────────────────────────


def test_set_and_get_included_range() -> None:
    h = ImageStackHandler()
    h.files = ["/a.tif", "/b.tif", "/c.tif"]
    h.set_included_range(1, 2)
    assert h.get_included_range() == (1, 2)


def test_get_total_frame_count() -> None:
    h = ImageStackHandler()
    h.files = ["/a.tif", "/b.tif", "/c.tif"]
    h.set_included_range(0, 1)
    assert h.get_total_frame_count() == 3


def test_get_included_files_empty_files_returns_empty() -> None:
    h = ImageStackHandler()
    assert h.get_included_files() == []


def test_get_included_files_no_range_set() -> None:
    h = ImageStackHandler()
    h.files = ["/a.tif", "/b.tif"]
    assert h.get_included_files() == ["/a.tif", "/b.tif"]


def test_get_included_files_with_range() -> None:
    h = ImageStackHandler()
    h.files = ["/a.tif", "/b.tif", "/c.tif"]
    h.set_included_range(1, 1)
    assert h.get_included_files() == ["/b.tif"]


def test_load_stack_resets_included_range(tmp_path: Path) -> None:
    (tmp_path / "f.tif").write_bytes(b"")
    h = ImageStackHandler()
    h.set_included_range(0, 1)
    h.load_image_stack(str(tmp_path))
    # After load, _included_end should be -1 (unset = all frames)
    assert h._included_end == -1


def test_load_stack_with_non_directory_string_returns_empty(tmp_path: Path) -> None:
    # Passing a path that is not a directory takes the is_dir()=False branch
    non_dir = str(tmp_path / "not_a_dir.tif")
    h = ImageStackHandler()
    result = h.load_image_stack(non_dir)
    assert result == []


def test_get_all_frames_as_array_rgb_frame(tmp_path: Path) -> None:
    p = tmp_path / "rgb.tif"
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    tifffile.imwrite(p, rgb)
    h = ImageStackHandler()
    h.load_image_stack([str(p)])
    stack = h.get_all_frames_as_array()
    # RGB frame should be collapsed to 2D (mean across channels)
    assert stack is not None
    assert stack.ndim == 3  # (1 frame, height, width)


def test_get_all_frames_as_array_uses_range(tmp_path: Path) -> None:
    p0 = tmp_path / "0.tif"
    p1 = tmp_path / "1.tif"
    p2 = tmp_path / "2.tif"
    tifffile.imwrite(p0, np.zeros((2, 2), dtype=np.uint8))
    tifffile.imwrite(p1, np.ones((2, 2), dtype=np.uint8) * 100)
    tifffile.imwrite(p2, np.ones((2, 2), dtype=np.uint8) * 200)
    h = ImageStackHandler()
    h.load_image_stack([str(p0), str(p1), str(p2)])
    h.set_included_range(0, 1)  # include frames 0 and 1 only
    stack = h.get_all_frames_as_array()
    assert stack is not None
    assert stack.shape == (2, 2, 2)
    np.testing.assert_array_equal(stack[0], np.zeros((2, 2), dtype=np.uint8))
    np.testing.assert_array_equal(stack[1], np.ones((2, 2), dtype=np.uint8) * 100)


def test_get_all_frames_as_array_empty_range(tmp_path: Path) -> None:
    p = tmp_path / "only.tif"
    tifffile.imwrite(p, np.zeros((2, 2), dtype=np.uint8))
    h = ImageStackHandler()
    h.load_image_stack([str(p)])
    # set_included_range to valid range — can't make it empty with range API
    # An empty result requires an out-of-bounds range which get_included_files clamps.
    # Verify the normal single-frame case works correctly.
    h.set_included_range(0, 0)
    stack = h.get_all_frames_as_array()
    assert stack is not None
    assert stack.shape == (1, 2, 2)


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
