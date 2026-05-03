"""Tests for ImageViewer range-based frame culling."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import tifffile
from PySide6.QtWidgets import QApplication

from ui.image_viewer import ImageViewer
from utils.file_handler import ImageStackHandler


@pytest.fixture
def app():
    if not QApplication.instance():
        return QApplication([])
    return QApplication.instance()


def _make_stack(tmp_path: Path, n: int = 5) -> list[str]:
    paths = []
    for i in range(n):
        p = tmp_path / f"frame_{i:03d}.tif"
        tifffile.imwrite(p, np.full((4, 4), i * 10, dtype=np.uint8))
        paths.append(str(p))
    return paths


@pytest.fixture
def viewer(app, tmp_path):
    files = _make_stack(tmp_path, 5)
    handler = ImageStackHandler()
    v = ImageViewer(handler)
    v.set_stack(files)
    return v


# ── Initial state ────────────────────────────────────────────────────────


class TestCullingInitialState:
    def test_range_covers_all_frames_on_load(self, viewer) -> None:
        start, end = viewer.get_cull_range()
        assert start == 0
        assert end == 4

    def test_filter_excluded_off_by_default(self, viewer) -> None:
        assert viewer._filter_excluded is False

    def test_cull_panel_hidden_by_default(self, viewer) -> None:
        assert viewer.cull_controls_panel.isVisible() is False


# ── get / set cull range ─────────────────────────────────────────────────


class TestGetSetCullRange:
    def test_set_and_get_round_trip(self, viewer) -> None:
        viewer.set_cull_range(1, 3)
        assert viewer.get_cull_range() == (1, 3)

    def test_set_clamps_to_valid_range(self, viewer) -> None:
        viewer.set_cull_range(-5, 100)
        start, end = viewer.get_cull_range()
        assert start == 0
        assert end == 4

    def test_set_updates_range_slider(self, viewer) -> None:
        viewer.set_cull_range(2, 4)
        assert viewer._range_slider.get_start() == 2
        assert viewer._range_slider.get_end() == 4

    def test_range_start_cannot_exceed_end(self, viewer) -> None:
        viewer.set_cull_range(3, 1)
        start, end = viewer.get_cull_range()
        assert start <= end


# ── Visible indices / navigation with filter ─────────────────────────────


class TestVisibleIndices:
    def test_all_visible_by_default(self, viewer) -> None:
        assert viewer._visible_indices == [0, 1, 2, 3, 4]

    def test_filter_off_shows_all_even_with_range(self, viewer) -> None:
        viewer.set_cull_range(1, 3)
        assert viewer._visible_indices == [0, 1, 2, 3, 4]

    def test_filter_on_shows_only_included_range(self, viewer) -> None:
        viewer.set_cull_range(1, 3)
        viewer._filter_excluded = True
        assert viewer._visible_indices == [1, 2, 3]

    def test_full_range_filter_on_shows_all(self, viewer) -> None:
        viewer._filter_excluded = True
        assert viewer._visible_indices == [0, 1, 2, 3, 4]


class TestSetFilterExcluded:
    def test_enables_filtering(self, viewer) -> None:
        viewer.set_cull_range(1, 3)
        viewer.set_filter_excluded(True)
        assert viewer._filter_excluded is True
        assert viewer._visible_indices == [1, 2, 3]

    def test_disables_filtering(self, viewer) -> None:
        viewer.set_cull_range(1, 3)
        viewer.set_filter_excluded(True)
        viewer.set_filter_excluded(False)
        assert viewer._visible_indices == [0, 1, 2, 3, 4]

    def test_noop_when_already_set(self, viewer) -> None:
        viewer.set_filter_excluded(False)
        assert viewer._filter_excluded is False

    def test_snaps_to_visible_frame(self, viewer) -> None:
        viewer.index = 0
        viewer.set_cull_range(1, 3)
        viewer.set_filter_excluded(True)
        assert viewer.index in viewer._visible_indices

    def test_slider_range_matches_visible(self, viewer) -> None:
        viewer.set_cull_range(1, 3)
        viewer.set_filter_excluded(True)
        assert viewer.slider.maximum() == 2  # 3 visible frames → slider 0..2


class TestNavigationWithFilter:
    def test_next_image_stays_in_range(self, viewer) -> None:
        viewer.set_cull_range(1, 3)
        viewer.set_filter_excluded(True)
        viewer.index = 1
        viewer.slider.setValue(0)
        viewer.next_image()
        assert viewer.index == 2

    def test_prev_image_stays_in_range(self, viewer) -> None:
        viewer.set_cull_range(1, 3)
        viewer.set_filter_excluded(True)
        viewer.index = 3
        viewer.slider.setValue(2)
        viewer.prev_image()
        assert viewer.index == 2

    def test_on_slider_maps_to_visible_index(self, viewer) -> None:
        viewer.set_cull_range(2, 4)
        viewer.set_filter_excluded(True)
        # visible = [2, 3, 4], slider pos 2 → raw index 4
        viewer._on_slider(2)
        assert viewer.index == 4

    def test_next_at_end_stays(self, viewer) -> None:
        viewer.set_cull_range(0, 2)
        viewer.set_filter_excluded(True)
        viewer.index = 2
        viewer.slider.setValue(2)
        viewer.next_image()
        assert viewer.index == 2

    def test_prev_at_start_stays(self, viewer) -> None:
        viewer.set_filter_excluded(True)
        viewer.index = 0
        viewer.slider.setValue(0)
        viewer.prev_image()
        assert viewer.index == 0


# ── Range slider signal ──────────────────────────────────────────────────


class TestRangeSliderSignal:
    def test_range_changed_emits_frameCullingChanged(self, viewer) -> None:
        received = []
        viewer.frameCullingChanged.connect(lambda s, e: received.append((s, e)))
        viewer._range_slider.range_changed.emit(1, 3)
        assert received == [(1, 3)]
        assert viewer.get_cull_range() == (1, 3)

    def test_frame_preview_jumps_viewer(self, viewer) -> None:
        viewer._range_slider.frame_preview_requested.emit(3)
        assert viewer.index == 3


# ── Reset ────────────────────────────────────────────────────────────────


class TestResetClearsCullRange:
    def test_reset_resets_cull_end_to_unset(self, viewer) -> None:
        viewer.set_cull_range(1, 3)
        viewer.set_filter_excluded(True)
        viewer.reset()
        assert viewer._cull_end == -1
        assert viewer._filter_excluded is False


# ── set_stack edge cases ─────────────────────────────────────────────────


class TestSetStackEdgeCases:
    def test_set_stack_empty_list_sets_cull_end_negative(self, app) -> None:
        handler = ImageStackHandler()
        v = ImageViewer(handler)
        v.set_stack([])
        assert v._cull_end == -1

    def test_set_stack_emits_stackLoaded_for_list(self, app, tmp_path) -> None:
        files = _make_stack(tmp_path, 2)
        handler = ImageStackHandler()
        v = ImageViewer(handler)
        received = []
        v.stackLoaded.connect(received.append)
        v.set_stack(files)
        assert len(received) == 1

    def test_set_stack_emits_stackLoaded_for_directory(self, app, tmp_path) -> None:
        _make_stack(tmp_path, 2)
        handler = ImageStackHandler()
        v = ImageViewer(handler)
        received = []
        v.stackLoaded.connect(received.append)
        v.set_stack(str(tmp_path))
        assert len(received) == 1

    def test_set_stack_empty_list_no_stackLoaded(self, app) -> None:
        handler = ImageStackHandler()
        v = ImageViewer(handler)
        received = []
        v.stackLoaded.connect(received.append)
        v.set_stack([])
        assert received == []


# ── set_cull_range with no stack ─────────────────────────────────────────


class TestSetCullRangeNoStack:
    def test_set_cull_range_no_stack_stores_values(self, app) -> None:
        handler = ImageStackHandler()
        v = ImageViewer(handler)
        # No stack loaded: total == 0, else branch is taken
        v.set_cull_range(2, 7)
        assert v._cull_start == 2
        assert v._cull_end == 7
