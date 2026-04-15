"""Tests for ImageViewer frame-culling features."""

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
    """Write *n* tiny 4x4 TIFF files and return their paths."""
    paths = []
    for i in range(n):
        p = tmp_path / f"frame_{i:03d}.tif"
        tifffile.imwrite(p, np.full((4, 4), i * 10, dtype=np.uint8))
        paths.append(str(p))
    return paths


@pytest.fixture
def viewer(app, tmp_path):
    """Create an ImageViewer loaded with a 5-frame stack."""
    files = _make_stack(tmp_path, 5)
    handler = ImageStackHandler()
    v = ImageViewer(handler)
    v.set_stack(files)
    return v


# ── Initial state ────────────────────────────────────────────────────────


class TestCullingInitialState:
    def test_excluded_frames_empty_on_init(self, viewer) -> None:
        assert viewer.get_excluded_frames() == set()

    def test_filter_excluded_off_by_default(self, viewer) -> None:
        assert viewer._filter_excluded is False

    def test_cull_panel_hidden_by_default(self, viewer) -> None:
        assert viewer.cull_controls_panel.isVisible() is False

    def test_cull_count_label_zero(self, viewer) -> None:
        assert "0" in viewer._cull_count_label.text()


# ── get / set excluded frames ────────────────────────────────────────────


class TestGetSetExcludedFrames:
    def test_set_and_get_round_trip(self, viewer) -> None:
        viewer.set_excluded_frames({1, 3})
        assert viewer.get_excluded_frames() == {1, 3}

    def test_get_returns_copy(self, viewer) -> None:
        viewer.set_excluded_frames({0})
        result = viewer.get_excluded_frames()
        result.add(99)
        assert 99 not in viewer.get_excluded_frames()

    def test_set_updates_cull_button_text(self, viewer) -> None:
        viewer.set_excluded_frames({0})
        viewer.index = 0
        viewer._refresh_cull_button()
        assert "Include" in viewer._cull_toggle_btn.text()

    def test_set_updates_count_label(self, viewer) -> None:
        viewer.set_excluded_frames({0, 2, 4})
        assert "3" in viewer._cull_count_label.text()

    def test_count_label_singular(self, viewer) -> None:
        viewer.set_excluded_frames({2})
        assert "1 frame excluded" in viewer._cull_count_label.text()


# ── Cull toggle ──────────────────────────────────────────────────────────


class TestCullToggle:
    def test_toggle_excludes_current_frame(self, viewer) -> None:
        viewer.index = 2
        viewer._on_cull_toggle()
        assert 2 in viewer.get_excluded_frames()

    def test_toggle_re_includes_excluded_frame(self, viewer) -> None:
        viewer.index = 2
        viewer._on_cull_toggle()  # exclude
        viewer._on_cull_toggle()  # re-include
        assert 2 not in viewer.get_excluded_frames()

    def test_toggle_emits_signal(self, viewer) -> None:
        spy = Mock()
        viewer.frameCullingChanged.connect(spy)
        viewer.index = 1
        viewer._on_cull_toggle()
        spy.assert_called_once()
        emitted = spy.call_args[0][0]
        assert isinstance(emitted, set)
        assert 1 in emitted

    def test_toggle_updates_button_label(self, viewer) -> None:
        viewer.index = 0
        viewer._on_cull_toggle()
        assert "Include" in viewer._cull_toggle_btn.text()
        viewer._on_cull_toggle()
        assert "Exclude" in viewer._cull_toggle_btn.text()


# ── Visible indices / navigation with filter ─────────────────────────────


class TestVisibleIndices:
    def test_all_visible_by_default(self, viewer) -> None:
        assert viewer._visible_indices == [0, 1, 2, 3, 4]

    def test_filter_off_shows_all_even_with_exclusions(self, viewer) -> None:
        viewer.set_excluded_frames({1, 3})
        assert viewer._visible_indices == [0, 1, 2, 3, 4]

    def test_filter_on_hides_excluded(self, viewer) -> None:
        viewer.set_excluded_frames({1, 3})
        viewer._filter_excluded = True
        assert viewer._visible_indices == [0, 2, 4]


class TestSetFilterExcluded:
    def test_enables_filtering(self, viewer) -> None:
        viewer.set_excluded_frames({1, 3})
        viewer.set_filter_excluded(True)
        assert viewer._filter_excluded is True
        assert viewer._visible_indices == [0, 2, 4]

    def test_disables_filtering(self, viewer) -> None:
        viewer.set_excluded_frames({1, 3})
        viewer.set_filter_excluded(True)
        viewer.set_filter_excluded(False)
        assert viewer._visible_indices == [0, 1, 2, 3, 4]

    def test_noop_when_already_set(self, viewer) -> None:
        viewer.set_filter_excluded(False)
        assert viewer._filter_excluded is False

    def test_snaps_to_visible_frame(self, viewer) -> None:
        viewer.index = 1
        viewer.set_excluded_frames({1})
        viewer.set_filter_excluded(True)
        assert viewer.index in viewer._visible_indices

    def test_slider_range_matches_visible(self, viewer) -> None:
        viewer.set_excluded_frames({0, 2, 4})
        viewer.set_filter_excluded(True)
        assert viewer.slider.maximum() == 1  # 2 visible frames → slider 0..1


class TestNavigationWithFilter:
    def test_next_image_skips_excluded(self, viewer) -> None:
        viewer.set_excluded_frames({1})
        viewer.set_filter_excluded(True)
        viewer.index = 0
        viewer.slider.setValue(0)
        viewer.next_image()
        assert viewer.index == 2

    def test_prev_image_skips_excluded(self, viewer) -> None:
        viewer.set_excluded_frames({1})
        viewer.set_filter_excluded(True)
        viewer.index = 2
        viewer.slider.setValue(1)
        viewer.prev_image()
        assert viewer.index == 0

    def test_on_slider_maps_to_visible_index(self, viewer) -> None:
        viewer.set_excluded_frames({1, 3})
        viewer.set_filter_excluded(True)
        # visible = [0, 2, 4], slider pos 2 → raw index 4
        viewer._on_slider(2)
        assert viewer.index == 4

    def test_next_at_end_stays(self, viewer) -> None:
        viewer.set_excluded_frames({3, 4})
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


# ── Reset ────────────────────────────────────────────────────────────────


class TestResetClearsExclusions:
    def test_reset_clears_excluded_frames(self, viewer) -> None:
        viewer.set_excluded_frames({0, 1, 2})
        viewer.set_filter_excluded(True)
        viewer.reset()
        assert viewer.get_excluded_frames() == set()
        assert viewer._filter_excluded is False
