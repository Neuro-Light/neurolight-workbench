"""Tests for DualHandleRangeSlider."""

from __future__ import annotations

from unittest.mock import Mock

import pytest
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QMouseEvent, QPaintEvent, QRegion
from PySide6.QtWidgets import QApplication

from ui.range_slider import DualHandleRangeSlider


@pytest.fixture
def app():
    if not QApplication.instance():
        return QApplication([])
    return QApplication.instance()


def _make_slider(app, total: int = 10, width: int = 300, height: int = 80) -> DualHandleRangeSlider:
    s = DualHandleRangeSlider()
    s.resize(width, height)
    s.set_total(total)
    return s


def _press(slider: DualHandleRangeSlider, x: float, y: float) -> None:
    ev = QMouseEvent(
        QMouseEvent.Type.MouseButtonPress,
        QPointF(x, y),
        QPointF(x, y),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    slider.mousePressEvent(ev)


def _move(slider: DualHandleRangeSlider, x: float, y: float) -> None:
    ev = QMouseEvent(
        QMouseEvent.Type.MouseMove,
        QPointF(x, y),
        QPointF(x, y),
        Qt.MouseButton.NoButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    slider.mouseMoveEvent(ev)


def _release(slider: DualHandleRangeSlider, x: float, y: float) -> None:
    ev = QMouseEvent(
        QMouseEvent.Type.MouseButtonRelease,
        QPointF(x, y),
        QPointF(x, y),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
    )
    slider.mouseReleaseEvent(ev)


# ── Initialization ────────────────────────────────────────────────────────


class TestInit:
    def test_default_total_is_zero(self, app) -> None:
        s = DualHandleRangeSlider()
        assert s._total == 0

    def test_default_start_end_zero(self, app) -> None:
        s = DualHandleRangeSlider()
        assert s._start == 0
        assert s._end == 0

    def test_default_dragging_none(self, app) -> None:
        s = DualHandleRangeSlider()
        assert s._dragging is None

    def test_minimum_height(self, app) -> None:
        s = DualHandleRangeSlider()
        assert s.minimumHeight() == 60


# ── set_total ─────────────────────────────────────────────────────────────


class TestSetTotal:
    def test_sets_total(self, app) -> None:
        s = _make_slider(app, total=5)
        assert s._total == 5

    def test_resets_start_to_zero(self, app) -> None:
        s = _make_slider(app, total=10)
        s.set_values(3, 7)
        s.set_total(10)
        assert s._start == 0

    def test_sets_end_to_last_frame(self, app) -> None:
        s = _make_slider(app, total=10)
        assert s._end == 9

    def test_negative_total_clamped_to_zero(self, app) -> None:
        s = DualHandleRangeSlider()
        s.set_total(-5)
        assert s._total == 0

    def test_zero_total_end_stays_zero(self, app) -> None:
        s = DualHandleRangeSlider()
        s.set_total(0)
        assert s._end == 0

    def test_single_frame_total(self, app) -> None:
        s = DualHandleRangeSlider()
        s.set_total(1)
        assert s._total == 1
        assert s._start == 0
        assert s._end == 0


# ── set_values ────────────────────────────────────────────────────────────


class TestSetValues:
    def test_valid_range(self, app) -> None:
        s = _make_slider(app, total=10)
        s.set_values(2, 7)
        assert s.get_start() == 2
        assert s.get_end() == 7

    def test_noop_when_total_zero(self, app) -> None:
        s = DualHandleRangeSlider()
        s.set_values(2, 7)
        assert s._start == 0
        assert s._end == 0

    def test_start_clamped_to_zero(self, app) -> None:
        s = _make_slider(app, total=10)
        s.set_values(-5, 8)
        assert s.get_start() == 0

    def test_end_clamped_to_last_frame(self, app) -> None:
        s = _make_slider(app, total=10)
        s.set_values(0, 100)
        assert s.get_end() == 9

    def test_start_cannot_exceed_end(self, app) -> None:
        s = _make_slider(app, total=10)
        s.set_values(8, 2)
        # start is clamped, then end is max(start, ...)
        assert s.get_start() <= s.get_end()

    def test_get_start_get_end(self, app) -> None:
        s = _make_slider(app, total=10)
        s.set_values(1, 6)
        assert s.get_start() == 1
        assert s.get_end() == 6


# ── Geometry helpers ──────────────────────────────────────────────────────


class TestGeometry:
    def test_usable_w_returns_width_minus_padding(self, app) -> None:
        s = DualHandleRangeSlider()
        s.resize(300, 80)
        assert s._usable_w() == 300 - 2 * DualHandleRangeSlider._PAD

    def test_usable_w_minimum_one(self, app) -> None:
        s = DualHandleRangeSlider()
        s.resize(0, 80)
        assert s._usable_w() >= 1

    def test_frame_to_x_first_frame(self, app) -> None:
        s = _make_slider(app, total=10, width=300)
        assert s._frame_to_x(0) == s._PAD

    def test_frame_to_x_last_frame(self, app) -> None:
        s = _make_slider(app, total=10, width=300)
        assert s._frame_to_x(9) == s._PAD + s._usable_w()

    def test_frame_to_x_single_frame(self, app) -> None:
        s = _make_slider(app, total=1, width=300)
        assert s._frame_to_x(0) == s._PAD + s._usable_w() // 2

    def test_x_to_frame_left_edge(self, app) -> None:
        s = _make_slider(app, total=10, width=300)
        assert s._x_to_frame(s._PAD) == 0

    def test_x_to_frame_right_edge(self, app) -> None:
        s = _make_slider(app, total=10, width=300)
        assert s._x_to_frame(s._PAD + s._usable_w()) == 9

    def test_x_to_frame_single_frame_always_zero(self, app) -> None:
        s = _make_slider(app, total=1, width=300)
        assert s._x_to_frame(150) == 0

    def test_x_to_frame_clamps_below_zero(self, app) -> None:
        s = _make_slider(app, total=10, width=300)
        assert s._x_to_frame(-1000) == 0

    def test_x_to_frame_clamps_above_max(self, app) -> None:
        s = _make_slider(app, total=10, width=300)
        assert s._x_to_frame(10000) == 9

    def test_track_y(self, app) -> None:
        s = DualHandleRangeSlider()
        expected = s._LABEL_H + s._HANDLE_RADIUS + 4
        assert s._track_y() == expected

    def test_frame_to_x_x_to_frame_round_trip(self, app) -> None:
        s = _make_slider(app, total=10, width=300)
        for frame in range(10):
            x = s._frame_to_x(frame)
            assert s._x_to_frame(x) == frame


# ── paintEvent ────────────────────────────────────────────────────────────


class TestPaintEvent:
    def test_no_paint_when_total_zero(self, app) -> None:
        s = DualHandleRangeSlider()
        s.resize(300, 80)
        # Should return immediately without error
        ev = QPaintEvent(QRegion(s.rect()))
        s.paintEvent(ev)

    def test_paint_does_not_crash_full_range(self, app) -> None:
        s = _make_slider(app, total=10, width=300)
        s.show()
        QApplication.processEvents()
        s.hide()

    def test_paint_does_not_crash_partial_range(self, app) -> None:
        s = _make_slider(app, total=10, width=300)
        s.set_values(2, 7)
        s.show()
        QApplication.processEvents()
        s.hide()

    def test_paint_does_not_crash_start_equals_end(self, app) -> None:
        s = _make_slider(app, total=10, width=300)
        s.set_values(5, 5)
        s.show()
        QApplication.processEvents()
        s.hide()

    def test_paint_does_not_crash_single_frame(self, app) -> None:
        s = _make_slider(app, total=1, width=300)
        s.show()
        QApplication.processEvents()
        s.hide()


# ── mousePressEvent ───────────────────────────────────────────────────────


class TestMousePressEvent:
    def test_no_drag_when_total_zero(self, app) -> None:
        s = DualHandleRangeSlider()
        s.resize(300, 80)
        _press(s, 150, 33)
        assert s._dragging is None

    def test_press_on_start_handle_selects_start(self, app) -> None:
        s = _make_slider(app, total=10, width=300)
        x_start = s._frame_to_x(0)
        _press(s, x_start, s._track_y())
        assert s._dragging == "start"

    def test_press_on_end_handle_selects_end(self, app) -> None:
        s = _make_slider(app, total=10, width=300)
        x_end = s._frame_to_x(9)
        _press(s, x_end, s._track_y())
        assert s._dragging == "end"

    def test_press_far_from_handles_no_drag(self, app) -> None:
        s = _make_slider(app, total=10, width=300)
        s.set_values(0, 9)
        # Click in the middle track area, far from both handles
        _press(s, s._frame_to_x(4), s._track_y())
        # Neither handle is at frame 4 (start=0, end=9), so no drag
        assert s._dragging is None

    def test_prefer_start_when_handles_equidistant(self, app) -> None:
        # When handles are at the same position, start is preferred
        s = _make_slider(app, total=10, width=300)
        s.set_values(5, 5)
        x = s._frame_to_x(5)
        _press(s, x, s._track_y())
        assert s._dragging == "start"


# ── mouseMoveEvent ────────────────────────────────────────────────────────


class TestMouseMoveEvent:
    def test_no_move_when_not_dragging(self, app) -> None:
        s = _make_slider(app, total=10, width=300)
        received = []
        s.frame_preview_requested.connect(received.append)
        _move(s, s._frame_to_x(5), s._track_y())
        assert received == []

    def test_no_move_when_total_zero_while_dragging(self, app) -> None:
        s = DualHandleRangeSlider()
        s.resize(300, 80)
        s._dragging = "start"
        received = []
        s.frame_preview_requested.connect(received.append)
        _move(s, 150, 33)
        assert received == []

    def test_drag_start_emits_preview(self, app) -> None:
        s = _make_slider(app, total=10, width=300)
        x_start = s._frame_to_x(0)
        _press(s, x_start, s._track_y())

        received = []
        s.frame_preview_requested.connect(received.append)
        x_mid = s._frame_to_x(4)
        _move(s, x_mid, s._track_y())

        assert 4 in received
        assert s._start == 4

    def test_drag_end_emits_preview(self, app) -> None:
        s = _make_slider(app, total=10, width=300)
        x_end = s._frame_to_x(9)
        _press(s, x_end, s._track_y())

        received = []
        s.frame_preview_requested.connect(received.append)
        x_new = s._frame_to_x(6)
        _move(s, x_new, s._track_y())

        assert 6 in received
        assert s._end == 6

    def test_start_cannot_exceed_end(self, app) -> None:
        s = _make_slider(app, total=10, width=300)
        s.set_values(0, 5)
        x_start = s._frame_to_x(0)
        _press(s, x_start, s._track_y())
        # Try dragging start past end
        _move(s, s._frame_to_x(8), s._track_y())
        assert s._start <= s._end

    def test_end_cannot_go_below_start(self, app) -> None:
        s = _make_slider(app, total=10, width=300)
        s.set_values(5, 9)
        x_end = s._frame_to_x(9)
        _press(s, x_end, s._track_y())
        # Try dragging end before start
        _move(s, s._frame_to_x(2), s._track_y())
        assert s._end >= s._start

    def test_no_signal_when_start_frame_unchanged(self, app) -> None:
        s = _make_slider(app, total=10, width=300)
        x_start = s._frame_to_x(0)
        _press(s, x_start, s._track_y())
        received = []
        s.frame_preview_requested.connect(received.append)
        # Move to same position — frame doesn't change
        _move(s, x_start, s._track_y())
        assert received == []

    def test_no_signal_when_end_frame_unchanged(self, app) -> None:
        s = _make_slider(app, total=10, width=300)
        x_end = s._frame_to_x(9)
        _press(s, x_end, s._track_y())
        received = []
        s.frame_preview_requested.connect(received.append)
        # Move to same position — end frame doesn't change
        _move(s, x_end, s._track_y())
        assert received == []


# ── mouseReleaseEvent ─────────────────────────────────────────────────────


class TestMouseReleaseEvent:
    def test_release_without_drag_does_not_emit(self, app) -> None:
        s = _make_slider(app, total=10, width=300)
        received = []
        s.range_changed.connect(lambda a, b: received.append((a, b)))
        _release(s, 150, 33)
        assert received == []

    def test_release_after_start_drag_emits_range(self, app) -> None:
        s = _make_slider(app, total=10, width=300)
        received = []
        s.range_changed.connect(lambda a, b: received.append((a, b)))

        _press(s, s._frame_to_x(0), s._track_y())
        _move(s, s._frame_to_x(3), s._track_y())
        _release(s, s._frame_to_x(3), s._track_y())

        assert len(received) == 1
        assert received[0] == (3, 9)

    def test_release_after_end_drag_emits_range(self, app) -> None:
        s = _make_slider(app, total=10, width=300)
        received = []
        s.range_changed.connect(lambda a, b: received.append((a, b)))

        _press(s, s._frame_to_x(9), s._track_y())
        _move(s, s._frame_to_x(6), s._track_y())
        _release(s, s._frame_to_x(6), s._track_y())

        assert len(received) == 1
        assert received[0] == (0, 6)

    def test_release_clears_dragging(self, app) -> None:
        s = _make_slider(app, total=10, width=300)
        _press(s, s._frame_to_x(0), s._track_y())
        assert s._dragging == "start"
        _release(s, s._frame_to_x(0), s._track_y())
        assert s._dragging is None

    def test_range_changed_emits_correct_int_types(self, app) -> None:
        s = _make_slider(app, total=10, width=300)
        received = []
        s.range_changed.connect(lambda a, b: received.append((type(a), type(b))))
        _press(s, s._frame_to_x(0), s._track_y())
        _release(s, s._frame_to_x(0), s._track_y())
        assert received == [(int, int)]


# ── Label text ────────────────────────────────────────────────────────────


class TestLabelText:
    def test_label_uses_one_indexed_frames(self, app) -> None:
        s = _make_slider(app, total=10, width=300)
        s.set_values(0, 9)
        # Labels are "Frame N" where N = 0-indexed + 1
        # Verify via internal state since rendering is visual
        assert s._start + 1 == 1
        assert s._end + 1 == 10

    def test_label_updates_during_drag(self, app) -> None:
        s = _make_slider(app, total=10, width=300)
        _press(s, s._frame_to_x(0), s._track_y())
        _move(s, s._frame_to_x(4), s._track_y())
        # After dragging to frame 4, label should show "Frame 5"
        assert s._start + 1 == 5

    def test_end_label_updates_during_drag(self, app) -> None:
        s = _make_slider(app, total=10, width=300)
        _press(s, s._frame_to_x(9), s._track_y())
        _move(s, s._frame_to_x(6), s._track_y())
        assert s._end + 1 == 7


# ── Edge cases ────────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_single_frame_widget_renders(self, app) -> None:
        s = _make_slider(app, total=1, width=300)
        assert s._start == 0
        assert s._end == 0
        s.show()
        QApplication.processEvents()
        s.hide()

    def test_full_drag_sequence(self, app) -> None:
        s = _make_slider(app, total=20, width=400)
        preview_frames = []
        range_calls = []
        s.frame_preview_requested.connect(preview_frames.append)
        s.range_changed.connect(lambda a, b: range_calls.append((a, b)))

        # Drag start from frame 0 to frame 5
        _press(s, s._frame_to_x(0), s._track_y())
        _move(s, s._frame_to_x(5), s._track_y())
        _release(s, s._frame_to_x(5), s._track_y())

        assert s.get_start() == 5
        assert s.get_end() == 19
        assert 5 in preview_frames
        assert (5, 19) in range_calls
