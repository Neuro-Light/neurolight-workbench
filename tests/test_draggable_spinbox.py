"""Tests for DraggableSpinBox and DraggableDoubleSpinBox — drag-to-adjust and tooltip."""

from __future__ import annotations

import pytest
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication

from ui.draggable_spinbox import DraggableDoubleSpinBox, DraggableSpinBox


@pytest.fixture
def app():
    if not QApplication.instance():
        return QApplication([])
    return QApplication.instance()


def _make_mouse_event(event_type, button, pos_x, pos_y, buttons=None):
    """Create a synthetic QMouseEvent."""
    if buttons is None:
        buttons = button
    return QMouseEvent(
        event_type,
        QPointF(pos_x, pos_y),
        QPointF(pos_x, pos_y),
        button,
        buttons,
        Qt.KeyboardModifier.NoModifier,
    )


# ── DraggableSpinBox ──────────────────────────────────────────────────────


class TestDraggableSpinBox:
    def test_initial_cursor(self, app) -> None:
        sb = DraggableSpinBox()
        assert sb.cursor().shape() == Qt.CursorShape.SizeHorCursor

    def test_tooltip_appends_drag_hint(self, app) -> None:
        sb = DraggableSpinBox()
        sb.setToolTip("Adjust value")
        assert "Drag horizontally" in sb.toolTip()

    def test_tooltip_does_not_double_hint(self, app) -> None:
        sb = DraggableSpinBox()
        sb.setToolTip("Adjust value")
        sb.setToolTip(sb.toolTip())
        assert sb.toolTip().count("Drag horizontally") == 1

    def test_drag_increases_value(self, app) -> None:
        sb = DraggableSpinBox()
        sb.setRange(0, 100)
        sb.setSingleStep(1)
        sb.setValue(50)

        press = _make_mouse_event(
            QMouseEvent.Type.MouseButtonPress, Qt.MouseButton.LeftButton, 100, 10
        )
        sb.mousePressEvent(press)

        move = _make_mouse_event(
            QMouseEvent.Type.MouseMove,
            Qt.MouseButton.NoButton,
            200,
            10,
            buttons=Qt.MouseButton.LeftButton,
        )
        sb.mouseMoveEvent(move)
        assert sb.value() > 50

    def test_drag_clamps_to_range(self, app) -> None:
        sb = DraggableSpinBox()
        sb.setRange(0, 10)
        sb.setSingleStep(1)
        sb.setValue(8)

        press = _make_mouse_event(
            QMouseEvent.Type.MouseButtonPress, Qt.MouseButton.LeftButton, 100, 10
        )
        sb.mousePressEvent(press)

        move = _make_mouse_event(
            QMouseEvent.Type.MouseMove,
            Qt.MouseButton.NoButton,
            2000,
            10,
            buttons=Qt.MouseButton.LeftButton,
        )
        sb.mouseMoveEvent(move)
        assert sb.value() <= 10

    def test_release_clears_drag_state(self, app) -> None:
        sb = DraggableSpinBox()
        sb.setRange(0, 100)
        sb.setValue(50)

        press = _make_mouse_event(
            QMouseEvent.Type.MouseButtonPress, Qt.MouseButton.LeftButton, 100, 10
        )
        sb.mousePressEvent(press)
        assert sb._drag_start_x is not None

        release = _make_mouse_event(
            QMouseEvent.Type.MouseButtonRelease, Qt.MouseButton.LeftButton, 100, 10
        )
        sb.mouseReleaseEvent(release)
        assert sb._drag_start_x is None
        assert sb._drag_start_value is None


# ── DraggableDoubleSpinBox ────────────────────────────────────────────────


class TestDraggableDoubleSpinBox:
    def test_initial_cursor(self, app) -> None:
        sb = DraggableDoubleSpinBox()
        assert sb.cursor().shape() == Qt.CursorShape.SizeHorCursor

    def test_tooltip_appends_drag_hint(self, app) -> None:
        sb = DraggableDoubleSpinBox()
        sb.setToolTip("Sigma value")
        assert "Drag horizontally" in sb.toolTip()

    def test_drag_changes_value(self, app) -> None:
        sb = DraggableDoubleSpinBox()
        sb.setRange(0.0, 100.0)
        sb.setSingleStep(0.5)
        sb.setDecimals(2)
        sb.setValue(50.0)

        press = _make_mouse_event(
            QMouseEvent.Type.MouseButtonPress, Qt.MouseButton.LeftButton, 100, 10
        )
        sb.mousePressEvent(press)

        move = _make_mouse_event(
            QMouseEvent.Type.MouseMove,
            Qt.MouseButton.NoButton,
            200,
            10,
            buttons=Qt.MouseButton.LeftButton,
        )
        sb.mouseMoveEvent(move)
        assert sb.value() != 50.0

    def test_drag_clamps_to_min(self, app) -> None:
        sb = DraggableDoubleSpinBox()
        sb.setRange(0.0, 10.0)
        sb.setSingleStep(1.0)
        sb.setValue(2.0)

        press = _make_mouse_event(
            QMouseEvent.Type.MouseButtonPress, Qt.MouseButton.LeftButton, 100, 10
        )
        sb.mousePressEvent(press)

        move = _make_mouse_event(
            QMouseEvent.Type.MouseMove,
            Qt.MouseButton.NoButton,
            -2000,
            10,
            buttons=Qt.MouseButton.LeftButton,
        )
        sb.mouseMoveEvent(move)
        assert sb.value() >= 0.0

    def test_release_clears_drag_state(self, app) -> None:
        sb = DraggableDoubleSpinBox()
        sb.setRange(0.0, 100.0)
        sb.setValue(50.0)

        press = _make_mouse_event(
            QMouseEvent.Type.MouseButtonPress, Qt.MouseButton.LeftButton, 100, 10
        )
        sb.mousePressEvent(press)

        release = _make_mouse_event(
            QMouseEvent.Type.MouseButtonRelease, Qt.MouseButton.LeftButton, 100, 10
        )
        sb.mouseReleaseEvent(release)
        assert sb._drag_start_x is None
        assert sb._drag_start_value is None

    def test_move_without_press_is_noop(self, app) -> None:
        sb = DraggableDoubleSpinBox()
        sb.setRange(0.0, 100.0)
        sb.setValue(50.0)

        move = _make_mouse_event(
            QMouseEvent.Type.MouseMove,
            Qt.MouseButton.NoButton,
            200,
            10,
            buttons=Qt.MouseButton.LeftButton,
        )
        sb.mouseMoveEvent(move)
        assert sb.value() == 50.0
