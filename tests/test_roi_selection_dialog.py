"""Tests for ROISelectionDialog and _ROIGraphicsView (zoom, polygon flow, overlays)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QKeyEvent, QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog

from core.roi import ROI, ROIHandle, ROIShape
from ui.roi_selection_dialog import ROISelectionDialog


def _mb_event(button: Qt.MouseButton) -> SimpleNamespace:
    return SimpleNamespace(button=lambda b=button: b)


def _empty_event() -> SimpleNamespace:
    return SimpleNamespace()


@pytest.fixture
def app():
    if not QApplication.instance():
        return QApplication([])
    return QApplication.instance()


def _wheel_event(view, *, angle_y: int = 120) -> QWheelEvent:
    pos = QPointF(view.width() / 2, view.height() / 2)
    return QWheelEvent(
        pos,
        pos,
        QPoint(0, 0),
        QPoint(0, angle_y),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.ScrollUpdate,
        False,
    )


def test_roi_dialog_opens_grayscale_and_rgb(app) -> None:
    gray = np.zeros((40, 50), dtype=np.uint8)
    dlg = ROISelectionDialog(gray)
    dlg._on_fit_view()
    assert "ROI Selection" in dlg.windowTitle()
    dlg.close()

    rgb = np.zeros((20, 20, 3), dtype=np.uint8)
    dlg2 = ROISelectionDialog(rgb, active_roi_label="ROI 2")
    dlg2._on_fit_view()
    assert "ROI 2" in dlg2.windowTitle()
    dlg2.close()


def test_roi_dialog_other_roi_polygon_overlay(app) -> None:
    img = np.zeros((60, 60), dtype=np.uint8)
    other = ROI(
        x=0,
        y=0,
        width=60,
        height=60,
        shape=ROIShape.POLYGON,
        points=[(5, 5), (55, 5), (55, 55), (5, 55)],
    )
    dlg = ROISelectionDialog(img, other_roi=other)
    dlg._on_fit_view()
    dlg._update_overlay()
    dlg.close()


def test_roi_dialog_zoom_toolbar_and_view_wheel(app) -> None:
    img = np.zeros((80, 80), dtype=np.uint8)
    dlg = ROISelectionDialog(img)
    dlg._on_fit_view()
    QTest.mouseClick(dlg._zoom_in_btn, Qt.MouseButton.LeftButton)
    QTest.mouseClick(dlg._zoom_out_btn, Qt.MouseButton.LeftButton)
    QTest.mouseClick(dlg._fit_btn, Qt.MouseButton.LeftButton)
    dlg._view.wheelEvent(_wheel_event(dlg._view, angle_y=120))
    dlg._view.wheelEvent(_wheel_event(dlg._view, angle_y=-120))
    dlg._view.wheelEvent(_wheel_event(dlg._view, angle_y=0))
    dlg.close()


def test_roi_dialog_polygon_draw_complete_accept(app) -> None:
    img = np.zeros((100, 100), dtype=np.uint8)
    dlg = ROISelectionDialog(img)
    dlg._on_fit_view()
    for xy in ((10.0, 10.0), (80.0, 10.0), (80.0, 80.0)):
        dlg._handle_mouse_press(_mb_event(Qt.MouseButton.LeftButton), QPointF(*xy))
    dlg._complete_polygon()
    assert dlg._current_roi is not None
    assert dlg._current_roi.shape == ROIShape.POLYGON
    QTest.mouseClick(dlg._accept_btn, Qt.MouseButton.LeftButton)
    assert dlg.result() == QDialog.DialogCode.Accepted
    dlg.close()


def test_roi_dialog_toggle_adjust_updates_ui(app) -> None:
    pts = [(5, 5), (90, 5), (90, 90), (5, 90)]
    roi = ROI.from_dict({"shape": "polygon", "points": pts})
    img = np.zeros((100, 100), dtype=np.uint8)
    dlg = ROISelectionDialog(img, existing_roi=roi)
    dlg._on_fit_view()
    QTest.mouseClick(dlg._adjust_btn, Qt.MouseButton.LeftButton)
    assert dlg._adjust_mode is True
    QTest.mouseClick(dlg._adjust_btn, Qt.MouseButton.LeftButton)
    assert dlg._adjust_mode is False
    dlg.close()


def test_roi_dialog_right_click_undo_point_while_drawing(app) -> None:
    img = np.zeros((50, 50), dtype=np.uint8)
    dlg = ROISelectionDialog(img)
    dlg._on_fit_view()
    dlg._handle_mouse_press(_mb_event(Qt.MouseButton.LeftButton), QPointF(5, 5))
    dlg._handle_mouse_press(_mb_event(Qt.MouseButton.RightButton), QPointF(40, 40))
    assert dlg._polygon_points == []
    dlg.close()


def test_roi_dialog_key_plus_minus_zoom_escape(app) -> None:
    img = np.zeros((40, 40), dtype=np.uint8)
    dlg = ROISelectionDialog(img)
    dlg._on_fit_view()
    dlg._handle_key_press(QKeyEvent(QKeyEvent.Type.KeyPress, int(Qt.Key.Key_Plus), Qt.KeyboardModifier.NoModifier))
    dlg._handle_key_press(QKeyEvent(QKeyEvent.Type.KeyPress, int(Qt.Key.Key_Minus), Qt.KeyboardModifier.NoModifier))
    dlg._handle_key_press(QKeyEvent(QKeyEvent.Type.KeyPress, int(Qt.Key.Key_Escape), Qt.KeyboardModifier.NoModifier))
    assert dlg.result() == QDialog.DialogCode.Rejected
    dlg.close()


def test_roi_dialog_fit_timer_runs_immediately_with_patch(app) -> None:
    img = np.zeros((30, 30), dtype=np.uint8)
    with patch("PySide6.QtCore.QTimer.singleShot", lambda _ms, fn: fn()):
        dlg = ROISelectionDialog(img)
        dlg.close()


def test_roi_dialog_middle_button_pan(app) -> None:
    img = np.zeros((40, 40), dtype=np.uint8)
    dlg = ROISelectionDialog(img)
    dlg._on_fit_view()
    v = dlg._view

    class _E:
        def __init__(self, typ, pos_f: QPointF, btn: Qt.MouseButton, buttons: Qt.MouseButton) -> None:
            self._typ = typ
            self._pos = pos_f
            self._btn = btn
            self._buttons = buttons

        def accept(self) -> None:
            pass

        def button(self) -> Qt.MouseButton:
            return self._btn

        def buttons(self) -> Qt.MouseButton:
            return self._buttons

        def position(self) -> QPointF:
            return self._pos

    v.mousePressEvent(_E("press", QPointF(10, 10), Qt.MouseButton.MiddleButton, Qt.MouseButton.MiddleButton))
    v.mouseMoveEvent(_E("move", QPointF(20, 15), Qt.MouseButton.NoButton, Qt.MouseButton.MiddleButton))
    v.mouseReleaseEvent(_E("rel", QPointF(20, 15), Qt.MouseButton.MiddleButton, Qt.MouseButton.NoButton))
    dlg.close()


def test_roi_dialog_preview_line_on_mouse_move_while_drawing(app) -> None:
    img = np.zeros((60, 60), dtype=np.uint8)
    dlg = ROISelectionDialog(img)
    dlg._on_fit_view()
    dlg._handle_mouse_press(_mb_event(Qt.MouseButton.LeftButton), QPointF(5, 5))
    dlg._handle_mouse_press(_mb_event(Qt.MouseButton.LeftButton), QPointF(30, 5))
    dlg._handle_mouse_move(_empty_event(), QPointF(40, 40))
    assert dlg._preview_pos is not None
    dlg.close()


def test_roi_dialog_double_click_finishes_polygon(app) -> None:
    img = np.zeros((80, 80), dtype=np.uint8)
    dlg = ROISelectionDialog(img)
    dlg._on_fit_view()
    for xy in ((10.0, 10.0), (70.0, 10.0), (40.0, 70.0)):
        dlg._handle_mouse_press(_mb_event(Qt.MouseButton.LeftButton), QPointF(*xy))
    dlg._handle_mouse_double_click(_mb_event(Qt.MouseButton.LeftButton), QPointF(40.0, 40.0))
    assert dlg._current_roi is not None
    dlg.close()


def test_roi_dialog_complete_polygon_near_closing_point_drops_last(app) -> None:
    img = np.zeros((100, 100), dtype=np.uint8)
    dlg = ROISelectionDialog(img)
    dlg._on_fit_view()
    dlg._polygon_points = [QPointF(10, 10), QPointF(80, 10), QPointF(80, 80), QPointF(12, 12)]
    dlg._complete_polygon()
    assert dlg._current_roi is not None
    assert len(dlg._current_roi.points or []) == 3
    dlg.close()


def test_roi_dialog_complete_polygon_ignored_with_few_points(app) -> None:
    img = np.zeros((40, 40), dtype=np.uint8)
    dlg = ROISelectionDialog(img)
    dlg._polygon_points = [QPointF(1, 1), QPointF(2, 2)]
    dlg._complete_polygon()
    assert dlg._current_roi is None
    dlg.close()


def test_roi_dialog_backspace_removes_last_vertex(app) -> None:
    img = np.zeros((50, 50), dtype=np.uint8)
    dlg = ROISelectionDialog(img)
    dlg._on_fit_view()
    dlg._handle_mouse_press(_mb_event(Qt.MouseButton.LeftButton), QPointF(5, 5))
    dlg._handle_mouse_press(_mb_event(Qt.MouseButton.LeftButton), QPointF(40, 40))
    ev = QKeyEvent(QKeyEvent.Type.KeyPress, int(Qt.Key.Key_Backspace), Qt.KeyboardModifier.NoModifier)
    dlg._handle_key_press(ev)
    assert len(dlg._polygon_points) == 1
    dlg.close()


def test_roi_dialog_get_roi(app) -> None:
    img = np.zeros((20, 20), dtype=np.uint8)
    pts = [(2, 2), (18, 2), (18, 18), (2, 18)]
    roi = ROI.from_dict({"shape": "polygon", "points": pts})
    dlg = ROISelectionDialog(img, existing_roi=roi)
    assert dlg.get_roi() is roi
    dlg.close()


def test_roi_dialog_adjust_mode_drag_vertex(app) -> None:
    img = np.zeros((100, 100), dtype=np.uint8)
    pts = [(20, 20), (80, 20), (80, 80), (20, 80)]
    roi = ROI.from_dict({"shape": "polygon", "points": pts})
    dlg = ROISelectionDialog(img, existing_roi=roi)
    dlg._on_fit_view()
    dlg._adjust_mode = True
    dlg._update_overlay()
    dlg._handle_mouse_press(_mb_event(Qt.MouseButton.LeftButton), QPointF(20, 20))
    assert dlg._dragging_handle != ROIHandle.NONE
    dlg._handle_mouse_move(_empty_event(), QPointF(25, 25))
    dlg._handle_mouse_release(_mb_event(Qt.MouseButton.LeftButton), QPointF(25, 25))
    dlg.close()


def test_roi_dialog_right_click_deletes_vertex_in_adjust_mode(app) -> None:
    img = np.zeros((100, 100), dtype=np.uint8)
    pts = [(20, 20), (80, 20), (80, 80), (20, 80)]
    roi = ROI.from_dict({"shape": "polygon", "points": pts})
    dlg = ROISelectionDialog(img, existing_roi=roi)
    dlg._on_fit_view()
    dlg._adjust_mode = True
    dlg._update_overlay()
    dlg._handle_mouse_press(_mb_event(Qt.MouseButton.RightButton), QPointF(20, 20))
    assert dlg._current_roi is not None
    assert len(dlg._current_roi.points or []) == 3
    dlg.close()


def test_roi_dialog_double_click_drops_duplicate_last_point(app) -> None:
    img = np.zeros((60, 60), dtype=np.uint8)
    dlg = ROISelectionDialog(img)
    dlg._on_fit_view()
    dlg._polygon_points = [QPointF(5, 5), QPointF(55, 5), QPointF(55, 55), QPointF(55, 55)]
    dlg._handle_mouse_double_click(_mb_event(Qt.MouseButton.LeftButton), QPointF(30, 30))
    assert dlg._current_roi is not None
    dlg.close()


def test_roi_dialog_zoom_hits_min_max_bounds(app) -> None:
    img = np.zeros((30, 30), dtype=np.uint8)
    dlg = ROISelectionDialog(img)
    dlg._on_fit_view()
    for _ in range(40):
        dlg._view.zoom_in()
    for _ in range(40):
        dlg._view.zoom_out()
    dlg.close()


def test_roi_dialog_view_routes_double_click_to_dialog(app) -> None:
    img = np.zeros((60, 60), dtype=np.uint8)
    dlg = ROISelectionDialog(img)
    dlg._on_fit_view()
    for xy in ((5.0, 5.0), (55.0, 5.0), (30.0, 55.0)):
        dlg._handle_mouse_press(_mb_event(Qt.MouseButton.LeftButton), QPointF(*xy))

    class _DE:
        def accept(self) -> None:
            pass

        def button(self) -> Qt.MouseButton:
            return Qt.MouseButton.LeftButton

        def position(self) -> QPointF:
            return QPointF(30, 30)

    dlg._view.mouseDoubleClickEvent(_DE())
    assert dlg._current_roi is not None
    dlg.close()
