"""Spinbox widgets that support drag-to-adjust in addition to typing and arrow buttons."""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QDoubleSpinBox, QSpinBox, QWidget

# Pixels of horizontal drag per one step (larger = less sensitive)
_DRAG_PIXELS_PER_STEP = 6


class DraggableSpinBox(QSpinBox):
    """QSpinBox that can be adjusted by clicking and dragging horizontally."""

    _DRAG_HINT = " Drag horizontally to adjust."

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._drag_start_x: Optional[int] = None
        self._drag_start_value: Optional[int] = None
        self.setCursor(Qt.CursorShape.SizeHorCursor)

    def setToolTip(self, text: str) -> None:
        if text and not text.rstrip().endswith(self._DRAG_HINT.strip()):
            text = (text.rstrip() + self._DRAG_HINT).strip()
        super().setToolTip(text)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start_x = event.globalPosition().x()
            self._drag_start_value = self.value()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if (
            self._drag_start_x is not None
            and self._drag_start_value is not None
            and event.buttons() & Qt.MouseButton.LeftButton
        ):
            delta_x = event.globalPosition().x() - self._drag_start_x
            steps = int(round(delta_x / _DRAG_PIXELS_PER_STEP))
            new_val = self._drag_start_value + steps * self.singleStep()
            new_val = max(self.minimum(), min(self.maximum(), new_val))
            self.setValue(int(round(new_val)))
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start_x = None
            self._drag_start_value = None
        super().mouseReleaseEvent(event)


class DraggableDoubleSpinBox(QDoubleSpinBox):
    """QDoubleSpinBox that can be adjusted by clicking and dragging horizontally."""

    _DRAG_HINT = " Drag horizontally to adjust."

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._drag_start_x: Optional[int] = None
        self._drag_start_value: Optional[float] = None
        self.setCursor(Qt.CursorShape.SizeHorCursor)

    def setToolTip(self, text: str) -> None:
        if text and not text.rstrip().endswith(self._DRAG_HINT.strip()):
            text = (text.rstrip() + self._DRAG_HINT).strip()
        super().setToolTip(text)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start_x = event.globalPosition().x()
            self._drag_start_value = self.value()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if (
            self._drag_start_x is not None
            and self._drag_start_value is not None
            and event.buttons() & Qt.MouseButton.LeftButton
        ):
            delta_x = event.globalPosition().x() - self._drag_start_x
            step = self.singleStep()
            steps = delta_x / _DRAG_PIXELS_PER_STEP
            new_val = self._drag_start_value + steps * step
            new_val = max(self.minimum(), min(self.maximum(), new_val))
            self.setValue(round(new_val, self.decimals()))
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start_x = None
            self._drag_start_value = None
        super().mouseReleaseEvent(event)
