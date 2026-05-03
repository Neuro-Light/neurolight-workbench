from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QWidget


class DualHandleRangeSlider(QWidget):
    """Dual-handle range slider for selecting a contiguous frame range.

    Signals:
        range_changed(start, end): Emitted on mouse release (0-indexed).
        frame_preview_requested(frame): Emitted live during drag (0-indexed).
    """

    range_changed = Signal(int, int)
    frame_preview_requested = Signal(int)

    _HANDLE_RADIUS = 9
    _TRACK_HEIGHT = 6
    _LABEL_H = 20
    _PAD = 16

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._total: int = 0
        self._start: int = 0
        self._end: int = 0
        self._dragging: str | None = None
        self.setMinimumHeight(60)
        self.setMouseTracking(True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_total(self, total: int) -> None:
        """Set total frames and reset range to include all."""
        self._total = max(0, total)
        self._start = 0
        self._end = max(0, self._total - 1)
        self.update()

    def set_values(self, start: int, end: int) -> None:
        """Set current range without emitting signals."""
        if self._total <= 0:
            return
        self._start = max(0, min(start, self._total - 1))
        self._end = max(self._start, min(end, self._total - 1))
        self.update()

    def get_start(self) -> int:
        return self._start

    def get_end(self) -> int:
        return self._end

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    def _usable_w(self) -> int:
        return max(1, self.width() - 2 * self._PAD)

    def _frame_to_x(self, frame: int) -> int:
        if self._total <= 1:
            return self._PAD + self._usable_w() // 2
        return self._PAD + round(frame / (self._total - 1) * self._usable_w())

    def _x_to_frame(self, x: int) -> int:
        if self._total <= 1:
            return 0
        ratio = (x - self._PAD) / self._usable_w()
        return max(0, min(self._total - 1, round(ratio * (self._total - 1))))

    def _track_y(self) -> int:
        return self._LABEL_H + self._HANDLE_RADIUS + 4

    # ------------------------------------------------------------------
    # Paint
    # ------------------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802
        if self._total <= 0:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        ty = self._track_y()
        hh = self._TRACK_HEIGHT // 2
        xs = self._frame_to_x(self._start)
        xe = self._frame_to_x(self._end)
        tl = self._PAD
        tr = self._PAD + self._usable_w()

        excluded_col = QColor(70, 70, 70)
        included_col = QColor(70, 140, 220)

        # left excluded region
        if xs > tl:
            painter.setPen(Qt.NoPen)
            painter.setBrush(excluded_col)
            painter.drawRoundedRect(tl, ty - hh, xs - tl, self._TRACK_HEIGHT, 3, 3)

        # included region
        inc_w = max(0, xe - xs)
        painter.setPen(Qt.NoPen)
        painter.setBrush(included_col)
        painter.drawRoundedRect(xs, ty - hh, inc_w, self._TRACK_HEIGHT, 3, 3)

        # right excluded region
        if xe < tr:
            painter.setPen(Qt.NoPen)
            painter.setBrush(excluded_col)
            painter.drawRoundedRect(xe, ty - hh, tr - xe, self._TRACK_HEIGHT, 3, 3)

        # handles and frame labels
        font = QFont()
        font.setPixelSize(11)
        painter.setFont(font)
        fm = painter.fontMetrics()

        for x, label in (
            (xs, f"Frame {self._start + 1}"),
            (xe, f"Frame {self._end + 1}"),
        ):
            r = self._HANDLE_RADIUS
            painter.setPen(QPen(QColor(180, 180, 180), 2))
            painter.setBrush(QColor(240, 240, 240))
            painter.drawEllipse(x - r, ty - r, r * 2, r * 2)

            painter.setPen(QColor(200, 200, 200))
            tw = fm.horizontalAdvance(label)
            tx = max(0, min(x - tw // 2, self.width() - tw))
            painter.drawText(tx, self._LABEL_H - 4, label)

        painter.end()

    # ------------------------------------------------------------------
    # Mouse events
    # ------------------------------------------------------------------

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if self._total <= 0:
            return
        x = event.position().x()
        xs = self._frame_to_x(self._start)
        xe = self._frame_to_x(self._end)
        hit = self._HANDLE_RADIUS + 4

        ds = abs(x - xs)
        de = abs(x - xe)

        if ds <= hit and (ds <= de or de > hit):
            self._dragging = "start"
        elif de <= hit:
            self._dragging = "end"

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._dragging is None or self._total <= 0:
            return
        frame = self._x_to_frame(int(event.position().x()))
        if self._dragging == "start":
            frame = min(frame, self._end)
            if frame != self._start:
                self._start = frame
                self.update()
                self.frame_preview_requested.emit(self._start)
        else:
            frame = max(frame, self._start)
            if frame != self._end:
                self._end = frame
                self.update()
                self.frame_preview_requested.emit(self._end)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if self._dragging is not None:
            self._dragging = None
            self.range_changed.emit(self._start, self._end)
