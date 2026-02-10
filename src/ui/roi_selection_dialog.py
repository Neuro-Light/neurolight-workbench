"""ROI Selection Dialog with zoom/pan support for precise polygon drawing."""
from __future__ import annotations

from typing import List, Optional

import numpy as np
from PySide6.QtCore import Qt, QPointF, QRectF
from PySide6.QtGui import (
    QPixmap,
    QPainter,
    QPen,
    QColor,
    QBrush,
    QPolygonF,
    QWheelEvent,
    QMouseEvent,
    QKeyEvent,
)
from PySide6.QtWidgets import (
    QDialog,
    QGraphicsView,
    QGraphicsScene,
    QGraphicsPixmapItem,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QWidget,
    QApplication,
    QSizePolicy,
)

from core.roi import ROI, ROIShape, ROIHandle, HandleResult
from utils.image_utils import numpy_to_qimage


# ---------------------------------------------------------------------------
# Custom QGraphicsView with zoom, pan, and ROI interaction
# ---------------------------------------------------------------------------

class _ROIGraphicsView(QGraphicsView):
    """QGraphicsView subclass that provides mouse-wheel zoom, pan,
    polygon drawing, and ROI adjustment interactions."""

    # Zoom limits
    MIN_ZOOM = 0.1
    MAX_ZOOM = 30.0

    def __init__(self, scene: QGraphicsScene, dialog: "ROISelectionDialog") -> None:
        super().__init__(scene)
        self._dialog = dialog
        self._zoom_factor = 1.0
        self._panning = False
        self._pan_start = QPointF()

        self.setRenderHint(QPainter.Antialiasing, True)
        self.setRenderHint(QPainter.SmoothPixmapTransform, True)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setMouseTracking(True)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)

    # ----- zoom -----

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        """Zoom in/out with mouse wheel."""
        angle = event.angleDelta().y()
        if angle == 0:
            return
        factor = 1.15 if angle > 0 else 1 / 1.15
        new_zoom = self._zoom_factor * factor
        if new_zoom < self.MIN_ZOOM or new_zoom > self.MAX_ZOOM:
            return
        self._zoom_factor = new_zoom
        self.scale(factor, factor)

    def zoom_in(self) -> None:
        factor = 1.25
        new_zoom = self._zoom_factor * factor
        if new_zoom > self.MAX_ZOOM:
            return
        self._zoom_factor = new_zoom
        self.scale(factor, factor)

    def zoom_out(self) -> None:
        factor = 1 / 1.25
        new_zoom = self._zoom_factor * factor
        if new_zoom < self.MIN_ZOOM:
            return
        self._zoom_factor = new_zoom
        self.scale(factor, factor)

    def fit_view(self) -> None:
        self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)
        # Recalculate zoom factor from the current transform
        self._zoom_factor = self.transform().m11()

    # ----- pan via middle-button or Space+left-button -----

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MiddleButton:
            self._panning = True
            self._pan_start = event.position()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return

        # Forward to dialog logic for ROI interaction
        scene_pos = self.mapToScene(event.position().toPoint())
        self._dialog._handle_mouse_press(event, scene_pos)
        event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._panning:
            delta = event.position() - self._pan_start
            self._pan_start = event.position()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - int(delta.x())
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - int(delta.y())
            )
            event.accept()
            return

        scene_pos = self.mapToScene(event.position().toPoint())
        self._dialog._handle_mouse_move(event, scene_pos)
        event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MiddleButton and self._panning:
            self._panning = False
            self.setCursor(Qt.ArrowCursor)
            event.accept()
            return

        scene_pos = self.mapToScene(event.position().toPoint())
        self._dialog._handle_mouse_release(event, scene_pos)
        event.accept()

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        scene_pos = self.mapToScene(event.position().toPoint())
        self._dialog._handle_mouse_double_click(event, scene_pos)
        event.accept()

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        self._dialog._handle_key_press(event)


# ---------------------------------------------------------------------------
# ROI Selection Dialog
# ---------------------------------------------------------------------------

class ROISelectionDialog(QDialog):
    """Modal dialog for ROI polygon selection with zoom and pan."""

    def __init__(
        self,
        image: np.ndarray,
        existing_roi: Optional[ROI] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("ROI Selection")
        self.setModal(True)

        # ---- state ----
        self._image = image
        self._image_height, self._image_width = image.shape[:2]
        self._polygon_points: List[QPointF] = []
        self._preview_pos: Optional[QPointF] = None
        self._current_roi: Optional[ROI] = existing_roi
        self._selection_mode: bool = existing_roi is None
        self._adjust_mode: bool = False
        self._dragging_handle: HandleResult = ROIHandle.NONE
        self._last_drag_pos: Optional[QPointF] = None

        # ---- build UI ----
        self._build_ui()

        # ---- load image into scene ----
        self._load_image()

        # ---- if existing ROI, draw it ----
        self._update_overlay()
        self._update_button_states()

        # ---- size dialog ----
        screen = QApplication.primaryScreen()
        if screen:
            geom = screen.availableGeometry()
            w = max(900, int(geom.width() * 0.8))
            h = max(700, int(geom.height() * 0.8))
            self.resize(w, h)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        # --- Graphics view ---
        self._scene = QGraphicsScene(self)
        self._view = _ROIGraphicsView(self._scene, self)
        self._view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self._view, 1)

        # --- Status bar ---
        self._status_label = QLabel(self._status_text())
        layout.addWidget(self._status_label)

        # --- Toolbar ---
        toolbar = QHBoxLayout()

        # Zoom buttons
        self._zoom_in_btn = QPushButton("Zoom In (+)")
        self._zoom_out_btn = QPushButton("Zoom Out (-)")
        self._fit_btn = QPushButton("Fit to View")
        self._zoom_in_btn.clicked.connect(self._view.zoom_in)
        self._zoom_out_btn.clicked.connect(self._view.zoom_out)
        self._fit_btn.clicked.connect(self._on_fit_view)
        toolbar.addWidget(self._zoom_in_btn)
        toolbar.addWidget(self._zoom_out_btn)
        toolbar.addWidget(self._fit_btn)

        toolbar.addStretch()

        # ROI action buttons
        self._complete_btn = QPushButton("Complete Polygon")
        self._complete_btn.clicked.connect(self._complete_polygon)

        self._adjust_btn = QPushButton("Adjust ROI")
        self._adjust_btn.clicked.connect(self._toggle_adjust_mode)

        self._accept_btn = QPushButton("Accept ROI")
        self._accept_btn.clicked.connect(self._accept)

        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.clicked.connect(self.reject)

        toolbar.addWidget(self._complete_btn)
        toolbar.addWidget(self._adjust_btn)
        toolbar.addWidget(self._accept_btn)
        toolbar.addWidget(self._cancel_btn)

        layout.addLayout(toolbar)

    # ------------------------------------------------------------------
    # Image loading
    # ------------------------------------------------------------------

    def _load_image(self) -> None:
        qimg = numpy_to_qimage(self._image)
        pix = QPixmap.fromImage(qimg)
        self._pixmap_item = QGraphicsPixmapItem(pix)
        self._scene.addItem(self._pixmap_item)
        self._scene.setSceneRect(QRectF(0, 0, pix.width(), pix.height()))
        # Fit after a short delay so the view has its final size
        from PySide6.QtCore import QTimer
        QTimer.singleShot(0, self._on_fit_view)

    def _on_fit_view(self) -> None:
        self._view.fit_view()

    # ------------------------------------------------------------------
    # Overlay drawing (ROI polygon on the scene)
    # ------------------------------------------------------------------

    def _update_overlay(self) -> None:
        """Redraw ROI overlay items on the scene."""
        # Remove old overlay items (everything except the pixmap)
        for item in list(self._scene.items()):
            if item is not self._pixmap_item:
                self._scene.removeItem(item)

        if self._selection_mode and self._polygon_points:
            self._draw_selection_overlay()
        elif self._current_roi is not None and not self._selection_mode:
            self._draw_roi_overlay()

        self._status_label.setText(self._status_text())

    def _draw_selection_overlay(self) -> None:
        """Draw the polygon being constructed (red dashed)."""
        pen = QPen(QColor(255, 50, 50), 2, Qt.DashLine)
        pen.setCosmetic(True)  # constant width regardless of zoom
        vertex_pen = QPen(QColor(255, 50, 50), 1)
        vertex_pen.setCosmetic(True)
        vertex_brush = QBrush(QColor(255, 100, 100, 180))

        pts = self._polygon_points

        # Edges
        if len(pts) >= 2:
            for i in range(len(pts) - 1):
                line = self._scene.addLine(
                    pts[i].x(), pts[i].y(),
                    pts[i + 1].x(), pts[i + 1].y(),
                    pen,
                )
                line.setZValue(10)
            # Preview line to cursor
            if self._preview_pos is not None:
                preview_line = self._scene.addLine(
                    pts[-1].x(), pts[-1].y(),
                    self._preview_pos.x(), self._preview_pos.y(),
                    pen,
                )
                preview_line.setZValue(10)

        # Vertices
        r = 4  # radius in scene units (will stay visually small because cosmetic pen)
        for pt in pts:
            ellipse = self._scene.addEllipse(
                pt.x() - r, pt.y() - r, 2 * r, 2 * r,
                vertex_pen, vertex_brush,
            )
            ellipse.setZValue(11)

    def _draw_roi_overlay(self) -> None:
        """Draw the finalised ROI (green solid)."""
        roi = self._current_roi
        if roi is None:
            return

        pen = QPen(QColor(0, 220, 0), 2, Qt.SolidLine)
        pen.setCosmetic(True)

        if roi.shape == ROIShape.POLYGON and roi.points:
            qpts = [QPointF(p[0], p[1]) for p in roi.points]
            polygon = self._scene.addPolygon(
                QPolygonF(qpts), pen, QBrush(Qt.NoBrush)
            )
            polygon.setZValue(10)

            # Adjustment handles
            if self._adjust_mode:
                handle_pen = QPen(QColor(255, 255, 0), 2)
                handle_pen.setCosmetic(True)
                handle_brush = QBrush(QColor(0, 255, 255, 180))
                hs = 5  # half-size of handle square in scene pixels
                for pt in qpts:
                    rect = self._scene.addRect(
                        pt.x() - hs, pt.y() - hs, 2 * hs, 2 * hs,
                        handle_pen, handle_brush,
                    )
                    rect.setZValue(11)

    # ------------------------------------------------------------------
    # Button state management
    # ------------------------------------------------------------------

    def _update_button_states(self) -> None:
        has_roi = self._current_roi is not None
        drawing = self._selection_mode
        enough_pts = len(self._polygon_points) >= 3

        self._complete_btn.setVisible(drawing and enough_pts)
        self._adjust_btn.setVisible(has_roi and not drawing)
        self._accept_btn.setEnabled(has_roi)

        if self._adjust_mode:
            self._adjust_btn.setText("Finish Adjusting")
        else:
            self._adjust_btn.setText("Adjust ROI")

    def _status_text(self) -> str:
        if self._selection_mode:
            n = len(self._polygon_points)
            return (
                f"Drawing polygon: {n} point(s).  "
                "Left-click to add points. Double-click or press Complete to finish. "
                "Right-click / Backspace to undo. Scroll wheel to zoom. Middle-click to pan."
            )
        if self._adjust_mode:
            return (
                "Adjust mode: drag vertices to move them, drag inside polygon to move all. "
                "Right-click a vertex to delete it (min 3). Press Finish Adjusting when done."
            )
        if self._current_roi is not None:
            return "ROI ready. Press Accept ROI to confirm, or Adjust ROI to modify."
        return "Click to begin drawing a polygon ROI."

    # ------------------------------------------------------------------
    # Polygon completion / ROI creation
    # ------------------------------------------------------------------

    def _complete_polygon(self) -> None:
        if len(self._polygon_points) < 3:
            return
        pts = list(self._polygon_points)
        # Remove duplicate close-to-first point
        if len(pts) > 3:
            dx = abs(pts[-1].x() - pts[0].x())
            dy = abs(pts[-1].y() - pts[0].y())
            if dx < 10 and dy < 10:
                pts.pop()

        points_tuples = [(int(p.x()), int(p.y())) for p in pts]
        self._current_roi = ROI.from_dict({"shape": "polygon", "points": points_tuples})
        self._selection_mode = False
        self._polygon_points.clear()
        self._preview_pos = None
        self._adjust_mode = False
        self._update_overlay()
        self._update_button_states()

    def _toggle_adjust_mode(self) -> None:
        self._adjust_mode = not self._adjust_mode
        if not self._adjust_mode:
            self._dragging_handle = ROIHandle.NONE
            self._last_drag_pos = None
        self._update_overlay()
        self._update_button_states()

    def _accept(self) -> None:
        if self._current_roi is not None:
            self.accept()  # QDialog.Accepted

    def get_roi(self) -> Optional[ROI]:
        return self._current_roi

    # ------------------------------------------------------------------
    # Mouse / key handlers (called from _ROIGraphicsView)
    # ------------------------------------------------------------------

    def _handle_mouse_press(self, event: QMouseEvent, scene_pos: QPointF) -> None:
        x, y = scene_pos.x(), scene_pos.y()

        # Clamp to image bounds
        x = max(0, min(self._image_width - 1, x))
        y = max(0, min(self._image_height - 1, y))

        # --- right-click ---
        if event.button() == Qt.RightButton:
            if self._selection_mode and self._polygon_points:
                self._polygon_points.pop()
                self._update_overlay()
                self._update_button_states()
            elif (
                self._adjust_mode
                and self._current_roi is not None
                and self._current_roi.shape == ROIShape.POLYGON
                and self._current_roi.points
                and len(self._current_roi.points) > 3
            ):
                handle = self._current_roi.get_handle_at_point(int(x), int(y), 8)
                if isinstance(handle, tuple) and handle[0] == ROIHandle.VERTEX:
                    _, vidx = handle
                    self._current_roi.delete_vertex(vidx)
                    self._update_overlay()
            return

        if event.button() != Qt.LeftButton:
            return

        # --- adjustment drag start ---
        if self._adjust_mode and self._current_roi is not None:
            handle = self._current_roi.get_handle_at_point(int(x), int(y), 8)
            if handle != ROIHandle.NONE:
                self._dragging_handle = handle
                self._last_drag_pos = QPointF(x, y)
                return

        # --- selection mode: add point ---
        if self._selection_mode:
            self._polygon_points.append(QPointF(x, y))
            self._update_overlay()
            self._update_button_states()

    def _handle_mouse_move(self, event: QMouseEvent, scene_pos: QPointF) -> None:
        x, y = scene_pos.x(), scene_pos.y()
        x = max(0, min(self._image_width - 1, x))
        y = max(0, min(self._image_height - 1, y))

        # Adjustment drag
        if (
            self._adjust_mode
            and self._dragging_handle != ROIHandle.NONE
            and self._last_drag_pos is not None
            and self._current_roi is not None
        ):
            dx = int(x - self._last_drag_pos.x())
            dy = int(y - self._last_drag_pos.y())
            self._current_roi.adjust_with_handle(
                self._dragging_handle, dx, dy, self._image_width, self._image_height
            )
            self._last_drag_pos = QPointF(x, y)
            self._update_overlay()
            return

        # Selection preview
        if self._selection_mode and self._polygon_points:
            self._preview_pos = QPointF(x, y)
            self._update_overlay()

    def _handle_mouse_release(self, event: QMouseEvent, scene_pos: QPointF) -> None:
        if event.button() == Qt.LeftButton and self._dragging_handle != ROIHandle.NONE:
            self._dragging_handle = ROIHandle.NONE
            self._last_drag_pos = None
            self._update_overlay()

    def _handle_mouse_double_click(self, event: QMouseEvent, scene_pos: QPointF) -> None:
        if event.button() != Qt.LeftButton:
            return
        if self._selection_mode and len(self._polygon_points) >= 3:
            # The preceding mousePressEvent already appended a point for this
            # same click; remove it so the double-click does not create a
            # duplicate vertex.  (_complete_polygon has its own close-to-first
            # dedup which stays unchanged.)
            self._polygon_points.pop()
            self._complete_polygon()

    def _handle_key_press(self, event: QKeyEvent) -> None:
        key = event.key()
        if key in (Qt.Key.Key_Backspace, Qt.Key.Key_Delete):
            if self._selection_mode and self._polygon_points:
                self._polygon_points.pop()
                self._update_overlay()
                self._update_button_states()
                event.accept()
                return
        if key == Qt.Key.Key_Escape:
            self.reject()
            event.accept()
            return
        # Pass plus/minus as zoom shortcuts
        if key == Qt.Key.Key_Plus or key == Qt.Key.Key_Equal:
            self._view.zoom_in()
            event.accept()
            return
        if key == Qt.Key.Key_Minus:
            self._view.zoom_out()
            event.accept()
            return
