from __future__ import annotations

from collections import OrderedDict
from typing import List, Optional, Tuple
from pathlib import Path
import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal, QRect, QPoint, QPointF, QEvent
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QBrush, QIcon, QPolygonF
from PySide6.QtWidgets import (
    QLabel,
    QSlider,
    QVBoxLayout,
    QWidget,
    QPushButton,
    QHBoxLayout,
    QStyle,
)

from utils.file_handler import ImageStackHandler
from core.roi import ROI, ROIShape, ROIHandle, HandleResult


class _LRUCache:
    def __init__(self, capacity: int = 20) -> None:
        self.capacity = capacity
        self.store: "OrderedDict[int, np.ndarray]" = OrderedDict()

    def get(self, key: int) -> Optional[np.ndarray]:
        if key not in self.store:
            return None
        value = self.store.pop(key)
        self.store[key] = value
        return value

    def set(self, key: int, value: np.ndarray) -> None:
        if key in self.store:
            self.store.pop(key)
        elif len(self.store) >= self.capacity:
            self.store.popitem(last=False)
        self.store[key] = value


class ImageViewer(QWidget):
    stackLoaded = Signal(str)
    roiSelected = Signal(object)  # Emits ROI object
    roiChanged = Signal(object)  # Emits ROI object when adjusted
    displaySettingsChanged = Signal(int, int)  # Emits (exposure, contrast) when display settings change

    def __init__(self, handler: ImageStackHandler) -> None:
        super().__init__()
        self.handler = handler
        self.index = 0
        self.cache = _LRUCache(20)

        # ROI selection state (polygon mode)
        self.roi_selection_mode = False
        self.roi_adjustment_mode = False  # User must explicitly enable adjustment
        self.roi_start_point = None
        self.roi_end_point = None
        self.polygon_points: List[QPoint] = []
        self.polygon_preview_pos: Optional[QPoint] = None  # Mouse position for live preview
        self.current_roi: Optional[ROI] = None
        self.selected_shape = ROIShape.POLYGON  # Polygon is the primary ROI tool
        
        # ROI adjustment state
        self.active_handle: HandleResult = ROIHandle.NONE
        self.last_mouse_pos = None
        self.can_adjust_roi = False  # Only true when user clicks "Adjust ROI"

        self.filename_label = QLabel("Load image to see data") #label for user to see if no image are selected
        self.filename_label.setAlignment(Qt.AlignCenter)
        self.filename_label.setWordWrap(True)
        self.filename_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        
        # Create image display area with upload button
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(320, 240)
        self.image_label.setMouseTracking(True)
        self.image_label.mousePressEvent = self._on_mouse_press
        self.image_label.mouseMoveEvent = self._on_mouse_move
        self.image_label.mouseReleaseEvent = self._on_mouse_release
        self.image_label.mouseDoubleClickEvent = self._on_mouse_double_click
        self.image_label.setFocusPolicy(Qt.StrongFocus)
        self.image_label.installEventFilter(self)
        
        # Upload button (visible when no images loaded)
        self.upload_btn = QPushButton("Open Images")
        # Add standard Qt file open icon
        icon = self.style().standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon)
        self.upload_btn.setIcon(icon)
        self.upload_btn.clicked.connect(self._open_upload_dialog)
        
        # Container for image label with upload button overlay
        self.image_container = QWidget()
        image_layout = QVBoxLayout(self.image_container)
        image_layout.setContentsMargins(0, 0, 0, 0)
        image_layout.addWidget(self.image_label)
        
        # Overlay upload button on image label
        self.upload_btn.setParent(self.image_label)
        self.upload_btn.show()
        
        # Schedule button centering after layout is complete
        from PySide6.QtCore import QTimer
        QTimer.singleShot(0, self._center_upload_button)

        self.prev_btn = QPushButton("Previous")
        self.next_btn = QPushButton("Next")
        self.roi_btn = QPushButton("Select ROI")
        self.complete_polygon_btn = QPushButton("Complete Polygon")
        self.complete_polygon_btn.setVisible(False)  # Visible only in selection mode when >= 3 points
        self.complete_polygon_btn.clicked.connect(self._complete_polygon)
        self.adjust_roi_btn = QPushButton("Adjust ROI")
        self.adjust_roi_btn.setVisible(False)  # Hidden until ROI exists
        
        self.prev_btn.clicked.connect(self.prev_image)
        self.next_btn.clicked.connect(self.next_image)
        self.roi_btn.clicked.connect(self._toggle_roi_mode)
        self.adjust_roi_btn.clicked.connect(self._toggle_adjustment_mode)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.valueChanged.connect(self._on_slider)

        ## Exposure/Contrast slider controls for the exposure
        # Label text at inital load
        self.exposure_label = QLabel("Exposure: 0")
        # Adds the slider movement as a horizontal slider "vertical setting also"
        self.exposure_slider = QSlider(Qt.Horizontal)
        # Setting the value in the slider, set value to -100 - 100 for more accuracy
        self.exposure_slider.setRange(-100, 100)
        # The slider will start 0 at the load in
        self.exposure_slider.setValue(0)
        # Handle the changeing value for the exposure slider
        self.exposure_slider.valueChanged.connect(self._on_adjustment_changed)

        # Label text at inital load for the contrast
        self.contrast_label = QLabel("Contrast: 0")
        # Adds the slider movement as a horizontal slider "vertical setting also"
        self.contrast_slider = QSlider(Qt.Horizontal)
        # Setting the values in the slider, set value to -100 - 100 for more accuracy
        self.contrast_slider.setRange(-100, 100)
        # Slider will start at 0 on load in
        self.contrast_slider.setValue(0)
        #handle the changing value for the contrast slider
        self.contrast_slider.valueChanged.connect(self._on_adjustment_changed)
        

        nav = QHBoxLayout()
        nav.addWidget(self.prev_btn)
        nav.addWidget(self.next_btn)
        nav.addWidget(self.roi_btn)
        nav.addWidget(self.complete_polygon_btn)
        nav.addWidget(self.adjust_roi_btn)

        # New box for the exposure and  contrast slider
        adjustments_layout = QVBoxLayout()
        # Add the label to the slider
        adjustments_layout.addWidget(self.exposure_label)
        # Add the slider to the container
        adjustments_layout.addWidget(self.exposure_slider)
        # Added the label to the slider
        adjustments_layout.addWidget(self.contrast_label)
        # Add the slider to the container
        adjustments_layout.addWidget(self.contrast_slider)

        layout = QVBoxLayout(self)
        # Image gets most of the space (stretch factor 1)
        layout.addWidget(self.image_container, 1)
        layout.addLayout(nav)
        layout.addWidget(self.slider)
        # Added the layout for the exposure and contrast
        layout.addLayout(adjustments_layout)
        # Metadata label should be compact (stretch factor 0, max height)
        self.filename_label.setMaximumHeight(50)
        layout.addWidget(self.filename_label, 0)
        # adjust the labels on sliders to go to 0 on load
        self._update_adjustment_labels()

        self.setAcceptDrops(True)
        self.setFocusPolicy(Qt.StrongFocus)

    def set_stack(self, files) -> None:
        self.handler.load_image_stack(files)
        self.slider.setRange(0, max(0, self.handler.get_image_count() - 1))
        self.index = 0
        self._show_current()
        
        # Hide upload button when images are loaded
        if self.handler.get_image_count() > 0:
            self.upload_btn.hide()
        
        # Determine directory path and emit
        directory: Optional[str] = None
        if isinstance(files, (list, tuple)) and files:
            directory = str(Path(files[0]).parent)
        elif isinstance(files, str):
            p = Path(files)
            directory = str(p if p.is_dir() else p.parent)
        if directory:
            self.stackLoaded.emit(directory)

    def reset_cache(self) -> None:
        """Reset the image cache."""
        self.cache = _LRUCache(20)

    def reset(self) -> None:
        """Reset the viewer to initial state."""
        # Clear handler files and reset navigation
        self.handler.files = []
        self.index = 0
        # Reset cache and ROI-related state
        self.cache = _LRUCache(20)
        self.current_roi = None
        self.roi_selection_mode = False
        self.roi_adjustment_mode = False
        self.can_adjust_roi = False
        self.roi_start_point = None
        self.roi_end_point = None
        self.polygon_points = []
        self.polygon_preview_pos = None
        self.active_handle = ROIHandle.NONE
        self.last_mouse_pos = None
        # Reset UI labels and slider
        self.image_label.clear()
        self.filename_label.setText("Load image to see data")
        self.slider.setRange(0, 0)
        self._update_roi_button_text()
        self.adjust_roi_btn.setVisible(False)
        self.complete_polygon_btn.setVisible(False)
        # Re-enable all controls
        self.prev_btn.setEnabled(True)
        self.next_btn.setEnabled(True)
        self.slider.setEnabled(True)
        self.roi_btn.setEnabled(True)
        # Show upload button again
        self.upload_btn.show()
        self._center_upload_button()

    def dragEnterEvent(self, event) -> None:  # noqa: N802
        md = event.mimeData()
        if md.hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event) -> None:  # noqa: N802
        urls = event.mimeData().urls()
        paths = [u.toLocalFile() for u in urls]
        if not paths:
            return
        
        # If a single directory dropped, use directory
        if len(paths) == 1 and Path(paths[0]).is_dir():
            self.set_stack(paths[0])
        else:
            # Filter to only allow TIF and GIF files
            allowed_extensions = {'.tif', '.tiff', '.gif'}
            filtered_paths = [
                p for p in paths 
                if Path(p).suffix.lower() in allowed_extensions
            ]
            
            if filtered_paths:
                self.set_stack(filtered_paths)
            else:
                # Show message if no valid files were dropped
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.warning(
                    self,
                    "Invalid Files",
                    "Only TIF and GIF files are supported."
                )

    def _numpy_to_qimage(self, arr: np.ndarray) -> QImage:
        if arr.ndim == 2:
            h, w = arr.shape
            fmt = (
                QImage.Format_Grayscale8
                if arr.dtype != np.uint16
                else QImage.Format_Grayscale16
            )
            bytes_per_line = arr.strides[0]
            return QImage(arr.data, w, h, bytes_per_line, fmt)
        if arr.ndim == 3:
            h, w, c = arr.shape
            if c == 3:
                return QImage(arr.data, w, h, 3 * w, QImage.Format_RGB888)
            if c == 4:
                return QImage(arr.data, w, h, 4 * w, QImage.Format_RGBA8888)
        raise ValueError("Unsupported image shape")

    # Function to update the silder value so the user can see what value they have
    def _update_adjustment_labels(self) -> None:
        # For exposure
        self.exposure_label.setText(f"Exposure: {self.exposure_slider.value()}")
        # For contrast
        self.contrast_label.setText(f"Contrast: {self.contrast_slider.value()}")

    '''Next three function convert to 8 bit and do exposure and contrast
    calculation. The order is a stated: Exposure/Contrast, then 8 bit converstion
    This will preserve the appearance and will allow for more adjustment'''
    #function to calculate teh amount of exposure and contrast the imahe
    # should get based on the slider value input
    def _apply_adjustments(self, arr: np.ndarray) -> np.ndarray:

        # Set the exposure and contrast sliders value to ev and cv
        ev = self.exposure_slider.value()
        cv = self.contrast_slider.value()
        #stores the orignal image data type
        orig_dtype = arr.dtype
        # convert the image to float32 and saves it as a new array
        # copy=false stop the automatic saving of this array from astype(),
        # if step to true then astype() will save a array if one value is off.
        new_arr = arr.astype(np.float32, copy=True)
        min_pixel = float(np.min(new_arr))
        max_pixel = float(np.max(new_arr))
        pixel_range = max_pixel - min_pixel
        # check to see if the pixel arent all equal
        # cant divied by 0 
        if pixel_range != 0:
            # normalize all the images in the array
            new_arr = (new_arr - min_pixel) / pixel_range
        else:
            # if the orignal image data type is a integer image
            # This is because float and integer data type normalize differently
            if np.issubdtype(orig_dtype, np.integer):
                #get the max out of the image arr
                max_possible = float(np.iinfo(orig_dtype).max)
                #normailze the image
                new_arr = new_arr / max_possible
            else:
                #the image is normalized
                new_arr = np.clip(new_arr, 0, 1)

        #creating exposure and contrast scalers
        exposure = 2 ** (ev / 50)               
        contrast = 1 + (cv / 100) 
        #0.5 to preserve the greyscale... if higher turns black...if lower turns white      
        new_arr = ((new_arr - 0.5) * contrast + 0.5) * exposure
        # this will set the max and min values to 0 to 1
        # you will get more uniformed ranges in contrast and exposure
        new_arr = np.clip(new_arr, 0, 1)
        # return the normalized array with edits
        return new_arr


    def _on_adjustment_changed(self, _value: int) -> None:
        self._update_adjustment_labels()
        self._show_current()
        # Emit signal so MainWindow can save to experiment
        self.displaySettingsChanged.emit(self.exposure_slider.value(), self.contrast_slider.value())


    # Function to convert to 8 bits
    def _ensure_uint8(self, arr: np.ndarray) -> np.ndarray:
        # Do the contrast and exposure edits first
        new_arr = self._apply_adjustments(arr)
        # convert to unit 8... 8 bit
        unit_8 = cv2.convertScaleAbs(new_arr, alpha=255.0, beta=0.0)
        #return the 8 bit image
        return unit_8

    def _show_current(self) -> None:
        count = self.handler.get_image_count()
        if count == 0:
            self.image_label.clear()
            self.filename_label.setText("Load image to see data")
            # Make sure upload button is visible and centered
            if not self.upload_btn.isVisible():
                self.upload_btn.show()
                # Delay centering to ensure layout is complete
                from PySide6.QtCore import QTimer
                QTimer.singleShot(10, self._center_upload_button)
            else:
                self._center_upload_button()
            return
        img = self.cache.get(self.index)
        if img is None:
            img = self.handler.get_image_at_index(self.index)
            self.cache.set(self.index, img)
        #show the 8 bit image on the workbench
        preview_img = self._ensure_uint8(img)
        qimg = self._numpy_to_qimage(preview_img)
        pix = QPixmap.fromImage(qimg)
        scaled_pix = pix.scaled(
            self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )

        if img.ndim >= 2:
            original_height, original_width = img.shape[0], img.shape[1]
            label_size = self.image_label.size()
            label_aspect = label_size.width() / label_size.height()
            original_aspect = original_width / original_height
            scale = (
                scaled_pix.height() / original_height
                if label_aspect > original_aspect
                else scaled_pix.width() / original_width
            )
        else:
            scale = 1.0

        # ============================================================
        # ROI: Draw polygon during selection mode
        # ============================================================
        if self.roi_selection_mode and self.polygon_points:
            painter = QPainter(scaled_pix)
            pen = QPen(Qt.red, 2, Qt.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)

            # Draw polygon edges
            pts = [QPointF(int(p.x() * scale), int(p.y() * scale)) for p in self.polygon_points]
            if len(pts) >= 2:
                for i in range(len(pts) - 1):
                    painter.drawLine(pts[i], pts[i + 1])
                # Preview line from last point to cursor
                if self.polygon_preview_pos is not None:
                    prev_pt = QPointF(
                        int(self.polygon_preview_pos.x() * scale),
                        int(self.polygon_preview_pos.y() * scale),
                    )
                    painter.drawLine(pts[-1], prev_pt)
            # Draw vertex circles
            for pt in pts:
                painter.drawEllipse(pt, 4, 4)
            painter.end()

        # ============================================================
        # ROI: Draw saved ROI when not in selection mode
        # ============================================================
        elif self.current_roi is not None and not self.roi_selection_mode:
            painter = QPainter(scaled_pix)
            pen = QPen(Qt.green, 2, Qt.SolidLine)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)

            if self.current_roi.shape == ROIShape.POLYGON and self.current_roi.points:
                qpts = [
                    QPointF(int(p[0] * scale), int(p[1] * scale))
                    for p in self.current_roi.points
                ]
                painter.drawPolygon(QPolygonF(qpts))

                if self.can_adjust_roi:
                    handle_size = 10
                    painter.setPen(QPen(QColor(255, 255, 0), 2))
                    painter.setBrush(QBrush(QColor(0, 255, 255)))
                    for pt in qpts:
                        painter.drawRect(
                            int(pt.x()) - handle_size // 2,
                            int(pt.y()) - handle_size // 2,
                            handle_size,
                            handle_size,
                        )
            else:
                x_scaled = int(self.current_roi.x * scale)
                y_scaled = int(self.current_roi.y * scale)
                w_scaled = int(self.current_roi.width * scale)
                h_scaled = int(self.current_roi.height * scale)
                painter.drawEllipse(x_scaled, y_scaled, w_scaled, h_scaled)

                if self.can_adjust_roi:
                    handle_size = 10
                    painter.setPen(QPen(QColor(255, 255, 0), 2))
                    painter.setBrush(QBrush(QColor(0, 255, 255)))
                    corners = [
                        (x_scaled, y_scaled),
                        (x_scaled + w_scaled, y_scaled),
                        (x_scaled, y_scaled + h_scaled),
                        (x_scaled + w_scaled, y_scaled + h_scaled),
                    ]
                    for cx, cy in corners:
                        painter.drawRect(
                            cx - handle_size // 2, cy - handle_size // 2,
                            handle_size, handle_size
                        )
            painter.end()

        self.image_label.setPixmap(scaled_pix)
        current_path = Path(self.handler.files[self.index])
        base_text = f"{self.index + 1}/{count}: \n{current_path.name}"
        if self.roi_selection_mode and self.polygon_points:
            base_text += f" | Polygon: {len(self.polygon_points)} points (dbl-click or Complete to finish)"
        self.filename_label.setText(base_text)

    def resizeEvent(self, event) -> None:  # noqa: N802
        # ROI FIX: Redraw on resize so ROI scale updates correctly
        # When the window/pane is resized, the scaled_pix size changes
        # By calling _show_current(), we recalculate the scale factor and redraw the ROI
        # at the correct position for the new display size
        super().resizeEvent(event)
        self._show_current()
        # Center upload button when window resizes
        if self.upload_btn.isVisible():
            self._center_upload_button()
    
    def _center_upload_button(self) -> None:
        """Center the upload button in the image label."""
        if not self.upload_btn.isVisible():
            return
            
        # Use parent (image_label) size for centering
        parent_width = self.image_label.width()
        parent_height = self.image_label.height()
        btn_width = self.upload_btn.width()
        btn_height = self.upload_btn.height()
        
        # Calculate center position
        x = max(0, (parent_width - btn_width) // 2)
        y = max(0, (parent_height - btn_height) // 2)
        
        self.upload_btn.move(x, y)
    
    def _open_upload_dialog(self) -> None:
        """Open file dialog to select TIF or GIF images."""
        from PySide6.QtWidgets import QFileDialog
        
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Image Files",
            "",
            "Image Files (*.tif *.tiff *.gif);;TIF Files (*.tif *.tiff);;GIF Files (*.gif);;All Files (*.*)"
        )
        
        if files:
            self.set_stack(files)

    def prev_image(self) -> None:
        if self.index > 0:
            self.index -= 1
            self.slider.blockSignals(True)
            self.slider.setValue(self.index)
            self.slider.blockSignals(False)
            self._show_current()

    def next_image(self) -> None:
        if self.index < max(0, self.handler.get_image_count() - 1):
            self.index += 1
            self.slider.blockSignals(True)
            self.slider.setValue(self.index)
            self.slider.blockSignals(False)
            self._show_current()

    def _on_slider(self, value: int) -> None:
        self.index = value
        self._show_current()

    def _toggle_roi_mode(self) -> None:
        """Toggle ROI selection mode."""
        from PySide6.QtWidgets import QMessageBox
        
        # If we're not in selection mode and there's an existing ROI, confirm before starting new selection
        if not self.roi_selection_mode and self.current_roi is not None:
            reply = QMessageBox.question(
                self,
                "Create New ROI",
                "Creating a new ROI will replace the existing one. Continue?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply == QMessageBox.No:
                return
            # Clear existing ROI to start fresh
            self.current_roi = None
            self.can_adjust_roi = False
            self.adjust_roi_btn.setVisible(False)
        
        self.roi_selection_mode = not self.roi_selection_mode
        if self.roi_selection_mode:
            self.image_label.setFocus()
        self.roi_btn.setText("Cancel ROI" if self.roi_selection_mode else (
            "New ROI" if self.current_roi is not None else "Select ROI"
        ))
        if not self.roi_selection_mode:
            self.roi_start_point = None
            self.roi_end_point = None
            self.polygon_points = []
            self.polygon_preview_pos = None
            self.complete_polygon_btn.setVisible(False)
        self._show_current()
    
    def _toggle_adjustment_mode(self) -> None:
        """Toggle ROI adjustment mode."""
        self.can_adjust_roi = not self.can_adjust_roi
        self.adjust_roi_btn.setText("Finish Adjusting" if self.can_adjust_roi else "Adjust ROI")
        
        # Disable/enable other controls based on adjustment mode
        self.prev_btn.setEnabled(not self.can_adjust_roi)
        self.next_btn.setEnabled(not self.can_adjust_roi)
        self.slider.setEnabled(not self.can_adjust_roi)
        self.roi_btn.setEnabled(not self.can_adjust_roi)
        
        # Exit adjustment mode properly
        if not self.can_adjust_roi:
            self.roi_adjustment_mode = False
            self.active_handle = ROIHandle.NONE
            self.last_mouse_pos = None
            # Emit final ROI state after adjustment is complete
            if self.current_roi is not None:
                self.roiSelected.emit(self.current_roi)
        
        self._show_current()
    
    def _update_roi_button_text(self) -> None:
        """Update ROI button text based on current state."""
        if self.roi_selection_mode:
            self.roi_btn.setText("Cancel ROI")
        elif self.current_roi is not None:
            self.roi_btn.setText("New ROI")
        else:
            self.roi_btn.setText("Select ROI")

        # Show Complete Polygon button when in selection mode with >= 3 points
        if self.roi_selection_mode and len(self.polygon_points) >= 3:
            self.complete_polygon_btn.setVisible(True)
        else:
            self.complete_polygon_btn.setVisible(False)

        # Show/hide adjust button based on ROI existence
        if self.current_roi is not None and not self.roi_selection_mode:
            self.adjust_roi_btn.setVisible(True)
        else:
            self.adjust_roi_btn.setVisible(False)

    def _complete_polygon(self) -> None:
        """Complete polygon and create ROI."""
        if not self.roi_selection_mode or len(self.polygon_points) < 3:
            return
        self._finish_polygon_roi()

    def _finish_polygon_roi(self) -> None:
        """Create ROI from polygon_points and exit selection mode."""
        if len(self.polygon_points) < 3:
            return
        pts = list(self.polygon_points)
        # If last point is very close to first, remove it (user closed by clicking near start)
        if len(pts) > 3:
            dx = abs(pts[-1].x() - pts[0].x())
            dy = abs(pts[-1].y() - pts[0].y())
            if dx < 10 and dy < 10:
                pts.pop()
        points_tuples = [(p.x(), p.y()) for p in pts]
        self.current_roi = ROI.from_dict({"shape": "polygon", "points": points_tuples})
        self.roiSelected.emit(self.current_roi)
        self.roi_selection_mode = False
        self.polygon_points = []
        self.polygon_preview_pos = None
        self.can_adjust_roi = False
        self._update_roi_button_text()
        self._show_current()

    def eventFilter(self, obj, event) -> bool:
        """Handle Backspace on image label to remove last polygon point."""
        if (
            obj == self.image_label
            and event.type() == QEvent.Type.KeyPress
            and event.key() in (Qt.Key.Backspace, Qt.Key.Delete)
            and self.roi_selection_mode
            and self.polygon_points
        ):
            self.polygon_points.pop()
            self._update_roi_button_text()
            self._show_current()
            return True
        return super().eventFilter(obj, event)

    def keyPressEvent(self, event) -> None:  # noqa: N802
        """Handle Backspace to remove last polygon point when viewer has focus."""
        if (
            self.roi_selection_mode
            and self.polygon_points
            and event.key() in (Qt.Key.Backspace, Qt.Key.Delete)
        ):
            self.polygon_points.pop()
            self._update_roi_button_text()
            self._show_current()
            event.accept()
            return
        super().keyPressEvent(event)

    def _on_mouse_double_click(self, event) -> None:
        """Complete polygon on double-click."""
        if event.button() != Qt.LeftButton:
            return
        if self.roi_selection_mode and len(self.polygon_points) >= 3:
            self._finish_polygon_roi()

    def _get_image_coords_from_mouse(self, event) -> Optional[Tuple[int, int, float]]:
        """Convert mouse coordinates to image coordinates and return scale."""
        # Check if there are any images loaded
        if self.handler.get_image_count() == 0:
            return None
        
        img = self.cache.get(self.index)
        if img is None:
            try:
                img = self.handler.get_image_at_index(self.index)
            except (IndexError, Exception):
                return None
        if img is None or img.ndim < 2:
            return None
            
        original_height, original_width = img.shape[0], img.shape[1]
        label_size = self.image_label.size()
        pixmap = self.image_label.pixmap()
        
        if not pixmap:
            return None
            
        scaled_pixmap_size = pixmap.size()
        label_aspect = label_size.width() / label_size.height()
        original_aspect = original_width / original_height

        if label_aspect > original_aspect:
            scale = scaled_pixmap_size.height() / original_height
        else:
            scale = scaled_pixmap_size.width() / original_width

        mouse_x = event.position().x()
        mouse_y = event.position().y()
        offset_x = (label_size.width() - scaled_pixmap_size.width()) / 2
        offset_y = (label_size.height() - scaled_pixmap_size.height()) / 2

        x = int((mouse_x - offset_x) / scale)
        y = int((mouse_y - offset_y) / scale)
        x = max(0, min(original_width - 1, x))
        y = max(0, min(original_height - 1, y))
        
        return (x, y, scale)
    
    def _on_mouse_press(self, event) -> None:
        """Handle mouse press for ROI selection and adjustment."""
        if self.handler.get_image_count() == 0:
            return

        coords = self._get_image_coords_from_mouse(event)
        if coords is None:
            return
        x, y, scale = coords

        # Right-click: remove last polygon point (selection) or delete vertex (adjustment)
        if event.button() == Qt.RightButton:
            if self.roi_selection_mode and self.polygon_points:
                self.polygon_points.pop()
                self._update_roi_button_text()
                self._show_current()
            elif (
                self.current_roi is not None
                and self.current_roi.shape == ROIShape.POLYGON
                and self.current_roi.points
                and len(self.current_roi.points) > 3
                and self.can_adjust_roi
            ):
                handle_size_image = max(2, int(10 / scale))
                handle_result = self.current_roi.get_handle_at_point(x, y, handle_size_image)
                if isinstance(handle_result, tuple) and handle_result[0] == ROIHandle.VERTEX:
                    _, vidx = handle_result
                    if self.current_roi.delete_vertex(vidx):
                        self.roiSelected.emit(self.current_roi)
                        self._show_current()
            return

        if event.button() != Qt.LeftButton:
            return

        # Check if adjusting existing ROI (only if adjustment mode is enabled)
        if self.current_roi is not None and not self.roi_selection_mode and self.can_adjust_roi:
            handle_size_image = max(2, int(10 / scale))
            handle_result = self.current_roi.get_handle_at_point(x, y, handle_size_image)
            if handle_result != ROIHandle.NONE:
                self.active_handle = handle_result
                self.roi_adjustment_mode = True
                self.last_mouse_pos = QPoint(x, y)
                return

        # Polygon selection mode: left-click adds point
        if self.roi_selection_mode:
            self.polygon_points.append(QPoint(x, y))
            self._update_roi_button_text()
            self._show_current()

    def _on_mouse_move(self, event) -> None:
        """Handle mouse move for ROI selection and adjustment."""
        if self.handler.get_image_count() == 0:
            return

        coords = self._get_image_coords_from_mouse(event)
        if coords is None:
            return
        x, y, _ = coords

        # Handle ROI adjustment (only if adjustment mode is enabled)
        if self.roi_adjustment_mode and self.last_mouse_pos is not None and self.can_adjust_roi:
            img = self.cache.get(self.index)
            if img is None:
                img = self.handler.get_image_at_index(self.index)
            if img is not None and img.ndim >= 2:
                original_height, original_width = img.shape[0], img.shape[1]
                dx = x - self.last_mouse_pos.x()
                dy = y - self.last_mouse_pos.y()
                self.current_roi.adjust_with_handle(
                    self.active_handle, dx, dy, original_width, original_height
                )
                self.last_mouse_pos = QPoint(x, y)
                self._show_current()
                self.roiChanged.emit(self.current_roi)

        # Polygon selection mode: update preview position
        elif self.roi_selection_mode:
            self.polygon_preview_pos = QPoint(x, y)
            self._show_current()

    def _on_mouse_release(self, event) -> None:
        """Handle mouse release for ROI adjustment completion."""
        if event.button() != Qt.LeftButton:
            return
        if self.handler.get_image_count() == 0:
            return

        # Handle ROI adjustment completion
        if self.roi_adjustment_mode:
            self.roi_adjustment_mode = False
            self.active_handle = ROIHandle.NONE
            self.last_mouse_pos = None
            if self.current_roi is not None:
                self.roiSelected.emit(self.current_roi)

    def get_current_roi(self) -> Optional[ROI]:
        """Get the current ROI object."""
        return self.current_roi

    def get_exposure(self) -> int:
        """Get the current exposure value (-100 to 100)."""
        return self.exposure_slider.value()

    def get_contrast(self) -> int:
        """Get the current contrast value (-100 to 100)."""
        return self.contrast_slider.value()

    def set_exposure(self, value: int) -> None:
        """Set the exposure value (-100 to 100)."""
        # Block signals to avoid triggering save during load
        self.exposure_slider.blockSignals(True)
        self.exposure_slider.setValue(value)
        self.exposure_slider.blockSignals(False)
        self._update_adjustment_labels()

    def set_contrast(self, value: int) -> None:
        """Set the contrast value (-100 to 100)."""
        # Block signals to avoid triggering save during load
        self.contrast_slider.blockSignals(True)
        self.contrast_slider.setValue(value)
        self.contrast_slider.blockSignals(False)
        self._update_adjustment_labels()

    def set_roi(self, roi: ROI) -> None:
        """
        Set the ROI from a saved ROI object.
        
        This method is called when loading an experiment with a saved ROI.
        The coordinates are in original image pixel space, not display/widget space.
        """
        self.current_roi = roi
        self.can_adjust_roi = False  # Start with adjustment disabled
        # Update button text
        self._update_roi_button_text()
        # Redraw to show the ROI with correct scaling
        self._show_current()

