from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QPainter, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QStyle,
    QVBoxLayout,
    QWidget,
)

from core.roi import ROI, ROIShape
from utils.file_handler import ImageStackHandler
from utils.image_utils import numpy_to_qimage


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
    displaySettingsChanged = Signal(
        int, int
    )  # Emits (exposure, contrast) when display settings change

    def __init__(self, handler: ImageStackHandler) -> None:
        super().__init__()
        self.handler = handler
        self.index = 0
        self.cache = _LRUCache(20)

        # ROI state (selection/adjustment happens in ROISelectionDialog)
        self.current_roi: Optional[ROI] = None

        self.filename_label = QLabel(
            "Load image to see data"
        )  # label for user to see if no image are selected
        self.filename_label.setAlignment(Qt.AlignCenter)
        self.filename_label.setWordWrap(True)
        self.filename_label.setTextInteractionFlags(Qt.TextSelectableByMouse)

        # Create image display area with upload button
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(320, 240)

        # Upload button (visible when no images loaded)
        self.upload_btn = QPushButton("Open Images")
        self.upload_btn.setProperty("class", "primary")
        # Add standard Qt file open icon
        icon = self.style().standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon)
        self.upload_btn.setIcon(icon)
        self.upload_btn.clicked.connect(self._open_upload_dialog)

        # Container for image label with upload button overlay
        self.image_container = QWidget()
        self.image_container.setObjectName("imageContainer")
        image_layout = QVBoxLayout(self.image_container)
        image_layout.setContentsMargins(0, 0, 0, 0)
        image_layout.addWidget(self.image_label)

        # Preview group: border and title around the image area
        self.preview_group = QGroupBox("Preview")
        preview_group_layout = QVBoxLayout(self.preview_group)
        preview_group_layout.setContentsMargins(8, 16, 8, 8)
        preview_group_layout.addWidget(self.image_container)

        # Overlay upload button on image label
        self.upload_btn.setParent(self.image_label)
        self.upload_btn.show()

        # Schedule button centering after layout is complete
        from PySide6.QtCore import QTimer

        QTimer.singleShot(0, self._center_upload_button)

        self.prev_btn = QPushButton("Previous")
        self.next_btn = QPushButton("Next")
        self.roi_btn = QPushButton("Select ROI")
        self.adjust_roi_btn = QPushButton("Adjust ROI")
        self.adjust_roi_btn.setVisible(False)  # Hidden until ROI exists

        self.prev_btn.clicked.connect(self.prev_image)
        self.next_btn.clicked.connect(self.next_image)
        self.roi_btn.clicked.connect(self._toggle_roi_mode)
        self.adjust_roi_btn.clicked.connect(self._toggle_adjustment_mode)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.valueChanged.connect(self._on_slider)

        # Frame slider row: label, slider, frame index (e.g. "1 / 100")
        self.frame_index_label = QLabel("— / —")
        self.frame_index_label.setMinimumWidth(48)

        ## Exposure/Contrast: in a collapsible panel, hidden by default
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
        # handle the changing value for the contrast slider
        self.contrast_slider.valueChanged.connect(self._on_adjustment_changed)

        nav = QHBoxLayout()
        nav.addWidget(self.prev_btn)
        nav.addWidget(self.next_btn)
        nav.addWidget(self.roi_btn)
        nav.addWidget(self.adjust_roi_btn)

        # Display options panel (exposure/contrast) - hidden until "Display options" is clicked
        self.adjustments_panel = QWidget()
        adjustments_layout = QVBoxLayout(self.adjustments_panel)
        adjustments_layout.setContentsMargins(0, 4, 0, 0)
        adjustments_layout.addWidget(self.exposure_label)
        adjustments_layout.addWidget(self.exposure_slider)
        adjustments_layout.addWidget(self.contrast_label)
        adjustments_layout.addWidget(self.contrast_slider)
        self.adjustments_panel.setVisible(False)

        self.display_options_btn = QPushButton("Display options")
        self.display_options_btn.setCheckable(True)
        self.display_options_btn.setChecked(False)
        self.display_options_btn.clicked.connect(self._toggle_display_options)

        layout = QVBoxLayout(self)
        layout.addWidget(self.preview_group, 1)
        layout.addLayout(nav)
        # Frame row: label, slider, index
        frame_row = QHBoxLayout()
        frame_row.addWidget(QLabel("Frame:"))
        frame_row.addWidget(self.slider, 1)
        frame_row.addWidget(self.frame_index_label)
        layout.addLayout(frame_row)
        layout.addWidget(self.display_options_btn)
        layout.addWidget(self.adjustments_panel)
        # Metadata label should be compact (stretch factor 0, max height)
        self.filename_label.setMaximumHeight(50)
        layout.addWidget(self.filename_label, 0)
        # adjust the labels on sliders to go to 0 on load
        self._update_adjustment_labels()

        self.setAcceptDrops(True)
        self.setFocusPolicy(Qt.StrongFocus)

    def set_stack(self, files) -> None:
        self.handler.load_image_stack(files)
        count = self.handler.get_image_count()
        self.slider.setRange(0, max(0, count - 1))
        self.index = 0
        self._update_frame_index_label(count)
        self._show_current()

        # Hide upload button when images are loaded
        if count > 0:
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
        # Reset cache and ROI state
        self.cache = _LRUCache(20)
        self.current_roi = None
        # Reset UI labels and slider
        self.image_label.clear()
        self.filename_label.setText("Load image to see data")
        self.frame_index_label.setText("— / —")
        self.slider.setRange(0, 0)
        self._update_roi_button_text()
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
            allowed_extensions = {".tif", ".tiff", ".gif"}
            filtered_paths = [p for p in paths if Path(p).suffix.lower() in allowed_extensions]

            if filtered_paths:
                self.set_stack(filtered_paths)
            else:
                # Show message if no valid files were dropped
                from PySide6.QtWidgets import QMessageBox

                QMessageBox.warning(self, "Invalid Files", "Only TIF and GIF files are supported.")

    # Function to update the silder value so the user can see what value they have
    def _update_adjustment_labels(self) -> None:
        # For exposure
        self.exposure_label.setText(f"Exposure: {self.exposure_slider.value()}")
        # For contrast
        self.contrast_label.setText(f"Contrast: {self.contrast_slider.value()}")

    """Next three function convert to 8 bit and do exposure and contrast
    calculation. The order is a stated: Exposure/Contrast, then 8 bit converstion
    This will preserve the appearance and will allow for more adjustment"""

    # function to calculate teh amount of exposure and contrast the imahe
    # should get based on the slider value input
    def _apply_adjustments(self, arr: np.ndarray) -> np.ndarray:
        # Set the exposure and contrast sliders value to ev and cv
        ev = self.exposure_slider.value()
        cv = self.contrast_slider.value()
        # stores the orignal image data type
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
                # get the max out of the image arr
                max_possible = float(np.iinfo(orig_dtype).max)
                # normailze the image
                new_arr = new_arr / max_possible
            else:
                # the image is normalized
                new_arr = np.clip(new_arr, 0, 1)

        # creating exposure and contrast scalers
        exposure = 2 ** (ev / 50)
        contrast = 1 + (cv / 100)
        # 0.5 to preserve the greyscale... if higher turns black...if lower turns white
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
        # return the 8 bit image
        return unit_8

    def _show_current(self) -> None:
        count = self.handler.get_image_count()
        if count == 0:
            self.image_label.clear()
            self.frame_index_label.setText("— / —")
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
        # show the 8 bit image on the workbench
        preview_img = self._ensure_uint8(img)
        qimg = numpy_to_qimage(preview_img)
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
        # ROI: Draw saved ROI overlay
        # ============================================================
        if self.current_roi is not None:
            painter = QPainter(scaled_pix)
            pen = QPen(Qt.green, 2, Qt.SolidLine)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)

            if self.current_roi.shape == ROIShape.POLYGON and self.current_roi.points:
                qpts = [
                    QPointF(int(p[0] * scale), int(p[1] * scale)) for p in self.current_roi.points
                ]
                painter.drawPolygon(QPolygonF(qpts))
            else:
                x_scaled = int(self.current_roi.x * scale)
                y_scaled = int(self.current_roi.y * scale)
                w_scaled = int(self.current_roi.width * scale)
                h_scaled = int(self.current_roi.height * scale)
                painter.drawEllipse(x_scaled, y_scaled, w_scaled, h_scaled)
            painter.end()

        self.image_label.setPixmap(scaled_pix)
        current_path = Path(self.handler.files[self.index])
        self._update_frame_index_label(count)
        self.filename_label.setText(f"{self.index + 1}/{count}: \n{current_path.name}")

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
            "Image Files (*.tif *.tiff *.gif);;"
            "TIF Files (*.tif *.tiff);;"
            "GIF Files (*.gif);;"
            "All Files (*.*)",
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

    def _update_frame_index_label(self, count: int) -> None:
        """Update the frame index label (e.g. '1 / 100' or '— / —')."""
        if count <= 0:
            self.frame_index_label.setText("— / —")
        else:
            self.frame_index_label.setText(f"{self.index + 1} / {count}")

    def _toggle_display_options(self) -> None:
        """Show or hide the exposure/contrast panel."""
        self.adjustments_panel.setVisible(self.display_options_btn.isChecked())
        self.display_options_btn.setText(
            "Hide display options" if self.display_options_btn.isChecked() else "Display options"
        )

    def _toggle_roi_mode(self) -> None:
        """Open ROI selection dialog for precise polygon drawing."""
        if self.handler.get_image_count() == 0:
            return

        from PySide6.QtWidgets import QMessageBox

        # If there's an existing ROI, confirm before starting new selection
        if self.current_roi is not None:
            reply = QMessageBox.question(
                self,
                "Create New ROI",
                "Creating a new ROI will replace the existing one. Continue?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply == QMessageBox.No:
                return

        self._open_roi_dialog(existing_roi=None)

    def _toggle_adjustment_mode(self) -> None:
        """Open ROI selection dialog with existing ROI for adjustment."""
        if self.current_roi is None:
            return
        self._open_roi_dialog(existing_roi=self.current_roi)

    def _open_roi_dialog(self, existing_roi: "Optional[ROI]" = None) -> None:
        """Open the ROI selection dialog and handle the result."""
        from PySide6.QtWidgets import QDialog

        from ui.roi_selection_dialog import ROISelectionDialog

        # Get current image (with exposure/contrast applied)
        img = self.cache.get(self.index)
        if img is None:
            img = self.handler.get_image_at_index(self.index)
        preview_img = self._ensure_uint8(img)

        dialog = ROISelectionDialog(
            image=preview_img,
            existing_roi=existing_roi,
            parent=self,
        )
        if dialog.exec() == QDialog.Accepted:
            roi = dialog.get_roi()
            if roi is not None:
                self.current_roi = roi
                self.roiSelected.emit(self.current_roi)
                self._update_roi_button_text()
                self._show_current()

    def _update_roi_button_text(self) -> None:
        """Update ROI button text based on current state."""
        if self.current_roi is not None:
            self.roi_btn.setText("New ROI")
        else:
            self.roi_btn.setText("Select ROI")

        # Show/hide adjust button based on ROI existence
        self.adjust_roi_btn.setVisible(self.current_roi is not None)

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
        # Update button text
        self._update_roi_button_text()
        # Redraw to show the ROI with correct scaling
        self._show_current()
