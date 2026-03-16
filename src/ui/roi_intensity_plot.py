from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional

if TYPE_CHECKING:
    from core.experiment_manager import Experiment

import numpy as np
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
)
from matplotlib.backends.backend_qtagg import (
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from core.roi import ROI, ROIShape
from ui.app_settings import get_roi_colors, get_theme
from ui.constants import ROI_DISPLAY_NAMES, ROI_KEYS
from ui.styles import get_mpl_theme


class ROIIntensityPlotWidget(QWidget):
    """Widget for plotting ROI intensity over time for up to two ROIs."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._intensity: Dict[str, Optional[np.ndarray]] = {
            "roi_1": None,
            "roi_2": None,
        }
        self._rois: Dict[str, Optional[ROI]] = {"roi_1": None, "roi_2": None}
        self.experiment: Optional["Experiment"] = None
        self._hover_cid = None

        layout = QVBoxLayout(self)

        # Status label
        self.status_label = QLabel("No ROI selected. Select an ROI in the image viewer.")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)

        # Toggle checkboxes for each ROI (with colour indicators)
        toggle_row = QHBoxLayout()
        self._checkboxes: Dict[str, QCheckBox] = {}
        colors = get_roi_colors()
        for key in ROI_KEYS:
            cb = QCheckBox(ROI_DISPLAY_NAMES[key])
            cb.setChecked(True)
            cb.toggled.connect(self._replot)
            self._checkboxes[key] = cb

            swatch = QLabel()
            swatch.setFixedSize(12, 12)
            pix = QPixmap(12, 12)
            pix.fill(QColor(colors[key]))
            swatch.setPixmap(pix)

            toggle_row.addWidget(swatch)
            toggle_row.addWidget(cb)
            toggle_row.addSpacing(12)

        toggle_row.addStretch()
        layout.addLayout(toggle_row)

        # Matplotlib figure and canvas
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.toolbar.setObjectName("mpl_nav_toolbar")
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        # Hover label
        self.hover_label = QLabel("Hover over plot for frame and intensity.")
        self.hover_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.hover_label)

        # Export buttons
        button_layout = QHBoxLayout()

        self.export_png_btn = QPushButton("Export PNG...")
        self.export_png_btn.clicked.connect(self._export_to_png)
        self.export_png_btn.setEnabled(False)
        self.export_png_btn.setToolTip("Save the current plot as a PNG image.")
        button_layout.addWidget(self.export_png_btn)

        self.export_btn = QPushButton("Export to CSV")
        self.export_btn.clicked.connect(self._export_to_csv)
        self.export_btn.setEnabled(False)
        button_layout.addWidget(self.export_btn)

        layout.addLayout(button_layout)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plot_intensity_time_series(
        self, roi_key: str, intensity_data: np.ndarray, roi: ROI
    ) -> None:
        """Store intensity data for *roi_key* and redraw."""
        self._intensity[roi_key] = intensity_data
        self._rois[roi_key] = roi
        self._replot()

    def clear_roi(self, roi_key: str) -> None:
        """Remove a single ROI's data and replot."""
        self._intensity[roi_key] = None
        self._rois[roi_key] = None
        self._replot()

    def clear_plot(self) -> None:
        """Clear everything and reset state."""
        if self._hover_cid is not None:
            self.canvas.mpl_disconnect(self._hover_cid)
            self._hover_cid = None
        self.figure.clear()
        self.canvas.draw()
        self._intensity = {"roi_1": None, "roi_2": None}
        self._rois = {"roi_1": None, "roi_2": None}
        self.status_label.setText("No ROI selected. Select an ROI in the image viewer.")
        self.hover_label.setText("Hover over plot for frame and intensity.")
        self.export_btn.setEnabled(False)
        self.export_png_btn.setEnabled(False)

    def refresh_theme(self) -> None:
        """Redraw the plot with the current app theme / ROI colours."""
        self._replot()

    # ------------------------------------------------------------------
    # Internal plotting
    # ------------------------------------------------------------------

    def _replot(self) -> None:
        """Redraw the graph based on available data and checkbox state."""
        self.figure.clear()

        visible: Dict[str, np.ndarray] = {}
        for key in ROI_KEYS:
            data = self._intensity.get(key)
            if data is not None and self._checkboxes[key].isChecked():
                visible[key] = data

        if not visible:
            has_any = any(d is not None for d in self._intensity.values())
            if has_any:
                self.status_label.setText("All ROI traces hidden. Check a box above to show.")
            else:
                self.status_label.setText("No ROI selected. Select an ROI in the image viewer.")
            self.canvas.draw_idle()
            self.export_btn.setEnabled(False)
            self.export_png_btn.setEnabled(False)
            return

        theme = get_mpl_theme(get_theme())
        ax = self.figure.add_subplot(111)

        color_map = {
            "roi_1": theme["roi_1_line_color"],
            "roi_2": theme["roi_2_line_color"],
        }

        for key, data in visible.items():
            frames = np.arange(len(data))
            roi = self._rois.get(key)
            label = ROI_DISPLAY_NAMES[key]
            if roi is not None and roi.shape == ROIShape.POLYGON and roi.points:
                label += f" ({len(roi.points)} pts)"
            ax.plot(frames, data, linewidth=2, color=color_map[key], label=label)

        ax.set_xlabel("Frame Number", fontsize=12)
        ax.set_ylabel("Mean Pixel Intensity", fontsize=12)
        ax.set_title("ROI Intensity Over Time", fontsize=14)
        if len(visible) > 1:
            ax.legend(loc="best")
        self._apply_theme(ax, theme)

        # Status text
        parts = []
        for key, data in visible.items():
            name = ROI_DISPLAY_NAMES[key]
            parts.append(f"{name}: {len(data)} frames, mean {np.mean(data):.2f}")
        self.status_label.setText(" | ".join(parts))

        self.export_btn.setEnabled(True)
        self.export_png_btn.setEnabled(True)

        if self._hover_cid is not None:
            self.canvas.mpl_disconnect(self._hover_cid)
        self._hover_cid = self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas.draw_idle()

    def _apply_theme(self, ax, theme: dict) -> None:
        self.figure.patch.set_facecolor(theme["figure_facecolor"])
        ax.set_facecolor(theme["axes_facecolor"])
        ax.tick_params(axis="both", colors=theme["text_color"])
        ax.xaxis.label.set_color(theme["text_color"])
        ax.yaxis.label.set_color(theme["text_color"])
        ax.title.set_color(theme["text_color"])
        for spine in ax.spines.values():
            spine.set_color(theme["axes_edgecolor"])
        ax.grid(True, alpha=0.35, color=theme["grid_color"])

    def _on_motion(self, event) -> None:
        if event.inaxes is None or event.xdata is None:
            self.hover_label.setText("Hover over plot for frame and intensity.")
            return
        frame_idx = int(round(event.xdata))
        parts = []
        for key in ROI_KEYS:
            data = self._intensity.get(key)
            if data is not None and self._checkboxes[key].isChecked():
                if 0 <= frame_idx < len(data):
                    parts.append(f"{ROI_DISPLAY_NAMES[key]}: {data[frame_idx]:.3f}")
        if parts:
            self.hover_label.setText(f"Frame {frame_idx}  ·  " + "  ·  ".join(parts))
        else:
            self.hover_label.setText("Hover over plot for frame and intensity.")

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _export_to_png(self) -> None:
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Plot as PNG",
            "roi_intensity.png",
            "PNG Files (*.png)",
        )
        if not file_path:
            return
        try:
            self.figure.savefig(
                file_path,
                dpi=150,
                facecolor=self.figure.get_facecolor(),
                edgecolor="none",
                bbox_inches="tight",
            )
            QMessageBox.information(self, "Export Successful", f"Plot saved to:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"Failed to save PNG:\n{e}")

    def _export_to_csv(self) -> None:
        experiment_name = self.experiment.name if self.experiment else "Experiment"
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Intensity Data",
            f"{experiment_name}_roi_intensity_data.csv",
            "CSV Files (*.csv)",
        )
        if not file_path:
            return
        try:
            columns = []
            header_parts = ["Frame"]
            for key in ROI_KEYS:
                data = self._intensity.get(key)
                if data is not None:
                    columns.append(data)
                    header_parts.append(f"{ROI_DISPLAY_NAMES[key]}_Mean_Intensity")

            if not columns:
                QMessageBox.warning(self, "No Data", "No intensity data to export.")
                return

            max_len = max(len(c) for c in columns)
            frames = np.arange(max_len)
            padded = []
            for c in columns:
                if len(c) < max_len:
                    padded_col = np.pad(c, (0, max_len - len(c)), constant_values=np.nan)
                else:
                    padded_col = c
                padded.append(padded_col)

            data_to_save = np.column_stack([frames] + padded)
            header = ",".join(header_parts)
            fmt = ",".join(["%d"] + ["%.6f"] * len(padded))
            np.savetxt(
                file_path,
                data_to_save,
                delimiter=",",
                header=header,
                comments="",
                fmt=fmt,
            )
            QMessageBox.information(
                self, "Export Successful", f"Intensity data exported to:\n{file_path}"
            )
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"Failed to export data:\n{e}")
