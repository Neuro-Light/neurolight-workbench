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
from scipy.signal import find_peaks

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
        self._pick_cid = None
        self._marker_annotation = None
        self._peak_data: list[tuple[int, float, str, int]] = []  # (frame, value, type, order)
        self._trough_data: list[tuple[int, float, str, int]] = []

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

        # Peak/Trough markers toggle
        self.show_peaks_checkbox = QCheckBox("Show Peaks/Troughs")
        self.show_peaks_checkbox.setChecked(False)
        self.show_peaks_checkbox.setToolTip("Overlay peak (maxima) and trough (minima) markers on the graph")
        self.show_peaks_checkbox.toggled.connect(self._on_show_peaks_toggled)
        toggle_row.addWidget(self.show_peaks_checkbox)

        # Peak numbering toggle (hidden until Show Peaks/Troughs is enabled)
        self.number_peaks_checkbox = QCheckBox("Number Markers")
        self.number_peaks_checkbox.setChecked(False)
        self.number_peaks_checkbox.setToolTip("Show order numbers (1, 2, 3...) on peak and trough markers")
        self.number_peaks_checkbox.toggled.connect(self._replot)
        self.number_peaks_checkbox.setVisible(False)
        toggle_row.addWidget(self.number_peaks_checkbox)

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

    def _on_show_peaks_toggled(self, checked: bool) -> None:
        """Show/hide the Number Markers checkbox based on Show Peaks/Troughs state."""
        self.number_peaks_checkbox.setVisible(checked)
        if not checked:
            self.number_peaks_checkbox.setChecked(False)
        self._replot()

    def plot_intensity_time_series(self, roi_key: str, intensity_data: np.ndarray, roi: ROI) -> None:
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

        # Overlay peak/trough markers if enabled
        self._peak_data = []
        self._trough_data = []
        if self.show_peaks_checkbox.isChecked():
            peak_color = theme.get("peak_marker_color", "#f97316")
            trough_color = theme.get("trough_marker_color", "#06b6d4")
            show_numbers = self.number_peaks_checkbox.isChecked()

            # First pass: collect all markers across visible ROIs (without order)
            raw_peaks: list[tuple[int, float]] = []
            raw_troughs: list[tuple[int, float]] = []
            for key, data in visible.items():
                frames = np.arange(len(data))
                peaks, troughs = self._find_peaks_and_troughs(data)
                if len(peaks) > 0:
                    ax.scatter(
                        frames[peaks],
                        data[peaks],
                        marker="^",
                        s=60,
                        color=peak_color,
                        zorder=5,
                        label="Peaks" if key == list(visible.keys())[0] else "",
                        edgecolors="white",
                        linewidths=0.5,
                        picker=True,
                        pickradius=5,
                    )
                    for idx in peaks:
                        raw_peaks.append((int(frames[idx]), float(data[idx])))
                if len(troughs) > 0:
                    ax.scatter(
                        frames[troughs],
                        data[troughs],
                        marker="v",
                        s=60,
                        color=trough_color,
                        zorder=5,
                        label="Troughs" if key == list(visible.keys())[0] else "",
                        edgecolors="white",
                        linewidths=0.5,
                        picker=True,
                        pickradius=5,
                    )
                    for idx in troughs:
                        raw_troughs.append((int(frames[idx]), float(data[idx])))

            # Second pass: sort by frame and assign chronological order numbers
            raw_peaks.sort(key=lambda x: x[0])
            raw_troughs.sort(key=lambda x: x[0])

            for order, (frame, value) in enumerate(raw_peaks, start=1):
                self._peak_data.append((frame, value, "peak", order))
            for order, (frame, value) in enumerate(raw_troughs, start=1):
                self._trough_data.append((frame, value, "trough", order))

            # Add number annotations if enabled (using sorted order)
            if show_numbers:
                for frame, value, _, order in self._peak_data:
                    ax.annotate(
                        str(order),
                        (frame, value),
                        textcoords="offset points",
                        xytext=(0, 8),
                        ha="center",
                        fontsize=8,
                        color=peak_color,
                        fontweight="bold",
                    )
                for frame, value, _, order in self._trough_data:
                    ax.annotate(
                        str(order),
                        (frame, value),
                        textcoords="offset points",
                        xytext=(0, -12),
                        ha="center",
                        fontsize=8,
                        color=trough_color,
                        fontweight="bold",
                    )

        ax.set_xlabel("Frame Number", fontsize=12)
        ax.set_ylabel("Mean Pixel Intensity", fontsize=12)
        ax.set_title("ROI Intensity Over Time", fontsize=14)
        if len(visible) > 1 or self.show_peaks_checkbox.isChecked():
            ax.legend(loc="best")
        self._apply_theme(ax, theme)

        # Status text
        parts = []
        warnings = []
        for key, data in visible.items():
            name = ROI_DISPLAY_NAMES[key]
            status = f"{name}: {len(data)} frames, mean {np.mean(data):.2f}"
            if self.show_peaks_checkbox.isChecked():
                peaks, troughs = self._find_peaks_and_troughs(data)
                status += f" ({len(peaks)} peaks, {len(troughs)} troughs)"
                if len(peaks) > 0 and len(troughs) == 0:
                    warnings.append(f"{name}: No troughs detected — signal may be mostly rising or troughs too subtle")
                elif len(troughs) > 0 and len(peaks) == 0:
                    warnings.append(f"{name}: No peaks detected — signal may be mostly falling or peaks too subtle")
                elif len(peaks) == 0 and len(troughs) == 0:
                    warnings.append(f"{name}: No peaks/troughs detected — signal may be too flat or noisy")
            parts.append(status)
        status_text = " | ".join(parts)
        if warnings:
            warning_text = " | ".join(warnings)
            status_text += (
                f'<br><span style="background-color: rgba(250, 204, 21, 0.25); '
                f'padding: 2px 6px; border-radius: 3px;">⚠ {warning_text}</span>'
            )
            self.status_label.setTextFormat(Qt.RichText)
        else:
            self.status_label.setTextFormat(Qt.PlainText)
        self.status_label.setText(status_text)

        self.export_btn.setEnabled(True)
        self.export_png_btn.setEnabled(True)

        # Create annotation for marker tooltips (hidden initially)
        self._marker_annotation = ax.annotate(
            "",
            xy=(0, 0),
            xytext=(10, 10),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.9),
            fontsize=9,
            visible=False,
            zorder=10,
        )

        if self._hover_cid is not None:
            self.canvas.mpl_disconnect(self._hover_cid)
        if self._pick_cid is not None:
            self.canvas.mpl_disconnect(self._pick_cid)
        self._hover_cid = self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self._pick_cid = self.canvas.mpl_connect("pick_event", self._on_pick)
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

    def _find_peaks_and_troughs(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Find local maxima (peaks) and minima (troughs) in the signal."""
        if len(data) < 3:
            return np.array([], dtype=int), np.array([], dtype=int)
        data_range = np.max(data) - np.min(data)
        prominence = data_range * 0.01 if data_range > 1e-6 else 1e-6
        distance = max(2, len(data) // 100)
        peaks, _ = find_peaks(data, prominence=prominence, distance=distance)
        troughs, _ = find_peaks(-data, prominence=prominence, distance=distance)
        return peaks, troughs

    def _on_motion(self, event) -> None:
        if event.inaxes is None or event.xdata is None:
            self.hover_label.setText("Hover over plot for frame and intensity.")
            if self._marker_annotation:
                self._marker_annotation.set_visible(False)
                self.canvas.draw_idle()
            return

        frame_idx = int(round(event.xdata))

        # Check if hovering near a marker and show tooltip
        marker_found = False
        if self.show_peaks_checkbox.isChecked() and self._marker_annotation:
            all_markers = self._peak_data + self._trough_data
            for m_frame, m_value, m_type, m_order in all_markers:
                if (
                    abs(event.xdata - m_frame) < 1.5
                    and abs(event.ydata - m_value)
                    < (self.figure.axes[0].get_ylim()[1] - self.figure.axes[0].get_ylim()[0]) * 0.05
                ):
                    marker_found = True
                    prev_frame = self._get_previous_marker_frame(m_frame, m_type)
                    interval_text = f"\nInterval: {m_frame - prev_frame} frames" if prev_frame is not None else ""
                    tooltip = f"{m_type.title()} #{m_order}\nFrame: {m_frame}\nValue: {m_value:.3f}{interval_text}"
                    self._marker_annotation.xy = (m_frame, m_value)
                    self._marker_annotation.set_text(tooltip)
                    self._marker_annotation.set_visible(True)
                    self.canvas.draw_idle()
                    break

            if not marker_found:
                self._marker_annotation.set_visible(False)
                self.canvas.draw_idle()

        # Update hover label with frame info
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

    def _get_previous_marker_frame(self, current_frame: int, marker_type: str) -> Optional[int]:
        """Get the frame number of the previous marker of the same type."""
        markers = self._peak_data if marker_type == "peak" else self._trough_data
        prev_frame = None
        for m_frame, _, _, _ in markers:
            if m_frame < current_frame:
                prev_frame = m_frame
            else:
                break
        return prev_frame

    def _on_pick(self, event) -> None:
        """Handle click on a marker to show details in hover label."""
        if not hasattr(event, "ind") or event.ind is None or len(event.ind) == 0:
            return

        artist = event.artist
        ind = event.ind[0]
        xdata = artist.get_offsets()[ind][0]
        ydata = artist.get_offsets()[ind][1]

        # Find which marker was clicked
        all_markers = self._peak_data + self._trough_data
        for m_frame, m_value, m_type, m_order in all_markers:
            if abs(xdata - m_frame) < 0.5 and abs(ydata - m_value) < 0.001:
                prev_frame = self._get_previous_marker_frame(m_frame, m_type)
                interval_text = f" | Interval from previous: {m_frame - prev_frame} frames" if prev_frame is not None else ""
                self.hover_label.setText(
                    f"Selected: {m_type.title()} #{m_order} at Frame {m_frame}, Value: {m_value:.3f}{interval_text}"
                )
                break

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
            QMessageBox.information(self, "Export Successful", f"Intensity data exported to:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"Failed to export data:\n{e}")
