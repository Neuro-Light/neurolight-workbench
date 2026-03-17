from __future__ import annotations

from typing import Optional

import numpy as np
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
)
from matplotlib.backends.backend_qtagg import (
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ui.draggable_spinbox import DraggableSpinBox

from ui.app_settings import get_theme
from ui.styles import get_mpl_theme

def _smooth_display(y: np.ndarray, window: int) -> np.ndarray:
    """Apply a moving average for display only (window in frames); does not alter exported data."""
    if window < 2 or len(y) < window:
        return y
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(y.astype(np.float64), kernel, mode="same").astype(np.float32)


class NeuronTrajectoryPlotWidget(QWidget):
    """Widget for plotting individual neuron intensity trajectories over time."""

    VIEW_BOTH = "both"
    VIEW_ROI1 = "roi_1"
    VIEW_ROI2 = "roi_2"

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.neuron_trajectories: Optional[np.ndarray] = None
        self.quality_mask: Optional[np.ndarray] = None
        self.neuron_locations: Optional[np.ndarray] = None
        self.roi_origin: Optional[np.ndarray] = None  # 0 = ROI 1, 1 = ROI 2 per neuron
        self._hover_cid = None

        layout = QVBoxLayout(self)

        # Status label
        self.status_label = QLabel("No neuron trajectories available. Run detection first.")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        # ROI view selector: Both / ROI 1 only / ROI 2 only
        roi_view_row = QHBoxLayout()
        roi_view_row.addWidget(QLabel("Show trajectories:"))
        self.roi_view_combo = QComboBox()
        self.roi_view_combo.addItem("Both (ROI 1 & 2)", self.VIEW_BOTH)
        self.roi_view_combo.addItem("ROI 1 only", self.VIEW_ROI1)
        self.roi_view_combo.addItem("ROI 2 only", self.VIEW_ROI2)
        self.roi_view_combo.setToolTip("Filter trajectories by ROI when detection was run on both ROIs.")
        self.roi_view_combo.currentIndexChanged.connect(self._update_plot)
        roi_view_row.addWidget(self.roi_view_combo)
        roi_view_row.addStretch()
        layout.addLayout(roi_view_row)

        # Display options group
        options_group = QGroupBox("Display Options")
        options_layout = QFormLayout()

        # Show good neurons checkbox
        self.show_good_checkbox = QCheckBox()
        self.show_good_checkbox.setChecked(True)
        self.show_good_checkbox.stateChanged.connect(self._update_plot)
        options_layout.addRow("Show Good Neurons:", self.show_good_checkbox)

        # Show bad neurons checkbox
        self.show_bad_checkbox = QCheckBox()
        self.show_bad_checkbox.setChecked(False)
        self.show_bad_checkbox.stateChanged.connect(self._update_plot)
        options_layout.addRow("Show Bad Neurons:", self.show_bad_checkbox)

        # Max neurons to display
        self.max_neurons_spin = DraggableSpinBox()
        self.max_neurons_spin.setRange(1, 1000)
        self.max_neurons_spin.setValue(50)
        self.max_neurons_spin.setToolTip("Maximum number of neurons to display (for performance)")
        self.max_neurons_spin.valueChanged.connect(self._update_plot)
        options_layout.addRow("Max Neurons to Display:", self.max_neurons_spin)

        # Show average checkbox
        self.show_average_checkbox = QCheckBox()
        self.show_average_checkbox.setChecked(True)
        self.show_average_checkbox.stateChanged.connect(self._update_plot)
        options_layout.addRow("Show Average:", self.show_average_checkbox)

        # Smoothing (display only; 0 = none)
        self.smoothing_spin = DraggableSpinBox()
        self.smoothing_spin.setRange(0, 51)
        self.smoothing_spin.setValue(0)
        self.smoothing_spin.setSpecialValueText("None")
        self.smoothing_spin.setToolTip(
            "Moving average window in frames for display only (0 = no smoothing). Export uses raw data."
        )
        self.smoothing_spin.valueChanged.connect(self._update_plot)
        options_layout.addRow("Smoothing (frames):", self.smoothing_spin)

        options_group.setLayout(options_layout)
        layout.addWidget(options_group)

        # Matplotlib figure and canvas (theme applied in _update_plot)
        self.figure = Figure(figsize=(10, 8))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.toolbar.setObjectName("mpl_nav_toolbar")
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        # Hover status: show frame and intensity when moving over plot
        self.hover_label = QLabel("Hover over plot for frame and intensity.")
        self.hover_label.setAlignment(Qt.AlignCenter)
        self.hover_label.setProperty("class", "plot-hover")
        layout.addWidget(self.hover_label)

        # Buttons layout
        buttons_layout = QHBoxLayout()

        self.export_png_btn = QPushButton("Export PNG...")
        self.export_png_btn.clicked.connect(self._export_to_png)
        self.export_png_btn.setEnabled(False)
        self.export_png_btn.setToolTip("Save the current plot as a PNG image.")
        buttons_layout.addWidget(self.export_png_btn)

        self.export_btn = QPushButton("Export Plot Data (CSV)")
        self.export_btn.clicked.connect(self._export_to_csv)
        self.export_btn.setEnabled(False)
        buttons_layout.addWidget(self.export_btn)

        layout.addLayout(buttons_layout)

    def plot_trajectories(
        self,
        neuron_trajectories: np.ndarray,
        quality_mask: Optional[np.ndarray] = None,
        neuron_locations: Optional[np.ndarray] = None,
        roi_origin: Optional[np.ndarray] = None,
    ) -> None:
        """
        Plot intensity trajectories for each detected neuron.

        Args:
            neuron_trajectories: 2D array (neurons x frames) of intensity time-series
            quality_mask: Boolean array indicating good (True) vs bad (False) neurons
            neuron_locations: Array of (y, x) coordinates for neurons (optional, for labeling)
            roi_origin: Optional 1D array of 0 (ROI 1) or 1 (ROI 2) per neuron for coloring/filtering
        """
        self.neuron_trajectories = neuron_trajectories
        self.quality_mask = quality_mask
        self.neuron_locations = neuron_locations
        self.roi_origin = roi_origin

        if neuron_trajectories is None or len(neuron_trajectories) == 0:
            self.status_label.setText("No neuron trajectories to display.")
            self.export_btn.setEnabled(False)
            return

        num_neurons, num_frames = neuron_trajectories.shape

        # Update status
        if quality_mask is not None:
            num_good = np.sum(quality_mask)
            num_bad = num_neurons - num_good
            self.status_label.setText(
                f"Displaying {num_neurons} neuron trajectories "
                f"({num_good} good, {num_bad} bad) across {num_frames} frames"
            )
        else:
            self.status_label.setText(
                f"Displaying {num_neurons} neuron trajectories across {num_frames} frames"
            )

        # Enable export buttons
        self.export_btn.setEnabled(True)
        self.export_png_btn.setEnabled(True)

        # Update plot
        self._update_plot()

    def _apply_theme(self, ax) -> None:
        """Apply current app theme to figure and axes."""
        theme = get_mpl_theme(get_theme())
        self.figure.patch.set_facecolor(theme["figure_facecolor"])
        ax.set_facecolor(theme["axes_facecolor"])
        ax.tick_params(axis="both", colors=theme["text_color"])
        ax.xaxis.label.set_color(theme["text_color"])
        ax.yaxis.label.set_color(theme["text_color"])
        ax.title.set_color(theme["text_color"])
        for spine in ax.spines.values():
            spine.set_color(theme["axes_edgecolor"])
        ax.grid(True, alpha=0.35, color=theme["grid_color"])
        leg = ax.get_legend()
        if leg:
            leg.get_frame().set_facecolor(theme["legend_facecolor"])
            leg.get_frame().set_edgecolor(theme["legend_edgecolor"])
            for t in leg.get_texts():
                t.set_color(theme["text_color"])

    def _on_motion(self, event) -> None:
        """Show frame and intensity under cursor in hover label."""
        if (
            self.neuron_trajectories is None
            or event.inaxes is None
            or event.xdata is None
            or event.ydata is None
        ):
            self.hover_label.setText("Hover over plot for frame and intensity.")
            return
        frame_idx = int(round(event.xdata))
        num_frames = self.neuron_trajectories.shape[1]
        if frame_idx < 0 or frame_idx >= num_frames:
            self.hover_label.setText("Hover over plot for frame and intensity.")
            return
        # Show mean intensity across displayed neurons at this frame (or nearest)
        intensity = float(np.mean(self.neuron_trajectories[:, frame_idx]))
        self.hover_label.setText(f"Frame {frame_idx}  ·  Intensity {intensity:.3f}")

    def _update_plot(self) -> None:
        """Update the trajectory plot based on current display options."""
        if self.neuron_trajectories is None or len(self.neuron_trajectories) == 0:
            return

        theme = get_mpl_theme(get_theme())

        # Clear previous plot
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        num_neurons, num_frames = self.neuron_trajectories.shape
        frames = np.arange(num_frames)

        # Get display options
        show_good = self.show_good_checkbox.isChecked()
        show_bad = self.show_bad_checkbox.isChecked()
        max_neurons = self.max_neurons_spin.value()
        show_average = self.show_average_checkbox.isChecked()
        smooth_window = self.smoothing_spin.value()

        def _display_series(y: np.ndarray) -> np.ndarray:
            return _smooth_display(y, smooth_window) if smooth_window >= 2 else y

        view_mode = self.roi_view_combo.currentData()

        # Build candidate indices (good and/or bad), then apply ROI filter and max limit
        if self.quality_mask is not None:
            good_indices = np.where(self.quality_mask)[0] if show_good else np.array([], dtype=np.intp)
            bad_indices = np.where(~self.quality_mask)[0] if show_bad else np.array([], dtype=np.intp)
            candidates = np.concatenate([good_indices, bad_indices])
        else:
            candidates = np.arange(num_neurons)

        # When we have ROI origin: filter by ROI first, then apply max_neurons per ROI (for Both) or total (for single ROI)
        if self.roi_origin is not None and len(candidates) > 0:
            roi_1_candidates = candidates[self.roi_origin[candidates] == 0]
            roi_2_candidates = candidates[self.roi_origin[candidates] == 1]
            if view_mode == self.VIEW_ROI1:
                neurons_to_plot = roi_1_candidates[:max_neurons].tolist()
            elif view_mode == self.VIEW_ROI2:
                neurons_to_plot = roi_2_candidates[:max_neurons].tolist()
            else:
                # VIEW_BOTH: take up to max_neurons from each ROI so both are represented
                neurons_to_plot = (
                    roi_1_candidates[:max_neurons].tolist()
                    + roi_2_candidates[:max_neurons].tolist()
                )
        else:
            # No ROI split: use first max_neurons good, first max_neurons bad
            neurons_to_plot = []
            if self.quality_mask is not None:
                if show_good:
                    neurons_to_plot.extend(good_indices[:max_neurons].tolist())
                if show_bad:
                    neurons_to_plot.extend(bad_indices[:max_neurons].tolist())
            else:
                neurons_to_plot = list(range(min(num_neurons, max_neurons)))
            if len(neurons_to_plot) > max_neurons:
                neurons_to_plot = neurons_to_plot[:max_neurons]

        # Plot individual neuron trajectories (theme-aware colors)
        use_roi_colors = (
            view_mode == self.VIEW_BOTH
            and self.roi_origin is not None
            and len(np.unique(self.roi_origin)) > 1
        )
        roi_1_color = theme["roi_1_line_color"]
        roi_2_color = theme["roi_2_line_color"]

        if use_roi_colors:
            roi_1_indices = [i for i in neurons_to_plot if self.roi_origin[i] == 0]
            roi_2_indices = [i for i in neurons_to_plot if self.roi_origin[i] == 1]
            for idx in roi_1_indices:
                ax.plot(
                    frames,
                    _display_series(self.neuron_trajectories[idx]),
                    color=roi_1_color,
                    alpha=0.4,
                    linewidth=0.8,
                    antialiased=True,
                    label="ROI 1" if idx == roi_1_indices[0] else "",
                )
            for idx in roi_2_indices:
                ax.plot(
                    frames,
                    _display_series(self.neuron_trajectories[idx]),
                    color=roi_2_color,
                    alpha=0.4,
                    linewidth=0.8,
                    antialiased=True,
                    label="ROI 2" if idx == roi_2_indices[0] else "",
                )
        elif self.quality_mask is not None:
            good_to_plot = [i for i in neurons_to_plot if self.quality_mask[i]]
            if good_to_plot and show_good:
                for idx in good_to_plot:
                    ax.plot(
                        frames,
                        _display_series(self.neuron_trajectories[idx]),
                        color=theme["good_color"],
                        alpha=0.4,
                        linewidth=0.8,
                        antialiased=True,
                        label="Good Neurons" if idx == good_to_plot[0] else "",
                    )
            bad_to_plot = [i for i in neurons_to_plot if not self.quality_mask[i]]
            if bad_to_plot and show_bad:
                for idx in bad_to_plot:
                    ax.plot(
                        frames,
                        _display_series(self.neuron_trajectories[idx]),
                        color=theme["bad_color"],
                        alpha=0.4,
                        linewidth=0.8,
                        antialiased=True,
                        label="Bad Neurons" if idx == bad_to_plot[0] else "",
                    )
        else:
            for idx in neurons_to_plot:
                ax.plot(
                    frames,
                    _display_series(self.neuron_trajectories[idx]),
                    color=theme["neutral_color"],
                    alpha=0.4,
                    linewidth=0.8,
                    antialiased=True,
                    label="Neurons" if idx == neurons_to_plot[0] else "",
                )

        if show_average and len(neurons_to_plot) > 0:
            avg_color = theme.get("avg_trajectory_color", theme.get("average_color", "#e879f9"))
            if use_roi_colors:
                avg_roi_1_color = theme.get(
                    "avg_trajectory_roi_1_color", roi_1_color
                )
                avg_roi_2_color = theme.get(
                    "avg_trajectory_roi_2_color", roi_2_color
                )
                if roi_1_indices:
                    avg_1 = np.mean(self.neuron_trajectories[roi_1_indices], axis=0)
                    ax.plot(
                        frames,
                        _display_series(avg_1),
                        color=avg_roi_1_color,
                        linewidth=2.5,
                        antialiased=True,
                        label="Average (ROI 1)",
                    )
                if roi_2_indices:
                    avg_2 = np.mean(self.neuron_trajectories[roi_2_indices], axis=0)
                    ax.plot(
                        frames,
                        _display_series(avg_2),
                        color=avg_roi_2_color,
                        linewidth=2.5,
                        antialiased=True,
                        label="Average (ROI 2)",
                    )
            elif self.quality_mask is not None and show_good:
                good_in_plot = [i for i in neurons_to_plot if self.quality_mask[i]]
                if good_in_plot:
                    avg_trajectory = np.mean(self.neuron_trajectories[good_in_plot], axis=0)
                    ax.plot(
                        frames,
                        _display_series(avg_trajectory),
                        color=avg_color,
                        linewidth=2.5,
                        antialiased=True,
                        label="Average (Good Neurons)",
                    )
            else:
                avg_trajectory = np.mean(self.neuron_trajectories[neurons_to_plot], axis=0)
                ax.plot(
                    frames,
                    _display_series(avg_trajectory),
                    color=avg_color,
                    linewidth=2.5,
                    antialiased=True,
                    label="Average",
                )

        ax.set_xlabel("Frame Number", fontsize=12)
        ax.set_ylabel("Intensity", fontsize=12)
        ax.set_title("Neuron Intensity Trajectories Over Time", fontsize=14)
        ax.legend(loc="best")
        self._apply_theme(ax)

        if getattr(self, "_hover_cid", None) is not None:
            self.canvas.mpl_disconnect(self._hover_cid)
        self._hover_cid = self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas.draw_idle()

    def _export_to_png(self) -> None:
        """Save the current plot as a PNG image."""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Plot as PNG",
            "neuron_trajectories.png",
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
            QMessageBox.information(
                self,
                "Export Successful",
                f"Plot saved to:\n{file_path}",
            )
        except Exception as e:
            QMessageBox.critical(
                self,
                "Export Failed",
                f"Failed to save PNG:\n{str(e)}",
            )

    def _export_to_csv(self) -> None:
        """Export trajectory data to CSV file."""
        if self.neuron_trajectories is None or len(self.neuron_trajectories) == 0:
            QMessageBox.warning(self, "No Data", "No trajectory data to export.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Trajectory Data", "neuron_trajectories.csv", "CSV Files (*.csv)"
        )

        if not file_path:
            return

        try:
            num_neurons, num_frames = self.neuron_trajectories.shape

            # Create CSV with frame numbers and all neuron trajectories
            # Format: Frame, Neuron_0, Neuron_1, ..., Neuron_N
            header_parts = ["Frame"]
            if self.quality_mask is not None:
                for i in range(num_neurons):
                    quality = "Good" if self.quality_mask[i] else "Bad"
                    header_parts.append(f"Neuron_{i}_{quality}")
            else:
                header_parts.extend([f"Neuron_{i}" for i in range(num_neurons)])

            header = ",".join(header_parts)

            # Create data array
            data = []
            for frame_idx in range(num_frames):
                row = [frame_idx]
                row.extend(self.neuron_trajectories[:, frame_idx])
                data.append(row)

            # Format string
            fmt_parts = ["%d"]  # Frame number
            fmt_parts.extend(["%.6f"] * num_neurons)  # Trajectory values
            fmt = ",".join(fmt_parts)

            np.savetxt(file_path, data, delimiter=",", header=header, comments="", fmt=fmt)

            QMessageBox.information(
                self, "Export Successful", f"Trajectory data exported to:\n{file_path}"
            )
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"Failed to export data:\n{str(e)}")

    def refresh_theme(self) -> None:
        """Redraw the plot with the current app theme (e.g. after theme change)."""
        if self.neuron_trajectories is not None and len(self.neuron_trajectories) > 0:
            self._update_plot()

    def clear_plot(self) -> None:
        """Clear the plot and reset state."""
        if getattr(self, "_hover_cid", None) is not None:
            self.canvas.mpl_disconnect(self._hover_cid)
            self._hover_cid = None
        self.figure.clear()
        self.canvas.draw()
        self.neuron_trajectories = None
        self.quality_mask = None
        self.neuron_locations = None
        self.roi_origin = None
        self.status_label.setText("No neuron trajectories available. Run detection first.")
        self.hover_label.setText("Hover over plot for frame and intensity.")
        self.export_btn.setEnabled(False)
        self.export_png_btn.setEnabled(False)
