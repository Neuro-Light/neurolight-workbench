from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
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

from core.lomb_scargle import compute_lomb_scargle
from ui.app_settings import get_theme
from ui.constants import DEFAULT_FRAME_INTERVAL_MINUTES, ROI_DISPLAY_NAMES, ROI_KEYS
from ui.draggable_spinbox import DraggableDoubleSpinBox
from ui.styles import get_mpl_theme


class LombScarglePlotWidget(QWidget):
    """
    Widget for computing and plotting Lomb–Scargle periodograms for ROI intensity time series.

    This widget consumes the same ROI intensity data used by ROIIntensityPlotWidget; it does
    not create a new data source. The main window should call set_intensity_time_series()
    whenever an ROI intensity trace is updated.
    """

    AXIS_FREQ = "frequency"
    AXIS_PERIOD = "period"

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        self._intensity: Dict[str, Optional[np.ndarray]] = {key: None for key in ROI_KEYS}
        self._results: Dict[str, Dict[str, float]] = {}
        self._last_frequency: Optional[np.ndarray] = None

        layout = QVBoxLayout(self)

        # Status label
        self.status_label = QLabel("No ROI intensity data available. Select an ROI in the image viewer.")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        # ROI selection + analysis options
        controls_group = QGroupBox("Lomb–Scargle Settings")
        controls_layout = QFormLayout()

        # ROI visibility / analysis checkboxes
        roi_row = QHBoxLayout()
        self._roi_checkboxes: Dict[str, QCheckBox] = {}
        for key in ROI_KEYS:
            cb = QCheckBox(ROI_DISPLAY_NAMES[key])
            cb.setChecked(True)
            cb.toggled.connect(self._update_plot)
            self._roi_checkboxes[key] = cb
            roi_row.addWidget(cb)
        roi_row.addStretch()
        controls_layout.addRow("Analyze ROIs:", roi_row)

        # Sampling interval in minutes per frame
        self.sampling_interval_spin = DraggableDoubleSpinBox()
        self.sampling_interval_spin.setRange(0.0001, 10_000.0)
        self.sampling_interval_spin.setDecimals(4)
        self.sampling_interval_spin.setSingleStep(0.5)
        self.sampling_interval_spin.setValue(DEFAULT_FRAME_INTERVAL_MINUTES)
        self.sampling_interval_spin.setToolTip(
            "Time between successive frames in minutes.\n"
            "Set this to match the experiment's acquisition interval."
        )
        self.sampling_interval_spin.valueChanged.connect(self._update_plot)
        controls_layout.addRow("Time Between Frames (minutes):", self.sampling_interval_spin)

        # X-axis mode: frequency vs period
        self.axis_mode_combo = QComboBox()
        self.axis_mode_combo.addItem("Frequency", self.AXIS_FREQ)
        self.axis_mode_combo.addItem("Period", self.AXIS_PERIOD)
        self.axis_mode_combo.currentIndexChanged.connect(self._update_plot)
        controls_layout.addRow("X-axis:", self.axis_mode_combo)

        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)

        # Matplotlib figure and canvas
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.toolbar.setObjectName("mpl_nav_toolbar")
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        # Short summary label for peak results
        self.summary_label = QLabel("No periodogram computed yet.")
        self.summary_label.setAlignment(Qt.AlignCenter)
        self.summary_label.setProperty("class", "plot-hover")
        self.summary_label.setStyleSheet("font-size: 14px; font-weight: 600; margin-top: 4px; margin-bottom: 4px;")
        layout.addWidget(self.summary_label)

        # Export buttons
        buttons_layout = QHBoxLayout()
        self.export_png_btn = QPushButton("Export PNG...")
        self.export_png_btn.clicked.connect(self._export_to_png)
        self.export_png_btn.setEnabled(False)
        self.export_png_btn.setToolTip("Save the current Lomb–Scargle plot as a PNG image.")
        buttons_layout.addWidget(self.export_png_btn)

        self.export_csv_btn = QPushButton("Export Periodogram (CSV)")
        self.export_csv_btn.clicked.connect(self._export_to_csv)
        self.export_csv_btn.setEnabled(False)
        buttons_layout.addWidget(self.export_csv_btn)

        layout.addLayout(buttons_layout)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_intensity_time_series(self, roi_key: str, intensity: np.ndarray) -> None:
        """
        Update the stored intensity time series for a given ROI and recompute plot.

        This should be called by the main window whenever ROI intensity is updated.
        """
        if roi_key not in ROI_KEYS:
            return
        self._intensity[roi_key] = np.asarray(intensity, dtype=float)
        self._update_plot()

    def clear_roi(self, roi_key: str) -> None:
        """Clear a specific ROI's intensity series and recompute."""
        if roi_key not in ROI_KEYS:
            return
        self._intensity[roi_key] = None
        self._results.pop(roi_key, None)
        self._update_plot()

    def clear_all(self) -> None:
        """Clear all ROI intensity series and reset state."""
        for key in ROI_KEYS:
            self._intensity[key] = None
        self._results.clear()
        self._last_frequency = None
        self.figure.clear()
        self.canvas.draw()
        self.status_label.setText("No ROI intensity data available. Select an ROI in the image viewer.")
        self.summary_label.setText("No periodogram computed yet.")
        self.export_png_btn.setEnabled(False)
        self.export_csv_btn.setEnabled(False)

    def get_peak_for_roi(self, roi_key: str) -> Optional[Dict[str, float]]:
        """
        Return peak frequency / power information for a given ROI, if available.

        Returns a dict with keys:
            - peak_frequency
            - peak_power
            - peak_period
        or None if no result is available for that ROI.
        """
        result = self._results.get(roi_key)
        if result is None:
            return None
        return {
            "peak_frequency": float(result["peak_frequency"]),
            "peak_power": float(result["peak_power"]),
            "peak_period": float(result["peak_period"]),
        }

    def get_all_peaks(self) -> Dict[str, Dict[str, float]]:
        """Return peak results for all ROIs that have been analyzed."""
        return {k: self.get_peak_for_roi(k) for k in ROI_KEYS if self.get_peak_for_roi(k) is not None}

    def set_frame_interval_minutes(self, minutes: float) -> None:
        """Set the time-between-frames spinbox to the given value in minutes."""
        if minutes > 0:
            self.sampling_interval_spin.setValue(minutes)

    def refresh_theme(self) -> None:
        """Redraw the plot with the current app theme (e.g. after theme change)."""
        if any(intensity is not None for intensity in self._intensity.values()):
            self._update_plot()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_theme(self, ax) -> None:
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

    def _update_plot(self) -> None:
        """Recompute Lomb–Scargle periodograms for selected ROIs and update the plot."""
        self.figure.clear()
        self._last_frequency = None

        visible: Dict[str, np.ndarray] = {}
        for key in ROI_KEYS:
            data = self._intensity.get(key)
            cb = self._roi_checkboxes.get(key)
            if data is not None and cb is not None and cb.isChecked():
                visible[key] = data

        if not visible:
            has_any = any(d is not None for d in self._intensity.values())
            if has_any:
                self.status_label.setText("All ROI traces hidden. Check a box above to analyze.")
            else:
                self.status_label.setText("No ROI intensity data available. Select an ROI in the image viewer.")
            self._results.clear()
            self.export_png_btn.setEnabled(False)
            self.export_csv_btn.setEnabled(False)
            self.canvas.draw_idle()
            self.summary_label.setText("No periodogram computed yet.")
            return

        dt = float(self.sampling_interval_spin.value())
        if dt <= 0:
            QMessageBox.warning(
                self,
                "Invalid Sampling Interval",
                "Time between frames must be positive.",
            )
            self._results.clear()
            self.export_png_btn.setEnabled(False)
            self.export_csv_btn.setEnabled(False)
            self.summary_label.setText("No periodogram computed yet.")
            return

        ax = self.figure.add_subplot(111)
        theme = get_mpl_theme(get_theme())

        # Track global peak for annotation
        global_peak_power = -np.inf
        global_peak_freq = 0.0
        global_peak_label = ""
        any_uneven = False

        self._results.clear()

        for key, series in visible.items():
            y = np.asarray(series, dtype=float).ravel()
            n = y.size
            if n < 4:
                self.status_label.setText(f"{ROI_DISPLAY_NAMES[key]}: insufficient data points for Lomb–Scargle.")
                continue

            # Build time vector from frame indices and sampling interval
            t = np.arange(n, dtype=float) * dt

            try:
                result = compute_lomb_scargle(t, y)
            except Exception as exc:
                QMessageBox.warning(
                    self,
                    "Lomb–Scargle Error",
                    f"Failed to compute Lomb–Scargle for {ROI_DISPLAY_NAMES[key]}:\n{exc}",
                )
                continue

            freq = result["frequency"]
            power = result["power"]

            # Store frequency grid for export; all ROIs share the same frequencies by construction.
            self._last_frequency = np.asarray(freq, dtype=float)

            self._results[key] = {
                "peak_frequency": float(result["peak_frequency"]),
                "peak_power": float(result["peak_power"]),
                "peak_period": float(result["peak_period"]),
            }

            any_uneven = any_uneven or bool(result.get("uneven_sampling", False))

            x_values: np.ndarray
            x_label: str
            axis_mode = self.axis_mode_combo.currentData()
            if axis_mode == self.AXIS_PERIOD:
                # Safely convert to period, avoiding division by zero.
                with np.errstate(divide="ignore", invalid="ignore"):
                    period = np.where(freq > 0, 1.0 / freq, np.inf)
                x_values = period
                x_label = "Period (minutes)"
            else:
                x_values = freq
                x_label = "Frequency (cycles / minute)"

            color = theme["roi_1_line_color"] if key == "roi_1" else theme["roi_2_line_color"]
            ax.plot(
                x_values,
                power,
                label=f"{ROI_DISPLAY_NAMES[key]}",
                color=color,
                linewidth=2.0,
                antialiased=True,
            )

            if result["peak_power"] > global_peak_power:
                global_peak_power = float(result["peak_power"])
                global_peak_freq = float(result["peak_frequency"])
                global_peak_label = ROI_DISPLAY_NAMES[key]

        if not self._results or self._last_frequency is None:
            # Nothing successfully computed
            self.canvas.draw_idle()
            self.export_png_btn.setEnabled(False)
            self.export_csv_btn.setEnabled(False)
            return

        # Annotate global peak
        if global_peak_power > 0 and np.isfinite(global_peak_freq):
            axis_mode = self.axis_mode_combo.currentData()
            if axis_mode == self.AXIS_PERIOD and global_peak_freq > 0:
                peak_x = 1.0 / global_peak_freq
                peak_x_label = "period"
            else:
                peak_x = global_peak_freq
                peak_x_label = "frequency"
            ax.axvline(peak_x, color=theme.get("avg_trajectory_color", "#e879f9"), linestyle="--", alpha=0.8)
            ax.text(
                peak_x,
                global_peak_power,
                f" Peak {peak_x_label}\n({global_peak_label})",
                color=theme["text_color"],
                ha="left",
                va="bottom",
                fontsize=10,
            )

        ax.set_xlabel(x_label if "x_label" in locals() else "Frequency (cycles / minute)", fontsize=12)
        ax.set_ylabel("Power", fontsize=12)
        ax.set_title("Lomb–Scargle Periodogram", fontsize=14)
        if len(visible) > 1:
            ax.legend(loc="best")
        self._apply_theme(ax)
        self.canvas.draw_idle()

        # Update status / summary labels
        parts = []
        for key, res in self._results.items():
            parts.append(
                f"{ROI_DISPLAY_NAMES[key]}: f* = {res['peak_frequency']:.4g}, "
                f"P(f*) = {res['peak_power']:.3g}, "
                f"T* = {res['peak_period']:.4g}"
            )
        self.summary_label.setText(" | ".join(parts))

        if any_uneven:
            self.status_label.setText(
                "Lomb–Scargle computed. Time sampling appears uneven for at least one series (handled natively)."
            )
        else:
            self.status_label.setText("Lomb–Scargle computed for selected ROIs.")

        # Enable exports now that we have valid content
        self.export_png_btn.setEnabled(True)
        self.export_csv_btn.setEnabled(True)

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------

    def _export_to_png(self) -> None:
        """Save the current Lomb–Scargle plot as a PNG image."""
        if self._last_frequency is None or not self._results:
            QMessageBox.warning(self, "No Data", "No periodogram data to export.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Lomb–Scargle Plot as PNG",
            "lomb_scargle_periodogram.png",
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
        """Export the current Lomb–Scargle periodogram samples to a CSV file."""
        if self._last_frequency is None or not self._results:
            QMessageBox.warning(self, "No Data", "No periodogram data to export.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Periodogram Data",
            "lomb_scargle_periodogram.csv",
            "CSV Files (*.csv)",
        )
        if not file_path:
            return

        try:
            freq = np.asarray(self._last_frequency, dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                period = np.where(freq > 0, 1.0 / freq, np.inf)

            # Assemble columns: frequency, period, power_ROI1, power_ROI2 (when available)
            columns = [freq, period]
            header_parts = ["Frequency", "Period"]

            # Recompute power arrays per ROI over the stored frequency grid
            # using the cached intensity series so CSV is always consistent.
            dt = float(self.sampling_interval_spin.value())
            for key in ROI_KEYS:
                series = self._intensity.get(key)
                if series is None or key not in self._results:
                    continue
                y = np.asarray(series, dtype=float).ravel()
                if y.size < 4:
                    continue
                t = np.arange(y.size, dtype=float) * dt
                result = compute_lomb_scargle(t, y, min_freq=freq[0], max_freq=freq[-1], num_freqs=freq.size)
                power = np.asarray(result["power"], dtype=float)
                columns.append(power)
                header_parts.append(f"{ROI_DISPLAY_NAMES[key]}_Power")

            if len(columns) <= 2:
                QMessageBox.warning(self, "No Data", "No ROI power data available to export.")
                return

            data_to_save = np.column_stack(columns)
            header = ",".join(header_parts)
            fmt = ",".join(["%.10g"] * data_to_save.shape[1])

            np.savetxt(
                file_path,
                data_to_save,
                delimiter=",",
                header=header,
                comments="",
                fmt=fmt,
            )

            QMessageBox.information(self, "Export Successful", f"Periodogram data exported to:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"Failed to export data:\n{e}")
