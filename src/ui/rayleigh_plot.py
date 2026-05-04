from __future__ import annotations

from typing import Optional

import numpy as np
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
)
from matplotlib.backends.backend_qtagg import (
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.collections import PathCollection
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, QTime
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTimeEdit,
    QVBoxLayout,
    QWidget,
)

from core.circular_stats import rao_spacing_test, rayleigh_test
from core.rayleigh_cycles import RayleighCycleData, compute_cycle_rayleigh_data
from ui.app_settings import get_theme
from ui.constants import DEFAULT_FRAME_INTERVAL_MINUTES
from ui.help_content import get_help_text
from ui.help_widgets import HelpIconButton
from ui.styles import get_mpl_theme


class RayLeighPlotWidget(QWidget):
    """Widget for plotting peak neuron times on a 24-hour circular (RayLeigh) plot."""

    # This plot is useful for visualizing the distribution of peak activity times across neurons,
    # especially in circadian rhythm studies. Each point on the circle represents a neuron, and
    # its angle corresponds to the time of day when that neuron is most active.
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.neuron_trajectories: Optional[np.ndarray] = None
        self.quality_mask: Optional[np.ndarray] = None
        self.roi_origin: Optional[np.ndarray] = None  # 0 = ROI 1, 1 = ROI 2 per neuron
        self._pick_cid: Optional[int] = None
        self._motion_cid: Optional[int] = None
        self._last_hover_text: str = ""
        self._last_pick_text: str = ""
        self._cycle_data: list[RayleighCycleData] = []
        self._current_cycle: Optional[RayleighCycleData] = None

        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Left sidebar: options (fixed width)
        sidebar = QFrame()
        sidebar.setMaximumWidth(380)
        sidebar.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(8, 8, 8, 8)

        self.status_label = QLabel("No neuron trajectories available. Run detection first.")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setWordWrap(True)
        sidebar_layout.addWidget(self.status_label)

        # ROI view selector: Both / ROI 1 only / ROI 2 only
        roi_row_widget = QWidget()
        roi_row_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        roi_row = QHBoxLayout(roi_row_widget)
        roi_row.setContentsMargins(0, 0, 0, 0)
        roi_row.addWidget(QLabel("Show neurons from:"))
        self.roi_view_combo = QComboBox()
        self.roi_view_combo.addItem("Both (ROI 1 & 2)", "both")
        self.roi_view_combo.addItem("ROI 1 only", "roi_1")
        self.roi_view_combo.addItem("ROI 2 only", "roi_2")
        self.roi_view_combo.setToolTip("Filter Rayleigh plot to neurons from ROI 1, ROI 2, or both.")
        self.roi_view_combo.currentIndexChanged.connect(self._plot)
        roi_row.addWidget(self.roi_view_combo, 1)
        sidebar_layout.addWidget(roi_row_widget, alignment=Qt.AlignHCenter)

        controls_group = QGroupBox("Time Settings")
        controls_group.setMaximumWidth(340)
        controls_group.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        controls_layout = QFormLayout()
        self.start_time_edit = QTimeEdit()
        self.start_time_edit.setDisplayFormat("HH:mm")
        self.start_time_edit.setTime(QTime(0, 0))
        self.start_time_edit.setReadOnly(True)
        self.start_time_edit.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.start_time_edit.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.start_time_edit.setToolTip("Time of first frame (read-only, from frame metadata)")
        controls_layout.addRow("Experiment Start Time:", self.start_time_edit)
        interval_row = QHBoxLayout()
        interval_row.setSpacing(4)
        self.interval_spin = QSpinBox()
        self.interval_spin.setRange(1, 86400)
        self.interval_spin.setValue(DEFAULT_FRAME_INTERVAL_MINUTES)
        self.interval_spin.setToolTip("Time between consecutive frames")
        interval_row.addWidget(self.interval_spin)
        self.interval_unit_combo = QComboBox()
        self.interval_unit_combo.addItem("sec", "seconds")
        self.interval_unit_combo.addItem("min", "minutes")
        self.interval_unit_combo.addItem("hr", "hours")
        self.interval_unit_combo.setCurrentIndex(1)  # Default to minutes
        self.interval_unit_combo.setToolTip("Unit for the interval value")
        interval_row.addWidget(self.interval_unit_combo)
        controls_layout.addRow("Interval Between Photos:", interval_row)
        self.plot_btn = QPushButton("Plot Rayleigh Plot")
        self.plot_btn.setEnabled(False)
        self.plot_btn.clicked.connect(self._plot)
        controls_layout.addRow(self.plot_btn)

        controls_group.setLayout(controls_layout)
        sidebar_layout.addWidget(controls_group, alignment=Qt.AlignHCenter)

        cycle_group = QGroupBox("Cycle Selection")
        cycle_group.setMaximumWidth(340)
        cycle_group.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        cycle_layout = QFormLayout()
        self.cycle_combo = QComboBox()
        self.cycle_combo.setEnabled(False)
        self.cycle_combo.currentIndexChanged.connect(self._plot_selected_cycle)
        self.cycle_combo.setToolTip("Select a trough-to-trough cycle for per-cycle Rayleigh analysis.")
        cycle_layout.addRow("Cycle:", self.cycle_combo)
        self.cycle_info_label = QLabel("No complete trough-to-trough cycles available.")
        self.cycle_info_label.setWordWrap(True)
        self.cycle_info_label.setAlignment(Qt.AlignCenter)
        cycle_layout.addRow(self.cycle_info_label)
        cycle_group.setLayout(cycle_layout)
        sidebar_layout.addWidget(cycle_group, alignment=Qt.AlignHCenter)

        # Cursor / selection readout (separate from the status banner at top)
        self.cursor_group = QGroupBox("Cursor / Selection")
        self.cursor_group.setMaximumWidth(340)
        self.cursor_group.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        cursor_layout = QVBoxLayout(self.cursor_group)
        self.cursor_label = QLabel("Hover over the plot to see θ and r.\nClick a dot to pin its peak time.")
        self.cursor_label.setAlignment(Qt.AlignCenter)
        self.cursor_label.setWordWrap(True)
        self.cursor_label.setStyleSheet("font-size: 13px; font-weight: 500;")
        cursor_layout.addWidget(self.cursor_label)
        sidebar_layout.addWidget(self.cursor_group, alignment=Qt.AlignHCenter)

        # Text summary of Rayleigh / Rao statistics (left panel, larger font, below options)
        stats_container = QVBoxLayout()
        stats_container.setSpacing(4)

        rayleigh_title_row_widget = QWidget()
        rayleigh_title_row = QHBoxLayout(rayleigh_title_row_widget)
        rayleigh_title_row.setContentsMargins(0, 0, 0, 0)
        rayleigh_title_row.setSpacing(6)
        rayleigh_title_row.addStretch()
        self.rayleigh_title_label = QLabel("Rayleigh:")
        self.rayleigh_title_label.setAlignment(Qt.AlignCenter)
        self.rayleigh_title_label.setStyleSheet("font-size: 16px; font-weight: 700; color: #4A90E2;")
        self.rayleigh_title_label.setToolTip(get_help_text("rayleigh.stats"))
        rayleigh_title_row.addWidget(self.rayleigh_title_label)
        rayleigh_title_row.addWidget(HelpIconButton("rayleigh.stats", accessible_name="Help: Rayleigh test"))
        rayleigh_title_row.addStretch()
        stats_container.addWidget(rayleigh_title_row_widget)

        self.rayleigh_stats_label = QLabel("")
        self.rayleigh_stats_label.setAlignment(Qt.AlignCenter)
        self.rayleigh_stats_label.setWordWrap(True)
        self.rayleigh_stats_label.setStyleSheet("font-size: 14px; font-weight: 500;")
        self.rayleigh_stats_label.setToolTip(get_help_text("rayleigh.stats"))
        stats_container.addWidget(self.rayleigh_stats_label)

        rao_title_row_widget = QWidget()
        rao_title_row = QHBoxLayout(rao_title_row_widget)
        rao_title_row.setContentsMargins(0, 0, 0, 0)
        rao_title_row.setSpacing(6)
        rao_title_row.addStretch()
        self.rao_title_label = QLabel("Rao:")
        self.rao_title_label.setAlignment(Qt.AlignCenter)
        self.rao_title_label.setStyleSheet("font-size: 16px; font-weight: 700; color: #4A90E2; margin-top: 6px;")
        self.rao_title_label.setToolTip(get_help_text("rao.stats"))
        rao_title_row.addWidget(self.rao_title_label)
        rao_title_row.addWidget(HelpIconButton("rao.stats", accessible_name="Help: Rao's spacing test"))
        rao_title_row.addStretch()
        stats_container.addWidget(rao_title_row_widget)

        self.rao_stats_label = QLabel("")
        self.rao_stats_label.setAlignment(Qt.AlignCenter)
        self.rao_stats_label.setWordWrap(True)
        self.rao_stats_label.setStyleSheet("font-size: 14px; font-weight: 500;")
        self.rao_stats_label.setToolTip(get_help_text("rao.stats"))
        stats_container.addWidget(self.rao_stats_label)

        sidebar_layout.addLayout(stats_container)

        export_group = QGroupBox("Export")
        export_group.setMaximumWidth(340)
        export_group.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        export_layout = QVBoxLayout(export_group)
        self.export_png_btn = QPushButton("Export Cycle PNG...")
        self.export_png_btn.setEnabled(False)
        self.export_png_btn.clicked.connect(self._export_current_cycle_png)
        export_layout.addWidget(self.export_png_btn)
        self.export_csv_btn = QPushButton("Export Cycle CSV...")
        self.export_csv_btn.setEnabled(False)
        self.export_csv_btn.clicked.connect(self._export_current_cycle_csv)
        export_layout.addWidget(self.export_csv_btn)
        sidebar_layout.addWidget(export_group, alignment=Qt.AlignHCenter)
        sidebar_layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidget(sidebar)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        main_layout.addWidget(scroll)

        # Right: plot (takes remaining space)
        plot_container = QWidget()
        plot_layout = QVBoxLayout(plot_container)
        plot_layout.setContentsMargins(4, 4, 4, 4)
        self.figure = Figure(figsize=(6, 6))
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.setToolTip(get_help_text("rayleigh.plot"))
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.toolbar.setObjectName("mpl_nav_toolbar")
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas)
        main_layout.addWidget(plot_container, 1)

        # Enable picking on the Rayleigh plot so users can click dots.
        self._pick_cid = self.canvas.mpl_connect("pick_event", self._on_pick)
        self._motion_cid = self.canvas.mpl_connect("motion_notify_event", self._on_motion)

    def get_experiment_start_time_minutes(self) -> int:
        """
        Return the experiment start time as minutes since midnight (0–1439).
        """
        start_time = self.start_time_edit.time()
        return start_time.hour() * 60 + start_time.minute()

    def set_experiment_start_time_minutes(self, minutes: int) -> None:
        """
        Set the experiment start time from minutes since midnight (0–1439).
        Values outside this range are wrapped modulo 24 hours.
        """
        minutes_int = int(minutes) % (24 * 60)
        hours = minutes_int // 60
        mins = minutes_int % 60
        self.start_time_edit.setTime(QTime(hours, mins))

    def _get_interval_minutes(self) -> float:
        """
        Return the photo interval converted to minutes based on the selected unit.
        """
        value = self.interval_spin.value()
        unit = self.interval_unit_combo.currentData()
        if unit == "seconds":
            return value / 60.0
        elif unit == "hours":
            return value * 60.0
        else:  # minutes
            return float(value)

    def _get_interval_display(self) -> str:
        """
        Return a display string for the interval (e.g., '5 min', '30 sec').
        """
        value = self.interval_spin.value()
        unit_text = self.interval_unit_combo.currentText()
        return f"{value} {unit_text}"

    # This method is called by the main application when new neuron trajectory data is available.
    def set_trajectory_data(
        self,
        neuron_trajectories: np.ndarray,
        quality_mask: Optional[np.ndarray] = None,
        roi_origin: Optional[np.ndarray] = None,
    ) -> None:
        self.neuron_trajectories = neuron_trajectories
        self.quality_mask = quality_mask
        self.roi_origin = roi_origin

        if neuron_trajectories is None or len(neuron_trajectories) == 0:
            self.status_label.setText("No neuron trajectories to display.")
            self.plot_btn.setEnabled(False)
            return

        num_neurons, num_frames = neuron_trajectories.shape
        if quality_mask is not None:
            num_good = np.sum(quality_mask)
            num_bad = num_neurons - num_good
            self.status_label.setText(
                f"Ready to plot {num_neurons} neurons ({num_good} good, {num_bad} bad) across {num_frames} frames"
            )
        else:
            self.status_label.setText(f"Ready to plot {num_neurons} neurons across {num_frames} frames")
        self.plot_btn.setEnabled(True)
        self._plot()

    # themes to fit other diagrams in the app, and to support dark mode.
    # It applies the theme colors to the figure background, axes background,
    # ticks, title, spines, grid, and legend.
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

    # This method generates the Rayleigh plot based on the neuron trajectories and quality mask.
    # It calculates the peak frame for each neuron, converts it to a time in minutes, and then
    # to an angle in radians. It adds a small radial jitter to reduce overplotting.
    # The neurons are colored based on their quality (good vs bad) if a quality mask is provided.
    # The plot is then styled according to the current theme and displayed on the canvas.
    def _plot(self) -> None:
        if self.neuron_trajectories is None or len(self.neuron_trajectories) == 0:
            QMessageBox.warning(self, "No Data", "No neuron trajectories available.")
            return

        interval_minutes = self._get_interval_minutes()

        # Build mask of neurons to include based on ROI selector and quality mask
        num_neurons = self.neuron_trajectories.shape[0]
        indices = np.arange(num_neurons)

        # ROI filter first (if roi_origin available)
        view_mode = self.roi_view_combo.currentData()
        if self.roi_origin is not None and len(self.roi_origin) == num_neurons:
            if view_mode == "roi_1":
                indices = indices[self.roi_origin == 0]
            elif view_mode == "roi_2":
                indices = indices[self.roi_origin == 1]
            # "both" keeps all indices

        # Then apply quality filter (only good neurons) if available
        if self.quality_mask is not None:
            good_mask = self.quality_mask.astype(bool)
            indices = indices[good_mask[indices]]

        if indices.size == 0:
            QMessageBox.information(
                self,
                "No Neurons",
                "No neurons match the current ROI / quality filters.",
            )
            self._cycle_data = []
            self._set_cycle_controls_enabled(False)
            self._clear_plot("No neurons match the current ROI / quality filters.")
            return

        filtered_trajectories = self.neuron_trajectories[indices]
        cycle_signal = np.mean(filtered_trajectories, axis=0)
        self._cycle_data = compute_cycle_rayleigh_data(
            signal=cycle_signal,
            neuron_trajectories=filtered_trajectories,
            interval_minutes=interval_minutes,
            neuron_indices=indices,
        )

        if not self._cycle_data:
            self._set_cycle_controls_enabled(False)
            self._clear_plot("No complete trough-to-trough cycles were detected.")
            self.status_label.setText("No complete trough-to-trough cycles were detected for the current filters.")
            return

        previous_cycle_index = self._current_cycle.cycle_index if self._current_cycle is not None else None
        self._rebuild_cycle_combo(previous_cycle_index)
        self._set_cycle_controls_enabled(True)
        self._plot_selected_cycle()

    def _rebuild_cycle_combo(self, preferred_cycle_index: Optional[int] = None) -> None:
        self.cycle_combo.blockSignals(True)
        self.cycle_combo.clear()
        selected_idx = 0
        for idx, cycle in enumerate(self._cycle_data):
            label = f"Day {cycle.cycle_index}"
            self.cycle_combo.addItem(label, cycle.cycle_index)
            if preferred_cycle_index is not None and cycle.cycle_index == preferred_cycle_index:
                selected_idx = idx
        self.cycle_combo.setCurrentIndex(selected_idx)
        self.cycle_combo.blockSignals(False)

    def _set_cycle_controls_enabled(self, enabled: bool) -> None:
        self.cycle_combo.setEnabled(enabled)
        self.export_png_btn.setEnabled(enabled)
        self.export_csv_btn.setEnabled(enabled)

    def _clear_plot(self, status_text: str) -> None:
        self.figure.clear()
        self.canvas.draw_idle()
        self.rayleigh_stats_label.setText("")
        self.rao_stats_label.setText("")
        self.cycle_info_label.setText(status_text)
        self._current_cycle = None
        self._last_hover_text = ""
        self._last_pick_text = ""
        self._update_cursor_box()

    def _plot_selected_cycle(self) -> None:
        if not self._cycle_data:
            self._set_cycle_controls_enabled(False)
            self._clear_plot("No complete trough-to-trough cycles available.")
            return

        cycle_idx = self.cycle_combo.currentIndex()
        if cycle_idx < 0 or cycle_idx >= len(self._cycle_data):
            cycle_idx = 0
            self.cycle_combo.setCurrentIndex(cycle_idx)
        cycle = self._cycle_data[cycle_idx]
        self._current_cycle = cycle

        # Slight radial jitter to reduce overplotting
        # for separation of points that have the same peak time.
        # for visuals clarity, not a data transformation.
        theta = cycle.theta
        peak_minutes = cycle.normalized_day_minutes
        jitter = np.array([1.0 - 0.04 * (i % 5) for i in range(len(theta))])

        self.figure.clear()
        ax = self.figure.add_subplot(111, projection="polar")
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_ylim(0, 1.05)
        ax.set_yticks([])
        # Show tick labels for all even hours around the 24-hour circle.
        # We label "24" instead of "0" at the top for readability.
        even_hours = np.arange(0, 24, 2)
        even_thetas = (even_hours / 24.0) * (2 * np.pi)
        even_labels = ["24" if h == 0 else str(h) for h in even_hours]
        ax.set_xticks(even_thetas)
        ax.set_xticklabels(even_labels)

        theme = get_mpl_theme(get_theme())
        cycle_roi_origin = self.roi_origin[cycle.neuron_indices] if self.roi_origin is not None else None

        # Color points by ROI when possible so Rayleigh plot visually matches ROI colours.
        if (
            cycle_roi_origin is not None
            and len(cycle_roi_origin) == len(cycle.neuron_indices)
            and np.any(cycle_roi_origin == 0)
            and np.any(cycle_roi_origin == 1)
        ):
            roi_1_color = theme["roi_1_line_color"]
            roi_2_color = theme["roi_2_line_color"]
            roi_flags = cycle_roi_origin
            roi_1_idx = roi_flags == 0
            roi_2_idx = roi_flags == 1
            if np.any(roi_1_idx):
                sc1 = ax.scatter(
                    theta[roi_1_idx],
                    jitter[roi_1_idx],
                    s=35,
                    color=roi_1_color,
                    alpha=0.8,
                    label="ROI 1",
                    picker=5,
                )
                sc1._rayleigh_peak_minutes = peak_minutes[roi_1_idx]  # type: ignore[attr-defined]
                sc1._rayleigh_peak_frames = cycle.peak_frames[roi_1_idx]  # type: ignore[attr-defined]
                sc1._rayleigh_roi = "ROI 1"  # type: ignore[attr-defined]
            if np.any(roi_2_idx):
                sc2 = ax.scatter(
                    theta[roi_2_idx],
                    jitter[roi_2_idx],
                    s=35,
                    color=roi_2_color,
                    alpha=0.8,
                    label="ROI 2",
                    picker=5,
                )
                sc2._rayleigh_peak_minutes = peak_minutes[roi_2_idx]  # type: ignore[attr-defined]
                sc2._rayleigh_peak_frames = cycle.peak_frames[roi_2_idx]  # type: ignore[attr-defined]
                sc2._rayleigh_roi = "ROI 2"  # type: ignore[attr-defined]
        else:
            # Fallback: single-colour points using the theme's good/neutral colour.
            sc = ax.scatter(
                theta,
                jitter,
                s=35,
                color=theme.get("neutral_color", theme.get("good_color", "#60a5fa")),
                alpha=0.8,
                label="Neurons",
                picker=5,
            )
            sc._rayleigh_peak_minutes = peak_minutes  # type: ignore[attr-defined]
            sc._rayleigh_peak_frames = cycle.peak_frames  # type: ignore[attr-defined]
            sc._rayleigh_roi = None  # type: ignore[attr-defined]

        ax.set_title(
            "Peak Times (Normalized Within Cycle)\n"
            f"Day {cycle.cycle_index}  |  Interval {self._get_interval_display()}",
            fontsize=12,
        )
        # --- Rayleigh and Rao statistics summary (sidebar only) -------------------
        rayleigh_text = ""
        rao_text = ""
        try:
            rayleigh = rayleigh_test(theta)
            rayleigh_text = f"r = {rayleigh['r']:.3f}, p ≈ {rayleigh['p_value']:.3g}"
        except Exception:
            rayleigh_text = ""

        try:
            angles_deg = np.degrees(theta) % 360.0
            rao = rao_spacing_test(angles_deg)
            rao_text = f"U = {rao['U']:.1f}, p {rao['p_value']}"
        except Exception:
            rao_text = ""

        self.rayleigh_stats_label.setText(rayleigh_text)
        self.rao_stats_label.setText(rao_text)
        self.cycle_info_label.setText(
            f"Day {cycle.cycle_index}: {cycle.cycle_length_frames} frames"
            f" ({cycle.cycle_length_minutes:.1f} min)\n"
            f"First peak frame = {cycle.first_peak_frame}; {len(cycle.peak_frames)} neuron peaks plotted."
        )

        ax.legend(loc="lower left", bbox_to_anchor=(1.05, 0.1))
        self._apply_theme(ax)
        self.canvas.draw_idle()

    def refresh_theme(self) -> None:
        """Redraw the plot with the current app theme (e.g. after theme change)."""
        if self.neuron_trajectories is not None and len(self.neuron_trajectories) > 0:
            self._plot()

    def _update_cursor_box(self) -> None:
        parts = []
        if self._last_hover_text:
            parts.append(self._last_hover_text)
        if self._last_pick_text:
            parts.append(self._last_pick_text)
        self.cursor_label.setText(
            "\n".join(parts) if parts else "Hover over the plot to see θ and r.\nClick a" + "dot to pin its peak time."
        )

    def _on_motion(self, event) -> None:
        """Update cursor readout based on hover position."""
        if event.inaxes is None or event.xdata is None or event.ydata is None:
            self._last_hover_text = ""
            self._update_cursor_box()
            return

        theta_rad = float(event.xdata)
        r_val = float(event.ydata)
        theta_deg = (np.degrees(theta_rad) % 360.0 + 360.0) % 360.0
        minutes = int(round((theta_deg / 360.0) * (24 * 60))) % (24 * 60)
        hh = minutes // 60
        mm = minutes % 60
        self._last_hover_text = f"Hover: θ = {theta_deg:.1f}°  ({hh:02d}:{mm:02d})\nHover: r = {r_val:.3f}"
        self._update_cursor_box()

    def _on_pick(self, event) -> None:
        """Handle clicks on Rayleigh plot points to show their peak times."""
        artist = event.artist
        if not isinstance(artist, PathCollection):
            return

        mins = getattr(artist, "_rayleigh_peak_minutes", None)
        frames = getattr(artist, "_rayleigh_peak_frames", None)
        roi_label = getattr(artist, "_rayleigh_roi", None)
        if mins is None:
            return

        indices = np.atleast_1d(event.ind)
        mins_arr = np.asarray(mins, dtype=float)[indices]
        frames_arr = np.asarray(frames, dtype=float)[indices] if frames is not None else None

        # Use the first selected point as the representative for theta / r.
        offsets = artist.get_offsets()
        theta_sel, r_sel = map(float, offsets[indices[0]])

        # Convert theta to degrees and time-of-day (HH:MM)
        theta_deg = (np.degrees(theta_sel) % 360.0 + 360.0) % 360.0
        m0 = float(mins_arr[0]) % (24 * 60)
        h0 = int(m0 // 60)
        minute0 = int(round(m0 % 60)) % 60
        t_str = f"{h0:02d}:{minute0:02d}"

        count = len(indices)
        roi_part = f" ({roi_label})" if roi_label else ""
        frame_part = ""
        if frames_arr is not None:
            frame_part = f", frame = {int(frames_arr[0])}"
        message = (
            f"Selected{roi_part}: {count} neuron(s)\n"
            f"normalized time = {t_str}{frame_part}, θ = {theta_deg:.1f}°, r = {r_sel:.3f}"
        )

        self._last_pick_text = message
        self._update_cursor_box()

    def _export_current_cycle_png(self) -> None:
        """Export the currently displayed cycle plot as a PNG image."""
        if self._current_cycle is None:
            QMessageBox.information(self, "No Cycle", "There is no cycle plot to export.")
            return

        cycle = self._current_cycle
        default_name = f"rayleigh_cycle_{cycle.cycle_index}.png"
        file_path, _ = QFileDialog.getSaveFileName(self, "Export Cycle PNG", default_name, "PNG Files (*.png)")
        if not file_path:
            return
        self.figure.savefig(file_path, dpi=300, bbox_inches="tight")

    def _export_current_cycle_csv(self) -> None:
        """Export the currently displayed cycle's normalized peak data to CSV."""
        if self._current_cycle is None:
            QMessageBox.information(self, "No Cycle", "There is no cycle dataset to export.")
            return

        cycle = self._current_cycle
        default_name = f"rayleigh_cycle_{cycle.cycle_index}.csv"
        file_path, _ = QFileDialog.getSaveFileName(self, "Export Cycle CSV", default_name, "CSV Files (*.csv)")
        if not file_path:
            return

        roi_labels = np.full(len(cycle.neuron_indices), "", dtype=object)
        if self.roi_origin is not None and len(self.roi_origin) > int(np.max(cycle.neuron_indices)):
            roi_values = self.roi_origin[cycle.neuron_indices]
            roi_labels = np.where(roi_values == 0, "ROI 1", np.where(roi_values == 1, "ROI 2", ""))

        theta_deg = (np.degrees(cycle.theta) % 360.0 + 360.0) % 360.0
        normalized_hours = cycle.normalized_day_minutes / 60.0
        rows = np.column_stack(
            [
                np.full(len(cycle.neuron_indices), cycle.cycle_index, dtype=int),
                np.full(len(cycle.neuron_indices), cycle.trough_start_frame, dtype=int),
                np.full(len(cycle.neuron_indices), cycle.trough_end_frame, dtype=int),
                np.full(len(cycle.neuron_indices), cycle.cycle_length_frames, dtype=int),
                np.full(len(cycle.neuron_indices), cycle.cycle_length_minutes, dtype=float),
                np.full(len(cycle.neuron_indices), cycle.first_peak_frame, dtype=int),
                cycle.neuron_indices,
                roi_labels,
                cycle.peak_frames,
                cycle.normalized_day_minutes,
                normalized_hours,
                cycle.theta,
                theta_deg,
            ]
        )
        np.savetxt(
            file_path,
            rows,
            delimiter=",",
            fmt="%s",
            header=(
                "cycle_index,trough_start_frame,trough_end_frame,cycle_length_frames,"
                "cycle_length_minutes,first_peak_frame,neuron_index,roi_label,peak_frame,"
                "normalized_day_minutes,normalized_day_hours,theta_radians,theta_degrees"
            ),
            comments="",
        )
