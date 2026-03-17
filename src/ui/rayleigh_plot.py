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
    QComboBox,
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
from ui.app_settings import get_theme
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
        controls_group.setMaximumWidth(280)
        controls_group.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        controls_layout = QFormLayout()
        self.start_time_edit = QTimeEdit()
        self.start_time_edit.setDisplayFormat("HH:mm")
        self.start_time_edit.setTime(QTime(0, 0))
        self.start_time_edit.setToolTip("Time of first frame (24-hour time)")
        controls_layout.addRow("Experiment Start Time:", self.start_time_edit)
        self.interval_minutes_spin = QSpinBox()
        self.interval_minutes_spin.setRange(1, 1440)
        self.interval_minutes_spin.setValue(7)
        self.interval_minutes_spin.setSuffix(" min")
        self.interval_minutes_spin.setToolTip("Time between consecutive frames in minutes")
        controls_layout.addRow("Interval Between Photos:", self.interval_minutes_spin)
        self.plot_btn = QPushButton("Plot Rayleigh Plot")
        self.plot_btn.setEnabled(False)
        self.plot_btn.clicked.connect(self._plot)
        controls_layout.addRow(self.plot_btn)

        controls_group.setLayout(controls_layout)
        sidebar_layout.addWidget(controls_group, alignment=Qt.AlignHCenter)

        # Cursor / selection readout (separate from the status banner at top)
        self.cursor_group = QGroupBox("Cursor / Selection")
        self.cursor_group.setMaximumWidth(280)
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

        self.rayleigh_title_label = QLabel("Rayleigh:")
        self.rayleigh_title_label.setAlignment(Qt.AlignCenter)
        self.rayleigh_title_label.setStyleSheet("font-size: 16px; font-weight: 700; color: #4A90E2;")
        stats_container.addWidget(self.rayleigh_title_label)

        self.rayleigh_stats_label = QLabel("")
        self.rayleigh_stats_label.setAlignment(Qt.AlignCenter)
        self.rayleigh_stats_label.setWordWrap(True)
        self.rayleigh_stats_label.setStyleSheet("font-size: 14px; font-weight: 500;")
        stats_container.addWidget(self.rayleigh_stats_label)

        self.rao_title_label = QLabel("Rao:")
        self.rao_title_label.setAlignment(Qt.AlignCenter)
        self.rao_title_label.setStyleSheet("font-size: 16px; font-weight: 700; color: #4A90E2; margin-top: 6px;")
        stats_container.addWidget(self.rao_title_label)

        self.rao_stats_label = QLabel("")
        self.rao_stats_label.setAlignment(Qt.AlignCenter)
        self.rao_stats_label.setWordWrap(True)
        self.rao_stats_label.setStyleSheet("font-size: 14px; font-weight: 500;")
        stats_container.addWidget(self.rao_stats_label)

        sidebar_layout.addLayout(stats_container)
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
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.toolbar.setObjectName("mpl_nav_toolbar")
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas)
        main_layout.addWidget(plot_container, 1)

        # Enable picking on the Rayleigh plot so users can click dots.
        self._pick_cid = self.canvas.mpl_connect("pick_event", self._on_pick)
        self._motion_cid = self.canvas.mpl_connect("motion_notify_event", self._on_motion)

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

        start_time = self.start_time_edit.time()
        start_minutes = start_time.hour() * 60 + start_time.minute()
        interval_minutes = int(self.interval_minutes_spin.value())

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
            self.figure.clear()
            self.canvas.draw_idle()
            return

        peak_frames = np.argmax(self.neuron_trajectories[indices], axis=1)
        peak_minutes = (start_minutes + peak_frames * interval_minutes) % (24 * 60)
        theta = (peak_minutes / (24 * 60)) * (2 * np.pi)

        # Slight radial jitter to reduce overplotting
        # for separation of points that have the same peak time.
        # for visuals clarity, not a data transformation.
        jitter = np.array([1.0 - 0.04 * (i % 5) for i in range(len(theta))])

        self.figure.clear()
        ax = self.figure.add_subplot(111, projection="polar")
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_ylim(0, 1.05)
        ax.set_yticks([])
        ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2])
        ax.set_xticklabels(["24", "6", "12", "18"])

        theme = get_mpl_theme(get_theme())

        # Color points by ROI when possible so Rayleigh plot visually matches ROI colours.
        if (
            self.roi_origin is not None
            and len(self.roi_origin) == num_neurons
            and np.any(self.roi_origin[indices] == 0)
            and np.any(self.roi_origin[indices] == 1)
        ):
            roi_1_color = theme["roi_1_line_color"]
            roi_2_color = theme["roi_2_line_color"]
            roi_flags = self.roi_origin[indices]
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
            sc._rayleigh_roi = None  # type: ignore[attr-defined]

        title_time = start_time.toString("HH:mm")
        ax.set_title(
            f"Peak Times (Modulo 24h)\nStart {title_time}  |  Interval {interval_minutes} min",
            fontsize=12,
        )
        # --- Rayleigh and Rao statistics summary (sidebar only) -------------------
        rayleigh_text = ""
        rao_text = ""
        try:
            rayleigh = rayleigh_test(theta)
            angles_deg = np.degrees(theta) % 360.0
            rao = rao_spacing_test(angles_deg)

            r = rayleigh["r"]
            p_rayleigh = rayleigh["p_value"]
            U = rao["U"]
            p_rao = rao["p_value"]

            rayleigh_text = f"r = {r:.3f}, p ≈ {p_rayleigh:.3g}"
            rao_text = f"U = {U:.1f}, p {p_rao}"
        except Exception:
            rayleigh_text = ""
            rao_text = ""

        self.rayleigh_stats_label.setText(rayleigh_text)
        self.rao_stats_label.setText(rao_text)

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
        roi_label = getattr(artist, "_rayleigh_roi", None)
        if mins is None:
            return

        indices = np.atleast_1d(event.ind)
        mins_arr = np.asarray(mins, dtype=float)[indices]

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
        message = f"Selected{roi_part}: {count} neuron(s)\ntime = {t_str}, θ = {theta_deg:.1f}°, r = {r_sel:.3f}"

        self._last_pick_text = message
        self._update_cursor_box()
