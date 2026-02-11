from __future__ import annotations

from typing import Optional

import numpy as np
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QFileDialog,
    QMessageBox,
)
from PySide6.QtCore import Qt

from ui.app_settings import get_theme
from ui.styles import get_mpl_theme


class ROIIntensityPlotWidget(QWidget):
    """Widget for plotting ROI intensity over time with CSV export capability."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.intensity_data: Optional[np.ndarray] = None
        self.frames_data: Optional[np.ndarray] = None
        self.experiment: Optional["Experiment"] = None
        self._hover_cid = None
        
        layout = QVBoxLayout(self)
        
        # Label for status
        self.status_label = QLabel("No ROI selected. Select an ROI in the image viewer.")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)
        
        # Matplotlib figure and canvas (theme applied when plotting)
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.toolbar.setObjectName("mpl_nav_toolbar")
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        
        # Hover label for coordinates
        self.hover_label = QLabel("Hover over plot for frame and intensity.")
        self.hover_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.hover_label)
        
        # Buttons layout
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

    def _on_motion(self, event) -> None:
        """Show frame and intensity under cursor."""
        if (
            self.intensity_data is None
            or event.inaxes is None
            or event.xdata is None
            or event.ydata is None
        ):
            self.hover_label.setText("Hover over plot for frame and intensity.")
            return
        frame_idx = int(round(event.xdata))
        n = len(self.intensity_data)
        if frame_idx < 0 or frame_idx >= n:
            self.hover_label.setText("Hover over plot for frame and intensity.")
            return
        self.hover_label.setText(
            f"Frame {frame_idx}  ·  Intensity {self.intensity_data[frame_idx]:.3f}"
        )

    def plot_intensity_time_series(
        self, 
        intensity_data: np.ndarray,
        roi_coords: tuple[int, int, int, int]
    ) -> None:
        """
        Plot mean intensity time series for the selected ROI.
        
        Args:
            intensity_data: 1D numpy array of mean intensities across frames
            roi_coords: Tuple of (x, y, width, height) for the ROI
        """
        self.intensity_data = intensity_data
        self.roi_coords = roi_coords
        
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        theme = get_mpl_theme(get_theme())
        
        frames = np.arange(len(intensity_data))
        ax.plot(frames, intensity_data, linewidth=2, color=theme["roi_line_color"])
        ax.set_xlabel("Frame Number", fontsize=12)
        ax.set_ylabel("Mean Pixel Intensity", fontsize=12)
        ax.set_title(
            f"ROI Intensity Over Time\n"
            f"ROI: ({roi_coords[0]}, {roi_coords[1]}) "
            f"{roi_coords[2]}x{roi_coords[3]}",
            fontsize=14,
        )
        self._apply_theme(ax)
        
        self.status_label.setText(
            f"ROI: ({roi_coords[0]}, {roi_coords[1]}) "
            f"{roi_coords[2]}x{roi_coords[3]} | "
            f"Frames: {len(intensity_data)} | "
            f"Mean Intensity: {np.mean(intensity_data):.2f}"
        )
        
        self.export_btn.setEnabled(True)
        self.export_png_btn.setEnabled(True)
        
        if self._hover_cid is not None:
            self.canvas.mpl_disconnect(self._hover_cid)
        self._hover_cid = self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas.draw_idle()

    def _export_to_png(self) -> None:
        """Save the current plot as a PNG image."""
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
        """Export intensity data to CSV file."""
        if self.intensity_data is None:
            QMessageBox.warning(self, "No Data", "No intensity data to export.")
            return
        
        # Check for experiment name properly set
        experiment_name = self.experiment.name if self.experiment else "Experiment"
        
        # Get save location
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Intensity Data",
            f"{experiment_name}_roi_intensity_data.csv",
            "CSV Files (*.csv)"
        )
        
        if not file_path:
            return
        
        try:
            # Create CSV with frame numbers and intensities
            # Format: Frame, Intensity
            data_to_save = np.column_stack([
                np.arange(len(self.intensity_data)),
                self.intensity_data
            ])
            
            # Save with header
            header = "Frame,Mean_Intensity"
            np.savetxt(
                file_path,
                data_to_save,
                delimiter=',',
                header=header,
                comments='',
                fmt='%d,%.6f'
            )
            
            QMessageBox.information(
                self,
                "Export Successful",
                f"Intensity data exported to:\n{file_path}"
            )
        except Exception as e:
            QMessageBox.critical(
                self,
                "Export Failed",
                f"Failed to export data:\n{str(e)}"
            )

    def refresh_theme(self) -> None:
        """Redraw the plot with the current app theme (e.g. after theme change)."""
        if self.intensity_data is not None and self.roi_coords is not None:
            self.plot_intensity_time_series(self.intensity_data, self.roi_coords)

    def clear_plot(self) -> None:
        """Clear the plot and reset state."""
        if self._hover_cid is not None:
            self.canvas.mpl_disconnect(self._hover_cid)
            self._hover_cid = None
        self.figure.clear()
        self.canvas.draw()
        self.intensity_data = None
        self.roi_coords = None
        self.status_label.setText("No ROI selected. Select an ROI in the image viewer.")
        self.hover_label.setText("Hover over plot for frame and intensity.")
        self.export_btn.setEnabled(False)
        self.export_png_btn.setEnabled(False)

