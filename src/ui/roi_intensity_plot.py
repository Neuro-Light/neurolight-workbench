from __future__ import annotations

from typing import Optional

import numpy as np

from core.roi import ROI, ROIShape
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QPushButton,
    QLabel,
    QFileDialog,
    QMessageBox,
)
from PySide6.QtCore import Qt


class ROIIntensityPlotWidget(QWidget):
    """Widget for plotting ROI intensity over time with CSV export capability."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.intensity_data: Optional[np.ndarray] = None
        self.frames_data: Optional[np.ndarray] = None
        self.roi: Optional[ROI] = None
        self.experiment: Optional["Experiment"] = None
        
        layout = QVBoxLayout(self)
        
        # Label for status
        self.status_label = QLabel("No ROI selected. Select an ROI in the image viewer.")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)
        
        # Matplotlib figure and canvas
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        # Buttons layout
        button_layout = QVBoxLayout()
        
        self.export_btn = QPushButton("Export to CSV")
        self.export_btn.clicked.connect(self._export_to_csv)
        self.export_btn.setEnabled(False)
        button_layout.addWidget(self.export_btn)
        
        layout.addLayout(button_layout)

    def plot_intensity_time_series(
        self,
        intensity_data: np.ndarray,
        roi: ROI
    ) -> None:
        """
        Plot mean intensity time series for the selected ROI.

        Args:
            intensity_data: 1D numpy array of mean intensities across frames
            roi: ROI object (polygon or ellipse)
        """
        self.intensity_data = intensity_data
        self.roi = roi

        # Clear previous plot
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        frames = np.arange(len(intensity_data))
        ax.plot(frames, intensity_data, linewidth=2, color='blue')
        ax.set_xlabel('Frame Number', fontsize=12)
        ax.set_ylabel('Mean Pixel Intensity', fontsize=12)

        if roi.shape == ROIShape.POLYGON and roi.points:
            roi_desc = f"Polygon ({len(roi.points)} points)"
        else:
            roi_desc = f"({roi.x}, {roi.y}) {roi.width}x{roi.height}"
        ax.set_title(f"ROI Intensity Over Time\n{roi_desc}", fontsize=14)
        ax.grid(True, alpha=0.3)

        self.status_label.setText(
            f"{roi_desc} | Frames: {len(intensity_data)} | "
            f"Mean Intensity: {np.mean(intensity_data):.2f}"
        )
        self.export_btn.setEnabled(True)
        self.canvas.draw()

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

    def clear_plot(self) -> None:
        """Clear the plot and reset state."""
        self.figure.clear()
        self.canvas.draw()
        self.intensity_data = None
        self.roi = None
        self.status_label.setText("No ROI selected. Select an ROI in the image viewer.")
        self.export_btn.setEnabled(False)

