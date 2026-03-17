from PySide6.QtWidgets import QLabel, QTabWidget, QVBoxLayout, QWidget

from ui.neuron_detection_widget import NeuronDetectionWidget
from ui.neuron_trajectory_plot import NeuronTrajectoryPlotWidget
from ui.lomb_scargle_plot import LombScarglePlotWidget
from ui.rayleigh_plot import RayLeighPlotWidget
from ui.roi_intensity_plot import ROIIntensityPlotWidget


class AnalysisPanel(QTabWidget):
    def __init__(self) -> None:
        super().__init__()
        self.roi_plot_widget = ROIIntensityPlotWidget()
        self.neuron_detection_widget = NeuronDetectionWidget()
        self.neuron_trajectory_plot_widget = NeuronTrajectoryPlotWidget()
        self.lomb_scargle_widget = LombScarglePlotWidget()
        self.rayleigh_plot_widget = RayLeighPlotWidget()
        # Tab order: Detection, ROI Intensity, Trajectories, Lomb–Scargle, Rayleigh/Rao, Statistics
        self._add_tab("Detection", self.neuron_detection_widget)
        self._add_tab("ROI Intensity", self.roi_plot_widget)
        self._add_tab("Trajectories", self.neuron_trajectory_plot_widget)
        self._add_tab("Lomb–Scargle", self.lomb_scargle_widget)
        self._add_tab("Rayleigh/Rao", self.rayleigh_plot_widget)
        self._add_tab("Statistics")

    def _add_tab(self, title: str, widget: QWidget | None = None) -> None:
        if widget is None:
            w = QWidget()
            layout = QVBoxLayout(w)
            layout.addWidget(QLabel(f"{title} (placeholder)"))
            self.addTab(w, title)
        else:
            self.addTab(widget, title)

    def get_roi_plot_widget(self) -> ROIIntensityPlotWidget:
        """Get the ROI intensity plot widget."""
        return self.roi_plot_widget

    def get_neuron_detection_widget(self) -> NeuronDetectionWidget:
        """Get the neuron detection widget."""
        return self.neuron_detection_widget

    def get_neuron_trajectory_plot_widget(self) -> NeuronTrajectoryPlotWidget:
        """Get the neuron trajectory plot widget."""
        return self.neuron_trajectory_plot_widget

    def get_lomb_scargle_widget(self) -> LombScarglePlotWidget:
        """Get the Lomb–Scargle periodogram widget."""
        return self.lomb_scargle_widget

    def get_rayleigh_plot_widget(self) -> RayLeighPlotWidget:
        """Get the RayLeigh plot widget."""
        return self.rayleigh_plot_widget
