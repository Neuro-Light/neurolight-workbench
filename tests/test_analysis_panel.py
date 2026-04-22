"""Tests for AnalysisPanel — tab creation and widget accessor methods."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import pytest
from PySide6.QtWidgets import QApplication

from ui.analysis_panel import AnalysisPanel
from ui.lomb_scargle_plot import LombScarglePlotWidget
from ui.neuron_detection_widget import NeuronDetectionWidget
from ui.neuron_trajectory_plot import NeuronTrajectoryPlotWidget
from ui.rayleigh_plot import RayLeighPlotWidget
from ui.roi_intensity_plot import ROIIntensityPlotWidget


@pytest.fixture
def app():
    if not QApplication.instance():
        return QApplication([])
    return QApplication.instance()


def test_panel_has_five_tabs(app) -> None:
    panel = AnalysisPanel()
    assert panel.count() == 5


def test_tab_labels(app) -> None:
    panel = AnalysisPanel()
    labels = [panel.tabText(i) for i in range(panel.count())]
    assert labels == ["Detection", "ROI Intensity", "Trajectories", "Lomb–Scargle", "Rayleigh/Rao"]


def test_get_roi_plot_widget_returns_correct_type(app) -> None:
    panel = AnalysisPanel()
    assert isinstance(panel.get_roi_plot_widget(), ROIIntensityPlotWidget)


def test_get_neuron_detection_widget_returns_correct_type(app) -> None:
    panel = AnalysisPanel()
    assert isinstance(panel.get_neuron_detection_widget(), NeuronDetectionWidget)


def test_get_neuron_trajectory_plot_widget_returns_correct_type(app) -> None:
    panel = AnalysisPanel()
    assert isinstance(panel.get_neuron_trajectory_plot_widget(), NeuronTrajectoryPlotWidget)


def test_get_lomb_scargle_widget_returns_correct_type(app) -> None:
    panel = AnalysisPanel()
    assert isinstance(panel.get_lomb_scargle_widget(), LombScarglePlotWidget)


def test_get_rayleigh_plot_widget_returns_correct_type(app) -> None:
    panel = AnalysisPanel()
    assert isinstance(panel.get_rayleigh_plot_widget(), RayLeighPlotWidget)


def test_add_tab_with_none_creates_placeholder(app) -> None:
    panel = AnalysisPanel()
    initial_count = panel.count()
    panel._add_tab("Placeholder Tab")
    assert panel.count() == initial_count + 1
    assert panel.tabText(initial_count) == "Placeholder Tab"
