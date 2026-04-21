"""Tests for MainWindow time-flow behaviors."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest
from PySide6.QtWidgets import QApplication, QDialog, QWidget

from core.experiment_manager import Experiment
from ui.main_window import MainWindow


@pytest.fixture
def app():
    if not QApplication.instance():
        return QApplication([])
    return QApplication.instance()


@pytest.fixture
def main_window(app):
    exp = Experiment(name="Time Flow Experiment")
    exp.settings = {}

    mock_viewer = QWidget()
    mock_viewer.upload_btn = Mock()
    mock_viewer.set_stack = Mock()
    mock_viewer.set_roi = Mock()
    mock_viewer.set_exposure = Mock()
    mock_viewer.set_contrast = Mock()
    mock_viewer.get_current_roi = Mock(return_value=None)
    mock_viewer.get_exposure = Mock(return_value=0)
    mock_viewer.get_contrast = Mock(return_value=0)
    mock_viewer.stackLoaded = Mock()
    mock_viewer.stackLoaded.connect = Mock()
    mock_viewer.roiSelected = Mock()
    mock_viewer.roiSelected.connect = Mock()
    mock_viewer.roiDeleted = Mock()
    mock_viewer.roiDeleted.connect = Mock()
    mock_viewer.displaySettingsChanged = Mock()
    mock_viewer.displaySettingsChanged.connect = Mock()
    mock_viewer.frameCullingChanged = Mock()
    mock_viewer.frameCullingChanged.connect = Mock()
    mock_viewer.set_filter_excluded = Mock()

    mock_analysis = QWidget()
    mock_analysis.get_rayleigh_plot_widget = Mock(return_value=Mock())
    mock_analysis.get_neuron_trajectory_plot_widget = Mock(return_value=Mock())
    mock_analysis.get_lomb_scargle_widget = Mock(return_value=Mock())
    mock_analysis.get_roi_plot_widget = Mock(return_value=Mock())
    mock_analysis.get_neuron_detection_widget = Mock(return_value=Mock())

    mock_stack_handler = Mock()
    mock_stack_handler.files = []
    mock_stack_handler.associate_with_experiment = Mock()
    mock_stack_handler.get_all_frames_as_array = Mock(return_value=None)

    with (
        patch("ui.main_window.ImageViewer", return_value=mock_viewer),
        patch("ui.main_window.AnalysisPanel", return_value=mock_analysis),
        patch("ui.main_window.ImageStackHandler", return_value=mock_stack_handler),
        patch("ui.main_window.DataAnalyzer", return_value=Mock()),
        patch("ui.main_window.QTimer.singleShot"),
    ):
        yield MainWindow(exp)


def test_apply_experiment_time_settings_applies_acquisition_to_all_widgets(main_window):
    rayleigh_widget = Mock()
    traj_widget = Mock()
    ls_widget = Mock()
    roi_widget = Mock()
    main_window.analysis.get_rayleigh_plot_widget = Mock(return_value=rayleigh_widget)
    main_window.analysis.get_neuron_trajectory_plot_widget = Mock(return_value=traj_widget)
    main_window.analysis.get_lomb_scargle_widget = Mock(return_value=ls_widget)
    main_window.analysis.get_roi_plot_widget = Mock(return_value=roi_widget)
    main_window.experiment.settings = {
        "acquisition": {
            "experiment_start_time": "12:34:56",
            "frame_interval_minutes": 7.5,
        }
    }

    main_window._apply_experiment_time_settings()

    rayleigh_widget.set_experiment_start_time_minutes.assert_called_once_with(12 * 60 + 34)
    traj_widget.set_time_settings.assert_called_once_with(interval_minutes=7.5, start_time="12:34:56")
    ls_widget.set_frame_interval_minutes.assert_called_once_with(7.5)
    roi_widget.set_frame_interval_minutes.assert_called_once_with(7.5)


def test_apply_experiment_time_settings_uses_legacy_minutes_fallback(main_window):
    rayleigh_widget = Mock()
    main_window.analysis.get_rayleigh_plot_widget = Mock(return_value=rayleigh_widget)
    main_window.experiment.settings = {"time": {"start_minutes": "120"}}

    main_window._apply_experiment_time_settings()

    rayleigh_widget.set_experiment_start_time_minutes.assert_called_once_with(120)


def test_confirm_start_time_from_loaded_stack_accept_updates_settings(main_window):
    main_window.stack_handler.files = ["/tmp/a.tif", "/tmp/z.tif"]
    main_window.experiment.settings = {}
    main_window.experiment.update_modified_date = Mock()
    dlg = Mock()
    dlg.exec.return_value = QDialog.Accepted
    dlg.selected_start_time.return_value = "06:30:00"

    with (
        patch("ui.main_window._get_exif_timestamp", side_effect=["06:00:00", "06:00:00", "06:00:00"]),
        patch("ui.main_window._ConfirmStartTimeDialog", return_value=dlg),
    ):
        main_window._confirm_start_time_from_loaded_stack()

    assert main_window.experiment.settings["acquisition"]["experiment_start_time"] == "06:30:00"
    main_window.experiment.update_modified_date.assert_called_once()


def test_confirm_start_time_from_loaded_stack_cancel_keeps_existing(main_window):
    main_window.stack_handler.files = ["/tmp/a.tif"]
    main_window.experiment.settings = {"acquisition": {"experiment_start_time": "08:00:00"}}
    dlg = Mock()
    dlg.exec.return_value = QDialog.Rejected

    with (
        patch("ui.main_window._get_exif_timestamp", return_value="08:00:00"),
        patch("ui.main_window._ConfirmStartTimeDialog", return_value=dlg),
    ):
        main_window._confirm_start_time_from_loaded_stack()

    assert main_window.experiment.settings["acquisition"]["experiment_start_time"] == "08:00:00"
