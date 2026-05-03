"""Tests for MainWindow frame-culling integration (persistence, restore, workflow gating)."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest
from PySide6.QtWidgets import QApplication, QWidget

from core.experiment_manager import Experiment
from ui.main_window import MainWindow
from ui.workflow import WorkflowStep


@pytest.fixture
def app():
    if not QApplication.instance():
        return QApplication([])
    return QApplication.instance()


def _make_main_window(app, experiment=None):
    exp = experiment or Experiment(name="Culling Test")
    exp.settings = exp.settings if exp.settings else {}

    mock_viewer = QWidget()
    mock_viewer.upload_btn = Mock()
    mock_viewer.set_stack = Mock()
    mock_viewer.set_roi = Mock()
    mock_viewer.set_exposure = Mock()
    mock_viewer.set_contrast = Mock()
    mock_viewer.get_current_roi = Mock(return_value=None)
    mock_viewer.get_exposure = Mock(return_value=0)
    mock_viewer.get_contrast = Mock(return_value=0)
    mock_viewer.set_cull_range = Mock()
    mock_viewer.set_filter_excluded = Mock()
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

    mock_analysis = QWidget()
    mock_analysis.get_roi_plot_widget = Mock(return_value=Mock())
    mock_analysis.get_neuron_detection_widget = Mock(return_value=Mock())
    mock_analysis.get_neuron_trajectory_plot_widget = Mock(return_value=Mock())
    mock_analysis.get_rayleigh_plot_widget = Mock(return_value=Mock())
    mock_analysis.get_lomb_scargle_widget = Mock(return_value=Mock())

    mock_stack_handler = Mock()
    mock_stack_handler.files = []
    mock_stack_handler.associate_with_experiment = Mock()
    mock_stack_handler.get_total_frame_count = Mock(return_value=10)
    mock_stack_handler.set_included_range = Mock()
    mock_stack_handler.get_included_range = Mock(return_value=(0, 9))

    with (
        patch("ui.main_window.ImageViewer", return_value=mock_viewer),
        patch("ui.main_window.AnalysisPanel", return_value=mock_analysis),
        patch("ui.main_window.ImageStackHandler", return_value=mock_stack_handler),
        patch("ui.main_window.DataAnalyzer", return_value=Mock()),
        patch("ui.main_window.QTimer.singleShot"),
    ):
        window = MainWindow(exp)
        return window


@pytest.fixture
def main_window(app):
    return _make_main_window(app)


# ── _on_frame_culling_changed ────────────────────────────────────────────


class TestOnFrameCullingChanged:
    def test_persists_range_to_settings(self, main_window) -> None:
        main_window._on_frame_culling_changed(2, 7)
        culling = main_window.experiment.settings["culling"]
        assert culling["start_frame"] == 2
        assert culling["end_frame"] == 7

    def test_syncs_range_to_stack_handler(self, main_window) -> None:
        main_window._on_frame_culling_changed(1, 8)
        main_window.stack_handler.set_included_range.assert_called_with(1, 8)

    def test_marks_cull_step_ready_with_valid_range(self, main_window) -> None:
        main_window.stack_handler.get_total_frame_count.return_value = 10
        main_window._on_frame_culling_changed(0, 9)
        assert main_window.workflow_manager.is_step_ready(WorkflowStep.CULL_FRAMES)

    def test_revokes_readiness_when_start_exceeds_end(self, main_window) -> None:
        main_window.stack_handler.get_total_frame_count.return_value = 10
        main_window._on_frame_culling_changed(5, 3)
        assert not main_window.workflow_manager.is_step_ready(WorkflowStep.CULL_FRAMES)

    def test_saves_experiment_when_path_known(self, main_window) -> None:
        main_window.current_experiment_path = "/tmp/test.nexp"
        with patch.object(main_window.manager, "save_experiment") as mock_save:
            main_window._on_frame_culling_changed(0, 9)
            mock_save.assert_called_once()

    def test_does_not_save_when_path_unknown(self, main_window) -> None:
        main_window.current_experiment_path = None
        with patch.object(main_window.manager, "save_experiment") as mock_save:
            main_window._on_frame_culling_changed(0, 9)
            mock_save.assert_not_called()

    def test_resets_downstream_when_cull_already_completed(self, main_window) -> None:
        wm = main_window.workflow_manager
        wm.completed_steps.add(WorkflowStep.CULL_FRAMES)
        wm.completed_steps.add(WorkflowStep.ALIGN_IMAGES)
        main_window.stack_handler.get_total_frame_count.return_value = 10
        main_window._on_frame_culling_changed(1, 8)
        assert WorkflowStep.ALIGN_IMAGES not in wm.completed_steps


# ── _restore_culling_state ───────────────────────────────────────────────


class TestRestoreCullingState:
    def test_restores_valid_range(self, main_window) -> None:
        main_window.stack_handler.get_total_frame_count.return_value = 10
        main_window.experiment.settings["culling"] = {"start_frame": 2, "end_frame": 7}
        main_window._restore_culling_state()
        main_window.stack_handler.set_included_range.assert_called_with(2, 7)
        main_window.viewer.set_cull_range.assert_called_with(2, 7)

    def test_defaults_to_full_range_when_missing(self, main_window) -> None:
        main_window.stack_handler.get_total_frame_count.return_value = 5
        main_window.experiment.settings = {}
        main_window._restore_culling_state()
        main_window.stack_handler.set_included_range.assert_called_with(0, 4)

    def test_handles_empty_culling_section(self, main_window) -> None:
        main_window.stack_handler.get_total_frame_count.return_value = 5
        main_window.experiment.settings["culling"] = {}
        main_window._restore_culling_state()
        main_window.stack_handler.set_included_range.assert_called_with(0, 4)

    def test_clamps_out_of_bounds_values(self, main_window) -> None:
        main_window.stack_handler.get_total_frame_count.return_value = 5
        main_window.experiment.settings["culling"] = {"start_frame": -1, "end_frame": 100}
        main_window._restore_culling_state()
        start, end = main_window.stack_handler.set_included_range.call_args[0]
        assert start >= 0
        assert end <= 4

    def test_handles_malformed_values(self, main_window) -> None:
        main_window.stack_handler.get_total_frame_count.return_value = 5
        main_window.experiment.settings["culling"] = {"start_frame": "bad", "end_frame": None}
        main_window._restore_culling_state()
        main_window.stack_handler.set_included_range.assert_called_with(0, 4)
