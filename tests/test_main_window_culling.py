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
    """Build a MainWindow with lightweight mocks, similar to existing test fixtures."""
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
    mock_viewer.set_excluded_frames = Mock()
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
    mock_stack_handler.set_excluded_frames = Mock()
    mock_stack_handler.get_excluded_frames = Mock(return_value=set())

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
    def test_persists_excluded_frames_to_settings(self, main_window) -> None:
        main_window._on_frame_culling_changed({2, 5})
        culling = main_window.experiment.settings["culling"]
        assert culling["excluded_frames"] == [2, 5]

    def test_syncs_exclusions_to_stack_handler(self, main_window) -> None:
        main_window._on_frame_culling_changed({0, 3})
        main_window.stack_handler.set_excluded_frames.assert_called_with({0, 3})

    def test_marks_cull_step_ready_when_not_all_excluded(self, main_window) -> None:
        main_window.stack_handler.get_total_frame_count.return_value = 10
        main_window._on_frame_culling_changed({1})
        assert main_window.workflow_manager.is_step_ready(WorkflowStep.CULL_FRAMES)

    def test_revokes_readiness_when_all_excluded(self, main_window) -> None:
        main_window.stack_handler.get_total_frame_count.return_value = 3
        main_window._on_frame_culling_changed({0, 1, 2})
        assert not main_window.workflow_manager.is_step_ready(WorkflowStep.CULL_FRAMES)

    def test_saves_experiment_when_path_known(self, main_window) -> None:
        main_window.current_experiment_path = "/tmp/test.nexp"
        with patch.object(main_window.manager, "save_experiment") as mock_save:
            main_window._on_frame_culling_changed({1})
            mock_save.assert_called_once()

    def test_does_not_save_when_path_unknown(self, main_window) -> None:
        main_window.current_experiment_path = None
        with patch.object(main_window.manager, "save_experiment") as mock_save:
            main_window._on_frame_culling_changed({1})
            mock_save.assert_not_called()

    def test_resets_downstream_when_cull_already_completed(self, main_window) -> None:
        wm = main_window.workflow_manager
        wm.completed_steps.add(WorkflowStep.CULL_FRAMES)
        wm.completed_steps.add(WorkflowStep.ALIGN_IMAGES)
        main_window.stack_handler.get_total_frame_count.return_value = 10
        main_window._on_frame_culling_changed({2})
        assert WorkflowStep.ALIGN_IMAGES not in wm.completed_steps


# ── _restore_culling_state ───────────────────────────────────────────────


class TestRestoreCullingState:
    def test_restores_valid_entries(self, main_window) -> None:
        main_window.experiment.settings["culling"] = {"excluded_frames": [1, 3, 7]}
        main_window._restore_culling_state()
        main_window.stack_handler.set_excluded_frames.assert_called_with({1, 3, 7})
        main_window.viewer.set_excluded_frames.assert_called_with({1, 3, 7})

    def test_skips_malformed_entries(self, main_window) -> None:
        main_window.experiment.settings["culling"] = {
            "excluded_frames": [0, "bad", None, 4, "also_bad"]
        }
        main_window._restore_culling_state()
        main_window.stack_handler.set_excluded_frames.assert_called_with({0, 4})

    def test_handles_empty_culling_section(self, main_window) -> None:
        main_window.experiment.settings = {}
        main_window._restore_culling_state()
        main_window.stack_handler.set_excluded_frames.assert_called_with(set())

    def test_handles_missing_excluded_frames_key(self, main_window) -> None:
        main_window.experiment.settings["culling"] = {}
        main_window._restore_culling_state()
        main_window.stack_handler.set_excluded_frames.assert_called_with(set())

    def test_coerces_string_ints(self, main_window) -> None:
        main_window.experiment.settings["culling"] = {
            "excluded_frames": ["2", "5"]
        }
        main_window._restore_culling_state()
        main_window.stack_handler.set_excluded_frames.assert_called_with({2, 5})
