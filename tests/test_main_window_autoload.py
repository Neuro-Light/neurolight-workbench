"""Tests for MainWindow._auto_load_experiment_data (stack + ROI + detection restore)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QWidget

from core.experiment_manager import Experiment
from core.roi import ROI
from ui.main_window import MainWindow


@pytest.fixture
def app():
    if not QApplication.instance():
        return QApplication([])
    return QApplication.instance()


def _make_main_window(
    experiment: Experiment,
    *,
    user_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    stack_array: np.ndarray | None,
) -> MainWindow:
    """Build MainWindow with heavy UI mocked; QTimer.singleShot runs callbacks immediately."""

    def _immediate_single_shot(_ms: int, fn: object) -> None:
        assert callable(fn)
        fn()

    monkeypatch.setattr(QTimer, "singleShot", staticmethod(_immediate_single_shot))

    mock_loading = MagicMock()
    mock_loading_cls = MagicMock(return_value=mock_loading)

    mock_viewer = QWidget()
    mock_viewer.index = 0
    mock_viewer.cache = MagicMock()
    mock_viewer.current_roi = None
    mock_viewer.image_label = MagicMock()
    mock_viewer.filename_label = MagicMock()
    mock_viewer.slider = MagicMock()
    mock_viewer.set_stack = MagicMock()
    mock_viewer.set_roi = MagicMock()
    mock_viewer.reset = MagicMock()
    mock_viewer.image_processor = MagicMock()
    mock_viewer.get_current_roi = MagicMock(return_value=None)
    mock_viewer.get_exposure = MagicMock(return_value=0)
    mock_viewer.get_contrast = MagicMock(return_value=0)
    mock_viewer.stackLoaded = MagicMock()
    mock_viewer.stackLoaded.connect = MagicMock()
    mock_viewer.roiSelected = MagicMock()
    mock_viewer.roiSelected.connect = MagicMock()
    mock_viewer.roiDeleted = MagicMock()
    mock_viewer.roiDeleted.connect = MagicMock()
    mock_viewer.displaySettingsChanged = MagicMock()
    mock_viewer.displaySettingsChanged.connect = MagicMock()
    mock_viewer.frameCullingChanged = MagicMock()
    mock_viewer.frameCullingChanged.connect = MagicMock()
    mock_viewer.set_filter_excluded = MagicMock()
    mock_viewer.set_exposure = MagicMock()
    mock_viewer.set_contrast = MagicMock()
    mock_viewer.set_cull_range = MagicMock()

    det = MagicMock()
    det.detectionCompleted = MagicMock()
    det.detectionCompleted.connect = MagicMock()
    det.reset_detection_state = MagicMock()
    det.load_detection_data = MagicMock()
    det.set_roi_mask = MagicMock()
    det.set_frame_data = MagicMock()
    det.set_image_processor = MagicMock()

    mock_roi_plot = MagicMock()
    mock_traj = MagicMock()
    mock_lomb = MagicMock()
    mock_ray = MagicMock()

    mock_analysis = QWidget()
    mock_analysis.roi_plot_widget = mock_roi_plot
    mock_analysis.get_roi_plot_widget = MagicMock(return_value=mock_roi_plot)
    mock_analysis.get_neuron_detection_widget = MagicMock(return_value=det)
    mock_analysis.get_neuron_trajectory_plot_widget = MagicMock(return_value=mock_traj)
    mock_analysis.get_lomb_scargle_widget = MagicMock(return_value=mock_lomb)
    mock_analysis.get_rayleigh_plot_widget = MagicMock(return_value=mock_ray)

    mock_stack_handler = MagicMock()
    mock_stack_handler.files = []
    mock_stack_handler.associate_with_experiment = MagicMock()
    mock_stack_handler.get_all_frames_as_array = MagicMock(return_value=stack_array)
    mock_stack_handler.get_total_frame_count = MagicMock(return_value=3 if stack_array is not None else 0)
    mock_stack_handler.set_included_range = MagicMock()

    mock_data_analyzer = MagicMock()
    mock_data_analyzer.extract_roi_intensity_time_series = MagicMock(
        return_value=np.array([1.0, 2.0, 3.0], dtype=np.float64)
    )

    with (
        patch("ui.main_window.ImageViewer", return_value=mock_viewer),
        patch("ui.main_window.AnalysisPanel", return_value=mock_analysis),
        patch("ui.main_window.ImageStackHandler", return_value=mock_stack_handler),
        patch("ui.main_window.DataAnalyzer", return_value=mock_data_analyzer),
        patch("ui.main_window.LoadingDialog", mock_loading_cls),
    ):
        return MainWindow(experiment, user_experiments_dir=user_dir)


def test_auto_load_stack_from_directory_only(app, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    img_dir = tmp_path / "stack"
    img_dir.mkdir()
    user_dir = tmp_path / "users" / "alice" / "experiments"
    user_dir.mkdir(parents=True)

    exp = Experiment(name="E")
    exp.image_stack_path = str(img_dir)
    exp.image_stack_files = []

    mw = _make_main_window(exp, user_dir=user_dir, monkeypatch=monkeypatch, stack_array=None)
    mw.viewer.set_stack.assert_called_once_with(str(img_dir))
    mw.viewer.set_roi.assert_not_called()


def test_auto_load_stack_from_saved_file_list(app, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tif = tmp_path / "a.tif"
    tif.write_bytes(b"")
    user_dir = tmp_path / "users" / "alice" / "experiments"
    user_dir.mkdir(parents=True)

    exp = Experiment(name="E")
    exp.image_stack_path = str(tmp_path)
    exp.image_stack_files = [str(tif)]

    mw = _make_main_window(exp, user_dir=user_dir, monkeypatch=monkeypatch, stack_array=None)
    mw.viewer.set_stack.assert_called_once_with([str(tif)])


def test_auto_load_fallback_to_directory_when_saved_files_missing(
    app, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    user_dir = tmp_path / "users" / "alice" / "experiments"
    user_dir.mkdir(parents=True)
    fallback = tmp_path / "fallback"
    fallback.mkdir()

    exp = Experiment(name="E")
    exp.image_stack_path = str(fallback)
    exp.image_stack_files = [str(tmp_path / "missing.tif")]

    mw = _make_main_window(exp, user_dir=user_dir, monkeypatch=monkeypatch, stack_array=None)
    mw.viewer.set_stack.assert_called_once_with(str(fallback))


def test_auto_load_restores_rois_and_detection_data(app, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stack_dir = tmp_path / "stack"
    stack_dir.mkdir()
    user_dir = tmp_path / "users" / "alice" / "experiments"
    user_dir.mkdir(parents=True)

    exp = Experiment(name="E")
    exp.image_stack_path = str(stack_dir)
    exp.image_stack_files = []
    exp.rois["roi_1"] = {"x": 0, "y": 0, "width": 4, "height": 4, "shape": "ellipse"}

    locs = np.array([[1.0, 1.0], [2.0, 2.0]], dtype=np.float64)
    traj = np.zeros((2, 3, 2), dtype=np.float64)
    qual = np.array([True, False], dtype=bool)
    exp.set_neuron_detection_data(
        neuron_locations=locs,
        neuron_trajectories=traj,
        quality_mask=qual,
        detection_params={"cell_size": 8},
    )

    frame_data = np.random.default_rng(0).random((3, 8, 8)).astype(np.float32)
    mw = _make_main_window(exp, user_dir=user_dir, monkeypatch=monkeypatch, stack_array=frame_data)

    det = mw.analysis.get_neuron_detection_widget.return_value
    assert mw.viewer.set_stack.called
    assert mw.viewer.set_roi.call_count >= 1
    det.load_detection_data.assert_called_once()
    call_kw = det.load_detection_data.call_args[1]
    assert np.array_equal(call_kw["neuron_locations"], locs)


def test_auto_load_detection_only_when_no_stack_path(app, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    user_dir = tmp_path / "users" / "alice" / "experiments"
    user_dir.mkdir(parents=True)

    exp = Experiment(name="E")
    exp.image_stack_path = None
    locs = np.array([[1.0, 1.0]], dtype=np.float64)
    exp.set_neuron_detection_data(
        neuron_locations=locs,
        neuron_trajectories=np.zeros((1, 2, 2), dtype=np.float64),
        quality_mask=np.array([True], dtype=bool),
    )

    mw = _make_main_window(exp, user_dir=user_dir, monkeypatch=monkeypatch, stack_array=None)
    det = mw.analysis.get_neuron_detection_widget.return_value
    mw.viewer.set_stack.assert_not_called()
    det.load_detection_data.assert_called_once()


def test_auto_load_roi_from_dict_fallback_on_invalid_polygon_points(
    app, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``ROI.from_dict`` failure uses bounding-box fallback in _auto_load_experiment_data."""
    stack_dir = tmp_path / "stack"
    stack_dir.mkdir()
    user_dir = tmp_path / "users" / "alice" / "experiments"
    user_dir.mkdir(parents=True)

    exp = Experiment(name="E")
    exp.image_stack_path = str(stack_dir)
    exp.rois["roi_1"] = {
        "shape": "polygon",
        "points": [["a", "b"], ["c", "d"], ["e", "f"]],
        "x": 0,
        "y": 0,
        "width": 10,
        "height": 10,
    }

    frame_data = np.ones((2, 6, 6), dtype=np.float32)
    mw = _make_main_window(exp, user_dir=user_dir, monkeypatch=monkeypatch, stack_array=frame_data)
    assert mw.viewer.set_roi.called
    first_call = mw.viewer.set_roi.call_args_list[0]
    assert isinstance(first_call[0][0], ROI)
