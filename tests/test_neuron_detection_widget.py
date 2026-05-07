from __future__ import annotations

import importlib
import sys
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication, QMessageBox, QWidget

# The test environment does not always install matplotlib, but this widget only
# needs lightweight stubs for construction in this test.
matplotlib_module = types.ModuleType("matplotlib")
backends_module = types.ModuleType("matplotlib.backends")
backend_qtagg_module = types.ModuleType("matplotlib.backends.backend_qtagg")
figure_module = types.ModuleType("matplotlib.figure")


class _DummyCanvas(QWidget):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def draw(self) -> None:
        pass


class _DummyFigure:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def clear(self) -> None:
        pass

    def add_subplot(self, *args, **kwargs) -> MagicMock:
        return MagicMock()


backend_qtagg_module.FigureCanvasQTAgg = _DummyCanvas
figure_module.Figure = _DummyFigure

sys.modules.setdefault("matplotlib", matplotlib_module)
sys.modules.setdefault("matplotlib.backends", backends_module)
sys.modules.setdefault("matplotlib.backends.backend_qtagg", backend_qtagg_module)
sys.modules.setdefault("matplotlib.figure", figure_module)

NeuronDetectionWidget = importlib.import_module("ui.neuron_detection_widget").NeuronDetectionWidget


@pytest.fixture
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        return QApplication([])
    return app


def test_max_absent_frames_defaults_to_stack_length(qapp: QApplication) -> None:
    widget = NeuronDetectionWidget()
    widget.set_frame_data(np.zeros((7, 12, 12), dtype=np.float32))

    assert widget.max_absent_frames_spin.minimum() == 0
    assert widget.max_absent_frames_spin.maximum() == 7
    assert widget.max_absent_frames_spin.value() == 7


def test_set_image_processor_assigns(qapp: QApplication) -> None:
    w = NeuronDetectionWidget()
    proc = MagicMock()
    w.set_image_processor(proc)
    assert w.image_processor is proc


def test_max_absent_frames_keeps_user_value_when_stack_changes(qapp: QApplication) -> None:
    w = NeuronDetectionWidget()
    w.set_frame_data(np.zeros((5, 4, 4), dtype=np.float32))
    w.max_absent_frames_spin.setValue(2)
    w.set_frame_data(np.zeros((10, 4, 4), dtype=np.float32))
    assert w.max_absent_frames_spin.maximum() == 10
    assert w.max_absent_frames_spin.value() == 2


def test_max_absent_frames_resets_when_still_at_default(qapp: QApplication) -> None:
    w = NeuronDetectionWidget()
    w.set_frame_data(np.zeros((5, 4, 4), dtype=np.float32))
    assert w.max_absent_frames_spin.value() == 5
    w.set_frame_data(np.zeros((8, 4, 4), dtype=np.float32))
    assert w.max_absent_frames_spin.value() == 8


def test_effective_mask_detect_both_falls_back_to_single_mask(qapp: QApplication) -> None:
    w = NeuronDetectionWidget()
    w.detect_mode_combo.setCurrentIndex(2)  # Detect Both ROIs
    m1 = np.zeros((4, 4), dtype=bool)
    m1[1:3, 1:3] = True
    w.set_roi_mask("roi_1", m1)
    w.set_roi_mask("roi_2", None)
    assert w._effective_mask() is m1


def test_compute_roi_origin_none_when_no_locations(qapp: QApplication) -> None:
    w = NeuronDetectionWidget()
    assert w._compute_roi_origin() is None


def test_compute_roi_origin_single_roi2(qapp: QApplication) -> None:
    w = NeuronDetectionWidget()
    w.neuron_locations = np.array([[1.0, 1.0], [2.0, 2.0]], dtype=np.float64)
    w.roi_masks["roi_1"] = None
    w.roi_masks["roi_2"] = np.ones((5, 5), dtype=bool)
    origin = w._compute_roi_origin()
    assert origin is not None
    assert np.all(origin == 1)


def test_compute_roi_origin_two_masks_assignment(qapp: QApplication) -> None:
    w = NeuronDetectionWidget()
    m1 = np.zeros((12, 12), dtype=bool)
    m2 = np.zeros((12, 12), dtype=bool)
    m1[0:4, 0:4] = True
    m2[8:12, 8:12] = True
    w.roi_masks["roi_1"] = m1
    w.roi_masks["roi_2"] = m2
    w.neuron_locations = np.array([[1.0, 1.0], [9.0, 9.0]], dtype=np.float64)
    origin = w._compute_roi_origin()
    assert origin.tolist() == [0, 1]


def test_load_detection_data_with_mean_frame_and_callbacks(qapp: QApplication) -> None:
    w = NeuronDetectionWidget()
    cb = MagicMock()
    w.set_trajectory_plot_callback(cb)
    locs = np.array([[1.0, 2.0]], dtype=np.float64)
    traj = np.zeros((1, 3, 2), dtype=np.float64)
    qual = np.array([True], dtype=bool)
    mean = np.ones((8, 8), dtype=np.float32)
    w.load_detection_data(
        locs,
        traj,
        qual,
        mean_frame=mean,
        detection_params={"cell_size": 12, "max_absent_frames": 3},
    )
    assert w.mean_frame is mean
    assert w.export_locations_btn.isEnabled()
    cb.assert_called_once()
    # No ROI masks: origin defaults to zeros (same length as locations).
    assert np.array_equal(cb.call_args.kwargs["roi_origin"], np.zeros(1, dtype=np.intp))


def test_load_detection_data_derives_frames_from_stack(qapp: QApplication) -> None:
    w = NeuronDetectionWidget()
    stack = np.random.default_rng(1).random((4, 6, 6)).astype(np.float32)
    w.set_frame_data(stack)
    mask = np.zeros((6, 6), dtype=bool)
    mask[2:5, 2:5] = True
    w.set_roi_mask("roi_1", mask)
    locs = np.array([[3.0, 3.0]], dtype=np.float64)
    traj = np.zeros((1, 4, 2), dtype=np.float64)
    qual = np.array([True], dtype=bool)
    w.load_detection_data(locs, traj, qual, mean_frame=None, detection_params=None)
    assert w.mean_frame is not None and w.mean_frame.shape == (6, 6)
    assert w._display_frame is not None


def test_load_detection_data_switches_to_both_when_saved_dual_roi(qapp: QApplication) -> None:
    w = NeuronDetectionWidget()
    m1 = np.zeros((8, 8), dtype=bool)
    m2 = np.zeros((8, 8), dtype=bool)
    m1[0:4, 0:4] = True
    m2[4:8, 4:8] = True
    w.set_roi_mask("roi_1", m1)
    w.set_roi_mask("roi_2", m2)
    locs = np.array([[1.0, 1.0], [6.0, 6.0]], dtype=np.float64)
    traj = np.zeros((2, 2, 2), dtype=np.float64)
    qual = np.array([True, True], dtype=bool)
    roi_origin = np.array([0, 1], dtype=np.intp)
    w.load_detection_data(locs, traj, qual, mean_frame=np.ones((8, 8), dtype=np.float32), roi_origin=roi_origin)
    assert w.detect_mode_combo.currentText() == "Detect Both ROIs"


def test_load_detection_data_uses_saved_roi_origin_in_callback(qapp: QApplication) -> None:
    w = NeuronDetectionWidget()
    cb = MagicMock()
    w.set_trajectory_plot_callback(cb)
    locs = np.array([[1.0, 1.0]], dtype=np.float64)
    traj = np.zeros((1, 2, 2), dtype=np.float64)
    qual = np.array([True], dtype=bool)
    roi_origin = np.array([1], dtype=np.intp)
    w.load_detection_data(locs, traj, qual, mean_frame=np.ones((4, 4), dtype=np.float32), roi_origin=roi_origin)
    cb.assert_called_once()
    assert np.array_equal(cb.call_args.kwargs["roi_origin"], roi_origin)


def test_reset_detection_state_clears(qapp: QApplication) -> None:
    w = NeuronDetectionWidget()
    w.neuron_locations = np.array([[0.0, 0.0]], dtype=np.float64)
    w.export_locations_btn.setEnabled(True)
    w.reset_detection_state()
    assert w.neuron_locations is None
    assert not w.export_locations_btn.isEnabled()


def test_clear_results_resets_masks(qapp: QApplication) -> None:
    w = NeuronDetectionWidget()
    w.set_roi_mask("roi_1", np.ones((3, 3), dtype=bool))
    w.neuron_locations = np.array([[0.0, 0.0]], dtype=np.float64)
    w.clear_results()
    assert w.roi_masks["roi_1"] is None
    assert w.neuron_locations is None


def test_update_ui_state_no_stack_message(qapp: QApplication) -> None:
    w = NeuronDetectionWidget()
    w.set_roi_mask("roi_1", np.ones((3, 3), dtype=bool))
    w.set_image_processor(MagicMock())
    w._update_ui_state()
    assert "No image stack" in w.status_label.text()


def test_run_detection_warns_when_not_ready(qapp: QApplication) -> None:
    w = NeuronDetectionWidget()
    with patch.object(QMessageBox, "warning") as warn:
        w._run_detection()
    warn.assert_called_once()


def test_export_locations_warns_when_empty(qapp: QApplication) -> None:
    w = NeuronDetectionWidget()
    with patch.object(QMessageBox, "warning") as warn:
        w._export_locations()
    warn.assert_called_once()


def test_export_locations_writes_csv(qapp: QApplication, tmp_path) -> None:
    w = NeuronDetectionWidget()
    w.experiment = MagicMock()
    w.experiment.name = "ExpA"
    w.neuron_locations = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    w.quality_mask = np.array([True, False], dtype=bool)
    target = tmp_path / "out.csv"
    with (
        patch("ui.neuron_detection_widget.QFileDialog.getSaveFileName", return_value=(str(target), "")),
        patch.object(QMessageBox, "information"),
    ):
        w._export_locations()
    text = target.read_text(encoding="utf-8")
    assert "Y,X,Quality" in text.replace(" ", "")
    assert "1,2,Good" in text


def test_export_trajectories_writes_npy(qapp: QApplication, tmp_path) -> None:
    w = NeuronDetectionWidget()
    w.experiment = MagicMock()
    w.experiment.name = "ExpA"
    arr = np.arange(6, dtype=np.float64).reshape(2, 3)
    w.neuron_trajectories = arr
    target = tmp_path / "t.npy"
    with (
        patch("ui.neuron_detection_widget.QFileDialog.getSaveFileName", return_value=(str(target), "")),
        patch.object(QMessageBox, "information"),
    ):
        w._export_trajectories()
    assert np.array_equal(np.load(str(target)), arr)


def test_export_all_writes_csv(qapp: QApplication, tmp_path) -> None:
    w = NeuronDetectionWidget()
    w.experiment = MagicMock()
    w.experiment.name = "ExpA"
    w.neuron_locations = np.array([[1.0, 2.0]], dtype=np.float64)
    w.quality_mask = np.array([True], dtype=bool)
    w.neuron_trajectories = np.array([[0.1, 0.2, 0.3]], dtype=np.float64)
    target = tmp_path / "all.csv"
    with (
        patch("ui.neuron_detection_widget.QFileDialog.getSaveFileName", return_value=(str(target), "")),
        patch.object(QMessageBox, "information"),
    ):
        w._export_all()
    lines = target.read_text(encoding="utf-8").strip().splitlines()
    assert "Frame_0" in lines[0]
    assert "1,2,Good" in lines[1]
