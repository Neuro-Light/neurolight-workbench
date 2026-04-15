"""Tests for DetectionWorker — signal emissions and error handling."""

from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from ui.detection_worker import DetectionWorker


@pytest.fixture
def app():
    if not QApplication.instance():
        return QApplication([])
    return QApplication.instance()


def _make_worker(processor_result=None, processor_error=None):
    mock_processor = Mock()
    if processor_error:
        mock_processor.detect_neurons_in_roi.side_effect = processor_error
    else:
        locations = np.array([[10, 20], [30, 40]])
        trajectories = np.random.rand(2, 5).astype(np.float32)
        quality = np.array([True, False])
        mock_processor.detect_neurons_in_roi.return_value = (
            processor_result or (locations, trajectories, quality)
        )

    frame_data = np.random.randint(0, 255, (5, 64, 64), dtype=np.uint8)
    roi_mask = np.ones((64, 64), dtype=np.uint8)
    params = {"cell_size": 6, "num_peaks": 100}

    return DetectionWorker(mock_processor, frame_data, roi_mask, params), mock_processor


def test_worker_emits_finished_on_success(app) -> None:
    worker, processor = _make_worker()
    results = []
    worker.finished.connect(lambda loc, traj, qm: results.append((loc, traj, qm)))
    worker.run()
    assert len(results) == 1
    loc, traj, qm = results[0]
    assert loc.shape == (2, 2)
    assert qm.shape == (2,)


def test_worker_emits_error_on_exception(app) -> None:
    worker, _ = _make_worker(processor_error=RuntimeError("detection failed"))
    errors = []
    worker.error.connect(lambda msg: errors.append(msg))
    worker.run()
    assert len(errors) == 1
    assert "detection failed" in errors[0]


def test_worker_emits_progress(app) -> None:
    mock_processor = Mock()

    def fake_detect(*args, **kwargs):
        cb = kwargs.get("progress_callback")
        if cb:
            cb(1, 5, "Step 1")
            cb(2, 5, "Step 2")
        return np.array([[0, 0]]), np.array([[1.0]]), np.array([True])

    mock_processor.detect_neurons_in_roi.side_effect = fake_detect

    frame_data = np.zeros((5, 8, 8), dtype=np.uint8)
    roi_mask = np.ones((8, 8), dtype=np.uint8)
    worker = DetectionWorker(mock_processor, frame_data, roi_mask, {})

    progress_msgs = []
    worker.progress.connect(lambda c, t, m: progress_msgs.append((c, t, m)))
    worker.run()
    assert (1, 5, "Step 1") in progress_msgs
    assert (2, 5, "Step 2") in progress_msgs


def test_worker_passes_params_to_processor(app) -> None:
    worker, processor = _make_worker()
    worker._params = {
        "cell_size": 12,
        "num_peaks": 500,
        "correlation_threshold": 0.6,
        "threshold_rel": 0.05,
        "apply_detrending": False,
        "use_max_projection": False,
        "preprocess_sigma": 2.0,
    }
    worker.run()
    call_kwargs = processor.detect_neurons_in_roi.call_args
    assert call_kwargs[1]["cell_size"] == 12
    assert call_kwargs[1]["num_peaks"] == 500
    assert call_kwargs[1]["correlation_threshold"] == 0.6
    assert call_kwargs[1]["apply_detrending"] is False
    assert call_kwargs[1]["preprocess_sigma"] == 2.0
