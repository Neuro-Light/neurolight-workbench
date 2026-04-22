"""Tests for ``AlignmentWorker`` — cancellation, sequential helpers, and full pipeline."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from ui.alignment_worker import AlignmentWorker


@pytest.fixture
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        return QApplication([])
    return app


# ---------------------------------------------------------------------------
# Cancellation API
# ---------------------------------------------------------------------------


def test_request_cancel_sets_flag(qapp: QApplication) -> None:
    w = AlignmentWorker(np.zeros((2, 4, 4), dtype=np.uint16))
    assert w._cancel_requested is False
    w.request_cancel()
    assert w._cancel_requested is True


def test_check_cancel_emits_signals_when_requested(qapp: QApplication) -> None:
    w = AlignmentWorker(np.zeros((2, 4, 4), dtype=np.uint16))
    w._cancel_requested = True
    progress_args: list[tuple[int, int, str]] = []
    cancelled_hits: list[bool] = []

    w.progress.connect(lambda a, b, msg: progress_args.append((a, b, msg)))
    w.cancelled.connect(lambda: cancelled_hits.append(True))

    assert w._check_cancel("unit_test_stage") is True
    assert cancelled_hits == [True]
    assert progress_args
    assert "unit_test_stage" in progress_args[0][2]


def test_check_cancel_returns_false_when_not_requested(qapp: QApplication) -> None:
    w = AlignmentWorker(np.zeros((2, 4, 4), dtype=np.uint16))
    cancelled_hits: list[bool] = []
    w.cancelled.connect(lambda: cancelled_hits.append(True))

    assert w._check_cancel("some_stage") is False
    assert cancelled_hits == []


# ---------------------------------------------------------------------------
# _register_sequential
# ---------------------------------------------------------------------------


def test_register_sequential_single_frame_returns_identity(qapp: QApplication) -> None:
    """A 1-frame stack has no frames to register; frame 0 should get the identity tmat."""
    w = AlignmentWorker(np.zeros((1, 4, 4), dtype=np.uint16), reference="first")
    sr = MagicMock()
    sr.register.return_value = np.eye(3, dtype=np.float64)
    stack = np.zeros((1, 4, 4), dtype=np.uint16)

    result = w._register_sequential(sr, stack, None, 1)

    assert result is not None
    assert result.shape == (1, 3, 3)
    np.testing.assert_array_almost_equal(result[0], np.eye(3))


def test_register_sequential_multiple_frames_first_reference(qapp: QApplication) -> None:
    w = AlignmentWorker(np.zeros((3, 4, 4), dtype=np.uint16), reference="first")
    sr = MagicMock()
    sr.register.return_value = np.eye(3, dtype=np.float64)
    stack = np.zeros((3, 4, 4), dtype=np.uint16)

    result = w._register_sequential(sr, stack, None, 3)

    assert result is not None
    assert result.shape == (3, 3, 3)
    # Frame 0 is always identity
    np.testing.assert_array_almost_equal(result[0], np.eye(3))


def test_register_sequential_mean_reference(qapp: QApplication) -> None:
    """'mean' reference should compute and use the mean frame."""
    w = AlignmentWorker(np.zeros((3, 4, 4), dtype=np.uint16), reference="mean")
    sr = MagicMock()
    sr.register.return_value = np.eye(3, dtype=np.float64)
    stack = np.zeros((3, 4, 4), dtype=np.uint16)

    result = w._register_sequential(sr, stack, None, 3)

    assert result is not None
    assert result.shape == (3, 3, 3)
    # register() must have been called at least once
    assert sr.register.call_count >= 1


def test_register_sequential_cancel_mid_loop_returns_none(qapp: QApplication) -> None:
    """Cancellation during the registration loop should return None and emit cancelled."""
    w = AlignmentWorker(np.zeros((5, 4, 4), dtype=np.uint16), reference="first")
    sr = MagicMock()
    sr.register.return_value = np.eye(3, dtype=np.float64)
    stack = np.zeros((5, 4, 4), dtype=np.uint16)

    w._cancel_requested = True
    cancelled_hits: list[bool] = []
    w.cancelled.connect(lambda: cancelled_hits.append(True))

    result = w._register_sequential(sr, stack, None, 5)

    assert result is None
    assert cancelled_hits  # cancellation signal was emitted


# ---------------------------------------------------------------------------
# _transform_sequential
# ---------------------------------------------------------------------------


def test_transform_sequential_returns_uint16_stack(qapp: QApplication) -> None:
    w = AlignmentWorker(np.zeros((3, 4, 4), dtype=np.uint16))
    sr = MagicMock()
    sr.transform.return_value = np.ones((4, 4), dtype=np.float64) * 1000.0
    stack = np.zeros((3, 4, 4), dtype=np.uint16)
    tmats = np.tile(np.eye(3), (3, 1, 1))

    result = w._transform_sequential(sr, stack, tmats, None, 3)

    assert result is not None
    assert result.shape == (3, 4, 4)
    assert result.dtype == np.uint16


def test_transform_sequential_clips_negative_values(qapp: QApplication) -> None:
    w = AlignmentWorker(np.zeros((2, 4, 4), dtype=np.uint16))
    sr = MagicMock()
    sr.transform.return_value = np.full((4, 4), -500.0, dtype=np.float64)
    stack = np.zeros((2, 4, 4), dtype=np.uint16)
    tmats = np.tile(np.eye(3), (2, 1, 1))

    result = w._transform_sequential(sr, stack, tmats, None, 2)

    assert result is not None
    assert result.min() == 0


def test_transform_sequential_clips_overflow_values(qapp: QApplication) -> None:
    w = AlignmentWorker(np.zeros((2, 4, 4), dtype=np.uint16))
    sr = MagicMock()
    sr.transform.return_value = np.full((4, 4), 99999.0, dtype=np.float64)
    stack = np.zeros((2, 4, 4), dtype=np.uint16)
    tmats = np.tile(np.eye(3), (2, 1, 1))

    result = w._transform_sequential(sr, stack, tmats, None, 2)

    assert result is not None
    assert result.max() == 65535


def test_transform_sequential_cancel_returns_none(qapp: QApplication) -> None:
    w = AlignmentWorker(np.zeros((3, 4, 4), dtype=np.uint16))
    sr = MagicMock()
    sr.transform.return_value = np.ones((4, 4), dtype=np.float64)
    stack = np.zeros((3, 4, 4), dtype=np.uint16)
    tmats = np.tile(np.eye(3), (3, 1, 1))

    w._cancel_requested = True
    cancelled_hits: list[bool] = []
    w.cancelled.connect(lambda: cancelled_hits.append(True))

    result = w._transform_sequential(sr, stack, tmats, None, 3)

    assert result is None
    assert cancelled_hits


# ---------------------------------------------------------------------------
# run() — full pipeline
# ---------------------------------------------------------------------------


@patch("pystackreg.StackReg")
def test_run_full_pipeline_emits_finished(mock_sr_cls: MagicMock, qapp: QApplication) -> None:
    """Happy-path run() with mocked StackReg should emit finished with correct shapes."""
    mock_sr = MagicMock()
    mock_sr.register.return_value = np.eye(3, dtype=np.float64)
    mock_sr.transform.return_value = np.zeros((4, 4), dtype=np.float64)
    mock_sr_cls.return_value = mock_sr

    stack = np.ones((2, 4, 4), dtype=np.uint16) * 100
    w = AlignmentWorker(stack, transform_type="rigid_body", reference="first", enable_multiprocessing=False)

    finished_args: list[tuple] = []
    error_args: list[str] = []
    w.finished.connect(lambda a, b, c: finished_args.append((a, b, c)))
    w.error.connect(lambda e: error_args.append(e))

    w.run()

    assert not error_args, f"Worker emitted error: {error_args}"
    assert len(finished_args) == 1
    aligned_stack, tmats, confidence_scores = finished_args[0]
    assert aligned_stack.shape == (2, 4, 4)
    assert len(confidence_scores) == 2


@patch("pystackreg.StackReg")
def test_run_confidence_scores_in_range(mock_sr_cls: MagicMock, qapp: QApplication) -> None:
    """Confidence scores must be in [0, 1]."""
    mock_sr = MagicMock()
    mock_sr.register.return_value = np.eye(3, dtype=np.float64)
    mock_sr.transform.return_value = np.zeros((4, 4), dtype=np.float64)
    mock_sr_cls.return_value = mock_sr

    stack = np.ones((3, 4, 4), dtype=np.uint16) * 200
    w = AlignmentWorker(stack, reference="first", enable_multiprocessing=False)

    finished_args: list[tuple] = []
    w.finished.connect(lambda a, b, c: finished_args.append((a, b, c)))
    w.run()

    assert finished_args
    _, _, confidence_scores = finished_args[0]
    assert len(confidence_scores) == 3
    for score in confidence_scores:
        assert 0.0 <= score <= 1.0


@patch("pystackreg.StackReg")
def test_run_previous_reference_strategy(mock_sr_cls: MagicMock, qapp: QApplication) -> None:
    """'previous' reference uses register_stack; pipeline should still finish."""
    mock_sr = MagicMock()
    mock_sr.register_stack.return_value = np.tile(np.eye(3), (3, 1, 1))
    mock_sr.transform.return_value = np.zeros((4, 4), dtype=np.float64)
    mock_sr_cls.return_value = mock_sr

    stack = np.ones((3, 4, 4), dtype=np.uint16) * 50
    w = AlignmentWorker(stack, reference="previous", enable_multiprocessing=False)

    finished_args: list[tuple] = []
    error_args: list[str] = []
    w.finished.connect(lambda a, b, c: finished_args.append((a, b, c)))
    w.error.connect(lambda e: error_args.append(e))

    w.run()

    assert not error_args, f"Worker emitted error: {error_args}"
    assert len(finished_args) == 1


@patch("pystackreg.StackReg")
def test_run_uniform_stack_skips_normalization(mock_sr_cls: MagicMock, qapp: QApplication) -> None:
    """A uniform stack (global_range == 0) uses copy path in de-normalization."""
    mock_sr = MagicMock()
    mock_sr.register.return_value = np.eye(3, dtype=np.float64)
    mock_sr.transform.return_value = np.zeros((4, 4), dtype=np.float64)
    mock_sr_cls.return_value = mock_sr

    # All zeros → global_range == 0
    stack = np.zeros((2, 4, 4), dtype=np.uint16)
    w = AlignmentWorker(stack, reference="first", enable_multiprocessing=False)

    finished_args: list[tuple] = []
    error_args: list[str] = []
    w.finished.connect(lambda a, b, c: finished_args.append((a, b, c)))
    w.error.connect(lambda e: error_args.append(e))

    w.run()

    assert not error_args, f"Worker emitted error: {error_args}"
    assert len(finished_args) == 1
    aligned_stack, _, _ = finished_args[0]
    assert aligned_stack.shape == stack.shape


@patch("pystackreg.StackReg")
def test_run_cancel_at_initialization_does_not_emit_finished(mock_sr_cls: MagicMock, qapp: QApplication) -> None:
    """Cancellation before normalization should stop the pipeline without emitting finished."""
    stack = np.ones((2, 4, 4), dtype=np.uint16)
    w = AlignmentWorker(stack, reference="first", enable_multiprocessing=False)
    w._cancel_requested = True

    cancelled_hits: list[bool] = []
    finished_hits: list[bool] = []
    w.cancelled.connect(lambda: cancelled_hits.append(True))
    w.finished.connect(lambda a, b, c: finished_hits.append(True))

    w.run()

    assert cancelled_hits
    assert not finished_hits


def test_run_emits_error_on_exception(qapp: QApplication) -> None:
    """If StackReg raises, the error signal should be emitted with the message."""
    stack = np.ones((2, 4, 4), dtype=np.uint16)
    w = AlignmentWorker(stack)

    error_args: list[str] = []
    finished_hits: list[bool] = []
    w.error.connect(lambda e: error_args.append(e))
    w.finished.connect(lambda a, b, c: finished_hits.append(True))

    with patch("pystackreg.StackReg", side_effect=RuntimeError("mocked pystackreg failure")):
        w.run()

    assert len(error_args) == 1
    assert "mocked pystackreg failure" in error_args[0]
    assert not finished_hits


@patch("pystackreg.StackReg")
def test_run_emits_progress_signals(mock_sr_cls: MagicMock, qapp: QApplication) -> None:
    """run() should emit at least one progress signal during alignment."""
    mock_sr = MagicMock()
    mock_sr.register.return_value = np.eye(3, dtype=np.float64)
    mock_sr.transform.return_value = np.zeros((4, 4), dtype=np.float64)
    mock_sr_cls.return_value = mock_sr

    stack = np.ones((2, 4, 4), dtype=np.uint16) * 100
    w = AlignmentWorker(stack, reference="first", enable_multiprocessing=False)

    progress_hits: list[tuple] = []
    w.progress.connect(lambda a, b, msg: progress_hits.append((a, b, msg)))

    w.run()

    assert progress_hits  # At least one progress signal emitted
