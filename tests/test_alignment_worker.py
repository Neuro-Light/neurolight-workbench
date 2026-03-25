"""Lightweight tests for ``AlignmentWorker`` cancellation behaviour (no full alignment run)."""

from __future__ import annotations

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
