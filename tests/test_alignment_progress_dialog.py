"""Tests for AlignmentProgressDialog — initial state, progress updates, and cancellation."""

from __future__ import annotations

import pytest
from PySide6.QtWidgets import QApplication

from ui.alignment_progress_dialog import AlignmentProgressDialog


@pytest.fixture
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        return QApplication([])
    return app


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------


def test_initial_cancelled_flag_is_false(qapp: QApplication) -> None:
    dialog = AlignmentProgressDialog(total_frames=10)
    assert dialog._cancelled is False


def test_is_cancelled_initially_false(qapp: QApplication) -> None:
    dialog = AlignmentProgressDialog(total_frames=10)
    assert dialog.is_cancelled() is False


def test_progress_bar_initial_value(qapp: QApplication) -> None:
    dialog = AlignmentProgressDialog(total_frames=10)
    assert dialog.progress_bar.value() == 0


def test_progress_bar_initial_maximum(qapp: QApplication) -> None:
    dialog = AlignmentProgressDialog(total_frames=10)
    assert dialog.progress_bar.maximum() == 10


def test_progress_bar_initial_minimum(qapp: QApplication) -> None:
    dialog = AlignmentProgressDialog(total_frames=10)
    assert dialog.progress_bar.minimum() == 0


def test_status_label_initial_text(qapp: QApplication) -> None:
    dialog = AlignmentProgressDialog(total_frames=5)
    assert dialog.status_label.text() == "Initializing alignment..."


def test_log_text_initially_empty(qapp: QApplication) -> None:
    dialog = AlignmentProgressDialog(total_frames=5)
    assert dialog.log_text.toPlainText() == ""


# ---------------------------------------------------------------------------
# update_progress
# ---------------------------------------------------------------------------


def test_update_progress_sets_bar_value(qapp: QApplication) -> None:
    dialog = AlignmentProgressDialog(total_frames=10)
    dialog.update_progress(5, 10, "Halfway there")
    assert dialog.progress_bar.value() == 5


def test_update_progress_updates_maximum(qapp: QApplication) -> None:
    dialog = AlignmentProgressDialog(total_frames=10)
    dialog.update_progress(0, 20, "New total")
    assert dialog.progress_bar.maximum() == 20


def test_update_progress_updates_status_label(qapp: QApplication) -> None:
    dialog = AlignmentProgressDialog(total_frames=10)
    dialog.update_progress(3, 10, "Registering frame 3")
    assert dialog.status_label.text() == "Registering frame 3"


def test_update_progress_appends_to_log(qapp: QApplication) -> None:
    dialog = AlignmentProgressDialog(total_frames=10)
    dialog.update_progress(1, 10, "First message")
    assert "First message" in dialog.log_text.toPlainText()


def test_update_progress_accumulates_log_entries(qapp: QApplication) -> None:
    dialog = AlignmentProgressDialog(total_frames=10)
    dialog.update_progress(1, 10, "Alpha")
    dialog.update_progress(2, 10, "Beta")
    log = dialog.log_text.toPlainText()
    assert "Alpha" in log
    assert "Beta" in log


def test_update_progress_zero_completed(qapp: QApplication) -> None:
    dialog = AlignmentProgressDialog(total_frames=10)
    dialog.update_progress(0, 10, "Starting...")
    assert dialog.progress_bar.value() == 0
    assert dialog.status_label.text() == "Starting..."


def test_update_progress_complete(qapp: QApplication) -> None:
    dialog = AlignmentProgressDialog(total_frames=10)
    dialog.update_progress(10, 10, "Alignment complete!")
    assert dialog.progress_bar.value() == 10
    assert dialog.status_label.text() == "Alignment complete!"


# ---------------------------------------------------------------------------
# reject / cancellation
# ---------------------------------------------------------------------------


def test_reject_sets_cancelled_flag(qapp: QApplication) -> None:
    dialog = AlignmentProgressDialog(total_frames=5)
    dialog.reject()
    assert dialog._cancelled is True


def test_is_cancelled_true_after_reject(qapp: QApplication) -> None:
    dialog = AlignmentProgressDialog(total_frames=5)
    dialog.reject()
    assert dialog.is_cancelled() is True


def test_reject_updates_status_label(qapp: QApplication) -> None:
    dialog = AlignmentProgressDialog(total_frames=5)
    dialog.reject()
    assert dialog.status_label.text() == "Cancelling..."


def test_reject_appends_cancellation_message_to_log(qapp: QApplication) -> None:
    dialog = AlignmentProgressDialog(total_frames=5)
    dialog.reject()
    assert "Cancellation requested" in dialog.log_text.toPlainText()


def test_reject_emits_rejected_signal(qapp: QApplication) -> None:
    dialog = AlignmentProgressDialog(total_frames=5)
    rejected_hits: list[bool] = []
    dialog.rejected.connect(lambda: rejected_hits.append(True))
    dialog.reject()
    assert rejected_hits == [True]
