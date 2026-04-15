"""Tests for DetectionProgressDialog — initialization, progress updates, and cancel."""

from __future__ import annotations

import pytest
from PySide6.QtWidgets import QApplication

from ui.detection_progress_dialog import DetectionProgressDialog


@pytest.fixture
def app():
    if not QApplication.instance():
        return QApplication([])
    return QApplication.instance()


def test_dialog_title(app) -> None:
    dlg = DetectionProgressDialog(total_steps=5)
    assert dlg.windowTitle() == "Detecting Neurons..."


def test_dialog_is_modal(app) -> None:
    dlg = DetectionProgressDialog(total_steps=3)
    assert dlg.isModal() is True


def test_dialog_minimum_size(app) -> None:
    dlg = DetectionProgressDialog()
    assert dlg.minimumWidth() >= 500
    assert dlg.minimumHeight() >= 220


def test_progress_bar_range(app) -> None:
    dlg = DetectionProgressDialog(total_steps=8)
    assert dlg.progress_bar.minimum() == 0
    assert dlg.progress_bar.maximum() == 8


def test_progress_bar_initial_value(app) -> None:
    dlg = DetectionProgressDialog(total_steps=5)
    assert dlg.progress_bar.value() == 0


def test_initial_status_label(app) -> None:
    dlg = DetectionProgressDialog()
    assert "Initializing" in dlg.status_label.text()


def test_log_text_is_readonly(app) -> None:
    dlg = DetectionProgressDialog()
    assert dlg.log_text.isReadOnly() is True


def test_update_progress_sets_bar_and_label(app) -> None:
    dlg = DetectionProgressDialog(total_steps=10)
    dlg.update_progress(3, 10, "Detecting frame 3/10")
    assert dlg.progress_bar.value() == 3
    assert dlg.status_label.text() == "Detecting frame 3/10"


def test_update_progress_clamps_to_max(app) -> None:
    dlg = DetectionProgressDialog(total_steps=5)
    dlg.update_progress(99, 5, "overflow")
    assert dlg.progress_bar.value() == 5


def test_update_progress_appends_to_log(app) -> None:
    dlg = DetectionProgressDialog(total_steps=5)
    dlg.update_progress(1, 5, "Step 1")
    dlg.update_progress(2, 5, "Step 2")
    log = dlg.log_text.toPlainText()
    assert "Step 1" in log
    assert "Step 2" in log


def test_update_progress_adjusts_max_when_total_changes(app) -> None:
    dlg = DetectionProgressDialog(total_steps=5)
    dlg.update_progress(1, 20, "reconfig")
    assert dlg.progress_bar.maximum() == 20


def test_cancel_button_rejects(app) -> None:
    dlg = DetectionProgressDialog(total_steps=5)
    rejected = []
    dlg.rejected.connect(lambda: rejected.append(True))
    dlg.show()
    dlg.cancel_button.click()
    assert rejected == [True]


def test_total_steps_zero_clamps_to_one(app) -> None:
    dlg = DetectionProgressDialog(total_steps=0)
    assert dlg.progress_bar.maximum() >= 1
