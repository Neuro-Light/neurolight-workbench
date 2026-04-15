"""Tests for LoadingDialog — initialization, status updates, and close."""

from __future__ import annotations

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from ui.loading_dialog import LoadingDialog


@pytest.fixture
def app():
    if not QApplication.instance():
        return QApplication([])
    return QApplication.instance()


def test_dialog_title(app) -> None:
    dlg = LoadingDialog()
    assert dlg.windowTitle() == "Loading Experiment..."


def test_dialog_is_non_modal(app) -> None:
    dlg = LoadingDialog()
    assert dlg.isModal() is False


def test_dialog_minimum_size(app) -> None:
    dlg = LoadingDialog()
    assert dlg.minimumWidth() >= 400
    assert dlg.minimumHeight() >= 150


def test_dialog_stays_on_top(app) -> None:
    dlg = LoadingDialog()
    assert dlg.windowFlags() & Qt.WindowStaysOnTopHint


def test_initial_labels(app) -> None:
    dlg = LoadingDialog()
    assert "Loading" in dlg.status_label.text()
    assert dlg.info_label.text() == "Please wait..."


def test_progress_bar_is_indeterminate(app) -> None:
    dlg = LoadingDialog()
    assert dlg.progress_bar.minimum() == 0
    assert dlg.progress_bar.maximum() == 0


def test_update_status_changes_labels(app) -> None:
    dlg = LoadingDialog()
    dlg.update_status("Processing...", "Step 2 of 5")
    assert dlg.status_label.text() == "Processing..."
    assert dlg.info_label.text() == "Step 2 of 5"


def test_update_status_without_info_keeps_existing(app) -> None:
    dlg = LoadingDialog()
    dlg.update_status("First", "details")
    dlg.update_status("Second")
    assert dlg.status_label.text() == "Second"
    assert dlg.info_label.text() == "details"


def test_close_dialog_accepts(app) -> None:
    dlg = LoadingDialog()
    accepted = []
    dlg.accepted.connect(lambda: accepted.append(True))
    dlg.show()
    dlg.close_dialog()
    assert accepted == [True]
