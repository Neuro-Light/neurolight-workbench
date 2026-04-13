"""Tests for MainWindow time-setting dialogs."""

import matplotlib

matplotlib.use("Agg")

from unittest.mock import patch

import pytest
from PySide6.QtCore import QTime
from PySide6.QtWidgets import QApplication

from core.experiment_manager import Experiment
from ui.main_window import _ConfirmStartTimeDialog, _ExperimentSettingsDialog


@pytest.fixture
def app():
    if not QApplication.instance():
        return QApplication([])
    return QApplication.instance()


def _make_experiment() -> Experiment:
    exp = Experiment(name="Exp A", principal_investigator="PI", description="desc")
    exp.settings = {
        "acquisition": {
            "frame_interval_minutes": 4.5,
            "experiment_start_time": "10:11:12",
        }
    }
    return exp


def test_experiment_settings_dialog_initializes_acquisition_fields(app):
    dlg = _ExperimentSettingsDialog(_make_experiment())

    assert dlg.frame_interval_spin.value() == pytest.approx(4.5)
    assert dlg.start_time_edit.time().toString("HH:mm:ss") == "10:11:12"


def test_experiment_settings_dialog_accept_captures_time_fields(app):
    dlg = _ExperimentSettingsDialog(_make_experiment())
    dlg.name_edit.setText("Updated")
    dlg.frame_interval_spin.setValue(6.25)
    dlg.start_time_edit.setTime(QTime(9, 30, 45))

    dlg._accept_dialog()

    assert dlg.name == "Updated"
    assert dlg.frame_interval_minutes == pytest.approx(6.25)
    assert dlg.experiment_start_time == "09:30:45"


def test_experiment_settings_dialog_rejects_empty_name(app):
    dlg = _ExperimentSettingsDialog(_make_experiment())
    dlg.name_edit.setText("   ")
    with patch("ui.main_window.QMessageBox.warning") as warn:
        dlg._accept_dialog()
    warn.assert_called_once()
    assert dlg.result() == 0  # Not accepted


def test_confirm_start_time_dialog_returns_selected_time(app):
    dlg = _ConfirmStartTimeDialog(
        suggested_time=QTime(7, 15, 0),
        metadata_source="frame0001.tif",
        timestamp_uniformity_note="Timestamps appear constant.",
    )
    dlg.start_time_edit.setTime(QTime(8, 45, 30))

    assert dlg.selected_start_time() == "08:45:30"
