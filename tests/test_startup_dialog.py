"""Tests for StartupDialog and NewExperimentDialog."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import Mock, patch

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QDialog

from ui.startup_dialog import NewExperimentDialog, RecentExperimentRow, StartupDialog


@pytest.fixture
def app():
    if not QApplication.instance():
        return QApplication([])
    return QApplication.instance()


# ── RecentExperimentRow ──────────────────────────────────────────────────


class TestRecentExperimentRow:
    def test_row_displays_name(self, app) -> None:
        row = RecentExperimentRow(name="Exp 1", path="/tmp/e.nexp", on_open=Mock())
        assert row.name_label.text() == "Exp 1"

    def test_row_has_options_button(self, app) -> None:
        row = RecentExperimentRow(name="Exp 1", path="/tmp/e.nexp", on_open=Mock())
        assert row.options_btn is not None
        assert row.options_btn.text() == "..."

    def test_double_click_calls_on_open(self, app) -> None:
        on_open = Mock()
        row = RecentExperimentRow(name="Exp 1", path="/tmp/e.nexp", on_open=on_open)
        from PySide6.QtCore import QPointF
        from PySide6.QtGui import QMouseEvent

        ev = QMouseEvent(
            QMouseEvent.Type.MouseButtonDblClick,
            QPointF(5, 5),
            QPointF(5, 5),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        row.mouseDoubleClickEvent(ev)
        on_open.assert_called_once()


# ── NewExperimentDialog ─────────────────────────────────────────────────


class TestNewExperimentDialog:
    def test_dialog_title(self, app) -> None:
        dlg = NewExperimentDialog()
        assert "New Experiment" in dlg.windowTitle()

    def test_dialog_is_modal(self, app) -> None:
        dlg = NewExperimentDialog()
        assert dlg.isModal() is True

    def test_default_date_is_today(self, app) -> None:
        dlg = NewExperimentDialog()
        assert datetime.utcnow().strftime("%Y-%m-%d") in dlg.date_edit.text()

    def test_default_frame_interval(self, app) -> None:
        dlg = NewExperimentDialog()
        assert dlg.frame_interval_spin.value() == pytest.approx(30.0)

    def test_analysis_combo_has_scn(self, app) -> None:
        dlg = NewExperimentDialog()
        assert dlg.analysis_combo.currentData() == "SCN"

    def test_accept_requires_name(self, app) -> None:
        dlg = NewExperimentDialog()
        dlg.name_edit.setText("")
        dlg._accept()
        assert dlg.output_path is None

    def test_accept_with_name_creates_output_path(self, app, tmp_path) -> None:
        dlg = NewExperimentDialog()
        dlg.name_edit.setText("TestExperiment")
        dlg.path_edit.setText(str(tmp_path))
        accepted = []
        dlg.accepted.connect(lambda: accepted.append(True))
        dlg._accept()
        assert dlg.output_path is not None
        assert dlg.output_path.endswith(".nexp")
        assert dlg.metadata["name"] == "TestExperiment"
        assert dlg.metadata["analysis_type"] == "SCN"

    def test_accept_rejects_duplicate_name(self, app, tmp_path) -> None:
        (tmp_path / "DuplicateExp.nexp").touch()
        dlg = NewExperimentDialog()
        dlg.name_edit.setText("DuplicateExp")
        dlg.path_edit.setText(str(tmp_path))
        dlg._accept()
        assert dlg.output_path is None

    def test_browse_sets_path(self, app) -> None:
        dlg = NewExperimentDialog()
        with patch("ui.startup_dialog.QFileDialog.getExistingDirectory", return_value="/new/path"):
            dlg._browse()
        assert dlg.path_edit.text() == "/new/path"

    def test_browse_noop_on_cancel(self, app) -> None:
        dlg = NewExperimentDialog()
        original = dlg.path_edit.text()
        with patch("ui.startup_dialog.QFileDialog.getExistingDirectory", return_value=""):
            dlg._browse()
        assert dlg.path_edit.text() == original


# ── StartupDialog ────────────────────────────────────────────────────────


class TestStartupDialog:
    def test_dialog_title(self, app) -> None:
        dlg = StartupDialog()
        assert "Experiment Manager" in dlg.windowTitle()

    def test_dialog_is_modal(self, app) -> None:
        dlg = StartupDialog()
        assert dlg.isModal() is True

    def test_dialog_minimum_width(self, app) -> None:
        dlg = StartupDialog()
        assert dlg.minimumWidth() >= 520

    def test_initial_experiment_is_none(self, app) -> None:
        dlg = StartupDialog()
        assert dlg.experiment is None
        assert dlg.experiment_path is None

    def test_recent_list_exists(self, app) -> None:
        dlg = StartupDialog()
        assert dlg.recent_list is not None

    def test_settings_button_exists(self, app) -> None:
        dlg = StartupDialog()
        assert dlg.settings_btn is not None

    def test_mp_checkbox_exists(self, app) -> None:
        dlg = StartupDialog()
        assert dlg.enable_mp_checkbox is not None

    def test_load_existing_noop_on_cancel(self, app) -> None:
        dlg = StartupDialog()
        with patch("ui.startup_dialog.QFileDialog.getOpenFileName", return_value=("", "")):
            dlg._load_existing()
        assert dlg.experiment is None

    def test_load_existing_bad_file_shows_warning(self, app, tmp_path) -> None:
        bad_file = tmp_path / "bad.nexp"
        bad_file.write_text("not json")
        dlg = StartupDialog()
        with (
            patch(
                "ui.startup_dialog.QFileDialog.getOpenFileName",
                return_value=(str(bad_file), ""),
            ),
            patch("ui.startup_dialog.QMessageBox.warning") as mock_warn,
        ):
            dlg._load_existing()
        mock_warn.assert_called_once()
        assert dlg.experiment is None

    def test_start_new_noop_on_cancel(self, app) -> None:
        dlg = StartupDialog()
        with patch.object(NewExperimentDialog, "exec", return_value=QDialog.Rejected):
            dlg._start_new()
        assert dlg.experiment is None

    def test_show_file_location_warns_when_missing(self, app) -> None:
        dlg = StartupDialog()
        with patch("ui.startup_dialog.QMessageBox.warning") as mock_warn:
            dlg._show_file_location("/nonexistent/path.nexp")
        mock_warn.assert_called_once()

    def test_open_settings_opens_dialog(self, app) -> None:
        dlg = StartupDialog()
        with patch("ui.startup_dialog.SettingsDialog") as MockSettings:
            MockSettings.return_value.exec.return_value = QDialog.Accepted
            dlg._open_settings()
        MockSettings.assert_called_once()

    def test_mp_toggle_persists(self, app) -> None:
        dlg = StartupDialog()
        with patch("ui.startup_dialog.set_enable_alignment_multiprocessing") as mock_set:
            dlg._on_alignment_mp_toggled(True)
        mock_set.assert_called_once_with(True)
