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


@pytest.fixture
def experiments_dir(tmp_path):
    """Per-user experiments directory, mirroring the real users/<name>/experiments/ layout."""
    d = tmp_path / "testuser" / "experiments"
    d.mkdir(parents=True)
    return d


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
    def test_dialog_title(self, app, experiments_dir) -> None:
        dlg = NewExperimentDialog(experiments_dir)
        assert "New Experiment" in dlg.windowTitle()

    def test_dialog_is_modal(self, app, experiments_dir) -> None:
        dlg = NewExperimentDialog(experiments_dir)
        assert dlg.isModal() is True

    def test_default_date_is_today(self, app, experiments_dir) -> None:
        expected = datetime.utcnow().strftime("%Y-%m-%d")
        dlg = NewExperimentDialog(experiments_dir)
        assert expected in dlg.date_edit.text()

    def test_default_frame_interval(self, app, experiments_dir) -> None:
        dlg = NewExperimentDialog(experiments_dir)
        assert dlg.frame_interval_spin.value() == pytest.approx(30.0)

    def test_analysis_combo_has_scn(self, app, experiments_dir) -> None:
        dlg = NewExperimentDialog(experiments_dir)
        assert dlg.analysis_combo.currentData() == "SCN"

    def test_accept_requires_name(self, app, experiments_dir) -> None:
        dlg = NewExperimentDialog(experiments_dir)
        dlg.name_edit.setText("")
        dlg._accept()
        assert dlg.output_path is None

    def test_accept_with_name_creates_output_path(self, app, experiments_dir) -> None:
        dlg = NewExperimentDialog(experiments_dir)
        dlg.name_edit.setText("TestExperiment")
        accepted = []
        dlg.accepted.connect(lambda: accepted.append(True))
        dlg._accept()
        assert dlg.output_path is not None
        assert dlg.output_path.endswith(".nexp")
        assert dlg.metadata["name"] == "TestExperiment"
        assert dlg.metadata["analysis_type"] == "SCN"
        assert "frame_interval_minutes" in dlg.metadata

    def test_accept_rejects_duplicate_name(self, app, experiments_dir) -> None:
        # Simulate an already-existing experiment folder+file
        exp_dir = experiments_dir / "DuplicateExp"
        exp_dir.mkdir(parents=True)
        (exp_dir / "DuplicateExp.nexp").touch()
        dlg = NewExperimentDialog(experiments_dir)
        dlg.name_edit.setText("DuplicateExp")
        dlg._accept()
        assert dlg.output_path is None

    def test_path_display_is_readonly(self, app, experiments_dir) -> None:
        dlg = NewExperimentDialog(experiments_dir)
        assert dlg._path_display.isReadOnly() is True

    def test_path_display_shows_experiments_dir(self, app, experiments_dir) -> None:
        dlg = NewExperimentDialog(experiments_dir)
        assert str(experiments_dir.resolve()) in dlg._path_display.text()


# ── StartupDialog ────────────────────────────────────────────────────────


class TestStartupDialog:
    def test_dialog_title(self, app, experiments_dir) -> None:
        dlg = StartupDialog(experiments_dir)
        assert "Experiment Manager" in dlg.windowTitle()

    def test_dialog_is_modal(self, app, experiments_dir) -> None:
        dlg = StartupDialog(experiments_dir)
        assert dlg.isModal() is True

    def test_dialog_minimum_width(self, app, experiments_dir) -> None:
        dlg = StartupDialog(experiments_dir)
        assert dlg.minimumWidth() >= 520

    def test_initial_experiment_is_none(self, app, experiments_dir) -> None:
        dlg = StartupDialog(experiments_dir)
        assert dlg.experiment is None
        assert dlg.experiment_path is None

    def test_recent_list_exists(self, app, experiments_dir) -> None:
        dlg = StartupDialog(experiments_dir)
        assert dlg.recent_list is not None

    def test_settings_button_exists(self, app, experiments_dir) -> None:
        dlg = StartupDialog(experiments_dir)
        assert dlg.settings_btn is not None

    def test_mp_checkbox_exists(self, app, experiments_dir) -> None:
        dlg = StartupDialog(experiments_dir)
        assert dlg.enable_mp_checkbox is not None

    def test_load_existing_noop_on_cancel(self, app, experiments_dir) -> None:
        dlg = StartupDialog(experiments_dir)
        with patch("ui.startup_dialog.QFileDialog.getOpenFileName", return_value=("", "")):
            dlg._load_existing()
        assert dlg.experiment is None

    def test_load_existing_bad_file_shows_warning(self, app, experiments_dir, tmp_path) -> None:
        bad_file = tmp_path / "bad.nexp"
        bad_file.write_text("not json")
        dlg = StartupDialog(experiments_dir)
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

    def test_start_new_noop_on_cancel(self, app, experiments_dir) -> None:
        dlg = StartupDialog(experiments_dir)
        with patch.object(NewExperimentDialog, "exec", return_value=QDialog.Rejected):
            dlg._start_new()
        assert dlg.experiment is None

    def test_show_file_location_warns_when_missing(self, app, experiments_dir) -> None:
        dlg = StartupDialog(experiments_dir)
        with patch("ui.startup_dialog.QMessageBox.warning") as mock_warn:
            dlg._show_file_location("/nonexistent/path.nexp")
        mock_warn.assert_called_once()

    def test_open_settings_opens_dialog(self, app, experiments_dir) -> None:
        dlg = StartupDialog(experiments_dir)
        with patch("ui.startup_dialog.SettingsDialog") as MockSettings:
            MockSettings.return_value.exec.return_value = QDialog.Accepted
            dlg._open_settings()
        MockSettings.assert_called_once()

    def test_mp_toggle_persists(self, app, experiments_dir) -> None:
        dlg = StartupDialog(experiments_dir)
        with patch("ui.startup_dialog.set_enable_alignment_multiprocessing") as mock_set:
            dlg._on_alignment_mp_toggled(True)
        mock_set.assert_called_once_with(True)
