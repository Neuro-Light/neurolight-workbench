"""Tests for StartupDialog and NewExperimentDialog."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import Mock, patch

import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtWidgets import QApplication, QDialog, QMessageBox

from ui.startup_dialog import NewExperimentDialog, RecentExperimentRow, StartupDialog, _public_recent_row_label


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


class TestPublicRecentRowLabel:
    def test_uses_owner_field(self) -> None:
        assert _public_recent_row_label({"name": "My Study", "owner": "alice"}, "/x/y.nexp") == "My Study - by alice"

    def test_owner_field_preserves_exact_casing(self) -> None:
        assert (
            _public_recent_row_label({"name": "My Study", "owner": "SoMeUsEr"}, "/x/y.nexp") == "My Study - by SoMeUsEr"
        )

    def test_parses_owner_from_public_copy_filename(self) -> None:
        assert (
            _public_recent_row_label({"name": "My Study"}, r"C:\users\Public\experiments\bob__MyStudy.nexp")
            == "My Study - by bob"
        )

    def test_parses_owner_casing_from_filename(self) -> None:
        assert (
            _public_recent_row_label(
                {"name": "My Study"},
                r"C:\users\Public\experiments\MiXeD__MyStudy.nexp",
            )
            == "My Study - by MiXeD"
        )

    def test_fallback_without_owner(self) -> None:
        assert _public_recent_row_label({"name": "Solo"}, "/no/underscore.nexp") == "Solo"


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
        expected = datetime.now(timezone.utc).strftime("%Y-%m-%d")
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

    def test_open_user_account_popup_switches_workspace(self, app, experiments_dir, tmp_path) -> None:
        dlg = StartupDialog(experiments_dir)
        next_experiments_dir = tmp_path / "other-user" / "experiments"
        next_experiments_dir.mkdir(parents=True)

        popup = Mock()
        popup.exec.return_value = QDialog.Accepted
        popup.switch_user_requested = True

        picker = Mock()
        picker.exec.return_value = QDialog.Accepted
        picker.selected_user_experiments_dir = next_experiments_dir
        picker.selected_user = "other-user"

        with (
            patch("ui.startup_dialog.UserAccountActionsDialog", return_value=popup),
            patch("ui.startup_dialog.UserSelectionDialog", return_value=picker),
            patch.object(dlg, "_refresh_recent") as mock_refresh,
        ):
            dlg._open_user_account_popup()

        assert dlg.experiments_dir == next_experiments_dir
        assert dlg._current_user_name == "other-user"
        assert dlg._current_user_btn.text() == "Current User: other-user"
        mock_refresh.assert_called_once()

    def test_open_user_account_popup_no_switch_when_picker_cancelled(self, app, experiments_dir) -> None:
        dlg = StartupDialog(experiments_dir)
        popup = Mock()
        popup.exec.return_value = QDialog.Accepted
        popup.switch_user_requested = True
        picker = Mock()
        picker.exec.return_value = QDialog.Rejected
        picker.selected_user_experiments_dir = None

        with (
            patch("ui.startup_dialog.UserAccountActionsDialog", return_value=popup),
            patch("ui.startup_dialog.UserSelectionDialog", return_value=picker),
            patch.object(dlg, "_refresh_recent") as mock_refresh,
        ):
            dlg._open_user_account_popup()

        assert dlg.experiments_dir == experiments_dir
        mock_refresh.assert_not_called()

    def test_show_file_location_opens_parent_directory(self, app, experiments_dir, tmp_path) -> None:
        dlg = StartupDialog(experiments_dir)
        exp_file = tmp_path / "folder" / "exp.nexp"
        exp_file.parent.mkdir(parents=True)
        exp_file.write_text("{}", encoding="utf-8")

        with patch("ui.startup_dialog.QDesktopServices.openUrl") as mock_open:
            dlg._show_file_location(str(exp_file))

        mock_open.assert_called_once()

    def test_get_item_for_path_finds_matching_row(self, app, experiments_dir) -> None:
        dlg = StartupDialog(experiments_dir)
        item = dlg._get_item_for_path("/missing")
        assert item is None

    def test_select_item_by_path_sets_current_item(self, app, experiments_dir) -> None:
        dlg = StartupDialog(experiments_dir)
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QListWidgetItem

        item = QListWidgetItem()
        item.setData(Qt.UserRole, "/tmp/exp.nexp")
        dlg.recent_list.addItem(item)
        dlg._select_item_by_path("/tmp/exp.nexp")
        assert dlg.recent_list.currentItem() == item

    def test_open_recent_by_path_calls_open_recent(self, app, experiments_dir) -> None:
        dlg = StartupDialog(experiments_dir)
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QListWidgetItem

        item = QListWidgetItem()
        item.setData(Qt.UserRole, "/tmp/exp.nexp")
        dlg.recent_list.addItem(item)
        with patch.object(dlg, "_open_recent") as mock_open_recent:
            dlg._open_recent_by_path("/tmp/exp.nexp")
        mock_open_recent.assert_called_once_with(item)

    def test_remove_from_list_for_path_deletes_item(self, app, experiments_dir) -> None:
        dlg = StartupDialog(experiments_dir)
        with patch.object(dlg, "_delete_experiment") as mock_delete:
            with patch.object(dlg, "_get_item_for_path", return_value=Mock()):
                dlg._remove_from_list_for_path("/tmp/exp.nexp")
        mock_delete.assert_called_once()

    def test_export_for_path_calls_export(self, app, experiments_dir) -> None:
        dlg = StartupDialog(experiments_dir)
        with patch.object(dlg, "_export_experiment") as mock_export:
            with patch.object(dlg, "_get_item_for_path", return_value=Mock()):
                dlg._export_for_path("/tmp/exp.nexp")
        mock_export.assert_called_once()

    def test_open_recent_load_failure_shows_warning(self, app, experiments_dir) -> None:
        dlg = StartupDialog(experiments_dir)
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QListWidgetItem

        item = QListWidgetItem()
        item.setData(Qt.UserRole, "/tmp/bad.nexp")
        with (
            patch.object(dlg.manager, "load_experiment", side_effect=ValueError("bad")),
            patch("ui.startup_dialog.QMessageBox.warning") as mock_warn,
            patch.object(dlg, "_refresh_recent") as mock_refresh,
        ):
            dlg._open_recent(item)
        mock_warn.assert_called_once()
        mock_refresh.assert_called_once()

    def test_delete_experiment_delete_file_success_shows_deleted(self, app, experiments_dir) -> None:
        dlg = StartupDialog(experiments_dir)
        item = Mock()
        item.data.return_value = "/tmp/path.nexp"
        with (
            patch("ui.startup_dialog.QMessageBox.warning", return_value=QMessageBox.Yes),
            patch.object(dlg.manager, "delete_experiment", return_value=True),
            patch.object(dlg, "_refresh_recent") as mock_refresh,
            patch("ui.startup_dialog.QMessageBox.information") as mock_info,
        ):
            dlg._delete_experiment(item, delete_file=True)
        mock_refresh.assert_called_once()
        mock_info.assert_called_once()

    def test_delete_experiment_failure_shows_warning(self, app, experiments_dir) -> None:
        dlg = StartupDialog(experiments_dir)
        item = Mock()
        item.data.return_value = "/tmp/path.nexp"
        with (
            patch.object(dlg.manager, "delete_experiment", return_value=False),
            patch("ui.startup_dialog.QMessageBox.warning") as mock_warn,
        ):
            dlg._delete_experiment(item, delete_file=False)
        mock_warn.assert_called_once()

    def test_export_experiment_no_path_noop(self, app, experiments_dir) -> None:
        dlg = StartupDialog(experiments_dir)
        item = Mock()
        item.data.return_value = None
        dlg._export_experiment(item)

    def test_export_experiment_failure_shows_critical(self, app, experiments_dir) -> None:
        dlg = StartupDialog(experiments_dir)
        item = Mock()
        item.data.return_value = "/tmp/path.nexp"
        with (
            patch.object(dlg.manager, "load_experiment", side_effect=ValueError("boom")),
            patch("ui.startup_dialog.QMessageBox.critical") as mock_critical,
        ):
            dlg._export_experiment(item)
        mock_critical.assert_called_once()

    def test_export_experiment_success_shows_information(self, app, experiments_dir, tmp_path) -> None:
        dlg = StartupDialog(experiments_dir)
        item = Mock()
        item.data.return_value = "/tmp/source.nexp"
        experiment = Mock()
        experiment.name = "ExpA"
        target = str(tmp_path / "exported.nexp")
        with (
            patch.object(dlg.manager, "load_experiment", return_value=experiment),
            patch("ui.startup_dialog.QFileDialog.getSaveFileName", return_value=(target, "")),
            patch.object(dlg.manager, "save_experiment", return_value=True),
            patch("ui.startup_dialog.QMessageBox.information") as mock_info,
        ):
            dlg._export_experiment(item)
        mock_info.assert_called_once()

    def test_show_context_menu_dispatches_open(self, app, experiments_dir) -> None:
        dlg = StartupDialog(experiments_dir)
        with (
            patch.object(dlg.recent_list, "itemAt", return_value=Mock()),
            patch("ui.startup_dialog.QMenu") as mock_menu_cls,
            patch.object(dlg, "_open_recent") as mock_open_recent,
        ):
            menu = mock_menu_cls.return_value
            open_action = Mock()
            delete_action = Mock()
            delete_file_action = Mock()
            export_action = Mock()
            menu.addAction.side_effect = [open_action, delete_action, delete_file_action, export_action]
            menu.exec.return_value = open_action
            dlg._show_context_menu(position=QPoint(0, 0))
        mock_open_recent.assert_called_once()

    def test_show_context_menu_dispatches_delete_file(self, app, experiments_dir) -> None:
        dlg = StartupDialog(experiments_dir)
        with (
            patch.object(dlg.recent_list, "itemAt", return_value=Mock()),
            patch("ui.startup_dialog.QMenu") as mock_menu_cls,
            patch.object(dlg, "_delete_experiment") as mock_delete,
        ):
            menu = mock_menu_cls.return_value
            open_action = Mock()
            delete_action = Mock()
            delete_file_action = Mock()
            export_action = Mock()
            menu.addAction.side_effect = [open_action, delete_action, delete_file_action, export_action]
            menu.exec.return_value = delete_file_action
            dlg._show_context_menu(position=QPoint(0, 0))
        mock_delete.assert_called_once()

    def test_show_context_menu_dispatches_export(self, app, experiments_dir) -> None:
        dlg = StartupDialog(experiments_dir)
        with (
            patch.object(dlg.recent_list, "itemAt", return_value=Mock()),
            patch("ui.startup_dialog.QMenu") as mock_menu_cls,
            patch.object(dlg, "_export_experiment") as mock_export,
        ):
            menu = mock_menu_cls.return_value
            open_action = Mock()
            delete_action = Mock()
            delete_file_action = Mock()
            export_action = Mock()
            menu.addAction.side_effect = [open_action, delete_action, delete_file_action, export_action]
            menu.exec.return_value = export_action
            dlg._show_context_menu(position=QPoint(0, 0))
        mock_export.assert_called_once()
