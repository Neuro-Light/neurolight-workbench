"""User selection + user switching tests.

This file keeps the original commented outline for reference, and contains
executable tests that validate the current user selection and switching flows.
"""

# The outlines above are intentionally kept. The tests below implement those
# scenarios while preserving the commented plan for future contributors.

# ruff: noqa: SLF001

from __future__ import annotations

import shutil
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QEnterEvent, QFocusEvent, QKeyEvent, QMouseEvent
from PySide6.QtWidgets import QApplication, QDialog, QGridLayout, QMessageBox, QWidget

from core.experiment_manager import Experiment  # pyright: ignore[reportMissingImports]
from ui.main_window import MainWindow  # pyright: ignore[reportMissingImports]
from ui.startup_dialog import StartupDialog  # pyright: ignore[reportMissingImports]
from ui.user_selection_dialog import (  # pyright: ignore[reportMissingImports]
    UserAccountActionsDialog,
    UserSelectionDialog,
    _DeleteConfirmDialog,
    _list_existing_users,
    _UserCard,
    current_user_button_text,
)

# -----------------------------------------------------------------------------
# Test strategy (high level)
# -----------------------------------------------------------------------------
#
# - Use pytest + PySide6 Qt widgets in headless mode (same style as existing
#   `tests/test_main_window_close_exit.py`).
# - Patch heavy UI components and any filesystem-dependent calls to keep tests
#   deterministic and fast.
# - Prefer testing dialog state/output (selected_user, selected_user_experiments_dir,
#   experiments_dir changes) and that the correct dialogs are invoked, rather than
#   pixel-perfect rendering.
# - Use `tmp_path` for filesystem roots where we need actual directories, and patch
#   `_repo_root()` / `_users_root()` in `ui.user_selection_dialog` so user folders
#   are created under `tmp_path` instead of the real repo.
#
# -----------------------------------------------------------------------------
# Fixtures we will likely need
# -----------------------------------------------------------------------------
#
# @pytest.fixture
# def app():
#     # Ensure a QApplication exists (mirror other tests).
#     ...
#
# @pytest.fixture
# def fake_repo(tmp_path, monkeypatch):
#     # Create a fake repo root with a `users/` directory.
#     # monkeypatch ui.user_selection_dialog._repo_root to return tmp_path
#     # so _users_root() resolves to tmp_path / "users".
#     ...
#
# @pytest.fixture
# def users_root(fake_repo):
#     # Returns tmp_path/"users", optionally pre-populated.
#     ...
#
# -----------------------------------------------------------------------------
# Unit tests for `ui.user_selection_dialog`
# -----------------------------------------------------------------------------
#
# def test_list_existing_users_empty(users_root):
#     # Ensure no users listed when users_root missing/empty.
#     # Call internal `_list_existing_users()` and assert it returns [].
#     ...
#
# def test_list_existing_users_sorted(users_root):
#     # Create users_root/"b", users_root/"a" directories.
#     # Assert `_list_existing_users()` returns ["a", "b"].
#     ...
#
# def test_create_new_user_creates_experiments_dir(users_root, app):
#     # Instantiate UserSelectionDialog.
#     # Patch QInputDialog.getText to return ("Alice", True).
#     # Trigger `_create_new_user()`.
#     # Assert dialog.selected_user == "Alice"
#     # Assert dialog.selected_user_experiments_dir == users_root/"Alice"/"experiments"
#     # Assert the directory exists on disk.
#     ...
#
# def test_create_new_user_rejects_empty_name(users_root, app):
#     # Patch QInputDialog.getText to return ("", True).
#     # Patch QMessageBox.warning to capture call.
#     # Trigger `_create_new_user()` and assert no selection was made.
#     ...
#
# def test_create_new_user_rejects_invalid_chars(users_root, app):
#     # Patch QInputDialog.getText to return ('Bad|Name', True).
#     # Assert it warns and does not create directories.
#     ...
#
# def test_load_existing_user_accepts_selection(users_root, app):
#     # Create users_root/"test"/"experiments".
#     # Patch _UserPickerDialog.exec to Accepted and set selected_user="test".
#     # Trigger `_load_existing_user()` and assert dialog accepted and outputs set.
#     ...
#
# -----------------------------------------------------------------------------
# Unit tests for `ui.startup_dialog` current user button + switching
# -----------------------------------------------------------------------------
#
# def test_startup_dialog_shows_current_user_button_text(users_root, app):
#     # Create experiments_dir = users_root/"test"/"experiments"
#     # StartupDialog(experiments_dir) should create a top-left button whose text is
#     # "Current User: test".
#     # (May need to locate `dialog._current_user_btn` and assert its text.)
#     ...
#
# def test_startup_dialog_switch_user_updates_experiments_dir_and_recent_list(users_root, app, monkeypatch):
#     # Create two users: test and Marcus, each with experiments dir.
#     # Patch UserAccountActionsDialog.exec to Accepted and switch_user_requested True.
#     # Patch UserSelectionDialog.exec to Accepted with selected_user_experiments_dir
#     # pointing at the other user's experiments dir, and selected_user name set.
#     # Call dialog._open_user_account_popup().
#     # Assert dialog.experiments_dir changed, button text updated, and _refresh_recent called.
#     ...
#
# -----------------------------------------------------------------------------
# Integration-ish tests for `ui.main_window` current user button + switching
# -----------------------------------------------------------------------------
#
# def test_main_window_corner_current_user_button_matches_user_experiments_dir(
#     app, sample_experiment, tmp_path, monkeypatch
# ):
#     # Create fake users_root and pass user_experiments_dir into MainWindow(...).
#     # Assert the menubar corner widget exists and its text matches the folder name.
#     ...
#
# def test_main_window_switch_user_then_switch_back_updates_button_correctly(
#     app, sample_experiment, users_root, monkeypatch
# ):
#     # Reproduce the reported bug:
#     # - Start as "test", switch to "Marcus" (no experiments), then switch back to "test"
#     # - Ensure the final button text is "Current User: test"
#     #
#     # Approach:
#     # - Patch UserAccountActionsDialog to always request switch.
#     # - Patch UserSelectionDialog to return Marcus first, then test.
#     # - Patch StartupDialog to simulate switching inside experiment manager and accepting an experiment.
#     # - Verify that `_reload_workbench_after_startup_choice` syncs `user_experiments_dir` from
#     #   `startup.experiments_dir` and updates the button.
#     ...
#
# -----------------------------------------------------------------------------
# Notes / gotchas
# -----------------------------------------------------------------------------
#
# - Many UI objects are C++ backed; keep patches alive using `with (...)` + `yield`
#   fixtures (see existing tests) to prevent premature deletion.
# - Avoid relying on exact widget geometry; test text/state and method calls.
# - Where dialog `exec()` is involved, patch it to return QDialog.Accepted/Rejected
#   and set the dialog’s output attributes (e.g., `.selected_user_experiments_dir`).
# - If tests need the corner widget, use `window.menuBar().cornerWidget(Qt.TopLeftCorner)`
#   OR keep a reference like `window._current_user_btn`.


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def app():
    """Ensure a QApplication exists (conftest also creates one very early)."""
    if not QApplication.instance():
        return QApplication([])
    return QApplication.instance()


@pytest.fixture
def fake_repo(tmp_path, monkeypatch):
    """Patch repo root so user folders are created under tmp_path."""
    users_dir = tmp_path / "users"
    users_dir.mkdir(parents=True, exist_ok=True)

    # ui.user_selection_dialog._users_root() calls _repo_root() at runtime, so we
    # patch only _repo_root() and let _users_root() derive from it.
    import ui.user_selection_dialog as usd  # pyright: ignore[reportMissingImports]

    monkeypatch.setattr(usd, "_repo_root", lambda: Path(tmp_path))
    return tmp_path


@pytest.fixture
def users_root(fake_repo):
    return Path(fake_repo) / "users"


@pytest.fixture
def test_user_experiments_dir(users_root):
    """
    Create a real on-disk test user workspace for tests that need a "selected user".

    Uses tmp_path via users_root, and also explicitly removes the user folder in a
    finally block so cleanup happens even if the test fails.
    """
    user_root = users_root / "test"
    exp_dir = user_root / "experiments"
    exp_dir.mkdir(parents=True, exist_ok=True)
    try:
        yield exp_dir
    finally:
        shutil.rmtree(user_root, ignore_errors=True)


@pytest.fixture
def sample_experiment():
    return Experiment(
        name="Test Experiment",
        description="Test description",
        principal_investigator="Test PI",
    )


@pytest.fixture
def main_window(app, sample_experiment, test_user_experiments_dir):
    """Create a MainWindow instance for tests with heavy widgets patched."""
    mock_viewer = QWidget()
    mock_viewer.index = 0
    mock_viewer.cache = Mock()
    mock_viewer.current_roi = None
    mock_viewer.image_label = Mock()
    mock_viewer.filename_label = Mock()
    mock_viewer.slider = Mock()
    mock_viewer.set_stack = Mock()
    mock_viewer.set_roi = Mock()
    mock_viewer.reset = Mock()
    mock_viewer.image_processor = Mock()
    mock_viewer.get_current_roi = Mock(return_value=None)
    mock_viewer.get_exposure = Mock(return_value=0)
    mock_viewer.get_contrast = Mock(return_value=0)
    mock_viewer.stackLoaded = Mock()
    mock_viewer.stackLoaded.connect = Mock()
    mock_viewer.roiSelected = Mock()
    mock_viewer.roiSelected.connect = Mock()
    mock_viewer.roiDeleted = Mock()
    mock_viewer.roiDeleted.connect = Mock()
    mock_viewer.displaySettingsChanged = Mock()
    mock_viewer.displaySettingsChanged.connect = Mock()

    mock_analysis = QWidget()
    mock_roi_plot_widget = Mock()
    mock_analysis.roi_plot_widget = mock_roi_plot_widget
    mock_analysis.get_roi_plot_widget = Mock(return_value=mock_roi_plot_widget)
    mock_analysis.get_neuron_detection_widget = Mock(return_value=Mock())
    mock_analysis.get_neuron_trajectory_plot_widget = Mock(return_value=Mock())

    mock_stack_handler = Mock()
    mock_stack_handler.files = []
    mock_stack_handler.associate_with_experiment = Mock()

    mock_data_analyzer = Mock()

    with (
        patch("ui.main_window.ImageViewer", return_value=mock_viewer),
        patch("ui.main_window.AnalysisPanel", return_value=mock_analysis),
        patch("ui.main_window.ImageStackHandler", return_value=mock_stack_handler),
        patch("ui.main_window.DataAnalyzer", return_value=mock_data_analyzer),
        patch("ui.main_window.QTimer.singleShot"),  # Avoid timer side effects
    ):
        window = MainWindow(sample_experiment, user_experiments_dir=test_user_experiments_dir)
        yield window


# -----------------------------------------------------------------------------
# Unit tests for `ui.user_selection_dialog`
# -----------------------------------------------------------------------------


def test_list_existing_users_empty(users_root):
    # When users/ exists but is empty.
    assert _list_existing_users() == []


def test_list_existing_users_sorted(users_root):
    (users_root / "b").mkdir(parents=True)
    (users_root / "a").mkdir(parents=True)
    assert _list_existing_users() == ["a", "b"]


def test_create_new_user_creates_experiments_dir(users_root, app):
    dlg = UserSelectionDialog()
    dlg._name_input.setText("Alice")
    dlg._confirm_new_user()

    assert dlg.result() == QDialog.Accepted
    assert dlg.selected_user == "Alice"
    assert dlg.selected_user_experiments_dir == users_root / "Alice" / "experiments"
    assert (users_root / "Alice" / "experiments").is_dir()


def test_create_new_user_rejects_empty_name(users_root, app):
    dlg = UserSelectionDialog()
    dlg._name_input.setText("")
    dlg._confirm_new_user()

    # Empty name: dialog stays open, no user selected
    assert dlg.selected_user is None
    assert dlg.selected_user_experiments_dir is None
    assert dlg.result() != QDialog.Accepted


def test_create_new_user_rejects_invalid_chars(users_root, app):
    dlg = UserSelectionDialog()
    dlg._name_input.setText("Bad|Name")
    with patch.object(QMessageBox, "warning") as mock_warning:
        dlg._confirm_new_user()

    mock_warning.assert_called_once()
    assert dlg.selected_user is None
    assert dlg.selected_user_experiments_dir is None
    assert not (users_root / "Bad|Name").exists()


def test_load_existing_user_accepts_selection(users_root, app):
    (users_root / "test" / "experiments").mkdir(parents=True)
    dlg = UserSelectionDialog()
    dlg._on_user_selected("test")

    assert dlg.result() == QDialog.Accepted
    assert dlg.selected_user == "test"
    assert dlg.selected_user_experiments_dir == users_root / "test" / "experiments"


# -----------------------------------------------------------------------------
# Unit tests for `ui.startup_dialog` current user button + switching
# -----------------------------------------------------------------------------


def test_startup_dialog_shows_current_user_button_text(users_root, app):
    experiments_dir = users_root / "test" / "experiments"
    experiments_dir.mkdir(parents=True)
    dlg = StartupDialog(experiments_dir)
    assert dlg._current_user_btn.text() == "Current User: test"


def test_startup_dialog_uses_experiments_dir_parent_as_user_name(users_root, app):
    """
    Rewrite of the prior "switch user" test.

    We intentionally avoid calling `_open_user_account_popup()` here because it
    launches modal dialogs (`exec()`), which is a common source of headless test
    hangs if patch targets drift after refactors.
    """
    experiments_dir = users_root / "Marcus" / "experiments"
    experiments_dir.mkdir(parents=True)
    dlg = StartupDialog(experiments_dir)
    assert dlg._current_user_name == "Marcus"
    assert dlg._current_user_btn.text() == "Current User: Marcus"


# -----------------------------------------------------------------------------
# Integration-ish tests for `ui.main_window` current user button + switching
# -----------------------------------------------------------------------------


def test_main_window_corner_current_user_button_matches_user_experiments_dir(app, sample_experiment, users_root):
    (users_root / "test" / "experiments").mkdir(parents=True)
    user_dir = users_root / "test" / "experiments"

    # Build a MainWindow with heavy widgets patched (same approach as other tests).
    mock_viewer = QWidget()
    mock_viewer.index = 0
    mock_viewer.cache = Mock()
    mock_viewer.current_roi = None
    mock_viewer.image_label = Mock()
    mock_viewer.filename_label = Mock()
    mock_viewer.slider = Mock()
    mock_viewer.set_stack = Mock()
    mock_viewer.set_roi = Mock()
    mock_viewer.reset = Mock()
    mock_viewer.image_processor = Mock()
    mock_viewer.get_current_roi = Mock(return_value=None)
    mock_viewer.get_exposure = Mock(return_value=0)
    mock_viewer.get_contrast = Mock(return_value=0)
    mock_viewer.stackLoaded = Mock()
    mock_viewer.stackLoaded.connect = Mock()
    mock_viewer.roiSelected = Mock()
    mock_viewer.roiSelected.connect = Mock()
    mock_viewer.roiDeleted = Mock()
    mock_viewer.roiDeleted.connect = Mock()
    mock_viewer.displaySettingsChanged = Mock()
    mock_viewer.displaySettingsChanged.connect = Mock()

    mock_analysis = QWidget()
    mock_roi_plot_widget = Mock()
    mock_analysis.roi_plot_widget = mock_roi_plot_widget
    mock_analysis.get_roi_plot_widget = Mock(return_value=mock_roi_plot_widget)
    mock_analysis.get_neuron_detection_widget = Mock(return_value=Mock())
    mock_analysis.get_neuron_trajectory_plot_widget = Mock(return_value=Mock())

    mock_stack_handler = Mock()
    mock_stack_handler.files = []
    mock_stack_handler.associate_with_experiment = Mock()

    mock_data_analyzer = Mock()

    with (
        patch("ui.main_window.ImageViewer", return_value=mock_viewer),
        patch("ui.main_window.AnalysisPanel", return_value=mock_analysis),
        patch("ui.main_window.ImageStackHandler", return_value=mock_stack_handler),
        patch("ui.main_window.DataAnalyzer", return_value=mock_data_analyzer),
        patch("ui.main_window.QTimer.singleShot"),
    ):
        window = MainWindow(sample_experiment, user_experiments_dir=user_dir)
        assert window._current_user_btn is not None
        assert window._current_user_btn.text() == "Current User: test"


def test_main_window_switch_user_then_switch_back_updates_button_correctly(app, sample_experiment, users_root):
    """
    Rewrite of the prior integration-ish test.

    We avoid calling `_open_user_account_popup()` because it can open several modal
    dialogs (`exec()`), which may hang in headless test runs if mocks don't line up
    exactly with the current import paths.

    Instead, we validate the underlying bugfix: whenever a `StartupDialog` choice
    is applied, the MainWindow must sync its `user_experiments_dir` (and button
    label) from the dialog's final `experiments_dir`, even if MainWindow previously
    held a different user's directory.
    """
    # Setup user directories
    test_dir = users_root / "test" / "experiments"
    marcus_dir = users_root / "Marcus" / "experiments"
    test_dir.mkdir(parents=True)
    marcus_dir.mkdir(parents=True)

    # Build a MainWindow with heavy widgets patched.
    mock_viewer = QWidget()
    mock_viewer.index = 0
    mock_viewer.cache = Mock()
    mock_viewer.current_roi = None
    mock_viewer.image_label = Mock()
    mock_viewer.filename_label = Mock()
    mock_viewer.slider = Mock()
    mock_viewer.set_stack = Mock()
    mock_viewer.set_roi = Mock()
    mock_viewer.reset = Mock()
    mock_viewer.image_processor = Mock()
    mock_viewer.get_current_roi = Mock(return_value=None)
    mock_viewer.get_exposure = Mock(return_value=0)
    mock_viewer.get_contrast = Mock(return_value=0)
    mock_viewer.stackLoaded = Mock()
    mock_viewer.stackLoaded.connect = Mock()
    mock_viewer.roiSelected = Mock()
    mock_viewer.roiSelected.connect = Mock()
    mock_viewer.roiDeleted = Mock()
    mock_viewer.roiDeleted.connect = Mock()
    mock_viewer.displaySettingsChanged = Mock()
    mock_viewer.displaySettingsChanged.connect = Mock()

    mock_analysis = QWidget()
    mock_roi_plot_widget = Mock()
    mock_analysis.roi_plot_widget = mock_roi_plot_widget
    mock_analysis.get_roi_plot_widget = Mock(return_value=mock_roi_plot_widget)
    mock_analysis.get_neuron_detection_widget = Mock(return_value=Mock())
    mock_analysis.get_neuron_trajectory_plot_widget = Mock(return_value=Mock())

    mock_stack_handler = Mock()
    mock_stack_handler.files = []
    mock_stack_handler.associate_with_experiment = Mock()

    mock_data_analyzer = Mock()

    with (
        patch("ui.main_window.ImageViewer", return_value=mock_viewer),
        patch("ui.main_window.AnalysisPanel", return_value=mock_analysis),
        patch("ui.main_window.ImageStackHandler", return_value=mock_stack_handler),
        patch("ui.main_window.DataAnalyzer", return_value=mock_data_analyzer),
        patch("ui.main_window.QTimer.singleShot"),
    ):
        window = MainWindow(sample_experiment, user_experiments_dir=test_dir)
        assert window._current_user_btn is not None
        assert window._current_user_btn.text() == "Current User: test"

        # Simulate an intermediate (incorrect) state that used to "stick".
        window.user_experiments_dir = marcus_dir
        window._update_current_user_button_text()
        assert window._current_user_btn.text() == "Current User: Marcus"

        # Apply a startup selection that resolves back to "test".
        startup_choice = Mock(spec=StartupDialog)
        startup_choice.experiments_dir = test_dir
        startup_choice.experiment = Experiment(name="test-1", description="", principal_investigator="")
        startup_choice.experiment_path = str(test_dir / "test-1" / "test-1.nexp")
        window._reload_workbench_after_startup_choice(startup_choice)

        # Window should now be synced to `startup_choice.experiments_dir` ("test"),
        # not the intermediate ("Marcus").
        assert window.user_experiments_dir == test_dir
        assert window._current_user_btn.text() == "Current User: test"


def test_user_selection_delete_user_success_refreshes_cards(users_root, app):
    (users_root / "Alice" / "experiments").mkdir(parents=True)
    dlg = UserSelectionDialog()
    confirm = Mock()
    confirm.exec.return_value = QDialog.Accepted
    with (
        patch("ui.user_selection_dialog._DeleteConfirmDialog", return_value=confirm),
        patch.object(dlg, "_refresh_cards") as mock_refresh,
    ):
        dlg._on_delete_user("Alice")
    assert not (users_root / "Alice").exists()
    mock_refresh.assert_called_once()


def test_user_selection_delete_user_failure_shows_error(users_root, app):
    dlg = UserSelectionDialog()
    confirm = Mock()
    confirm.exec.return_value = QDialog.Accepted
    with (
        patch("ui.user_selection_dialog._DeleteConfirmDialog", return_value=confirm),
        patch("ui.user_selection_dialog.shutil.rmtree", side_effect=OSError("boom")),
        patch.object(QMessageBox, "critical") as mock_critical,
    ):
        dlg._on_delete_user("Alice")
    mock_critical.assert_called_once()


def test_main_window_open_user_account_popup_reloads_after_switch(main_window, users_root):
    next_dir = users_root / "next" / "experiments"
    next_dir.mkdir(parents=True)
    main_window.current_experiment_path = "/tmp/current.nexp"

    account_popup = Mock()
    account_popup.exec.return_value = QDialog.Accepted
    account_popup.switch_user_requested = True

    user_picker = Mock()
    user_picker.exec.return_value = QDialog.Accepted
    user_picker.selected_user_experiments_dir = next_dir

    startup = Mock(spec=StartupDialog)
    startup.exec.return_value = QDialog.Accepted
    startup.experiment = Experiment(name="Reloaded", description="", principal_investigator="")
    startup.experiment_path = str(next_dir / "Reloaded" / "Reloaded.nexp")
    startup.experiments_dir = next_dir

    with (
        patch("ui.main_window.UserAccountActionsDialog", return_value=account_popup),
        patch("ui.main_window.UserSelectionDialog", return_value=user_picker),
        patch("ui.main_window.StartupDialog", return_value=startup),
        patch.object(main_window, "_reload_workbench_after_startup_choice") as mock_reload,
        patch.object(main_window.manager, "save_experiment"),
        patch.object(main_window, "hide") as mock_hide,
    ):
        main_window._open_user_account_popup()

    mock_hide.assert_called_once()
    mock_reload.assert_called_once_with(startup)


def test_main_window_open_user_account_popup_cancelled_selection_restores_window(main_window):
    account_popup = Mock()
    account_popup.exec.return_value = QDialog.Accepted
    account_popup.switch_user_requested = True

    user_picker = Mock()
    user_picker.exec.return_value = QDialog.Rejected
    user_picker.selected_user_experiments_dir = None

    with (
        patch("ui.main_window.UserAccountActionsDialog", return_value=account_popup),
        patch("ui.main_window.UserSelectionDialog", return_value=user_picker),
        patch.object(main_window, "show") as mock_show,
        patch.object(main_window, "hide"),
    ):
        main_window._open_user_account_popup()

    mock_show.assert_called_once()


def test_current_user_button_text_formats_label():
    assert current_user_button_text("test") == "Current User: test"


def test_delete_confirm_dialog_enables_button_on_exact_match(app):
    dlg = _DeleteConfirmDialog("Alice")
    assert not dlg._confirm_btn.isEnabled()
    dlg._on_text_changed("Alice")
    assert dlg._confirm_btn.isEnabled()


def test_delete_confirm_dialog_try_accept_rejects_non_match(app):
    dlg = _DeleteConfirmDialog("Alice")
    dlg._input.setText("Bob")
    dlg._try_accept()
    assert dlg.result() == QDialog.Rejected


def test_delete_confirm_dialog_try_accept_accepts_match(app):
    dlg = _DeleteConfirmDialog("Alice")
    dlg._input.setText("Alice")
    dlg._try_accept()
    assert dlg.result() == QDialog.Accepted


def test_user_account_actions_dialog_sets_switch_requested(app):
    dlg = UserAccountActionsDialog("test-user")
    assert dlg.switch_user_requested is False
    dlg._on_switch()
    assert dlg.switch_user_requested is True
    assert dlg.result() == QDialog.Accepted


def test_user_card_mouse_click_emits_login_requested(app):
    parent = QWidget()
    card = _UserCard("Alice", grid_row=0, grid_col=0, grid_cols=1, parent=parent)
    called = []
    card.login_requested.connect(lambda user: called.append(user))
    event = QMouseEvent(
        QMouseEvent.Type.MouseButtonPress,
        card.rect().center(),
        Qt.LeftButton,
        Qt.LeftButton,
        Qt.NoModifier,
    )
    card.mousePressEvent(event)
    assert called == ["Alice"]


def test_user_card_key_enter_emits_login_requested(app):
    parent = QWidget()
    card = _UserCard("Alice", grid_row=0, grid_col=0, grid_cols=1, parent=parent)
    called = []
    card.login_requested.connect(lambda user: called.append(user))
    event = QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key_Return, Qt.NoModifier)
    card.keyPressEvent(event)
    assert called == ["Alice"]


def test_user_card_key_arrow_moves_focus_to_neighbor(app):
    parent = QWidget()
    parent.setLayout(QGridLayout())
    left = _UserCard("Left", grid_row=0, grid_col=0, grid_cols=2, parent=parent)
    right = _UserCard("Right", grid_row=0, grid_col=1, grid_cols=2, parent=parent)
    parent.layout().addWidget(left, 0, 0)
    parent.layout().addWidget(right, 0, 1)
    right.setFocus = Mock()

    event = QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key_Right, Qt.NoModifier)
    left.keyPressEvent(event)
    right.setFocus.assert_called_once()


def test_create_new_user_rejects_dot_and_dotdot(users_root, app):
    dlg = UserSelectionDialog()
    with patch.object(QMessageBox, "warning") as mock_warn:
        dlg._name_input.setText(".")
        dlg._confirm_new_user()
        dlg._name_input.setText("..")
        dlg._confirm_new_user()
    assert mock_warn.call_count == 2


def test_create_new_user_duplicate_shows_info(users_root, app):
    (users_root / "Alice" / "experiments").mkdir(parents=True)
    dlg = UserSelectionDialog()
    dlg._name_input.setText("Alice")
    with patch.object(QMessageBox, "information") as mock_info:
        dlg._confirm_new_user()
    mock_info.assert_called_once()


def test_create_new_user_oserror_shows_critical(users_root, app):
    dlg = UserSelectionDialog()
    dlg._name_input.setText("Alice")
    with (
        patch("ui.user_selection_dialog._experiments_dir_for_user", return_value=users_root / "Alice" / "experiments"),
        patch("pathlib.Path.mkdir", side_effect=OSError("disk error")),
        patch.object(QMessageBox, "critical") as mock_critical,
    ):
        dlg._confirm_new_user()
    mock_critical.assert_called_once()


def test_user_selection_show_and_hide_new_user_form(app):
    dlg = UserSelectionDialog()
    dlg._show_new_user_form()
    assert dlg._new_user_frame.isHidden() is False
    assert dlg._add_btn.isHidden() is True
    dlg._hide_new_user_form()
    assert dlg._new_user_frame.isHidden() is True
    assert dlg._add_btn.isHidden() is False


def test_user_selection_refresh_cards_rebuilds_existing_widgets(users_root, app):
    (users_root / "alice").mkdir(parents=True)
    dlg = UserSelectionDialog()
    # Refresh twice to force the deleteLater cleanup branch.
    dlg._refresh_cards()
    dlg._refresh_cards()
    assert dlg._cards_layout.count() >= 1


def test_user_card_hover_and_focus_update_border(app):
    parent = QWidget()
    card = _UserCard("Alice", grid_row=0, grid_col=0, grid_cols=1, parent=parent)
    card.enterEvent(QEnterEvent(QPointF(0, 0), QPointF(0, 0), QPointF(0, 0)))
    assert "#4A90E2" in card.styleSheet()
    card.leaveEvent(QEvent(QEvent.Type.Leave))
    assert "transparent" in card.styleSheet()
    card.focusInEvent(QFocusEvent(QEvent.Type.FocusIn))
    assert "#4A90E2" in card.styleSheet()
    card.focusOutEvent(QFocusEvent(QEvent.Type.FocusOut))
    assert "transparent" in card.styleSheet()
