"""Tests for the Public User feature: sync, permissions, visibility flag, and credits."""

from __future__ import annotations

import builtins
import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QCloseEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QComboBox, QDialog, QMessageBox, QPushButton, QSpinBox, QToolButton, QWidget

from core.experiment_manager import Experiment, ExperimentManager
from core.roi import ROI
from ui.main_window import MainWindow
from ui.public_user_dialog import (
    PUBLIC_USER_NAME,
    ReadOnlyGuard,
    ensure_public_user_exists,
    is_public_user,
    register_public_experiment,
    sync_public_experiments,
    unregister_public_experiment,
)
from ui.startup_dialog import RecentExperimentRow, StartupDialog, _public_recent_row_label
from ui.user_selection_dialog import UserSelectionDialog, _list_existing_users, _UserCard
from ui.workflow import WorkflowManager, WorkflowStepper


@pytest.fixture
def app():
    if not QApplication.instance():
        return QApplication([])
    return QApplication.instance()


def _minimal_nexp(*, name: str = "Exp", is_public: bool = False) -> dict:
    return {
        "version": "1.0",
        "experiment": {
            "name": name,
            "image_stack": {"path": "", "file_list": [], "count": 0},
            "is_public": is_public,
        },
    }


@pytest.fixture
def fake_users_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Redirect ``users/`` layout under tmp_path (must match ``_repo_root`` layout)."""
    root = tmp_path / "repo_root"
    root.mkdir()
    (root / "users").mkdir()
    monkeypatch.setattr("ui.public_user_dialog._repo_root", lambda: root)
    return root


# ── public_user_dialog helpers ─────────────────────────────────────────────


def test_is_public_user_exact_string_only() -> None:
    assert is_public_user(PUBLIC_USER_NAME) is True
    assert is_public_user("public") is False
    assert is_public_user("PUBLIC") is False
    assert is_public_user("alice") is False


def test_ensure_public_user_creates_directories(fake_users_root: Path) -> None:
    exp_dir = ensure_public_user_exists()
    assert exp_dir == fake_users_root / "users" / PUBLIC_USER_NAME / "experiments"
    assert exp_dir.is_dir()
    recent = fake_users_root / "users" / PUBLIC_USER_NAME / "recent_experiments.json"
    assert recent.is_file()
    assert json.loads(recent.read_text(encoding="utf-8")) == {"recent": []}


def test_ensure_public_user_profile_appears_in_user_list(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """After ensure_public_user_exists(), the Public profile is listed like any user."""
    root = tmp_path / "repo_root"
    root.mkdir()
    (root / "users").mkdir()
    monkeypatch.setattr("ui.public_user_dialog._repo_root", lambda: root)
    monkeypatch.setattr("ui.user_selection_dialog._repo_root", lambda: root)
    assert _list_existing_users() == []
    ensure_public_user_exists()
    assert PUBLIC_USER_NAME in _list_existing_users()


def test_on_delete_user_does_not_remove_public_profile(app, monkeypatch: pytest.MonkeyPatch) -> None:
    """Public User cannot be deleted (handler no-ops before confirm or rmtree)."""
    removed: list[Path] = []
    monkeypatch.setattr(
        "ui.user_selection_dialog.shutil.rmtree",
        lambda p, *args, **kwargs: removed.append(Path(p)),
    )
    dlg = UserSelectionDialog()
    dlg._on_delete_user(PUBLIC_USER_NAME)
    assert removed == []


def test_sync_public_experiments_copies_only_nested_is_public(
    fake_users_root: Path,
) -> None:
    alice_exp = fake_users_root / "users" / "alice" / "experiments"
    alice_exp.mkdir(parents=True)
    (alice_exp / "study.nexp").write_text(
        json.dumps(_minimal_nexp(name="My Study", is_public=True)),
        encoding="utf-8",
    )
    (alice_exp / "private.nexp").write_text(
        json.dumps(_minimal_nexp(name="Secret", is_public=False)),
        encoding="utf-8",
    )
    # Top-level is_public must not count (only experiment.is_public)
    bad = _minimal_nexp(name="Bad", is_public=False)
    bad["is_public"] = True
    (alice_exp / "spoof.nexp").write_text(json.dumps(bad), encoding="utf-8")

    sync_public_experiments()

    pub_dir = fake_users_root / "users" / PUBLIC_USER_NAME / "experiments"
    assert (pub_dir / "alice__study.nexp").is_file()
    assert not (pub_dir / "alice__private.nexp").exists()
    assert not (pub_dir / "alice__spoof.nexp").exists()

    recent = json.loads(
        (fake_users_root / "users" / PUBLIC_USER_NAME / "recent_experiments.json").read_text(encoding="utf-8")
    )
    assert len(recent["recent"]) == 1
    entry = recent["recent"][0]
    assert entry["name"] == "My Study"
    assert entry["owner"] == "alice"
    assert entry["path"].endswith("alice__study.nexp")


def test_sync_public_experiments_owner_preserves_mixed_case_folder_name(
    fake_users_root: Path,
) -> None:
    udir = fake_users_root / "users" / "MiXeDCase" / "experiments"
    udir.mkdir(parents=True)
    (udir / "e.nexp").write_text(json.dumps(_minimal_nexp(name="E", is_public=True)), encoding="utf-8")
    sync_public_experiments()
    recent = json.loads(
        (fake_users_root / "users" / PUBLIC_USER_NAME / "recent_experiments.json").read_text(encoding="utf-8")
    )
    assert recent["recent"][0]["owner"] == "MiXeDCase"
    label = _public_recent_row_label(recent["recent"][0], recent["recent"][0]["path"])
    assert label == "E - by MiXeDCase"


def test_sync_public_experiments_removes_stale_copy_when_made_private(
    fake_users_root: Path,
) -> None:
    alice_exp = fake_users_root / "users" / "alice" / "experiments"
    alice_exp.mkdir(parents=True)
    path = alice_exp / "gone.nexp"
    path.write_text(json.dumps(_minimal_nexp(name="Gone", is_public=True)), encoding="utf-8")
    sync_public_experiments()
    pub_copy = fake_users_root / "users" / PUBLIC_USER_NAME / "experiments" / "alice__gone.nexp"
    assert pub_copy.is_file()

    path.write_text(json.dumps(_minimal_nexp(name="Gone", is_public=False)), encoding="utf-8")
    sync_public_experiments()
    assert not pub_copy.exists()
    recent = json.loads(
        (fake_users_root / "users" / PUBLIC_USER_NAME / "recent_experiments.json").read_text(encoding="utf-8")
    )
    assert recent["recent"] == []


def test_user_selection_card_hides_delete_for_public_profile(app) -> None:
    parent = QWidget()
    card = _UserCard(PUBLIC_USER_NAME, grid_row=0, grid_col=0, grid_cols=1, parent=parent)
    parent.show()
    card.show()
    app.processEvents()
    del_btns = card.findChildren(QToolButton)
    assert len(del_btns) == 1
    assert del_btns[0].isVisible() is False


def test_user_selection_card_shows_delete_for_normal_profile(app) -> None:
    parent = QWidget()
    card = _UserCard("alice", grid_row=0, grid_col=0, grid_cols=1, parent=parent)
    parent.show()
    card.show()
    app.processEvents()
    del_btns = card.findChildren(QToolButton)
    assert len(del_btns) == 1
    assert del_btns[0].isVisible() is True


def test_read_only_guard_reverts_enable(app) -> None:
    w = QPushButton("x")
    w.setEnabled(True)
    g = ReadOnlyGuard.lock(w)
    assert w.isEnabled() is False
    w.setEnabled(True)
    app.processEvents()
    assert w.isEnabled() is False
    w.removeEventFilter(g)


def test_read_only_guard_event_filter_no_reentrant_disable_when_already_off(app) -> None:
    w = QPushButton("x")
    g = ReadOnlyGuard(w)
    w.installEventFilter(g)
    w.setEnabled(False)
    assert g.eventFilter(w, QEvent(QEvent.Type.EnabledChange)) is False


def test_ensure_public_user_skips_recent_write_when_file_already_exists(fake_users_root: Path) -> None:
    ensure_public_user_exists()
    recent = fake_users_root / "users" / PUBLIC_USER_NAME / "recent_experiments.json"
    before = recent.read_text(encoding="utf-8")
    ensure_public_user_exists()
    assert recent.read_text(encoding="utf-8") == before


def test_sync_public_experiments_returns_when_users_path_not_a_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "repo_root"
    root.mkdir()
    (root / "users").write_text("not-a-directory", encoding="utf-8")
    monkeypatch.setattr("ui.public_user_dialog._repo_root", lambda: root)
    sync_public_experiments()
    assert not (root / "users" / PUBLIC_USER_NAME / "experiments").exists()


def test_sync_public_experiments_skips_invalid_json_and_non_dict_experiment(
    fake_users_root: Path,
) -> None:
    alice = fake_users_root / "users" / "alice" / "experiments"
    alice.mkdir(parents=True)
    (alice / "bad.nexp").write_text("not json {", encoding="utf-8")
    (alice / "weird.nexp").write_text(
        json.dumps({"version": "1.0", "experiment": "not-a-dict"}),
        encoding="utf-8",
    )
    (alice / "good.nexp").write_text(json.dumps(_minimal_nexp(name="Ok", is_public=True)), encoding="utf-8")
    sync_public_experiments()
    pub = fake_users_root / "users" / PUBLIC_USER_NAME / "experiments"
    assert (pub / "alice__good.nexp").is_file()
    assert len(list(pub.glob("*.nexp"))) == 1


def test_sync_public_experiments_skips_copy_oserror(fake_users_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    alice = fake_users_root / "users" / "alice" / "experiments"
    alice.mkdir(parents=True)
    (alice / "p.nexp").write_text(json.dumps(_minimal_nexp(is_public=True)), encoding="utf-8")
    monkeypatch.setattr("ui.public_user_dialog.shutil.copy2", Mock(side_effect=OSError("denied")))
    sync_public_experiments()
    pub = fake_users_root / "users" / PUBLIC_USER_NAME / "experiments"
    assert not any(pub.iterdir())


def test_sync_public_experiments_skips_subdir_without_experiments_folder(fake_users_root: Path) -> None:
    (fake_users_root / "users" / "bob").mkdir()
    sync_public_experiments()
    recent = json.loads(
        (fake_users_root / "users" / PUBLIC_USER_NAME / "recent_experiments.json").read_text(encoding="utf-8")
    )
    assert recent["recent"] == []


def test_register_public_experiment_prepends_and_deduplicates(fake_users_root: Path) -> None:
    ensure_public_user_exists()
    recent = fake_users_root / "users" / PUBLIC_USER_NAME / "recent_experiments.json"
    p1 = fake_users_root / "a.nexp"
    p1.write_text("{}", encoding="utf-8")
    register_public_experiment(str(p1), "Alpha")
    register_public_experiment(str(p1), "Alpha again")
    data = json.loads(recent.read_text(encoding="utf-8"))
    assert len(data["recent"]) == 1
    assert data["recent"][0]["name"] == "Alpha again"


def test_register_public_experiment_corrupt_read_falls_back(fake_users_root: Path) -> None:
    ensure_public_user_exists()
    recent = fake_users_root / "users" / PUBLIC_USER_NAME / "recent_experiments.json"
    recent.write_text("not json", encoding="utf-8")
    p = fake_users_root / "z.nexp"
    p.write_text("{}", encoding="utf-8")
    register_public_experiment(str(p), "Zed")
    data = json.loads(recent.read_text(encoding="utf-8"))
    assert len(data["recent"]) == 1


def test_unregister_public_experiment_noop_when_missing_recent(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    root = tmp_path / "r"
    root.mkdir()
    monkeypatch.setattr("ui.public_user_dialog._repo_root", lambda: root)
    unregister_public_experiment("/no/such/path.nexp")


def test_unregister_public_experiment_removes_path(fake_users_root: Path) -> None:
    ensure_public_user_exists()
    p = fake_users_root / "x.nexp"
    p.write_text("{}", encoding="utf-8")
    register_public_experiment(str(p))
    unregister_public_experiment(str(p))
    data = json.loads(
        (fake_users_root / "users" / PUBLIC_USER_NAME / "recent_experiments.json").read_text(encoding="utf-8")
    )
    assert data["recent"] == []


def test_repo_root_is_repository_containing_src_layout() -> None:
    """Exercise real ``_repo_root()`` (not redirected by tests)."""
    import ui.public_user_dialog as pud

    root = pud._repo_root()
    assert (root / "src" / "ui" / "public_user_dialog.py").is_file()


def test_ensure_public_user_exists_swallows_oserror_on_mkdir(
    fake_users_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Path, "mkdir", Mock(side_effect=OSError("denied")))
    # Must not raise; returns path under fake root even when mkdir fails.
    out = ensure_public_user_exists()
    assert out == fake_users_root / "users" / PUBLIC_USER_NAME / "experiments"


def _patch_open_fail_recent_write(monkeypatch: pytest.MonkeyPatch) -> None:
    real_open = builtins.open

    def _open(file, *args, **kwargs):
        mode = args[0] if args else kwargs.get("mode", "r")
        if mode == "w" and "recent_experiments.json" in str(file):
            raise OSError("write denied")
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", _open)


def test_sync_public_experiments_swallows_oserror_on_recent_rewrite(
    fake_users_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    alice = fake_users_root / "users" / "alice" / "experiments"
    alice.mkdir(parents=True)
    (alice / "p.nexp").write_text(json.dumps(_minimal_nexp(is_public=True)), encoding="utf-8")
    _patch_open_fail_recent_write(monkeypatch)
    sync_public_experiments()


def test_sync_public_experiments_swallows_oserror_on_stale_unlink(
    fake_users_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    alice = fake_users_root / "users" / "alice" / "experiments"
    alice.mkdir(parents=True)
    nexp = alice / "gone.nexp"
    nexp.write_text(json.dumps(_minimal_nexp(is_public=True)), encoding="utf-8")
    sync_public_experiments()
    pub = fake_users_root / "users" / PUBLIC_USER_NAME / "experiments"
    copy_path = pub / "alice__gone.nexp"
    assert copy_path.is_file()
    nexp.write_text(json.dumps(_minimal_nexp(is_public=False)), encoding="utf-8")
    orig_unlink = Path.unlink

    def _bad_unlink(self: Path, *a: object, **kw: object) -> None:
        if self == copy_path:
            raise OSError("in use")
        return orig_unlink(self, *a, **kw)

    monkeypatch.setattr(Path, "unlink", _bad_unlink)
    sync_public_experiments()


def test_register_public_experiment_swallows_oserror_on_write(
    fake_users_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ensure_public_user_exists()
    p = fake_users_root / "w.nexp"
    p.write_text("{}", encoding="utf-8")
    _patch_open_fail_recent_write(monkeypatch)
    register_public_experiment(str(p), "W")


def test_unregister_public_experiment_corrupt_read_then_write_ok(fake_users_root: Path) -> None:
    ensure_public_user_exists()
    recent = fake_users_root / "users" / PUBLIC_USER_NAME / "recent_experiments.json"
    recent.write_text("{not json", encoding="utf-8")
    p = fake_users_root / "u.nexp"
    p.write_text("{}", encoding="utf-8")
    unregister_public_experiment(str(p))
    data = json.loads(recent.read_text(encoding="utf-8"))
    assert data["recent"] == []


def test_unregister_public_experiment_swallows_oserror_on_write(
    fake_users_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ensure_public_user_exists()
    p = fake_users_root / "y.nexp"
    p.write_text("{}", encoding="utf-8")
    register_public_experiment(str(p))
    _patch_open_fail_recent_write(monkeypatch)
    unregister_public_experiment(str(p))


# ── ExperimentManager: no writes into Public copies tree ───────────────────


@pytest.fixture
def recent_file_global(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    path = tmp_path / "recent_experiments.json"
    monkeypatch.setattr("core.experiment_manager.RECENT_FILE", path)
    return path


def test_save_experiment_rejected_under_public_experiments(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, recent_file_global: Path
) -> None:
    monkeypatch.setattr("core.experiment_manager._repo_root", lambda: tmp_path)
    pub = tmp_path / "users" / PUBLIC_USER_NAME / "experiments"
    pub.mkdir(parents=True)
    target = pub / "alice__x.nexp"
    mgr = ExperimentManager(recent_file_global)
    assert mgr.save_experiment(Experiment(name="N"), str(target)) is False
    assert not target.exists()


# ── StartupDialog: Public mode disables create/load ───────────────────────


def test_startup_dialog_public_recent_list_shows_credit_line(app, tmp_path: Path) -> None:
    user_root = tmp_path / "users" / PUBLIC_USER_NAME
    exp_dir = user_root / "experiments"
    exp_dir.mkdir(parents=True)
    nexp = exp_dir / "alice__mine.nexp"
    nexp.write_text(json.dumps(_minimal_nexp(name="Mine", is_public=False)), encoding="utf-8")
    recent_path = user_root / "recent_experiments.json"
    recent_path.write_text(
        json.dumps(
            {
                "recent": [
                    {
                        "path": str(nexp.resolve()),
                        "name": "Mine",
                        "owner": "alice",
                        "last_opened": "2026-01-01T00:00:00+00:00",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    dlg = StartupDialog(exp_dir)
    rows = dlg.findChildren(RecentExperimentRow)
    assert len(rows) == 1
    assert rows[0].name_label.text() == "Mine - by alice"


def test_startup_dialog_public_user_disables_new_and_load(app, tmp_path: Path) -> None:
    exp_dir = tmp_path / "users" / PUBLIC_USER_NAME / "experiments"
    exp_dir.mkdir(parents=True)
    dlg = StartupDialog(exp_dir)
    texts = [b.text() for b in dlg.findChildren(QPushButton)]
    assert "Start New Experiment" in texts
    new_btn = next(b for b in dlg.findChildren(QPushButton) if b.text() == "Start New Experiment")
    load_btn = next(b for b in dlg.findChildren(QPushButton) if b.text() == "Load Existing Experiment")
    assert new_btn.isEnabled() is False
    assert load_btn.isEnabled() is False


def test_startup_dialog_public_user_start_new_shows_message_only(app, tmp_path: Path) -> None:
    exp_dir = tmp_path / "users" / PUBLIC_USER_NAME / "experiments"
    exp_dir.mkdir(parents=True)
    dlg = StartupDialog(exp_dir)
    with patch.object(QMessageBox, "information", return_value=None) as inf:
        dlg._start_new()
    inf.assert_called_once()
    with patch.object(QMessageBox, "information", return_value=None) as inf2:
        dlg._load_existing()
    inf2.assert_called_once()


def test_startup_dialog_public_user_mouse_clicks_do_not_start_new_or_load(app, tmp_path: Path) -> None:
    """Disabled New/Load buttons must not emit ``clicked`` (no dialog / file picker)."""
    exp_dir = tmp_path / "users" / PUBLIC_USER_NAME / "experiments"
    exp_dir.mkdir(parents=True)
    dlg = StartupDialog(exp_dir)
    new_btn = next(b for b in dlg.findChildren(QPushButton) if b.text() == "Start New Experiment")
    load_btn = next(b for b in dlg.findChildren(QPushButton) if b.text() == "Load Existing Experiment")
    with (
        patch("ui.startup_dialog.NewExperimentDialog") as mock_new_cls,
        patch(
            "ui.startup_dialog.QFileDialog.getOpenFileName",
            return_value=("/fake.nexp", "Neurolight Experiment (*.nexp)"),
        ) as mock_open,
    ):
        QTest.mouseClick(new_btn, Qt.MouseButton.LeftButton)
        QTest.mouseClick(load_btn, Qt.MouseButton.LeftButton)
        app.processEvents()
    mock_new_cls.assert_not_called()
    mock_open.assert_not_called()


# ── WorkflowStepper read-only toggling ─────────────────────────────────────


def test_workflow_stepper_read_only_unlock_restores_buttons(app) -> None:
    exp = Experiment(name="T")
    wm = WorkflowManager(exp)
    stepper = WorkflowStepper(wm)
    first = next(iter(stepper._step_buttons.values()))
    was = first.isEnabled()
    stepper.set_read_only(True)
    assert first.isEnabled() is False
    stepper.set_read_only(False)
    assert first.isEnabled() == was


def test_workflow_stepper_read_only_mouse_click_does_not_emit_clicked(app) -> None:
    exp = Experiment(name="T")
    wm = WorkflowManager(exp)
    stepper = WorkflowStepper(wm)
    first = next(iter(stepper._step_buttons.values()))
    slot = Mock()
    first.clicked.connect(slot)
    stepper.set_read_only(True)
    QTest.mouseClick(first, Qt.MouseButton.LeftButton)
    app.processEvents()
    slot.assert_not_called()


# ── MainWindow: public mode permissions and switching ─────────────────────


class _DetStub:
    def __init__(self) -> None:
        self.detect_mode_combo = QComboBox()
        self.cell_size_spin = QSpinBox()
        self.num_peaks_spin = QSpinBox()
        self.correlation_threshold_spin = QSpinBox()
        self.max_absent_frames_spin = QSpinBox()
        self.threshold_rel_spin = QSpinBox()
        self.max_projection_checkbox = QWidget()
        self.preprocess_sigma_spin = QSpinBox()
        self.detrending_checkbox = QWidget()
        self.detect_btn = QPushButton()


class _LombStub:
    def __init__(self) -> None:
        self.sampling_interval_spin = QSpinBox()


class _RayleighStub:
    def __init__(self) -> None:
        self.start_time_edit = QWidget()
        self.interval_spin = QSpinBox()
        self.interval_unit_combo = QComboBox()
        self.plot_btn = QPushButton()


@pytest.fixture
def patched_main_window(app, tmp_path: Path):
    sample_experiment = Experiment(name="Test Experiment", description="")
    user_experiments_dir = tmp_path / "users" / "alice" / "experiments"
    user_experiments_dir.mkdir(parents=True)

    mock_viewer = QWidget()
    mock_viewer.upload_btn = QPushButton()
    for attr in (
        "display_controls_panel",
        "cull_controls_panel",
    ):
        setattr(mock_viewer, attr, QWidget())
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
    mock_viewer.frameCullingChanged = Mock()
    mock_viewer.frameCullingChanged.connect = Mock()
    mock_viewer.set_filter_excluded = Mock()

    det = _DetStub()
    lomb = _LombStub()
    ray = _RayleighStub()

    mock_analysis = QWidget()
    mock_analysis.roi_plot_widget = Mock()
    mock_analysis.get_roi_plot_widget = Mock(return_value=mock_analysis.roi_plot_widget)
    mock_analysis.get_neuron_detection_widget = Mock(return_value=det)
    mock_analysis.get_neuron_trajectory_plot_widget = Mock(return_value=Mock())
    mock_analysis.get_lomb_scargle_widget = Mock(return_value=lomb)
    mock_analysis.get_rayleigh_plot_widget = Mock(return_value=ray)

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
        window = MainWindow(sample_experiment, user_experiments_dir=user_experiments_dir)
        window._det_stub = det  # type: ignore[attr-defined]
        window._lomb_stub = lomb  # type: ignore[attr-defined]
        yield window


@pytest.fixture
def patched_main_window_public(app, tmp_path: Path):
    sample_experiment = Experiment(name="Pub Exp", description="")
    public_exp_dir = tmp_path / "users" / PUBLIC_USER_NAME / "experiments"
    public_exp_dir.mkdir(parents=True)

    mock_viewer = QWidget()
    mock_viewer.upload_btn = QPushButton()
    mock_viewer.display_controls_panel = QWidget()
    mock_viewer.cull_controls_panel = QWidget()
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
    mock_viewer.frameCullingChanged = Mock()
    mock_viewer.frameCullingChanged.connect = Mock()
    mock_viewer.set_filter_excluded = Mock()

    det = _DetStub()
    lomb = _LombStub()
    ray = _RayleighStub()

    mock_analysis = QWidget()
    mock_analysis.roi_plot_widget = Mock()
    mock_analysis.get_roi_plot_widget = Mock(return_value=mock_analysis.roi_plot_widget)
    mock_analysis.get_neuron_detection_widget = Mock(return_value=det)
    mock_analysis.get_neuron_trajectory_plot_widget = Mock(return_value=Mock())
    mock_analysis.get_lomb_scargle_widget = Mock(return_value=lomb)
    mock_analysis.get_rayleigh_plot_widget = Mock(return_value=ray)

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
        window = MainWindow(sample_experiment, user_experiments_dir=public_exp_dir)
        window._det_stub = det  # type: ignore[attr-defined]
        window._lomb_stub = lomb  # type: ignore[attr-defined]
        yield window


def test_main_window_public_user_disables_save_and_locks_detection(
    patched_main_window_public,
) -> None:
    mw = patched_main_window_public
    assert mw._is_public_user_mode() is True
    assert mw._action_save is not None
    assert mw._action_save.isEnabled() is False
    det: _DetStub = mw._det_stub  # type: ignore[assignment]
    assert det.detect_btn.isEnabled() is False
    assert mw._lomb_stub.sampling_interval_spin.isEnabled() is False  # type: ignore[attr-defined]


def test_main_window_public_user_clicks_do_not_trigger_save_or_detection(patched_main_window_public, app) -> None:
    """Disabled Save action and Detect button must not emit when activated or clicked."""
    mw = patched_main_window_public
    save_slot = Mock()
    mw._action_save.triggered.connect(save_slot)
    mw._action_save.trigger()
    app.processEvents()
    save_slot.assert_not_called()

    det: _DetStub = mw._det_stub  # type: ignore[assignment]
    det_slot = Mock()
    det.detect_btn.clicked.connect(det_slot)
    QTest.mouseClick(det.detect_btn, Qt.MouseButton.LeftButton)
    app.processEvents()
    det_slot.assert_not_called()


def test_main_window_switch_from_public_to_normal_restores_save(patched_main_window_public, tmp_path: Path) -> None:
    mw = patched_main_window_public
    assert mw._action_save.isEnabled() is False
    lomb_before: _LombStub = mw._lomb_stub  # type: ignore[assignment]
    assert lomb_before.sampling_interval_spin.isEnabled() is False
    normal = tmp_path / "users" / "alice" / "experiments"
    normal.mkdir(parents=True)
    mw.user_experiments_dir = normal
    mw._sync_public_user_mode()
    assert mw._is_public_user_mode() is False
    assert mw._action_save.isEnabled() is True
    lomb_after: _LombStub = mw._lomb_stub  # type: ignore[assignment]
    assert lomb_after.sampling_interval_spin.isEnabled() is True


def test_main_window_switch_from_normal_to_public_applies_restrictions(patched_main_window, tmp_path: Path) -> None:
    mw = patched_main_window
    assert mw._is_public_user_mode() is False
    assert mw._action_save.isEnabled() is True
    pub = tmp_path / "users" / PUBLIC_USER_NAME / "experiments"
    pub.mkdir(parents=True)
    mw.user_experiments_dir = pub
    mw._sync_public_user_mode()
    assert mw._is_public_user_mode() is True
    assert mw._action_save.isEnabled() is False


def test_main_window_no_visibility_button_for_public_user(patched_main_window_public) -> None:
    assert patched_main_window_public._visibility_btn is None


def test_main_window_public_update_visibility_button_noops_without_widget(
    patched_main_window_public,
) -> None:
    patched_main_window_public._update_visibility_button()


def test_main_window_apply_public_restrictions_noop_when_not_public(patched_main_window) -> None:
    mw = patched_main_window
    mw._public_user_guards.clear()
    mw._apply_public_user_restrictions()
    assert mw._public_user_guards == []


def test_main_window_is_public_false_when_user_dir_unset(patched_main_window) -> None:
    mw = patched_main_window
    mw.user_experiments_dir = None
    assert mw._is_public_user_mode() is False


def test_main_window_public_save_save_as_stack_settings_crop_align_show_readonly(
    patched_main_window_public, app
) -> None:
    mw = patched_main_window_public
    with patch.object(QMessageBox, "information", return_value=None) as inf:
        mw._save()
        mw._save_as()
        mw._open_image_stack()
        mw._open_experiment_settings()
        mw._crop_stack_to_roi()
        mw._align_images()
    assert inf.call_count == 6


def test_main_window_public_autosave_and_save_roi_are_noops(patched_main_window_public) -> None:
    mw = patched_main_window_public
    mw.current_experiment_path = "/tmp/fake.nexp"
    with patch.object(mw.manager, "save_experiment", new=Mock()) as save:
        mw.autosave_experiment()
        save.assert_not_called()
    with patch.object(mw.manager, "save_experiment", new=Mock()) as save2:
        mw._save_roi_to_experiment("roi_1", ROI(x=0, y=0, width=2, height=2))
        save2.assert_not_called()


def test_main_window_public_close_event_does_not_save(patched_main_window_public, app, tmp_path: Path) -> None:
    mw = patched_main_window_public
    mw.current_experiment_path = str(tmp_path / "x.nexp")
    with patch.object(mw.manager, "save_experiment", new=Mock()) as save:
        with patch.object(QMessageBox, "question", return_value=QMessageBox.Yes):
            ev = QCloseEvent()
            mw.closeEvent(ev)
    save.assert_not_called()
    assert ev.isAccepted()


def test_main_window_public_close_experiment_calls_sync(
    patched_main_window_public, monkeypatch: pytest.MonkeyPatch
) -> None:
    mw = patched_main_window_public
    calls: list[int] = []
    monkeypatch.setattr("ui.public_user_dialog.sync_public_experiments", lambda: calls.append(1))

    class _StubStartup:
        def __init__(self, *a, **k) -> None:
            pass

        def exec(self) -> int:
            return QDialog.Rejected

    with patch.object(QMessageBox, "question", return_value=QMessageBox.Yes):
        with patch("ui.main_window.StartupDialog", _StubStartup):
            with patch("ui.main_window.QApplication.quit", new=Mock()):
                mw._close_experiment()
    assert calls == [1]


def test_main_window_visibility_button_for_normal_user(patched_main_window) -> None:
    assert patched_main_window._visibility_btn is not None


def test_toggle_experiment_visibility_public_sets_flag_and_saves(
    patched_main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_users_root: Path
) -> None:
    """Using fake repo root so sync_public_experiments does not touch the real project."""
    monkeypatch.setattr("ui.public_user_dialog._repo_root", lambda: fake_users_root)
    mw = patched_main_window
    mw.experiment.is_public = False
    nexp = tmp_path / "alice_exp.nexp"
    nexp.write_text(json.dumps(_minimal_nexp(name="Test Experiment", is_public=False)), encoding="utf-8")
    mw.current_experiment_path = str(nexp)
    sync_calls: list[int] = []

    def _track_sync() -> None:
        sync_calls.append(1)

    monkeypatch.setattr("ui.public_user_dialog.sync_public_experiments", _track_sync)

    with patch.object(mw.manager, "save_experiment", return_value=True) as mock_save:
        with patch.object(QMessageBox, "question", return_value=QMessageBox.Yes):
            mw._toggle_experiment_visibility()
    assert mw.experiment.is_public is True
    mock_save.assert_called_once()
    assert sync_calls == [1]

    with patch.object(mw.manager, "save_experiment", return_value=True) as mock_save2:
        mw._toggle_experiment_visibility()
    assert mw.experiment.is_public is False
    assert mock_save2.called


def test_experiment_is_public_round_trips_json() -> None:
    exp = Experiment(name="X", is_public=True)
    blob = json.loads(json.dumps(exp.to_json()))
    restored = Experiment.from_json(blob)
    assert restored.is_public is True
    exp2 = Experiment(name="Y", is_public=False)
    blob2 = json.loads(json.dumps(exp2.to_json()))
    assert Experiment.from_json(blob2).is_public is False
