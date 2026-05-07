"""Tests for ``main.main()`` launcher behavior (Public User sync on login)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PySide6.QtWidgets import QDialog

from core.experiment_manager import Experiment


class _FakeApp:
    def __init__(self, *a, **k) -> None:
        pass

    def setWindowIcon(self, *a, **k) -> None:
        pass

    def setStyleSheet(self, *a, **k) -> None:
        pass

    def exec(self) -> int:
        return 0


def test_main_calls_sync_when_public_user_logs_in(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    sync_calls: list[int] = []
    monkeypatch.setattr("main.sync_public_experiments", lambda: sync_calls.append(1))
    monkeypatch.setattr("main.ensure_public_user_exists", lambda: None)

    pub_exp = tmp_path / "users" / "Public" / "experiments"
    pub_exp.mkdir(parents=True)

    class _UserDlg:
        selected_user_experiments_dir = pub_exp
        selected_user = "Public"

        def exec(self) -> int:
            return QDialog.Accepted

    exp_path = str(tmp_path / "e.nexp")

    class _StartupDlg:
        Accepted = QDialog.Accepted

        def __init__(self, *a, **k) -> None:
            self.experiment = Experiment(name="E")
            self.experiment_path = exp_path

        def exec(self) -> int:
            return QDialog.Accepted

    mw = MagicMock()
    timer_instance = MagicMock()

    monkeypatch.setattr("main.QApplication", _FakeApp)
    monkeypatch.setattr("main.UserSelectionDialog", lambda *a, **k: _UserDlg())
    monkeypatch.setattr("main.StartupDialog", _StartupDlg)
    monkeypatch.setattr("main.MainWindow", lambda *a, **k: mw)
    monkeypatch.setattr("main.get_theme", lambda: "light")
    monkeypatch.setattr("main.get_stylesheet", lambda _theme: "")
    monkeypatch.setattr("main.QTimer", lambda: timer_instance)

    import main as main_mod

    assert main_mod.main() == 0
    assert sync_calls == [1]
    mw.show.assert_called_once()
    timer_instance.start.assert_called_once()


def test_main_does_not_sync_for_non_public_user(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    sync_calls: list[int] = []
    monkeypatch.setattr("main.sync_public_experiments", lambda: sync_calls.append(1))
    monkeypatch.setattr("main.ensure_public_user_exists", lambda: None)

    alice_exp = tmp_path / "users" / "alice" / "experiments"
    alice_exp.mkdir(parents=True)

    class _UserDlg:
        selected_user_experiments_dir = alice_exp
        selected_user = "alice"

        def exec(self) -> int:
            return QDialog.Accepted

    exp_path = str(tmp_path / "e.nexp")

    class _StartupDlg:
        Accepted = QDialog.Accepted

        def __init__(self, *a, **k) -> None:
            self.experiment = Experiment(name="E")
            self.experiment_path = exp_path

        def exec(self) -> int:
            return QDialog.Accepted

    mw = MagicMock()

    monkeypatch.setattr("main.QApplication", _FakeApp)
    monkeypatch.setattr("main.UserSelectionDialog", lambda *a, **k: _UserDlg())
    monkeypatch.setattr("main.StartupDialog", _StartupDlg)
    monkeypatch.setattr("main.MainWindow", lambda *a, **k: mw)
    monkeypatch.setattr("main.get_theme", lambda: "light")
    monkeypatch.setattr("main.get_stylesheet", lambda _theme: "")
    monkeypatch.setattr("main.QTimer", lambda: MagicMock())

    import main as main_mod

    assert main_mod.main() == 0
    assert sync_calls == []


def test_main_returns_zero_when_user_picks_cancel(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("main.ensure_public_user_exists", lambda: None)

    class _UserDlg:
        def exec(self) -> int:
            return QDialog.Rejected

    main_cls = MagicMock()
    monkeypatch.setattr("main.QApplication", _FakeApp)
    monkeypatch.setattr("main.UserSelectionDialog", lambda *a, **k: _UserDlg())
    monkeypatch.setattr("main.MainWindow", main_cls)
    monkeypatch.setattr("main.get_theme", lambda: "light")
    monkeypatch.setattr("main.get_stylesheet", lambda _theme: "")

    import main as main_mod

    assert main_mod.main() == 0
    main_cls.assert_not_called()


def test_main_swallows_set_current_experiment_path_errors(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Startup may pass a path helper; failures there must not abort launch."""
    monkeypatch.setattr("main.ensure_public_user_exists", lambda: None)
    monkeypatch.setattr("main.sync_public_experiments", lambda: None)

    alice_exp = tmp_path / "users" / "alice" / "experiments"
    alice_exp.mkdir(parents=True)

    class _UserDlg:
        selected_user_experiments_dir = alice_exp

        def exec(self) -> int:
            return QDialog.Accepted

    exp_path = str(tmp_path / "e.nexp")

    class _StartupDlg:
        Accepted = QDialog.Accepted

        def __init__(self, *a, **k) -> None:
            self.experiment = Experiment(name="E")
            self.experiment_path = exp_path

        def exec(self) -> int:
            return QDialog.Accepted

    mw = MagicMock()
    mw.set_current_experiment_path.side_effect = RuntimeError("path helper unavailable")

    monkeypatch.setattr("main.QApplication", _FakeApp)
    monkeypatch.setattr("main.UserSelectionDialog", lambda *a, **k: _UserDlg())
    monkeypatch.setattr("main.StartupDialog", _StartupDlg)
    monkeypatch.setattr("main.MainWindow", lambda *a, **k: mw)
    monkeypatch.setattr("main.get_theme", lambda: "light")
    monkeypatch.setattr("main.get_stylesheet", lambda _theme: "")
    monkeypatch.setattr("main.QTimer", lambda: MagicMock())

    import main as main_mod

    assert main_mod.main() == 0
    mw.set_current_experiment_path.assert_called_once()
    mw.show.assert_called_once()
