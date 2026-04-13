"""User selection entry point (identity only; not security).

This dialog is the first UX step after launch. It isolates each researcher's
workspace: experiments, settings, and data stay separate for organizational
clarity. There are no passwords, encryption, cloud sync, or roles—those are
out of scope.

**Intended behavior (this module)**

1. **When it appears**  
   Shown on every application start, before the experiment manager / workbench.
   Other code will orchestrate timing; this file owns the screen and its
   interactions.

2. **Primary surface**  
   Present existing users as a clear, scannable list of names. The user picks
   one row to enter that workspace.

3. **"Load Existing User"**  
   Reveal or focus the list of persisted profiles (read from local storage;
   path/format will live elsewhere). Selecting an entry confirms that user for
   this session. The dialog should finish in an *accepted* state only when a
   valid user has been chosen, and expose that choice (e.g. user id / profile
   key) for callers.

4. **"Create New User"**  
   Start a short flow: prompt for a case-sensitive name. When confirmed, create
   a corresponding folder structure at:

       `<project-root>/users/<user>/experiments/` (outside `src/`)

   After creation, either select the new user automatically or return to the
   list with the new entry visible.

5. **Optional product behavior** (may be implemented here or via settings)  
   Remember the last selected user for faster re-entry, while still showing
   this screen on launch so identity is explicit.

6. **What this dialog does *not* do**  
   It does not open the experiment manager, load `.nexp` files, or enforce
   access control. It only establishes *which local workspace user* is active;
   downstream code uses that to resolve per-user experiment directories and
   config.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


def _repo_root() -> Path:
    # This file lives at src/ui/user_selection_dialog.py
    return Path(__file__).resolve().parents[2]


def _users_root() -> Path:
    # <project-root>/users/<user>/experiments — next to src/, not inside it
    return _repo_root() / "users"


def _experiments_dir_for_user(user_name: str) -> Path:
    return _users_root() / user_name / "experiments"


def _list_existing_users() -> list[str]:
    root = _users_root()
    if not root.exists():
        return []
    try:
        return sorted([p.name for p in root.iterdir() if p.is_dir()])
    except OSError:
        return []


class _UserPickerDialog(QDialog):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Select User")
        self.setModal(True)
        self.setMinimumWidth(420)

        self.selected_user: Optional[str] = None

        heading = QLabel("Select an existing user")
        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QAbstractItemView.SingleSelection)
        self.list_widget.itemDoubleClicked.connect(self._accept_current)

        for name in _list_existing_users():
            self.list_widget.addItem(QListWidgetItem(name))

        select_btn = QPushButton("Select")
        cancel_btn = QPushButton("Cancel")
        select_btn.clicked.connect(self._accept_current)
        cancel_btn.clicked.connect(self.reject)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_row.addWidget(cancel_btn)
        btn_row.addWidget(select_btn)

        layout = QVBoxLayout()
        layout.addWidget(heading)
        layout.addWidget(self.list_widget)
        layout.addLayout(btn_row)
        self.setLayout(layout)

    def _accept_current(self) -> None:
        item = self.list_widget.currentItem()
        if item is None:
            QMessageBox.information(self, "Select User", "Please select a user.")
            return
        self.selected_user = item.text()
        self.accept()


class UserSelectionDialog(QDialog):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("User Selection")

        # Exposed outputs for the caller (wired later in main.py):
        self.selected_user: Optional[str] = None
        self.selected_user_experiments_dir: Optional[Path] = None

        load_btn = QPushButton("Load Existing User")
        create_btn = QPushButton("Create New User")
        load_btn.clicked.connect(self._load_existing_user)
        create_btn.clicked.connect(self._create_new_user)

        layout = QVBoxLayout()
        layout.addWidget(load_btn)
        layout.addWidget(create_btn)
        layout.addStretch()
        self.setLayout(layout)

    def _choose_user(self, user_name: str) -> None:
        self.selected_user = user_name
        self.selected_user_experiments_dir = _experiments_dir_for_user(user_name)
        self.accept()

    def _load_existing_user(self) -> None:
        if not _list_existing_users():
            QMessageBox.warning(
                self,
                "No users",
                "There are no created users yet. Please create one using Create New User.",
            )
            return
        picker = _UserPickerDialog(self)
        if picker.exec() != QDialog.Accepted or not picker.selected_user:
            return
        self._choose_user(picker.selected_user)

    def _create_new_user(self) -> None:
        name, ok = QInputDialog.getText(self, "Create New User", "User name (case-sensitive):")
        if not ok:
            return
        name = name.strip()
        if not name:
            QMessageBox.warning(self, "Create New User", "Name cannot be empty.")
            return
        if any(ch in name for ch in '\\/:*?"<>|'):
            QMessageBox.warning(
                self,
                "Create New User",
                'Name cannot contain any of: \\ / : * ? " < > |',
            )
            return

        exp_dir = _experiments_dir_for_user(name)
        try:
            exp_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            # Note: Windows filesystems are typically case-insensitive, so "Alice" and "alice"
            # may collide even if treated as distinct strings in Python.
            QMessageBox.information(self, "Create New User", f'User "{name}" already exists.')
            return
        except OSError as e:
            QMessageBox.critical(self, "Create New User", f"Could not create user folder:\n{e}")
            return

        self._choose_user(name)
