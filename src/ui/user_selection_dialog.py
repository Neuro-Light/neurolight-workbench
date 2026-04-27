"""User selection entry point (identity only; not security).

This module owns the first screen users see on launch: a full-size onboarding
screen where they pick an existing profile or create a new one. There are no
passwords, encryption, cloud sync, or roles — this is purely organizational.

Public API (unchanged from original, so callers need no updates):
  - UserSelectionDialog   — full-screen login/picker
  - UserAccountActionsDialog — "Switch User / Cancel" popover
  - current_user_button_text(name) — badge label helper
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

# ---------------------------------------------------------------------------
# Public helper (used by StartupDialog and MainWindow for the badge label)
# ---------------------------------------------------------------------------


def current_user_button_text(user_name: str) -> str:
    return f"Current User: {user_name}"


# ---------------------------------------------------------------------------
# Path helpers (unchanged logic)
# ---------------------------------------------------------------------------


def _repo_root() -> Path:
    # This file lives at src/ui/user_selection_dialog.py
    return Path(__file__).resolve().parents[2]


def _users_root() -> Path:
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


# ---------------------------------------------------------------------------
# Avatar color palette
# ---------------------------------------------------------------------------

_AVATAR_COLORS = [
    "#4A90E2",  # blue (primary)
    "#7B68EE",  # medium slate blue
    "#3CB371",  # medium sea green
    "#E07B54",  # terracotta
    "#E0A850",  # amber
    "#5BA3C9",  # steel blue
    "#C97DB5",  # mauve
    "#7BAAB0",  # cadet blue
]


def _avatar_color(name: str) -> str:
    return _AVATAR_COLORS[hash(name) % len(_AVATAR_COLORS)]


# ---------------------------------------------------------------------------
# Delete-with-typed-confirmation dialog
# ---------------------------------------------------------------------------


class _DeleteConfirmDialog(QDialog):
    """Requires the user to type the username before deletion is allowed."""

    def __init__(self, user_name: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Delete User")
        self.setModal(True)
        self.setMinimumWidth(420)
        self._user_name = user_name

        warning = QLabel(
            f"<b>This cannot be undone.</b> All experiments for <b>{user_name}</b> will be permanently deleted."
        )
        warning.setWordWrap(True)
        warning.setContentsMargins(0, 0, 0, 4)

        instruction = QLabel(f"Type <b>{user_name}</b> to confirm:")

        self._input = QLineEdit()
        self._input.setPlaceholderText(user_name)
        self._input.textChanged.connect(self._on_text_changed)
        self._input.returnPressed.connect(self._try_accept)

        self._confirm_btn = QPushButton("Delete")
        self._confirm_btn.setProperty("class", "primary")
        self._confirm_btn.setEnabled(False)
        self._confirm_btn.clicked.connect(self.accept)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_row.addWidget(cancel_btn)
        btn_row.addWidget(self._confirm_btn)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 20)
        layout.setSpacing(12)
        layout.addWidget(warning)
        layout.addWidget(instruction)
        layout.addWidget(self._input)
        layout.addSpacing(4)
        layout.addLayout(btn_row)

    def _on_text_changed(self, text: str) -> None:
        self._confirm_btn.setEnabled(text == self._user_name)

    def _try_accept(self) -> None:
        if self._input.text() == self._user_name:
            self.accept()


# ---------------------------------------------------------------------------
# Individual user card
# ---------------------------------------------------------------------------

_CARD_W = 152
_CARD_H = 178
_COLS = 4


class _UserCard(QFrame):
    """Clickable card: avatar circle, name, and a delete button."""

    login_requested: Signal = Signal(str)
    delete_requested: Signal = Signal(str)

    def __init__(
        self, user_name: str, grid_row: int, grid_col: int, grid_cols: int, parent: Optional[QWidget] = None
    ) -> None:
        super().__init__(parent)
        self._user_name = user_name
        self._grid_row = grid_row
        self._grid_col = grid_col
        self._grid_cols = grid_cols
        self._hovered = False
        self._focused = False

        self.setObjectName("userCard")
        self.setFixedSize(_CARD_W, _CARD_H)
        self.setCursor(Qt.PointingHandCursor)
        self.setFocusPolicy(Qt.StrongFocus)
        self._update_border()

        # Avatar — first letter(s) of each word, max 2 chars
        initials = "".join(w[0].upper() for w in user_name.split()[:2]) or user_name[:2].upper()
        color = _avatar_color(user_name)

        avatar = QLabel(initials)
        avatar.setFixedSize(68, 68)
        avatar.setAlignment(Qt.AlignCenter)
        avatar.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        avatar.setStyleSheet(
            f"background-color: {color}; border-radius: 34px;"
            " color: #FFFFFF; font-size: 22px; font-weight: 700;"
            " border: none;"
        )

        name_lbl = QLabel(user_name)
        name_lbl.setAlignment(Qt.AlignCenter)
        name_lbl.setWordWrap(True)
        name_lbl.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        name_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        # Delete button overlaid at top-right
        self._del_btn = QToolButton(self)
        self._del_btn.setText("✕")
        self._del_btn.setFixedSize(22, 22)
        self._del_btn.setToolTip(f"Delete {user_name}")
        self._del_btn.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self._del_btn.move(_CARD_W - 26, 4)
        self._del_btn.clicked.connect(lambda: self.delete_requested.emit(self._user_name))

        inner = QVBoxLayout(self)
        inner.setContentsMargins(8, 16, 8, 12)
        inner.setSpacing(10)
        inner.addStretch()
        inner.addWidget(avatar, 0, Qt.AlignCenter)
        inner.addWidget(name_lbl, 0, Qt.AlignCenter)
        inner.addStretch()

    # --- Border state management ---

    def _update_border(self) -> None:
        highlighted = self._hovered or self._focused
        color = "#4A90E2" if highlighted else "transparent"
        # Scoped to objectName so child widgets are not affected
        self.setStyleSheet(f"QFrame#userCard {{ border: 2px solid {color}; border-radius: 12px; }}")

    # --- Mouse & focus events ---

    def enterEvent(self, event) -> None:
        self._hovered = True
        self._update_border()
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:
        self._hovered = False
        self._update_border()
        super().leaveEvent(event)

    def focusInEvent(self, event) -> None:
        self._focused = True
        self._update_border()
        super().focusInEvent(event)

    def focusOutEvent(self, event) -> None:
        self._focused = False
        self._update_border()
        super().focusOutEvent(event)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self.login_requested.emit(self._user_name)
        super().mousePressEvent(event)

    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key in (Qt.Key_Return, Qt.Key_Enter, Qt.Key_Space):
            self.login_requested.emit(self._user_name)
            return
        # Arrow-key navigation within the grid
        if key in (Qt.Key_Right, Qt.Key_Left, Qt.Key_Up, Qt.Key_Down):
            dr, dc = {
                Qt.Key_Right: (0, 1),
                Qt.Key_Left: (0, -1),
                Qt.Key_Down: (1, 0),
                Qt.Key_Up: (-1, 0),
            }[key]
            grid: QGridLayout = self.parentWidget().layout()
            item = grid.itemAtPosition(self._grid_row + dr, self._grid_col + dc)
            if item and item.widget():
                item.widget().setFocus()
            return
        super().keyPressEvent(event)


# ---------------------------------------------------------------------------
# Improved UserAccountActionsDialog (same public interface, better look)
# ---------------------------------------------------------------------------


class UserAccountActionsDialog(QDialog):
    """Clean modal: shows signed-in user, offers Switch User or Cancel."""

    def __init__(self, user_name: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Account")
        self.setModal(True)
        self.setMinimumWidth(280)
        self.switch_user_requested = False

        signed_in = QLabel("Signed in as")
        signed_in.setAlignment(Qt.AlignCenter)

        name_lbl = QLabel(f"<b>{user_name}</b>")
        name_lbl.setAlignment(Qt.AlignCenter)
        name_lbl.setProperty("class", "section-heading")

        divider = QFrame()
        divider.setFrameShape(QFrame.HLine)

        switch_btn = QPushButton("Switch User")
        switch_btn.setProperty("class", "primary")
        switch_btn.clicked.connect(self._on_switch)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 20)
        layout.setSpacing(10)
        layout.addWidget(signed_in)
        layout.addWidget(name_lbl)
        layout.addWidget(divider)
        layout.addSpacing(4)
        layout.addWidget(switch_btn)
        layout.addWidget(cancel_btn)

    def _on_switch(self) -> None:
        self.switch_user_requested = True
        self.accept()


# ---------------------------------------------------------------------------
# Main login screen
# ---------------------------------------------------------------------------


class UserSelectionDialog(QDialog):
    """Full-size 'Who's using NeuroLight?' onboarding screen.

    Replaces the original chain of small dialogs with a single cohesive screen.
    Public interface is identical to the original so callers need no changes:
      - .exec() → QDialog.Accepted / Rejected
      - .selected_user: Optional[str]
      - .selected_user_experiments_dir: Optional[Path]
    """

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("NeuroLight")
        self.setMinimumSize(680, 500)
        self.setModal(True)

        self.selected_user: Optional[str] = None
        self.selected_user_experiments_dir: Optional[Path] = None

        self._build_ui()

    # --- UI construction ---

    def _build_ui(self) -> None:
        title = QLabel("Who's using NeuroLight?")
        title.setAlignment(Qt.AlignCenter)
        title.setProperty("class", "dialog-title")

        subtitle = QLabel("Select your profile to continue")
        subtitle.setAlignment(Qt.AlignCenter)

        # Scrollable card grid
        self._cards_widget = QWidget()
        self._cards_layout = QGridLayout(self._cards_widget)
        self._cards_layout.setSpacing(16)
        self._cards_layout.setContentsMargins(20, 12, 20, 12)
        self._cards_layout.setAlignment(Qt.AlignTop | Qt.AlignHCenter)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._cards_widget)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        # Inline new-user form (hidden until "Add User" is clicked)
        self._new_user_frame = QFrame()
        self._new_user_frame.setObjectName("newUserFormRow")
        self._new_user_frame.setVisible(False)

        form_lbl = QLabel("New user name:")
        self._name_input = QLineEdit()
        self._name_input.setPlaceholderText("Enter name…")
        self._name_input.setMinimumWidth(220)
        self._name_input.setMaximumWidth(340)
        self._name_input.returnPressed.connect(self._confirm_new_user)

        create_btn = QPushButton("Create")
        create_btn.setProperty("class", "primary")
        create_btn.clicked.connect(self._confirm_new_user)

        cancel_inline_btn = QPushButton("Cancel")
        cancel_inline_btn.clicked.connect(self._hide_new_user_form)

        form_row = QHBoxLayout(self._new_user_frame)
        form_row.setContentsMargins(0, 4, 0, 4)
        form_row.setSpacing(10)
        form_row.addStretch()
        form_row.addWidget(form_lbl)
        form_row.addWidget(self._name_input)
        form_row.addWidget(create_btn)
        form_row.addWidget(cancel_inline_btn)
        form_row.addStretch()

        # "Add User" button row
        self._add_btn = QPushButton("+ Add User")
        self._add_btn.setProperty("class", "tab-action")
        self._add_btn.setFixedWidth(130)
        self._add_btn.clicked.connect(self._show_new_user_form)

        add_row = QHBoxLayout()
        add_row.addStretch()
        add_row.addWidget(self._add_btn)
        add_row.addStretch()

        root = QVBoxLayout(self)
        root.setContentsMargins(36, 32, 36, 28)
        root.setSpacing(10)
        root.addWidget(title)
        root.addWidget(subtitle)
        root.addSpacing(6)
        root.addWidget(scroll, 1)
        root.addWidget(self._new_user_frame)
        root.addLayout(add_row)

        self._refresh_cards()

    # --- Card grid ---

    def _refresh_cards(self) -> None:
        # Remove all existing card widgets
        while self._cards_layout.count():
            item = self._cards_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        users = _list_existing_users()

        if not users:
            placeholder = QLabel("No profiles yet — add one below.")
            placeholder.setAlignment(Qt.AlignCenter)
            placeholder.setObjectName("emptyPlaceholder")
            self._cards_layout.addWidget(placeholder, 0, 0, Qt.AlignCenter)
            return

        cols = min(_COLS, len(users))
        for i, name in enumerate(users):
            row, col = divmod(i, cols)
            card = _UserCard(name, grid_row=row, grid_col=col, grid_cols=cols)
            card.login_requested.connect(self._on_user_selected)
            card.delete_requested.connect(self._on_delete_user)
            self._cards_layout.addWidget(card, row, col, Qt.AlignCenter)

        # Give focus to the first card for keyboard users
        first_item = self._cards_layout.itemAt(0)
        if first_item and first_item.widget():
            first_item.widget().setFocus()

    # --- Inline new-user form ---

    def _show_new_user_form(self) -> None:
        self._add_btn.setVisible(False)
        self._name_input.clear()
        self._new_user_frame.setVisible(True)
        self._name_input.setFocus()

    def _hide_new_user_form(self) -> None:
        self._new_user_frame.setVisible(False)
        self._add_btn.setVisible(True)

    def _confirm_new_user(self) -> None:
        name = self._name_input.text().strip()
        if not name:
            self._name_input.setFocus()
            return
        if name in {".", ".."}:
            QMessageBox.warning(self, "Invalid Name", 'Name cannot be "." or "..".')
            return
        if any(ch in name for ch in '\\/:*?"<>|'):
            QMessageBox.warning(
                self,
                "Invalid Name",
                'Name cannot contain: \\ / : * ? " < > |',
            )
            return

        exp_dir = _experiments_dir_for_user(name)
        try:
            exp_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            QMessageBox.information(self, "Create User", f'User "{name}" already exists.')
            return
        except OSError as e:
            QMessageBox.critical(self, "Create User", f"Could not create user folder:\n{e}")
            return

        self._hide_new_user_form()
        self._on_user_selected(name)

    # --- Selection and deletion ---

    def _on_user_selected(self, user_name: str) -> None:
        self.selected_user = user_name
        self.selected_user_experiments_dir = _experiments_dir_for_user(user_name)
        self.accept()

    def _on_delete_user(self, user_name: str) -> None:
        dlg = _DeleteConfirmDialog(user_name, self)
        if dlg.exec() != QDialog.Accepted:
            return
        user_dir = _users_root() / user_name
        try:
            shutil.rmtree(user_dir)
        except OSError as e:
            QMessageBox.critical(self, "Delete User", f"Could not delete:\n{e}")
            return
        self._refresh_cards()
