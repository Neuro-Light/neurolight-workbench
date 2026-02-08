"""Settings / Preferences dialog."""

from __future__ import annotations

from typing import Optional

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from ui.app_settings import get_theme, set_theme
from ui.styles import get_stylesheet


class SettingsDialog(QDialog):
    """Application settings dialog (theme, etc.)."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Preferences")
        self.setModal(True)
        self.setMinimumWidth(400)

        layout = QVBoxLayout(self)

        # Theme group
        theme_group = QGroupBox("Appearance")
        theme_layout = QVBoxLayout()
        self.dark_radio = QRadioButton("Dark mode")
        self.light_radio = QRadioButton("Light mode")
        theme_layout.addWidget(self.dark_radio)
        theme_layout.addWidget(self.light_radio)
        theme_group.setLayout(theme_layout)
        layout.addWidget(theme_group)

        # Set current selection
        current = get_theme()
        if current == "light":
            self.light_radio.setChecked(True)
        else:
            self.dark_radio.setChecked(True)

        # Info label
        info = QLabel("Theme changes apply immediately.")
        info.setWordWrap(True)
        layout.addWidget(info)

        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self._apply_and_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _apply_and_accept(self) -> None:
        """Save theme and reapply stylesheet."""
        theme = "light" if self.light_radio.isChecked() else "dark"
        set_theme(theme)

        from PySide6.QtWidgets import QApplication
        QApplication.instance().setStyleSheet(get_stylesheet(theme))

        self.accept()
