"""Settings / Preferences dialog."""

from __future__ import annotations

from typing import Optional

from PySide6.QtGui import QColor, QPixmap
from PySide6.QtWidgets import (
    QColorDialog,
    QDialog,
    QDialogButtonBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from ui.app_settings import (
    get_avg_trajectory_color,
    get_roi_colors,
    get_theme,
    set_avg_trajectory_color,
    set_roi_color,
    set_theme,
)
from ui.styles import get_stylesheet

# Theme values shown in Preferences (single selection, same blue-box style)
THEME_VALUES = (
    ("dark", "Dark mode"),
    ("light", "Light mode"),
    ("dark_high_contrast", "Dark high contrast"),
    ("light_high_contrast", "Light high contrast"),
)

ROI_LABELS = {"roi_1": "ROI 1", "roi_2": "ROI 2"}


class SettingsDialog(QDialog):
    """Application settings dialog (theme, ROI colors, etc.)."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Preferences")
        self.setModal(True)
        self.setMinimumWidth(400)

        layout = QVBoxLayout(self)

        # Theme group: four options with same selection style (blue box)
        theme_group = QGroupBox("Appearance")
        theme_layout = QVBoxLayout()
        self.theme_radios = {}
        for value, label in THEME_VALUES:
            radio = QRadioButton(label)
            radio.setToolTip(
                "Increase contrast for text, borders, and backgrounds."
                if "high contrast" in label
                else None
            )
            self.theme_radios[value] = radio
            theme_layout.addWidget(radio)
        theme_group.setLayout(theme_layout)
        layout.addWidget(theme_group)

        # Set current selection
        current = get_theme()
        self.theme_radios.get(current, self.theme_radios["dark"]).setChecked(True)

        # ROI Colors group
        roi_group = QGroupBox("ROI Colors")
        roi_layout = QVBoxLayout()
        self._roi_swatches: dict[str, QLabel] = {}
        self._roi_colors: dict[str, str] = get_roi_colors()

        for roi_key, label_text in ROI_LABELS.items():
            row = QHBoxLayout()
            label = QLabel(label_text)
            swatch = QLabel()
            swatch.setFixedSize(24, 24)
            self._set_swatch_color(swatch, self._roi_colors[roi_key])
            self._roi_swatches[roi_key] = swatch
            change_btn = QPushButton("Change...")
            change_btn.setFixedWidth(90)
            change_btn.clicked.connect(
                lambda _checked=False, k=roi_key: self._pick_roi_color(k)
            )
            row.addWidget(label)
            row.addWidget(swatch)
            row.addWidget(change_btn)
            row.addStretch()
            roi_layout.addLayout(row)

        roi_group.setLayout(roi_layout)
        layout.addWidget(roi_group)

        # Average Trajectory Color group
        traj_group = QGroupBox("Graph Colors")
        traj_layout = QVBoxLayout()
        self._avg_traj_color = get_avg_trajectory_color()

        avg_row = QHBoxLayout()
        avg_row.addWidget(QLabel("Average Trajectory Line"))
        self._avg_traj_swatch = QLabel()
        self._avg_traj_swatch.setFixedSize(24, 24)
        self._set_swatch_color(self._avg_traj_swatch, self._avg_traj_color)
        avg_change_btn = QPushButton("Change...")
        avg_change_btn.setFixedWidth(90)
        avg_change_btn.clicked.connect(self._pick_avg_traj_color)
        avg_row.addWidget(self._avg_traj_swatch)
        avg_row.addWidget(avg_change_btn)
        avg_row.addStretch()
        traj_layout.addLayout(avg_row)

        traj_group.setLayout(traj_layout)
        layout.addWidget(traj_group)

        # Info label
        info = QLabel("Theme and color changes apply immediately.")
        info.setWordWrap(True)
        layout.addWidget(info)

        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._apply_and_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @staticmethod
    def _set_swatch_color(swatch: QLabel, hex_color: str) -> None:
        pix = QPixmap(24, 24)
        pix.fill(QColor(hex_color))
        swatch.setPixmap(pix)

    def _pick_roi_color(self, roi_key: str) -> None:
        current = QColor(self._roi_colors[roi_key])
        color = QColorDialog.getColor(
            current, self, f"Choose {ROI_LABELS[roi_key]} Color"
        )
        if color.isValid():
            hex_color = color.name()
            self._roi_colors[roi_key] = hex_color
            self._set_swatch_color(self._roi_swatches[roi_key], hex_color)

    def _pick_avg_traj_color(self) -> None:
        current = QColor(self._avg_traj_color)
        color = QColorDialog.getColor(current, self, "Choose Average Trajectory Color")
        if color.isValid():
            self._avg_traj_color = color.name()
            self._set_swatch_color(self._avg_traj_swatch, self._avg_traj_color)

    def _apply_and_accept(self) -> None:
        """Save theme + ROI colors + graph colors and reapply stylesheet."""
        theme = "dark"
        for value, radio in self.theme_radios.items():
            if radio.isChecked():
                theme = value
                break
        set_theme(theme)

        for roi_key, hex_color in self._roi_colors.items():
            set_roi_color(roi_key, hex_color)

        set_avg_trajectory_color(self._avg_traj_color)

        from PySide6.QtWidgets import QApplication

        app = QApplication.instance()
        if app is not None:
            app.setStyleSheet(get_stylesheet(theme))

        self.accept()
