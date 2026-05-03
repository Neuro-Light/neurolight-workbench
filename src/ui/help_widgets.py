from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QFontMetrics
from PySide6.QtWidgets import QToolButton, QToolTip, QWidget

from ui.help_content import get_help_text


class HelpIconButton(QToolButton):
    """Small non-obstructive help icon that shows a tooltip/popover on hover/click."""

    def __init__(
        self,
        help_id: str,
        *,
        tooltip: Optional[str] = None,
        parent: Optional[QWidget] = None,
        accessible_name: Optional[str] = None,
    ) -> None:
        super().__init__(parent)
        self._help_id = help_id
        self._tooltip = tooltip if tooltip is not None else get_help_text(help_id)

        # Visuals: small icon-like button that doesn't steal attention.
        self.setText("ⓘ")
        self.setAutoRaise(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setProperty("class", "help-icon")
        self.setToolTip(self._tooltip)

        name = accessible_name or f"Help: {help_id}"
        self.setAccessibleName(name)
        self.setAccessibleDescription(self._tooltip)

        # Keep footprint tiny.
        fm = QFontMetrics(self.font())
        size = max(18, fm.height() + 4)
        self.setFixedSize(size, size)

        # Clicking should also show the tooltip (useful for touch/trackpad users).
        self.clicked.connect(self._show_tooltip_now)

    def _show_tooltip_now(self) -> None:
        # Show tooltip slightly below the icon to avoid covering labels.
        pos = self.mapToGlobal(QPoint(self.width() // 2, self.height()))
        QToolTip.showText(pos, self._tooltip, self)
