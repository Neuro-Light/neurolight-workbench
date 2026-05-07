"""Tests for help tooltip helpers and fallback help text."""

from __future__ import annotations

from unittest.mock import patch

from ui.help_content import get_help_text
from ui.help_widgets import HelpIconButton


def test_get_help_text_unknown_id_returns_fallback() -> None:
    text = get_help_text("__no_such_help_id__")
    assert "not available" in text.lower()


def test_help_icon_button_click_shows_tooltip() -> None:
    btn = HelpIconButton("workflow.overview", tooltip="Custom tip")
    with patch("ui.help_widgets.QToolTip.showText") as show:
        btn._show_tooltip_now()
        show.assert_called_once()
        _, tip, w = show.call_args[0]
        assert tip == "Custom tip"
        assert w is btn
