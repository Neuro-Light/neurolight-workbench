"""Tests for theme palettes and stylesheet helpers (``ui.styles``)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from ui.styles import (
    THEME_DARK,
    THEME_DARK_HIGH_CONTRAST,
    THEME_LIGHT,
    THEME_LIGHT_HIGH_CONTRAST,
    _palette,
    get_mpl_theme,
    get_stylesheet,
)


@pytest.mark.parametrize(
    "theme",
    [
        THEME_DARK,
        THEME_LIGHT,
        THEME_DARK_HIGH_CONTRAST,
        THEME_LIGHT_HIGH_CONTRAST,
    ],
)
def test_palette_and_stylesheet_for_all_themes(theme: str) -> None:
    pal = _palette(theme)
    assert "bg" in pal and "text" in pal
    ss = get_stylesheet(theme)
    assert len(ss) > 500


def test_palette_unknown_falls_back_to_dark() -> None:
    pal = _palette("not-a-real-theme")
    assert pal == _palette(THEME_DARK)


@patch("ui.app_settings.get_roi_colors", return_value={"roi_1": "#111", "roi_2": "#222"})
@patch("ui.app_settings.get_avg_trajectory_color", return_value="#333")
@patch("ui.app_settings.get_avg_trajectory_roi_colors", return_value={"roi_1": "#444", "roi_2": "#555"})
@patch("ui.app_settings.get_peak_marker_color", return_value="#666")
@patch("ui.app_settings.get_trough_marker_color", return_value="#777")
def test_get_mpl_theme_dark_and_light_branches(
    *_mocks,
) -> None:
    dark = get_mpl_theme(THEME_DARK)
    light = get_mpl_theme(THEME_LIGHT)
    assert dark["good_color"] != light["good_color"]
    hc = get_mpl_theme(THEME_DARK_HIGH_CONTRAST)
    assert "neutral_color" in hc
