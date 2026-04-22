"""Tests for settings load/save (``ui.app_settings``) using an isolated settings path."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Generator

import pytest

import ui.app_settings as app_settings


@pytest.fixture
def isolated_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Generator[Path, None, None]:
    config = tmp_path / "cfg"
    settings_file = config / "settings.json"
    monkeypatch.setattr(app_settings, "CONFIG_DIR", config)
    monkeypatch.setattr(app_settings, "SETTINGS_FILE", settings_file)
    yield settings_file


def test_load_settings_missing_file_uses_defaults(isolated_settings: Path) -> None:
    s = app_settings.load_settings()
    assert s["theme"] == "dark"
    assert s["enable_alignment_multiprocessing"] is False


def test_load_settings_merges_json_file(isolated_settings: Path) -> None:
    isolated_settings.parent.mkdir(parents=True, exist_ok=True)
    isolated_settings.write_text(json.dumps({"theme": "light", "roi_1_color": "#ff0000"}), encoding="utf-8")
    s = app_settings.load_settings()
    assert s["theme"] == "light"
    assert s["roi_1_color"] == "#ff0000"
    assert "roi_2_color" in s


def test_load_settings_corrupt_json_ignored(isolated_settings: Path) -> None:
    isolated_settings.parent.mkdir(parents=True, exist_ok=True)
    isolated_settings.write_text("{ not json", encoding="utf-8")
    s = app_settings.load_settings()
    assert s["theme"] == app_settings.DEFAULTS["theme"]


def test_save_and_roundtrip(isolated_settings: Path) -> None:
    app_settings.save_settings({"theme": "light", "enable_alignment_multiprocessing": True})
    assert isolated_settings.is_file()
    s = app_settings.load_settings()
    assert s["theme"] == "light"
    assert s["enable_alignment_multiprocessing"] is True


def test_get_theme_migrates_legacy_light_high_contrast(isolated_settings: Path) -> None:
    isolated_settings.parent.mkdir(parents=True, exist_ok=True)
    isolated_settings.write_text(json.dumps({"theme": "light", "high_contrast": True}), encoding="utf-8")
    theme = app_settings.get_theme()
    assert theme == "light_high_contrast"
    data = json.loads(isolated_settings.read_text(encoding="utf-8"))
    assert data.get("theme") == "light_high_contrast"
    assert "high_contrast" not in data


def test_get_theme_migrates_legacy_dark_high_contrast(isolated_settings: Path) -> None:
    isolated_settings.parent.mkdir(parents=True, exist_ok=True)
    isolated_settings.write_text(json.dumps({"theme": "dark", "high_contrast": True}), encoding="utf-8")
    theme = app_settings.get_theme()
    assert theme == "dark_high_contrast"
    data = json.loads(isolated_settings.read_text(encoding="utf-8"))
    assert data.get("theme") == "dark_high_contrast"
    assert "high_contrast" not in data


def test_set_theme_and_get_roi_colors(isolated_settings: Path) -> None:
    app_settings.set_theme("light")
    assert app_settings.get_theme() == "light"
    app_settings.set_roi_color("roi_1", "#abcdef")
    colors = app_settings.get_roi_colors()
    assert colors["roi_1"] == "#abcdef"


def test_avg_trajectory_color_roundtrip(isolated_settings: Path) -> None:
    app_settings.set_avg_trajectory_color("#112233")
    assert app_settings.get_avg_trajectory_color() == "#112233"
    app_settings.set_avg_trajectory_roi_color("roi_2", "#445566")
    assert app_settings.get_avg_trajectory_roi_colors()["roi_2"] == "#445566"


def test_set_roi_color_invalid_key(isolated_settings: Path) -> None:
    with pytest.raises(ValueError, match="Invalid roi_key"):
        app_settings.set_roi_color("roi_99", "#000000")


def test_set_avg_trajectory_roi_color_invalid_key(isolated_settings: Path) -> None:
    with pytest.raises(ValueError, match="Invalid roi_key"):
        app_settings.set_avg_trajectory_roi_color("roi_99", "#000000")


def test_alignment_mp_get_set(isolated_settings: Path) -> None:
    assert app_settings.get_enable_alignment_multiprocessing() is False
    app_settings.set_enable_alignment_multiprocessing(True)
    assert app_settings.get_enable_alignment_multiprocessing() is True


def test_peak_marker_color_roundtrip(isolated_settings: Path) -> None:
    default = app_settings.get_peak_marker_color()
    assert default == app_settings.DEFAULTS["peak_marker_color"]
    app_settings.set_peak_marker_color("#ff00ff")
    assert app_settings.get_peak_marker_color() == "#ff00ff"


def test_trough_marker_color_roundtrip(isolated_settings: Path) -> None:
    default = app_settings.get_trough_marker_color()
    assert default == app_settings.DEFAULTS["trough_marker_color"]
    app_settings.set_trough_marker_color("#00ffff")
    assert app_settings.get_trough_marker_color() == "#00ffff"
