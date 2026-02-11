"""Application settings persistence (theme, etc.)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

CONFIG_DIR = Path.home() / ".neurolight"
SETTINGS_FILE = CONFIG_DIR / "settings.json"

DEFAULTS: Dict[str, Any] = {
    "theme": "dark",
}


def load_settings() -> Dict[str, Any]:
    """Load application settings from disk."""
    settings = DEFAULTS.copy()
    try:
        if SETTINGS_FILE.exists():
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    settings.update(data)
    except (json.JSONDecodeError, OSError):
        pass
    return settings


def save_settings(settings: Dict[str, Any]) -> None:
    """Save application settings to disk."""
    try:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2)
    except OSError:
        pass


def get_theme() -> str:
    """Get the current theme: dark, light, dark_high_contrast, or light_high_contrast."""
    settings = load_settings()
    theme = settings.get("theme", "dark")
    high_contrast = bool(settings.get("high_contrast", False))
    # Migrate legacy theme + high_contrast into single theme value
    if theme == "dark" and high_contrast:
        theme = "dark_high_contrast"
        settings["theme"] = theme
        if "high_contrast" in settings:
            del settings["high_contrast"]
        save_settings(settings)
    elif theme == "light" and high_contrast:
        theme = "light_high_contrast"
        settings["theme"] = theme
        if "high_contrast" in settings:
            del settings["high_contrast"]
        save_settings(settings)
    return theme


def set_theme(theme: str) -> None:
    """Save the theme preference (dark, light, dark_high_contrast, or light_high_contrast)."""
    settings = load_settings()
    settings["theme"] = theme
    if "high_contrast" in settings:
        del settings["high_contrast"]
    save_settings(settings)
