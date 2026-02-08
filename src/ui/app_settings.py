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
    """Get the current theme (dark or light)."""
    return load_settings().get("theme", "dark")


def set_theme(theme: str) -> None:
    """Save the theme preference."""
    settings = load_settings()
    settings["theme"] = theme
    save_settings(settings)
