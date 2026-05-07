"""Public User management.

The Public User is a special permanent read-only account that can view
experiments explicitly marked public by their owners.  It cannot be deleted
from the user-selection screen and has no write access to experiments.

Public API:
  - PUBLIC_USER_NAME            — constant name for the public account
  - ReadOnlyGuard               — event filter that keeps a widget permanently disabled
  - ensure_public_user_exists() — create the account directory if absent
  - is_public_user(name)        — True when name == PUBLIC_USER_NAME
  - sync_public_experiments()   — scan all users and copy public .nexp files to Public folder
  - register_public_experiment(path, name) — expose an experiment to the Public User
  - unregister_public_experiment(path)     — hide an experiment from the Public User
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from PySide6.QtCore import QEvent, QObject
from PySide6.QtWidgets import QWidget

PUBLIC_USER_NAME = "Public"


class ReadOnlyGuard(QObject):
    """Event filter that permanently keeps a widget disabled.

    Install this on any button that internal widget code might re-enable
    during a replot or data-load cycle.  Safe to install multiple times on
    the same widget — duplicate installations are harmless.
    """

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        # If something tried to enable the widget, immediately revert it.
        if event.type() == QEvent.Type.EnabledChange and obj.isEnabled():
            obj.setEnabled(False)
        return False

    @classmethod
    def lock(cls, widget: QWidget) -> ReadOnlyGuard:
        """Disable *widget* and install the guard.  Returns the guard instance."""
        guard = cls(widget)
        widget.installEventFilter(guard)
        widget.setEnabled(False)
        return guard


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def _repo_root() -> Path:
    # This file lives at src/ui/public_user_dialog.py
    return Path(__file__).resolve().parents[2]


def _public_user_dir() -> Path:
    return _repo_root() / "users" / PUBLIC_USER_NAME


def _public_experiments_dir() -> Path:
    return _public_user_dir() / "experiments"


def _public_recent_file() -> Path:
    return _public_user_dir() / "recent_experiments.json"


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


def ensure_public_user_exists() -> Path:
    """Create the Public user directory structure if it does not exist.

    Returns the experiments directory path so callers can pass it to
    StartupDialog directly.
    """
    exp_dir = _public_experiments_dir()
    try:
        exp_dir.mkdir(parents=True, exist_ok=True)
        recent = _public_recent_file()
        if not recent.exists():
            recent.write_text(json.dumps({"recent": []}, indent=2), encoding="utf-8")
    except OSError:
        # Best-effort setup: ignore filesystem errors and let callers continue.
        # Public mode may be unavailable if the directory cannot be created.
        pass
    return exp_dir


def is_public_user(name: str) -> bool:
    """Return True when *name* is the Public User account."""
    return name == PUBLIC_USER_NAME


def sync_public_experiments() -> None:
    """Rebuild the Public User's experiment list from all users' public experiments.

    Walks every non-Public user directory under ``users/``, reads each ``.nexp``
    file, and copies those flagged ``is_public=True`` into the Public User's
    experiments folder using a collision-free name (``<owner>__<stem>.nexp``).
    The Public User's ``recent_experiments.json`` is then rewritten to exactly
    this set.  Stale copies from previously-public experiments are deleted.
    """
    pub_exp_dir = ensure_public_user_exists()
    users_dir = _repo_root() / "users"
    if not users_dir.is_dir():
        return

    recent_entries: list[dict] = []
    keep_names: set[str] = set()

    for user_dir in sorted(users_dir.iterdir()):
        if not user_dir.is_dir() or user_dir.name == PUBLIC_USER_NAME:
            continue
        exp_dir = user_dir / "experiments"
        if not exp_dir.is_dir():
            continue
        for nexp_file in sorted(exp_dir.rglob("*.nexp")):
            try:
                with open(nexp_file, encoding="utf-8") as fh:
                    data = json.load(fh)
            except Exception:
                continue
            exp = data.get("experiment") or {}
            if not isinstance(exp, dict):
                exp = {}
            if not exp.get("is_public", False):
                continue

            # Collision-free filename: <owner>__<original stem>.nexp
            dest_name = f"{user_dir.name}__{nexp_file.stem}.nexp"
            dest = pub_exp_dir / dest_name
            try:
                shutil.copy2(str(nexp_file), str(dest))
            except OSError:
                continue

            keep_names.add(dest_name)
            recent_entries.append(
                {
                    "path": str(dest.resolve()),
                    "name": exp.get("name", nexp_file.stem),
                    "owner": user_dir.name,
                    "last_opened": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                }
            )

    # Rewrite the recent list with exactly the current public set
    try:
        with open(_public_recent_file(), "w", encoding="utf-8") as fh:
            json.dump({"recent": recent_entries}, fh, indent=2)
    except OSError:
        pass

    # Remove stale copies that are no longer public
    for f in pub_exp_dir.iterdir():
        if f.suffix == ".nexp" and f.name not in keep_names:
            try:
                f.unlink()
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Experiment registration
# ---------------------------------------------------------------------------


def register_public_experiment(experiment_path: str, experiment_name: str | None = None) -> None:
    """Add *experiment_path* to the Public user's accessible experiment list.

    Safe to call multiple times for the same path (de-duplicates).
    """
    ensure_public_user_exists()
    recent_file = _public_recent_file()
    resolved = str(Path(experiment_path).resolve())

    entry = {
        "path": resolved,
        "name": experiment_name or Path(resolved).stem,
        "last_opened": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    try:
        with open(recent_file, encoding="utf-8") as f:
            data = json.load(f) or {"recent": []}
    except Exception:
        data = {"recent": []}

    # Remove any stale duplicate then prepend the fresh entry.
    data["recent"] = [e for e in data.get("recent", []) if e.get("path") != resolved]
    data["recent"].insert(0, entry)

    try:
        with open(recent_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except OSError:
        pass


def unregister_public_experiment(experiment_path: str) -> None:
    """Remove *experiment_path* from the Public user's accessible list."""
    recent_file = _public_recent_file()
    if not recent_file.exists():
        return

    resolved = str(Path(experiment_path).resolve())

    try:
        with open(recent_file, encoding="utf-8") as f:
            data = json.load(f) or {"recent": []}
    except Exception:
        data = {"recent": []}

    data["recent"] = [e for e in data.get("recent", []) if e.get("path") != resolved]

    try:
        with open(recent_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except OSError:
        pass
