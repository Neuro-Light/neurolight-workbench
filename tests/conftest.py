"""
Pytest configuration and session-wide fixtures.

Qt-backend segfault prevention
-------------------------------
lomb_scargle_plot.py imports FigureCanvasQTAgg / NavigationToolbar2QT directly
from matplotlib.backends.backend_qtagg at module level.  Importing that module
triggers a two-stage crash:

  Stage 1  backend_qt.py:14 pulls in matplotlib.backends.qt_editor.figureoptions
           → qt_editor/_formlayout.py:59 defines class ColorButton(QPushButton)
             with a QtCore.Signal — Qt GUI code executed at class-definition time.

  Stage 2  backend_qt.py:216 defines class FigureCanvasQT(QWidget).
           On macOS, PySide6's Shiboken metaclass calls into the Cocoa platform
           plugin at the moment any QWidget subclass is first defined in Python.
           That call segfaults unless a QApplication already exists.

Two complementary fixes (test-infrastructure only; no src/ files touched):

  1. Stub matplotlib.backends.qt_editor.figureoptions (and siblings) in
     sys.modules so the _formlayout.py import chain never runs (Stage 1).

  2. Create a QApplication singleton here, before any test module is collected,
     so the platform plugin is initialised when FigureCanvasQT is defined
     (Stage 2).  The widget tests that declare an `app` fixture check
     QApplication.instance() first, so they reuse this same singleton.

MPLBACKEND=Agg (set via pytest-env in pyproject.toml) additionally ensures that
matplotlib.pyplot never auto-selects the Qt backend during normal test execution.
"""

import sys
import unittest.mock

# ── Stage 1: stub the Qt figure-options editor ────────────────────────────────
# These modules are only used at runtime when the toolbar "Edit curves" button is
# clicked — they are never triggered in tests.  Stubbing them prevents _formlayout
# from being imported (and trying to define Signal-bearing Qt classes) before any
# QApplication exists.
_QT_EDITOR_STUBS = [
    "matplotlib.backends.qt_editor",
    "matplotlib.backends.qt_editor.figureoptions",
    "matplotlib.backends.qt_editor._formlayout",
]
for _mod in _QT_EDITOR_STUBS:
    sys.modules.setdefault(_mod, unittest.mock.MagicMock())

# ── Stage 2: create QApplication before collection ───────────────────────────
# PySide6 requires QApplication to exist before the first QWidget subclass is
# defined at the Python level (macOS Cocoa initialisation).  Creating it here
# means it is in place by the time backend_qt.py defines FigureCanvasQT.
try:
    from PySide6.QtWidgets import QApplication

    if not QApplication.instance():
        _qapp = QApplication(sys.argv or [""])
except Exception:
    # Non-graphical / headless environments: fall back gracefully.
    # Tests that actually render widgets will be skipped or fail on their own.
    pass
