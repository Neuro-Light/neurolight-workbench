import multiprocessing
import os
import sys
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication, QDialog

from ui.app_settings import get_enable_alignment_multiprocessing

# If alignment multiprocessing is enabled in a frozen app, avoid CPU thread
# oversubscription (OpenMP/BLAS) which can cause instability or hard crashes
# during SciPy-heavy workloads (e.g., neuron detection).
if get_enable_alignment_multiprocessing() or os.environ.get("NEUROLIGHT_ENABLE_MP", "").strip() == "1":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from ui.app_settings import get_theme
from ui.main_window import MainWindow
from ui.startup_dialog import StartupDialog
from ui.styles import get_stylesheet
from ui.user_selection_dialog import UserSelectionDialog

# Required for frozen apps on Windows (no-op elsewhere); helps avoid
# multiprocessing issues when the app is packaged.
multiprocessing.freeze_support()

# Application icon (project root / logo.png)
_LOGO_PATH = Path(__file__).resolve().parent.parent / "logo.png"


def main() -> int:
    app = QApplication(sys.argv)
    if _LOGO_PATH.is_file():
        app.setWindowIcon(QIcon(str(_LOGO_PATH)))
    theme = get_theme()
    app.setStyleSheet(get_stylesheet(theme))

    # User selection first, then experiment manager
    user_dialog = UserSelectionDialog()
    if user_dialog.exec() != QDialog.Accepted:
        return 0

    startup = StartupDialog(user_dialog.selected_user_experiments_dir)
    result = startup.exec()

    if result != StartupDialog.Accepted or startup.experiment is None:
        return 0

    # Create main window with experiment context
    recent_file = user_dialog.selected_user_experiments_dir.parent / "recent_experiments.json"
    main_window = MainWindow(startup.experiment, recent_file=recent_file)
    # Keep track of the active user's experiments directory so "Close Experiment"
    # can return to the right experiment manager root.
    try:
        main_window.user_experiments_dir = user_dialog.selected_user_experiments_dir
    except Exception:
        pass
    # Carry over the .nexp path so autosaves and path updates persist
    try:
        main_window.set_current_experiment_path(startup.experiment_path)  # type: ignore[attr-defined]
    except Exception:
        pass
    main_window.show()

    # Auto-save timer placeholder (configurable later)
    autosave_timer = QTimer()
    autosave_timer.setInterval(5 * 60 * 1000)
    autosave_timer.timeout.connect(main_window.autosave_experiment)
    autosave_timer.start()

    return app.exec()


if __name__ == "__main__":
    # Required for frozen apps on Windows (no-op elsewhere); must be first
    # in the main block so child re-execution is handled correctly.
    multiprocessing.freeze_support()
    sys.exit(main())
