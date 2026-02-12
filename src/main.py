import sys
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from ui.app_settings import get_theme
from ui.main_window import MainWindow
from ui.startup_dialog import StartupDialog
from ui.styles import get_stylesheet

# Application icon (project root / logo.png)
_LOGO_PATH = Path(__file__).resolve().parent.parent / "logo.png"


def main() -> int:
    app = QApplication(sys.argv)
    if _LOGO_PATH.is_file():
        app.setWindowIcon(QIcon(str(_LOGO_PATH)))
    theme = get_theme()
    app.setStyleSheet(get_stylesheet(theme))

    # Show startup dialog (modal)
    startup = StartupDialog()
    result = startup.exec()

    if result != StartupDialog.Accepted or startup.experiment is None:
        return 0

    # Create main window with experiment context
    main_window = MainWindow(startup.experiment)
    # Carry over the .nexp path so autosaves and path updates persist
    try:
        main_window.current_experiment_path = startup.experiment_path  # type: ignore[attr-defined]
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
    sys.exit(main())
