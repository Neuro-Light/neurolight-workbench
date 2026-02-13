from __future__ import annotations

import os
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from core.experiment_manager import Experiment, ExperimentManager
from ui.settings_dialog import SettingsDialog

EXPERIMENTS_DIR = Path(__file__).resolve().parents[2] / "experiments"
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

# Options button label for recent-experiment rows (text only, no icon)
OPTIONS_LABEL = "..."


class RecentExperimentRow(QWidget):
    """Single row in the recent experiments list: centered name + options button (...)."""

    def __init__(
        self,
        name: str,
        path: str,
        on_open: Callable[[], None],
        on_click: Callable[[], None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("recentExperimentRow")
        self._on_open = on_open
        self._on_click = on_click or (lambda: None)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 4, 4)
        layout.setSpacing(8)

        layout.addStretch()
        self.name_label = QLabel(name)
        self.name_label.setAlignment(Qt.AlignCenter)
        self.name_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        layout.addWidget(self.name_label, 1)
        layout.addStretch()

        # QToolButton with text only so the platform doesn't substitute an icon (e.g. upside-down U)
        self.options_btn = QToolButton()
        self.options_btn.setText(OPTIONS_LABEL)
        self.options_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.options_btn.setFixedSize(36, 22)
        self.options_btn.setToolTip("Options")
        self.options_btn.setProperty("class", "tab-action")
        layout.addWidget(self.options_btn)

    def mousePressEvent(self, event) -> None:
        if (
            event.button() == Qt.LeftButton
            and self.childAt(event.position().toPoint()) != self.options_btn
        ):
            self._on_click()
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self._on_open()
        super().mouseDoubleClickEvent(event)


class NewExperimentDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Neurolight - New Experiment")
        self.setModal(True)
        self.setMinimumWidth(500)

        self.name_edit = QLineEdit()
        self.pi_edit = QLineEdit()
        self.desc_edit = QPlainTextEdit()
        self.date_edit = QLineEdit(datetime.now(timezone.utc).strftime("%Y-%m-%d"))

        # Analysis type selection (future-proof for multiple analysis pipelines)
        self.analysis_combo = QComboBox()
        self.analysis_combo.addItem("SCN", "SCN")  # Current supported analysis
        # Make sure the combo box is wide enough to show full labels
        self.analysis_combo.setMinimumWidth(180)

        self.path_edit = QLineEdit(str(EXPERIMENTS_DIR))
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._browse)

        path_row = QHBoxLayout()
        path_row.addWidget(self.path_edit)
        path_row.addWidget(browse_btn)

        form = QFormLayout()
        form.addRow("Experiment Name*", self.name_edit)
        form.addRow("Analysis Type", self.analysis_combo)
        form.addRow("Principal Investigator", self.pi_edit)
        form.addRow("Description", self.desc_edit)
        form.addRow("Date", self.date_edit)

        container = QVBoxLayout()
        container.addLayout(form)

        path_container = QVBoxLayout()
        path_container.addWidget(QLabel("Save Location"))
        path_container.addLayout(path_row)
        container.addLayout(path_container)

        buttons = QDialogButtonBox(QDialogButtonBox.Cancel | QDialogButtonBox.Ok)
        create_btn = buttons.button(QDialogButtonBox.Ok)
        create_btn.setText("Create")
        create_btn.setProperty("class", "primary")
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self._accept)
        container.addWidget(buttons)

        self.setLayout(container)

        self.output_path: str | None = None
        self.metadata: dict = {}

    def _browse(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self, "Select Save Location", self.path_edit.text()
        )
        if directory:
            self.path_edit.setText(directory)

    def _accept(self) -> None:
        name = self.name_edit.text().strip()
        if not name:
            self.name_edit.setFocus()
            return
        base_dir = Path(self.path_edit.text().strip() or str(EXPERIMENTS_DIR))
        base_dir.mkdir(parents=True, exist_ok=True)
        file_path = base_dir / f"{name}.nexp"
        if file_path.exists():
            self.name_edit.setFocus()
            return
        self.output_path = str(file_path)
        self.metadata = {
            "name": name,
            "description": self.desc_edit.toPlainText().strip(),
            "principal_investigator": self.pi_edit.text().strip(),
            "created_date": datetime.now(timezone.utc),
            "analysis_type": self.analysis_combo.currentData(),
        }
        self.accept()


class StartupDialog(QDialog):
    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("experimentManagerDialog")
        self.setWindowTitle("Neurolight - Experiment Manager")
        self.setModal(True)
        self.setMinimumWidth(520)
        self.experiment: Experiment | None = None
        self.experiment_path: str | None = None
        self.manager = ExperimentManager()

        title = QLabel("Neurolight - Experiment Manager")
        title.setAlignment(Qt.AlignCenter)
        title.setProperty("class", "dialog-title")

        new_btn = QPushButton("Start New Experiment")
        new_btn.setProperty("class", "tab-action")
        load_btn = QPushButton("Load Existing Experiment")
        load_btn.setProperty("class", "tab-action")

        new_btn.clicked.connect(self._start_new)
        load_btn.clicked.connect(self._load_existing)

        self.recent_list = QListWidget()
        self.recent_list.itemDoubleClicked.connect(self._open_recent)
        self.recent_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.recent_list.customContextMenuRequested.connect(self._show_context_menu)
        self._refresh_recent()

        # Bottom bar: Preferences on the right (Delete/Export are on each row's ... menu)
        buttons_layout = QHBoxLayout()
        buttons_layout.addStretch()
        self.settings_btn = QPushButton("Preferences...")
        self.settings_btn.clicked.connect(self._open_settings)
        buttons_layout.addWidget(self.settings_btn)

        layout = QVBoxLayout()
        layout.addWidget(title)
        layout.addWidget(new_btn)
        layout.addWidget(load_btn)

        # Dividing line between main actions and Recent Experiments
        divider = QFrame()
        divider.setFrameShape(QFrame.HLine)
        divider.setFixedHeight(2)
        divider.setObjectName("experimentManagerDivider")
        layout.addWidget(divider)

        recent_heading = QLabel("Recent Experiments")
        recent_heading.setAlignment(Qt.AlignCenter)
        recent_heading.setProperty("class", "section-heading")
        layout.addWidget(recent_heading)
        layout.addWidget(self.recent_list)
        layout.addLayout(buttons_layout)
        self.setLayout(layout)

    def _refresh_recent(self) -> None:
        self.recent_list.clear()
        for rec in self.manager.get_recent_experiments():
            path = rec.get("path") or ""
            name = rec.get("name") or Path(path).stem if path else ""
            list_item = QListWidgetItem()
            list_item.setData(Qt.UserRole, path)
            list_item.setSizeHint(self._row_size_hint())
            self.recent_list.addItem(list_item)

            row = RecentExperimentRow(
                name=name,
                path=path,
                on_open=lambda p=path: self._open_recent_by_path(p),
                on_click=lambda p=path: self._select_item_by_path(p),
                parent=self.recent_list,
            )
            row.options_btn.clicked.connect(
                lambda checked=False, p=path, b=row.options_btn: self._show_options_for_path(p, b)
            )
            self.recent_list.setItemWidget(list_item, row)
        self.recent_list.clearSelection()

    def _row_size_hint(self):
        from PySide6.QtCore import QSize

        return QSize(0, 52)

    def _get_item_for_path(self, path: str):
        for i in range(self.recent_list.count()):
            item = self.recent_list.item(i)
            if item and item.data(Qt.UserRole) == path:
                return item
        return None

    def _select_item_by_path(self, path: str) -> None:
        item = self._get_item_for_path(path)
        if item is not None:
            self.recent_list.setCurrentItem(item)

    def _open_recent_by_path(self, path: str) -> None:
        item = self._get_item_for_path(path)
        if item is not None:
            self._open_recent(item)

    def _show_options_for_path(self, path: str, options_button: QPushButton | None) -> None:
        menu = QMenu(self)
        menu.addAction("Delete", lambda: self._remove_from_list_for_path(path))
        menu.addAction("Export", lambda: self._export_for_path(path))
        menu.addAction("Show file location", lambda: self._show_file_location(path))

        if options_button and options_button.isVisible():
            menu.exec(options_button.mapToGlobal(options_button.rect().bottomLeft()))
        else:
            menu.exec(self.recent_list.mapToGlobal(self.recent_list.rect().center()))

    def _remove_from_list_for_path(self, path: str) -> None:
        item = self._get_item_for_path(path)
        if item is not None:
            self._delete_experiment(item, delete_file=False)

    def _export_for_path(self, path: str) -> None:
        item = self._get_item_for_path(path)
        if item is not None:
            self._export_experiment(item)

    def _show_file_location(self, path: str) -> None:
        if not path or not os.path.isfile(path):
            QMessageBox.warning(
                self,
                "File location",
                "The experiment file could not be found.",
            )
            return
        parent_dir = str(Path(path).resolve().parent)
        QDesktopServices.openUrl(QUrl.fromLocalFile(parent_dir))

    def _start_new(self) -> None:
        dlg = NewExperimentDialog(self)
        if dlg.exec() == QDialog.Accepted and dlg.output_path:
            exp = self.manager.create_new_experiment(dlg.metadata)
            try:
                self.manager.save_experiment(exp, dlg.output_path)
                # Refresh recent list to show the new experiment
                self._refresh_recent()
            except Exception:
                return
            self.experiment = exp
            self.experiment_path = dlg.output_path
            self.accept()

    def _load_existing(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Experiment", str(EXPERIMENTS_DIR), "Neurolight Experiment (*.nexp)"
        )
        if not file_path:
            return
        try:
            self.experiment = self.manager.load_experiment(file_path)
            self.experiment_path = file_path
            # Refresh recent list to update the order
            self._refresh_recent()
            self.accept()
        except Exception as e:
            QMessageBox.warning(
                self, "Load Failed", f"Failed to load experiment:\n{file_path}\n\n{str(e)}"
            )
            self._refresh_recent()

    def _open_recent(self, item: QListWidgetItem) -> None:
        path = item.data(Qt.UserRole)
        if not path:
            return
        try:
            self.experiment = self.manager.load_experiment(path)
            self.experiment_path = path
            self.accept()
        except Exception:
            QMessageBox.warning(
                self,
                "Load Failed",
                f"Failed to load experiment:\n{path}\n\n"
                "The file may have been deleted or is corrupted.",
            )
            self._refresh_recent()

    def _open_settings(self) -> None:
        """Open the Preferences dialog."""
        dlg = SettingsDialog(self)
        dlg.exec()

    def _show_context_menu(self, position) -> None:
        """Show context menu for right-click on experiment list."""
        item = self.recent_list.itemAt(position)
        if item is None:
            return

        menu = QMenu(self)
        open_action = menu.addAction("Open")
        delete_action = menu.addAction("Delete from List")
        delete_file_action = menu.addAction("Delete File and Remove from List")
        export_action = menu.addAction("Export")

        action = menu.exec(self.recent_list.mapToGlobal(position))

        if action == open_action:
            self._open_recent(item)
        elif action == delete_action:
            self._delete_experiment(item, delete_file=False)
        elif action == delete_file_action:
            self._delete_experiment(item, delete_file=True)
        elif action == export_action:
            self._export_experiment(item)

    def _delete_experiment(self, item: QListWidgetItem, delete_file: bool = False) -> None:
        """Delete an experiment from the list and optionally from disk."""
        path = item.data(Qt.UserRole)
        if not path:
            return

        if delete_file:
            reply = QMessageBox.warning(
                self,
                "Delete Experiment File",
                f"Are you sure you want to permanently delete:\n{path}\n\n"
                "This action cannot be undone!",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return

        try:
            if self.manager.delete_experiment(path, delete_file=delete_file):
                self._refresh_recent()
                if delete_file:
                    QMessageBox.information(
                        self, "Deleted", "Experiment file and entry have been deleted."
                    )
                else:
                    QMessageBox.information(
                        self, "Removed", "Experiment has been removed from the recent list."
                    )
            else:
                QMessageBox.warning(self, "Delete Failed", "Failed to delete experiment.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred while deleting:\n{str(e)}")

    def _export_experiment(self, item: QListWidgetItem) -> None:
        """Export an experiment to a .nexp file."""
        path = item.data(Qt.UserRole)
        if not path:
            return

        try:
            # Load the experiment
            experiment = self.manager.load_experiment(path)

            # Get export location
            default_name = f"{experiment.name}_export.nexp"
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Export Experiment",
                default_name,
                "Neurolight Experiment (*.nexp);;All Files (*)",
            )

            if not file_path:
                return

            # Ensure .nexp extension
            if not file_path.endswith(".nexp"):
                file_path += ".nexp"

            # Export experiment data using the manager's save method
            # This ensures the file format matches the native .nexp format
            if self.manager.save_experiment(experiment, file_path):
                QMessageBox.information(
                    self, "Export Successful", f"Experiment exported to:\n{file_path}"
                )
            else:
                QMessageBox.warning(self, "Export Failed", "Failed to export experiment.")
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"Failed to export experiment:\n{str(e)}")
