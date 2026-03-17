"""Dialog showing progress during neuron detection."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QLabel,
    QProgressBar,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
)


class DetectionProgressDialog(QDialog):
    """Dialog showing progress during neuron detection (steps and messages)."""

    def __init__(self, parent=None, total_steps: int = 5):
        super().__init__(parent)
        self.setWindowTitle("Detecting Neurons...")
        self.setWindowFlags(Qt.Dialog | Qt.WindowTitleHint | Qt.CustomizeWindowHint)
        self.setModal(True)
        self.setMinimumWidth(500)
        self.setMinimumHeight(220)

        layout = QVBoxLayout(self)

        self.status_label = QLabel("Initializing detection...")
        layout.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(max(1, total_steps))
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(120)
        layout.addWidget(self.log_text)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        layout.addWidget(self.cancel_button)

    def update_progress(self, completed: int, total: int, message: str) -> None:
        """Update progress bar and status message."""
        if total > 0:
            self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(min(completed, self.progress_bar.maximum()))
        self.status_label.setText(message)
        self.log_text.append(message)
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())
