from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from PySide6.QtCore import QTime, QTimer, QUrl
from PySide6.QtGui import QAction, QCloseEvent, QDesktopServices
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QSplitter,
    QTimeEdit,
    QVBoxLayout,
    QWidget,
)

from core.data_analyzer import DataAnalyzer
from core.experiment_manager import Experiment, ExperimentManager
from core.image_processor import ImageProcessor
from core.roi import ROI, ROIShape
from ui.alignment_dialog import AlignmentDialog
from ui.alignment_progress_dialog import AlignmentProgressDialog
from ui.alignment_worker import AlignmentWorker
from ui.analysis_panel import AnalysisPanel
from ui.app_settings import get_enable_alignment_multiprocessing
from ui.image_viewer import ImageViewer
from ui.loading_dialog import LoadingDialog
from ui.settings_dialog import SettingsDialog
from ui.startup_dialog import StartupDialog
from ui.workflow import STEP_DEFINITIONS, WorkflowManager, WorkflowStep, WorkflowStepper
from utils.file_handler import ImageStackHandler, _get_exif_timestamp

# Set up logger for main window
logger = logging.getLogger(__name__)

# Configure logging to file if not already configured
_log_file = Path.home() / ".neurolight" / "neurolight.log"
try:
    _log_file.parent.mkdir(parents=True, exist_ok=True)
except OSError:
    # In restricted environments (some CI/sandboxes), the home directory may be read-only.
    # Logging should never prevent the UI (or tests) from importing.
    pass

# Check if logging is already configured at the module level
if not logger.handlers:
    handler: logging.Handler
    try:
        # Create file handler with append mode
        handler = logging.FileHandler(_log_file, encoding="utf-8", mode="a")
    except OSError:
        handler = logging.NullHandler()
    handler.setLevel(logging.WARNING)

    # Create formatter - logger.exception() automatically includes traceback
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)
    logger.propagate = False  # Prevent duplicate logs


class _ExperimentSettingsDialog(QDialog):
    """Dialog to edit experiment metadata and acquisition time settings."""

    def __init__(self, experiment: Experiment, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Experiment Settings")
        self.setModal(True)
        self.setMinimumWidth(400)
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.name_edit = QLineEdit()
        self.name_edit.setText(experiment.name)
        self.name_edit.setPlaceholderText("Experiment name")
        form.addRow("Name", self.name_edit)
        self.pi_edit = QLineEdit()
        self.pi_edit.setText(experiment.principal_investigator)
        self.pi_edit.setPlaceholderText("Principal investigator")
        form.addRow("Principal Investigator", self.pi_edit)
        self.desc_edit = QPlainTextEdit()
        self.desc_edit.setPlainText(experiment.description)
        self.desc_edit.setPlaceholderText("Description")
        self.desc_edit.setMaximumHeight(120)
        form.addRow("Description", self.desc_edit)

        acquisition = experiment.settings.get("acquisition") or {}
        interval_value = acquisition.get("frame_interval_minutes")
        if interval_value is None:
            interval_value = 30.0
        try:
            interval_value = float(interval_value)
        except (TypeError, ValueError):
            interval_value = 30.0
        self.frame_interval_spin = QDoubleSpinBox()
        self.frame_interval_spin.setRange(0.0001, 10000.0)
        self.frame_interval_spin.setDecimals(4)
        self.frame_interval_spin.setSingleStep(0.5)
        self.frame_interval_spin.setValue(interval_value)
        self.frame_interval_spin.setToolTip("Time between successive frames in minutes.")
        form.addRow("Interval between frames (minutes)", self.frame_interval_spin)

        self.start_time_edit = QTimeEdit()
        self.start_time_edit.setDisplayFormat("HH:mm:ss")
        self.start_time_edit.setToolTip("Time of first frame (24-hour clock).")
        self.start_time_edit.setTime(_parse_time_string(acquisition.get("experiment_start_time")) or QTime(0, 0, 0))
        form.addRow("Experiment start time", self.start_time_edit)
        layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._accept_dialog)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _accept_dialog(self) -> None:
        name = self.name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "Experiment Settings", "Name is required.")
            self.name_edit.setFocus()
            return
        self.name = name
        self.principal_investigator = self.pi_edit.text().strip()
        self.description = self.desc_edit.toPlainText().strip()
        self.frame_interval_minutes = float(self.frame_interval_spin.value())
        self.experiment_start_time = self.start_time_edit.time().toString("HH:mm:ss")
        self.accept()


class _ConfirmStartTimeDialog(QDialog):
    """Dialog to confirm or correct metadata-derived experiment start time."""

    def __init__(
        self,
        suggested_time: QTime,
        metadata_source: str,
        timestamp_uniformity_note: Optional[str],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Confirm Experiment Start Time")
        self.setModal(True)
        self.setMinimumWidth(460)

        layout = QVBoxLayout(self)
        info = QLabel(
            "Start time was inferred from image metadata.\n"
            f"Source: {metadata_source}\n"
            "Confirm or adjust this before running time-based analyses."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        if timestamp_uniformity_note:
            warn = QLabel(timestamp_uniformity_note)
            warn.setWordWrap(True)
            warn.setStyleSheet("color: #f59e0b; font-weight: 600;")
            layout.addWidget(warn)

        form = QFormLayout()
        self.start_time_edit = QTimeEdit()
        self.start_time_edit.setDisplayFormat("HH:mm:ss")
        self.start_time_edit.setTime(suggested_time)
        form.addRow("Experiment start time", self.start_time_edit)
        layout.addLayout(form)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def selected_start_time(self) -> str:
        return self.start_time_edit.time().toString("HH:mm:ss")


def _parse_time_string(value: object) -> Optional[QTime]:
    """Parse time strings like HH:MM[:SS] into QTime."""
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    for fmt in ("HH:mm:ss", "HH:mm"):
        t = QTime.fromString(text, fmt)
        if t.isValid():
            if fmt == "HH:mm":
                return QTime(t.hour(), t.minute(), 0)
            return t
    return None


def _time_string_to_minutes(value: object) -> Optional[int]:
    """Convert HH:MM[:SS] time strings to minutes since midnight."""
    qtime = _parse_time_string(value)
    if qtime is None:
        return None
    return qtime.hour() * 60 + qtime.minute()


class MainWindow(QMainWindow):
    def __init__(self, experiment: Experiment) -> None:
        super().__init__()
        self.experiment = experiment
        self.manager = ExperimentManager()
        self.current_experiment_path: Optional[str] = None
        self.image_processor = ImageProcessor(experiment)
        self._alignment_worker: Optional[AlignmentWorker] = None

        # Cached references to controls affected by workflow gating
        self._action_open_stack: Optional[QAction] = None
        self._action_align_images: Optional[QAction] = None

        # Initialize debounced save timer for display settings
        self._display_settings_timer = QTimer()
        self._display_settings_timer.setSingleShot(True)
        self._display_settings_timer.timeout.connect(self._persist_display_settings)
        self._pending_exposure: Optional[int] = None
        self._pending_contrast: Optional[int] = None

        self.setWindowTitle(f"Neurolight - {self.experiment.name}")
        self.resize(1200, 800)

        # Guided workflow
        self.workflow_manager = WorkflowManager(self.experiment, self)
        self.workflow_manager.state_changed.connect(self._save_workflow_progress)
        self.workflow_stepper = WorkflowStepper(self.workflow_manager, self)
        # Wire stepper actions
        self.workflow_stepper.requestAlignImages.connect(self._align_images)

        self._init_menu()
        self._init_layout()

    # ------------------------------------------------------------------
    # Workflow integration
    # ------------------------------------------------------------------

    def _init_workflow_bindings(self) -> None:
        """
        Wire workflow manager state to enable/disable relevant UI controls.

        Only the current step's primary controls are interactive. File-level
        actions like closing or exiting the experiment remain unaffected.
        """

        def refresh_controls() -> None:
            current = self.workflow_manager.current_step
            current_index = STEP_DEFINITIONS[current].index

            # Step 1: Load Image Stack
            enable_load = current == WorkflowStep.LOAD_IMAGES
            if self._action_open_stack is not None:
                self._action_open_stack.setEnabled(enable_load)
            # Upload button in viewer
            if hasattr(self, "viewer") and hasattr(self.viewer, "upload_btn"):
                self.viewer.upload_btn.setEnabled(enable_load)

            # Step 2: Edit Contrast & Exposure — show/hide entire display panel
            show_edit = current == WorkflowStep.EDIT_IMAGES
            if hasattr(self, "viewer"):
                panel = getattr(self.viewer, "display_controls_panel", None)
                if panel is not None:
                    panel.setVisible(show_edit)

            # Step 3: Cull Frames — show/hide cull controls panel
            show_cull = current == WorkflowStep.CULL_FRAMES
            if hasattr(self, "viewer"):
                panel = getattr(self.viewer, "cull_controls_panel", None)
                if panel is not None:
                    panel.setVisible(show_cull)
                # After the cull step, hide excluded frames from navigation
                cull_index = STEP_DEFINITIONS[WorkflowStep.CULL_FRAMES].index
                _set_filter = getattr(self.viewer, "set_filter_excluded", None)
                if callable(_set_filter):
                    _set_filter(current_index > cull_index)

            # Step 4: Align Images
            enable_align = current == WorkflowStep.ALIGN_IMAGES
            if self._action_align_images is not None:
                self._action_align_images.setEnabled(enable_align)

            # Step 5: Select ROI — show/hide entire ROI controls panel
            show_roi = current == WorkflowStep.SELECT_ROI
            if hasattr(self, "viewer"):
                panel = getattr(self.viewer, "roi_controls_panel", None)
                if panel is not None:
                    panel.setVisible(show_roi)

            # Step 6: Detect Neurons
            enable_detect = current == WorkflowStep.DETECT_NEURONS
            if hasattr(self, "analysis"):
                try:
                    detection_widget = self.analysis.get_neuron_detection_widget()
                    if hasattr(detection_widget, "detect_btn"):
                        detection_widget.detect_btn.setEnabled(enable_detect)
                except Exception:
                    pass

            # Hide the analysis panel until ROI has been selected and the
            # "Select ROI" step has been completed (i.e. downstream of step 4).
            if hasattr(self, "analysis"):
                roi_index = STEP_DEFINITIONS[WorkflowStep.SELECT_ROI].index
                # Show once either:
                # - current step is beyond Select ROI, or
                # - Select ROI is in completed steps (user finished step 4)
                show_analysis = (
                    current_index > roi_index or WorkflowStep.SELECT_ROI in self.workflow_manager.completed_steps
                )
                self.analysis.setVisible(show_analysis)

        # Connect workflow manager notifications
        self.workflow_manager.step_changed.connect(lambda _step: refresh_controls())
        self.workflow_manager.state_changed.connect(refresh_controls)

        # Apply initial state
        refresh_controls()

    def _save_workflow_progress(self) -> None:
        if not self.current_experiment_path:
            return
        try:
            # Persist any time-related settings that are cheap to capture
            if hasattr(self, "analysis"):
                self._capture_experiment_time_settings()
            self.manager.save_experiment(self.experiment, self.current_experiment_path)
        except Exception:
            pass

    def set_current_experiment_path(self, path: Optional[str], *, persist_workflow: bool = True) -> None:
        self.current_experiment_path = path
        if persist_workflow and path:
            self._save_workflow_progress()

    def _init_menu(self) -> None:
        menubar = self.menuBar()

        self._file_menu = menubar.addMenu("File")
        save_action = QAction("Save Experiment", self)
        save_as_action = QAction("Save Experiment As...", self)
        close_action = QAction("Close Experiment", self)
        exit_action = QAction("Exit", self)
        open_stack_action = QAction("Open Image Stack", self)
        export_results_action = QAction("Export Results", self)

        # Keep references to actions we will control via workflow
        self._action_open_stack = open_stack_action

        save_action.setShortcut("Ctrl+S")

        save_action.triggered.connect(self._save)
        save_as_action.triggered.connect(self._save_as)
        close_action.triggered.connect(self._close_experiment)
        exit_action.triggered.connect(self._exit_experiment)
        open_stack_action.triggered.connect(self._open_image_stack)
        export_results_action.triggered.connect(self._export_experiment)

        self._file_menu.addAction(save_action)
        self._file_menu.addAction(save_as_action)
        self._file_menu.addSeparator()
        self._file_menu.addAction(open_stack_action)
        self._file_menu.addAction(export_results_action)
        self._file_menu.addSeparator()
        self._file_menu.addAction(close_action)
        self._file_menu.addAction(exit_action)

        self._edit_menu = menubar.addMenu("Edit")
        settings_action = QAction("Preferences...", self)
        settings_action.setShortcut("Ctrl+,")
        settings_action.triggered.connect(self._open_settings)
        self._edit_menu.addAction(settings_action)
        experiment_settings_action = QAction("Experiment Settings...", self)
        experiment_settings_action.triggered.connect(self._open_experiment_settings)
        self._edit_menu.addAction(experiment_settings_action)
        self._tools_menu = menubar.addMenu("Tools")

        # Add crop action
        crop_action = QAction("Crop Stack to ROI", self)
        crop_action.triggered.connect(self._crop_stack_to_roi)
        self._tools_menu.addAction(crop_action)

        # Add alignment action
        align_action = QAction("Align Images", self)
        align_action.triggered.connect(self._align_images)
        self._tools_menu.addAction(align_action)
        self._action_align_images = align_action

        self._tools_menu.addAction("Generate GIF")
        self._tools_menu.addAction("Run Analysis")
        help_meun = menubar.addMenu("Help")
        about_action = help_meun.addAction("About")
        about_action.triggered.connect(self.open_website)

    def open_website(self):
        QDesktopServices.openUrl(
            QUrl("https://sce.nau.edu/capstone/projects/CS/2026/NeuroNauts_F25/project_overview.html")
        )

    def _open_settings(self) -> None:
        dlg = SettingsDialog(self)
        if dlg.exec() == QDialog.Accepted:
            # Theme / colours were applied; refresh all visuals
            self.analysis.get_neuron_trajectory_plot_widget().refresh_theme()
            try:
                # Rayleigh plot should update its colors with theme changes as well
                self.analysis.get_rayleigh_plot_widget().refresh_theme()
            except Exception:
                # Some tests or older analysis panels may not expose this widget
                pass
            try:
                # Lomb–Scargle periodogram should also adopt the new theme
                self.analysis.get_lomb_scargle_widget().refresh_theme()
            except Exception:
                # Some tests or lightweight analysis panels may not expose this widget
                pass
            self.analysis.get_roi_plot_widget().refresh_theme()
            self.viewer.refresh_roi_selector_icons()
            self.viewer._show_current()  # redraw ROI overlays with new colours

    def _open_experiment_settings(self) -> None:
        # Open the Experiment Settings dialog to edit metadata and acquisition timing.
        if self.experiment is None or not self.current_experiment_path:
            QMessageBox.information(
                self,
                "Experiment Settings",
                "No experiment is loaded. Open or create an experiment first.",
            )
            return
        dlg = _ExperimentSettingsDialog(self.experiment, self)
        if dlg.exec() == QDialog.Accepted:
            self.experiment.name = dlg.name
            self.experiment.description = dlg.description
            self.experiment.principal_investigator = dlg.principal_investigator
            if "acquisition" not in self.experiment.settings:
                self.experiment.settings["acquisition"] = {}
            self.experiment.settings["acquisition"]["frame_interval_minutes"] = dlg.frame_interval_minutes
            self.experiment.settings["acquisition"]["experiment_start_time"] = dlg.experiment_start_time
            self.experiment.update_modified_date()
            self._apply_experiment_time_settings()
            try:
                self.manager.save_experiment(self.experiment, self.current_experiment_path)
                QMessageBox.information(self, "Saved", "Experiment settings saved.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save: {e}")

    def _sync_rois_to_experiment(self) -> None:
        """Copy both ROIs from the viewer into the experiment for persistence."""
        for key in ("roi_1", "roi_2"):
            roi = self.viewer.get_current_roi(key)
            self.experiment.rois[key] = roi.to_dict() if roi is not None else None

    def closeEvent(self, event: QCloseEvent) -> None:
        reply = QMessageBox.question(
            self,
            "Exit Application",
            "Are you sure you want to exit the application?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if reply == QMessageBox.Yes:
            # Stop alignment worker if running
            if self._alignment_worker is not None and self._alignment_worker.isRunning():
                self._alignment_worker.request_cancel()
                self._alignment_worker.wait(5000)
            self._flush_pending_display_settings()
            self._sync_rois_to_experiment()
            self._capture_display_settings()
            self._capture_experiment_time_settings()
            if self.current_experiment_path:
                try:
                    self.manager.save_experiment(self.experiment, self.current_experiment_path)
                except Exception as e:
                    logger.exception(f"Failed to save experiment during close: {self.current_experiment_path}")
                    self._show_save_error_feedback(str(e))
            event.accept()
        else:
            event.ignore()

    def _show_save_error_feedback(self, error_message: str) -> None:
        """
        Show non-blocking feedback when save fails during close.
        Uses status bar message to inform user without blocking the close flow.

        Args:
            error_message: The error message string (also logged via logger.exception)
        """
        log_path = _log_file
        status_message = f"Save failed during close. Error logged to: {log_path}"

        # Show status bar message (non-blocking, brief display)
        # This will be visible briefly before the window closes if there's any delay
        status_bar = self.statusBar()
        if status_bar:
            # Show message for 5 seconds to increase visibility
            status_bar.showMessage(status_message, 5000)

        # Note: The full exception with traceback is already logged via logger.exception()
        # in the closeEvent handler. We intentionally don't show a modal dialog here
        # as it would block the close flow. The error is fully logged to file, and
        # the status bar message provides immediate feedback. Users can check the
        # log file at ~/.neurolight/neurolight.log for full error details.

    def _init_layout(self) -> None:
        splitter = QSplitter()

        # Left panel: image viewer
        self.stack_handler = ImageStackHandler()
        self.stack_handler.associate_with_experiment(self.experiment)
        self.viewer = ImageViewer(self.stack_handler)
        self.viewer.stackLoaded.connect(self._on_stack_loaded)

        # Right panel: analysis dashboard
        self.analysis = AnalysisPanel()
        self.analysis.get_roi_plot_widget().experiment = self.experiment

        # Set up neuron detection widget
        detection_widget = self.analysis.get_neuron_detection_widget()
        detection_widget.set_image_processor(self.image_processor)
        detection_widget.experiment = self.experiment

        # Notify workflow manager when detection completes
        try:
            detection_widget.detectionCompleted.connect(
                lambda: self.workflow_manager.complete_step_if_current(WorkflowStep.DETECT_NEURONS)
            )
        except Exception:
            # In tests the widget may be heavily mocked; ignore connection errors
            pass

        # Connect detection widget to trajectory and Rayleigh plots. Some tests use a
        # lightweight AnalysisPanel double that does not expose every plot.
        trajectory_plot_widget = self.analysis.get_neuron_trajectory_plot_widget()
        rayleigh_plot_getter = getattr(self.analysis, "get_rayleigh_plot_widget", None)
        rayleigh_plot_widget = rayleigh_plot_getter() if callable(rayleigh_plot_getter) else None

        def _update_neuron_plots(
            trajectories,
            quality_mask,
            locations,
            roi_origin=None,
        ) -> None:
            trajectory_plot_widget.plot_trajectories(
                trajectories,
                quality_mask,
                locations,
                roi_origin=roi_origin,
            )
            # Rayleigh plot ignores ROI; it just needs trajectories and quality mask.
            if rayleigh_plot_widget is not None:
                rayleigh_plot_widget.set_trajectory_data(
                    trajectories,
                    quality_mask,
                    roi_origin=roi_origin,
                )

        detection_widget.set_trajectory_plot_callback(_update_neuron_plots)

        # Connect detection widget to save experiment callback
        detection_widget.set_save_experiment_callback(self._save_neuron_detection)

        # Connect ROI selection to analysis and saving (signal: str, ROI)
        self.viewer.roiSelected.connect(self._on_roi_selected)
        self.viewer.roiSelected.connect(self._save_roi_to_experiment)
        self.viewer.roiDeleted.connect(self._on_roi_deleted)

        # Connect display settings changes to debounced saving
        self.viewer.displaySettingsChanged.connect(self._on_display_settings_changed)

        # Connect frame culling changes
        _culling_signal = getattr(self.viewer, "frameCullingChanged", None)
        if _culling_signal is not None:
            _culling_signal.connect(self._on_frame_culling_changed)

        # Create data analyzer
        self.data_analyzer = DataAnalyzer(self.experiment)

        splitter.addWidget(self.viewer)
        splitter.addWidget(self.analysis)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 7)

        # Wrap workflow stepper + splitter in a vertical layout
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.workflow_stepper)
        layout.addWidget(splitter)

        self.setCentralWidget(container)

        # Initialize workflow-driven enable/disable state
        self._init_workflow_bindings()

        # Auto-load image stack/ROI if experiment has saved data
        self._auto_load_experiment_data()

    def _apply_saved_display_settings(self) -> None:
        if not hasattr(self, "viewer") or not hasattr(self.viewer, "set_exposure"):
            return

        display_settings = self.experiment.settings.get("display") or {}

        def _normalize(value: Optional[int]) -> int:
            try:
                return max(-100, min(100, int(value)))  # type: ignore[arg-type]
            except (TypeError, ValueError):
                return 0

        exposure = _normalize(display_settings.get("exposure"))
        contrast = _normalize(display_settings.get("contrast"))

        self.viewer.set_exposure(exposure)
        self.viewer.set_contrast(contrast)

    def _apply_experiment_time_settings(self) -> None:
        """
        Apply saved experiment time settings (e.g. Rayleigh start time) to widgets.
        """
        if not hasattr(self, "analysis"):
            return
        # Rayleigh plot start time (from acquisition metadata if available)
        acquisition = self.experiment.settings.get("acquisition") or {}
        start_time = acquisition.get("experiment_start_time")
        start_minutes = _time_string_to_minutes(start_time)
        if start_minutes is None:
            # Backward compatibility for experiments that only have saved minutes.
            time_settings = self.experiment.settings.get("time") or {}
            legacy_minutes = time_settings.get("start_minutes")
            if legacy_minutes is not None:
                try:
                    start_minutes = int(legacy_minutes)
                except (TypeError, ValueError):
                    start_minutes = None

        try:
            rayleigh_getter = getattr(self.analysis, "get_rayleigh_plot_widget", None)
            rayleigh_widget = rayleigh_getter() if callable(rayleigh_getter) else None
        except Exception:
            rayleigh_widget = None
        if rayleigh_widget is not None:
            if start_minutes is not None:
                try:
                    rayleigh_widget.set_experiment_start_time_minutes(start_minutes)
                except Exception:
                    logger.exception("Failed to apply experiment start time to Rayleigh widget.")

        # Frame interval and start time for trajectory and Lomb-Scargle plots
        frame_interval = acquisition.get("frame_interval_minutes")

        try:
            traj_widget = self.analysis.get_neuron_trajectory_plot_widget()
            traj_widget.set_time_settings(
                interval_minutes=float(frame_interval) if frame_interval is not None else 30.0,
                start_time=start_time,
            )
        except Exception:
            logger.exception("Failed to configure neuron trajectory time settings.")

        try:
            ls_getter = getattr(self.analysis, "get_lomb_scargle_widget", None)
            ls_widget = ls_getter() if callable(ls_getter) else None
            if ls_widget is not None and frame_interval is not None:
                ls_widget.set_frame_interval_minutes(float(frame_interval))
        except Exception:
            logger.exception("Failed to apply frame interval to Lomb-Scargle widget.")

        try:
            roi_widget = self.analysis.get_roi_plot_widget()
            if frame_interval is not None:
                roi_widget.set_frame_interval_minutes(float(frame_interval))
        except Exception:
            pass

    def _auto_load_experiment_data(self) -> None:
        """Auto-load image stack, ROI, and display settings if experiment has saved data."""
        # Always apply saved or neutral display settings up front
        self._apply_saved_display_settings()
        # Apply saved experiment time (e.g. Rayleigh start time) if available
        self._apply_experiment_time_settings()
        try:
            detection_widget = self.analysis.get_neuron_detection_widget()
            detection_widget.reset_detection_state()
            self.analysis.get_neuron_trajectory_plot_widget().clear_plot()
        except Exception:
            pass
        try:
            path = self.experiment.image_stack_path
            if path:
                # Show loading dialog
                loading_dialog = LoadingDialog(self)
                loading_dialog.show()
                loading_dialog.update_status("Loading image stack...", "This may take a few seconds")

                # Process events to show the dialog
                QApplication.processEvents()

                def load_stack_and_roi(p=path):
                    try:
                        # Check if we have a saved file list (selected files)
                        saved_files = self.experiment.image_stack_files
                        if saved_files and len(saved_files) > 0:
                            # Verify files still exist
                            existing_files = [f for f in saved_files if Path(f).exists()]
                            if len(existing_files) > 0:
                                loading_dialog.update_status(
                                    "Loading image stack...",
                                    f"Loading {len(existing_files)} selected images",
                                )
                                QApplication.processEvents()
                                self.viewer.set_stack(existing_files)
                            else:
                                # Files don't exist, fall back to directory
                                loading_dialog.update_status(
                                    "Loading image stack...",
                                    f"Selected files not found, loading from directory: {p}",
                                )
                                QApplication.processEvents()
                                self.viewer.set_stack(p)
                        else:
                            # No saved file list, load from directory
                            loading_dialog.update_status("Loading image stack...", f"Loading images from: {p}")
                            QApplication.processEvents()
                            self.viewer.set_stack(p)

                        # Restore culling state
                        self._restore_culling_state()

                        # Load both ROIs from experiment.rois
                        has_any_roi = any(self.experiment.rois.get(k) for k in ("roi_1", "roi_2"))
                        if has_any_roi:
                            loading_dialog.update_status("Loading ROIs...", "Restoring ROI selections")
                            QApplication.processEvents()

                            loaded_rois: dict = {}
                            for roi_key in ("roi_1", "roi_2"):
                                roi_data = self.experiment.rois.get(roi_key)
                                if roi_data is None:
                                    continue
                                try:
                                    loaded_rois[roi_key] = ROI.from_dict(roi_data)
                                except Exception as exc:
                                    logger.warning(
                                        "ROI.from_dict failed for %s (data=%r): %s; using default ellipse fallback",
                                        roi_key,
                                        roi_data,
                                        exc,
                                    )
                                    loaded_rois[roi_key] = ROI(
                                        x=roi_data.get("x", 0),
                                        y=roi_data.get("y", 0),
                                        width=roi_data.get("width", 100),
                                        height=roi_data.get("height", 100),
                                        shape=ROIShape.ELLIPSE,
                                    )

                            def load_rois_and_plot():
                                try:
                                    loading_dialog.update_status(
                                        "Restoring ROIs and graphs...",
                                        "Loading analysis data",
                                    )
                                    QApplication.processEvents()

                                    for rk, roi_obj in loaded_rois.items():
                                        self.viewer.set_roi(roi_obj, key=rk)
                                        self._on_roi_selected(rk, roi_obj)

                                    detection_data = self.experiment.get_neuron_detection_data()
                                    if detection_data:
                                        loading_dialog.update_status(
                                            "Loading neuron detection data...",
                                            "Restoring detected neurons and trajectories",
                                        )
                                        QApplication.processEvents()
                                        self._load_neuron_detection_data()

                                    loading_dialog.close_dialog()
                                except Exception:
                                    loading_dialog.close_dialog()

                            QTimer.singleShot(200, load_rois_and_plot)
                        else:
                            loading_dialog.close_dialog()
                    except Exception:
                        loading_dialog.close_dialog()
                        # Silently fail - loading might have issues

                QTimer.singleShot(0, load_stack_and_roi)
            else:
                # No image stack path, check if there's neuron detection data to load
                detection_data = self.experiment.get_neuron_detection_data()
                if detection_data:
                    # Show loading dialog for neuron detection data only
                    loading_dialog = LoadingDialog(self)
                    loading_dialog.show()
                    loading_dialog.update_status(
                        "Loading neuron detection data...",
                        "Restoring detected neurons and trajectories",
                    )
                    QApplication.processEvents()

                    try:
                        self._load_neuron_detection_data()
                        loading_dialog.close_dialog()
                    except Exception:
                        loading_dialog.close_dialog()
        except Exception:
            pass

        # After any auto-load attempt, refresh workflow state from experiment data
        # (e.g., if ROI or detection data already exist).
        try:
            self.workflow_manager.refresh_state()
        except Exception:
            # Failing to refresh workflow should not break core functionality
            pass

    def _open_image_stack(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Select Image Stack Folder", "")
        if not directory:
            return
        self.viewer.set_stack(directory)
        # Reset downstream steps because the data source changed
        self.workflow_manager.reset_from_step(WorkflowStep.EDIT_IMAGES)

    def _on_stack_loaded(self, directory_path: str) -> None:
        # ImageStackHandler already updates experiment association for path/count
        self.stack_handler.associate_with_experiment(self.experiment)
        self._confirm_start_time_from_loaded_stack()
        self._apply_experiment_time_settings()

        # Update detection widget with frame data
        frame_data = self.stack_handler.get_all_frames_as_array()
        detection_widget = self.analysis.get_neuron_detection_widget()
        detection_widget.set_frame_data(frame_data)

        # Persist immediately if we know the path to the .nexp
        if self.current_experiment_path:
            try:
                self.manager.save_experiment(self.experiment, self.current_experiment_path)
            except Exception:
                pass

        # Mark first workflow step complete once a stack is available
        self.workflow_manager.complete_step_if_current(WorkflowStep.LOAD_IMAGES)

        # Cull step is always ready (zero exclusions is a valid choice)
        self.workflow_manager.mark_step_ready(WorkflowStep.CULL_FRAMES)

    def _confirm_start_time_from_loaded_stack(self) -> None:
        """Always prompt to confirm start time immediately after loading a stack."""
        if not self.stack_handler.files:
            return
        first_file = self.stack_handler.files[0]
        inferred = _get_exif_timestamp(first_file)
        qtime = _parse_time_string(inferred)

        if qtime is None:
            # Fall back to currently saved acquisition time, then midnight.
            acquisition = self.experiment.settings.get("acquisition") or {}
            qtime = _parse_time_string(acquisition.get("experiment_start_time")) or QTime(0, 0, 0)

        uniformity_note = None
        if len(self.stack_handler.files) > 1:
            sample_paths = [self.stack_handler.files[0], self.stack_handler.files[-1]]
            sampled = [_get_exif_timestamp(p) for p in sample_paths]
            if sampled[0] is not None and sampled[0] == sampled[1]:
                uniformity_note = (
                    "Warning: sampled image timestamps match exactly. "
                    "If this stack was pre-aligned/exported, adjust start time manually here."
                )
        if inferred is None:
            missing_note = (
                "No parseable timestamp was found in first-frame metadata. "
                "Please confirm the correct experiment start time."
            )
            uniformity_note = f"{uniformity_note}\n{missing_note}" if uniformity_note else missing_note

        dlg = _ConfirmStartTimeDialog(
            suggested_time=qtime,
            metadata_source=Path(first_file).name,
            timestamp_uniformity_note=uniformity_note,
            parent=self,
        )
        if dlg.exec() != QDialog.Accepted:
            return

        if "acquisition" not in self.experiment.settings:
            self.experiment.settings["acquisition"] = {}
        self.experiment.settings["acquisition"]["experiment_start_time"] = dlg.selected_start_time()
        self.experiment.update_modified_date()

    def _ensure_detection_data_saved(self) -> None:
        """
        Ensure neuron detection data is saved to the experiment.

        Checks if detection data exists in the experiment, and if not,
        retrieves it from the detection widget and saves it.
        """
        detection_data = self.experiment.get_neuron_detection_data()
        if detection_data is None or len(detection_data) == 0:
            # Data wasn't set, try to get it from the detection widget
            detection_widget = self.analysis.get_neuron_detection_widget()
            if (
                hasattr(detection_widget, "neuron_locations")
                and detection_widget.neuron_locations is not None
                and len(detection_widget.neuron_locations) > 0
            ):
                # Get detection params from widget if available
                detection_params = None
                if hasattr(detection_widget, "cell_size_spin"):
                    detection_params = {
                        "cell_size": detection_widget.cell_size_spin.value(),
                        "num_peaks": detection_widget.num_peaks_spin.value(),
                        "correlation_threshold": (detection_widget.correlation_threshold_spin.value()),
                        "threshold_rel": detection_widget.threshold_rel_spin.value(),
                        "apply_detrending": detection_widget.detrending_checkbox.isChecked(),
                    }
                roi_origin = detection_widget._compute_roi_origin()
                self.experiment.set_neuron_detection_data(
                    neuron_locations=detection_widget.neuron_locations,
                    neuron_trajectories=detection_widget.neuron_trajectories,
                    quality_mask=detection_widget.quality_mask,
                    mean_frame=detection_widget.mean_frame,
                    detection_params=detection_params,
                    roi_origin=roi_origin,
                )

    def _save_neuron_detection(self) -> None:
        """Save experiment when neuron detection completes."""
        if self.current_experiment_path:
            try:
                self._sync_rois_to_experiment()
                self._capture_display_settings()
                self._capture_experiment_time_settings()
                self._ensure_detection_data_saved()
                self.manager.save_experiment(self.experiment, self.current_experiment_path)
            except Exception as e:
                logger.error(f"Failed to save neuron detection data: {e}", exc_info=True)

    def _save(self) -> None:
        if not self.current_experiment_path:
            self._save_as()
            return
        self._flush_pending_display_settings()
        self._sync_rois_to_experiment()
        self._capture_display_settings()
        self._capture_experiment_time_settings()
        try:
            self.manager.save_experiment(self.experiment, self.current_experiment_path)
            QMessageBox.information(self, "Saved", "Experiment saved successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def _save_as(self) -> None:
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Experiment As", "", "Neurolight Experiment (*.nexp)")
        if not file_path:
            return
        if not file_path.endswith(".nexp"):
            file_path += ".nexp"
        self._flush_pending_display_settings()
        self._sync_rois_to_experiment()
        self._capture_display_settings()
        self._capture_experiment_time_settings()
        try:
            self.manager.save_experiment(self.experiment, file_path)
            self.set_current_experiment_path(file_path, persist_workflow=False)
            QMessageBox.information(self, "Saved", "Experiment saved successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def _close_experiment(self) -> None:
        """
        Close the current experiment and navigate to the home page (StartupDialog).
        This keeps the user in the application and allows them to select a new experiment.
        """
        # Prompt user if there are unsaved changes
        reply = QMessageBox.question(
            self,
            "Close Experiment",
            "Are you sure you want to close this experiment and return to the home page?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if reply == QMessageBox.No:
            return

        self._flush_pending_display_settings()
        self._sync_rois_to_experiment()
        self._capture_display_settings()
        if self.current_experiment_path:
            try:
                self.manager.save_experiment(self.experiment, self.current_experiment_path)
            except Exception:
                pass

        # Hide the main window
        self.hide()

        # Show startup dialog
        startup = StartupDialog()
        result = startup.exec()

        if result == QDialog.Accepted and startup.experiment is not None:
            # User selected a new experiment - replace current experiment
            self.experiment = startup.experiment
            self.set_current_experiment_path(startup.experiment_path, persist_workflow=False)
            self.workflow_manager.attach_experiment(self.experiment)
            self.setWindowTitle(f"Neurolight - {self.experiment.name}")

            # Reset viewer state
            self.viewer.reset()

            # Clear analysis panel (ROI intensity and trajectory graphs)
            self.analysis.roi_plot_widget.clear_plot()
            self.analysis.get_neuron_trajectory_plot_widget().clear_plot()

            # Reassociate handler and data analyzer with new experiment
            self.stack_handler.associate_with_experiment(self.experiment)
            self.data_analyzer = DataAnalyzer(self.experiment)

            # Auto-load image stack/ROI/display settings if experiment has saved data
            self._auto_load_experiment_data()

            # Show the window again
            self.show()
        else:
            # User canceled - exit the application
            QApplication.quit()

    def _exit_experiment(self) -> None:
        """
        Exit the entire application.
        This closes the application completely.
        """
        reply = QMessageBox.question(
            self,
            "Exit Experiment",
            "Are you sure you want to exit the application?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if reply == QMessageBox.Yes:
            self._flush_pending_display_settings()
            self._sync_rois_to_experiment()
            self._capture_display_settings()
            self._capture_experiment_time_settings()
            if self.current_experiment_path:
                try:
                    self.manager.save_experiment(self.experiment, self.current_experiment_path)
                except Exception:
                    pass
            QApplication.quit()

    def _on_roi_selected(self, roi_key: str, roi: ROI) -> None:
        """Handle ROI selection and extract intensity time series for *roi_key*."""
        try:
            frame_data = self.stack_handler.get_all_frames_as_array()
            if frame_data is None:
                QMessageBox.warning(
                    self,
                    "No Image Data",
                    "No image stack loaded. Please load an image stack first.",
                )
                return

            frame_min = np.min(frame_data)
            frame_max = np.max(frame_data)
            if frame_max > 1.0:
                frame_data = (
                    (frame_data - frame_min) / (frame_max - frame_min) if frame_max != frame_min else frame_data
                )

            intensity_data = self.data_analyzer.extract_roi_intensity_time_series(frame_data, roi=roi)

            roi_plot_widget = self.analysis.get_roi_plot_widget()
            roi_plot_widget.plot_intensity_time_series(roi_key, intensity_data, roi=roi)

            # Forward the same intensity data to the Lomb–Scargle periodogram widget so it
            # can reuse the existing ROI intensity pipeline without introducing a new source.
            try:
                lomb_scargle_widget = self.analysis.get_lomb_scargle_widget()
                lomb_scargle_widget.set_intensity_time_series(roi_key, intensity_data)
            except Exception:
                # Some tests or lightweight analysis panels may not expose this widget
                pass

            detection_widget = self.analysis.get_neuron_detection_widget()
            if frame_data.ndim == 3:
                _, height, width = frame_data.shape
                roi_mask_uint8 = roi.create_mask(width, height)
                roi_mask = (roi_mask_uint8 > 0).astype(bool)
                detection_widget.set_roi_mask(roi_key, roi_mask)
                detection_widget.set_frame_data(frame_data)

            self.workflow_manager.mark_step_ready(WorkflowStep.SELECT_ROI)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to analyze ROI:\n{str(e)}")

    def _on_roi_deleted(self, roi_key: str) -> None:
        """Handle ROI deletion: clear data from plot, detection, and experiment."""
        self.experiment.rois[roi_key] = None
        self.analysis.get_roi_plot_widget().clear_roi(roi_key)
        self.analysis.get_neuron_detection_widget().set_roi_mask(roi_key, None)
        if self.current_experiment_path:
            try:
                self.manager.save_experiment(self.experiment, self.current_experiment_path)
            except Exception:
                pass

    def _load_neuron_detection_data(self) -> None:
        """Load saved neuron detection data from experiment."""
        detection_data = self.experiment.get_neuron_detection_data()
        try:
            detection_widget = self.analysis.get_neuron_detection_widget()
        except Exception:
            detection_widget = None

        if detection_widget is None:
            return

        if detection_data is None:
            detection_widget.reset_detection_state()
            self.analysis.get_neuron_trajectory_plot_widget().clear_plot()
            return

        try:
            # Check if we have all required data (mean_frame is optional - can be recalculated)
            if (
                "neuron_locations" in detection_data
                and "neuron_trajectories" in detection_data
                and "quality_mask" in detection_data
            ):
                # Load the data into the detection widget
                # mean_frame is optional - will be recalculated if not present
                detection_widget.load_detection_data(
                    neuron_locations=detection_data["neuron_locations"],
                    neuron_trajectories=detection_data["neuron_trajectories"],
                    quality_mask=detection_data["quality_mask"],
                    mean_frame=detection_data.get("mean_frame"),  # Optional - can be recalculated
                    detection_params=detection_data.get("detection_params"),
                    roi_origin=detection_data.get("roi_origin"),  # Per-neuron ROI (0=ROI1, 1=ROI2)
                )
                # Advance workflow to analysis if detection data is restored.
                self.workflow_manager.mark_step_ready(WorkflowStep.DETECT_NEURONS)
                self.workflow_manager.complete_step_if_current(WorkflowStep.DETECT_NEURONS)
        except Exception:
            # Silently fail - detection data might be corrupted
            pass

    def _capture_display_settings(self) -> None:
        """
        Capture current display settings (exposure, contrast) from viewer into experiment.

        This method consolidates the logic for updating display settings in the experiment
        and should be called before saving, closing, or exiting.
        """
        if "display" not in self.experiment.settings:
            self.experiment.settings["display"] = {}
        self.experiment.settings["display"]["exposure"] = self.viewer.get_exposure()
        self.experiment.settings["display"]["contrast"] = self.viewer.get_contrast()

    def _capture_experiment_time_settings(self) -> None:
        """
        Capture experiment time settings into the experiment.

        Rayleigh start time is now sourced from acquisition metadata (first frame timestamp),
        so this method mirrors that value into legacy "time.start_minutes" for compatibility.
        """
        acquisition = self.experiment.settings.get("acquisition") or {}
        start_time = acquisition.get("experiment_start_time")
        start_minutes = _time_string_to_minutes(start_time)
        if start_minutes is None:
            return

        if "time" not in self.experiment.settings:
            self.experiment.settings["time"] = {}
        self.experiment.settings["time"]["start_minutes"] = start_minutes

    def _on_display_settings_changed(self, exposure: int, contrast: int) -> None:
        """
        Handle display settings changes with debounced saving.

        This method is called when the user adjusts exposure or contrast sliders.
        It stores the pending values and starts/resets a timer for debounced saving.
        """
        self._pending_exposure = exposure
        self._pending_contrast = contrast
        # Restart the timer (debounce: wait 500ms after last change before saving)
        self._display_settings_timer.stop()
        self._display_settings_timer.start(500)
        if self.workflow_manager.current_step == WorkflowStep.EDIT_IMAGES:
            self.workflow_manager.mark_step_ready(WorkflowStep.EDIT_IMAGES)

    def _on_frame_culling_changed(self, excluded: set) -> None:
        """Persist excluded frames to experiment and sync to the stack handler."""
        if "culling" not in self.experiment.settings:
            self.experiment.settings["culling"] = {}
        self.experiment.settings["culling"]["excluded_frames"] = sorted(excluded)
        self.stack_handler.set_excluded_frames(excluded)

        # Keep neuron detection input in sync with the included-only stack.
        # Detection operates on the frame_data provided to the widget; if culling
        # changes after load, we must refresh it.
        try:
            if hasattr(self, "analysis"):
                detection_widget = self.analysis.get_neuron_detection_widget()
                detection_widget.set_frame_data(self.stack_handler.get_all_frames_as_array())
        except Exception:
            # Some tests use lightweight/mocked panels; ignore refresh failures.
            pass

        total = self.stack_handler.get_total_frame_count()
        all_excluded = total > 0 and len(excluded) >= total

        if all_excluded:
            # Cannot advance with zero included frames — remove readiness
            self.workflow_manager._ready_steps.discard(WorkflowStep.CULL_FRAMES)
            self.workflow_manager.state_changed.emit()
        else:
            self.workflow_manager.mark_step_ready(WorkflowStep.CULL_FRAMES)

        # If culling changed after downstream steps completed, invalidate them
        if WorkflowStep.CULL_FRAMES in self.workflow_manager.completed_steps:
            self.workflow_manager.reset_from_step(WorkflowStep.CULL_FRAMES)
            if not all_excluded:
                self.workflow_manager.mark_step_ready(WorkflowStep.CULL_FRAMES)

        if self.current_experiment_path:
            try:
                self.manager.save_experiment(self.experiment, self.current_experiment_path)
            except Exception:
                pass

    def _restore_culling_state(self) -> None:
        """Restore excluded frames from experiment settings into viewer and handler."""
        culling = self.experiment.settings.get("culling") or {}
        excluded_list = culling.get("excluded_frames", [])
        excluded: set[int] = set()
        skipped = 0
        for item in excluded_list:
            try:
                excluded.add(int(item))
            except (ValueError, TypeError):
                skipped += 1
        if skipped:
            logger.warning("Skipped %d malformed excluded-frame entries", skipped)
        self.stack_handler.set_excluded_frames(excluded)
        self.viewer.set_excluded_frames(excluded)

        # Ensure detection uses the included-only stack after restoring culling.
        try:
            if hasattr(self, "analysis"):
                detection_widget = self.analysis.get_neuron_detection_widget()
                detection_widget.set_frame_data(self.stack_handler.get_all_frames_as_array())
        except Exception:
            pass

    def _flush_pending_display_settings(self) -> None:
        """
        Immediately flush any pending display settings without waiting for the timer.

        This is called when manually saving, closing, or exiting to ensure
        all pending changes are persisted immediately.
        """
        # Stop the timer and flush immediately
        self._display_settings_timer.stop()
        if self._pending_exposure is not None and self._pending_contrast is not None:
            self._persist_display_settings()

    def _persist_display_settings(self) -> None:
        """
        Persist pending display settings to experiment and save to file.

        This method is called by the debounce timer after the user stops adjusting sliders,
        or immediately when flushing pending settings.
        """
        if self._pending_exposure is not None and self._pending_contrast is not None:
            # Update experiment with pending values
            if "display" not in self.experiment.settings:
                self.experiment.settings["display"] = {}
            self.experiment.settings["display"]["exposure"] = self._pending_exposure
            self.experiment.settings["display"]["contrast"] = self._pending_contrast

            # Clear pending values
            self._pending_exposure = None
            self._pending_contrast = None

            # Persist to file if we have a path
            if self.current_experiment_path:
                try:
                    self.manager.save_experiment(self.experiment, self.current_experiment_path)
                except Exception:
                    pass

    def _save_roi_to_experiment(self, roi_key: str, roi: ROI) -> None:
        """Save a specific ROI to experiment and persist to .nexp file."""
        self.experiment.rois[roi_key] = roi.to_dict()
        if self.current_experiment_path:
            try:
                self._ensure_detection_data_saved()
                self.manager.save_experiment(self.experiment, self.current_experiment_path)
            except Exception:
                pass

    def autosave_experiment(self) -> None:
        if not self.experiment.settings.get("processing", {}).get("auto_save", True):
            return
        if not self.current_experiment_path:
            return
        self._flush_pending_display_settings()
        self._sync_rois_to_experiment()
        self._capture_display_settings()
        self._capture_experiment_time_settings()
        try:
            self.manager.save_experiment(self.experiment, self.current_experiment_path)
        except Exception:
            pass

    def _crop_stack_to_roi(self) -> None:
        """Crop the image stack to the active ROI and save as new stack."""
        current_roi = self.viewer.get_current_roi()
        if current_roi is None:
            QMessageBox.warning(self, "No ROI Selected", "Please select an ROI before cropping.")
            return

        try:
            # Load all frames
            frame_data = self.stack_handler.get_all_frames_as_array()
            if frame_data is None:
                QMessageBox.warning(self, "No Image Data", "No image stack loaded.")
                return

            # Ask user for output directory
            output_dir = QFileDialog.getExistingDirectory(self, "Select Output Directory for Cropped Stack", "")
            if not output_dir:
                return

            # Crop stack (apply mask for both ellipse and polygon)
            cropped_stack = self.image_processor.crop_stack_to_roi(frame_data, current_roi, apply_mask=True)

            # Save cropped frames
            from PIL import Image

            output_path = Path(output_dir)
            included_files = self.stack_handler.get_included_files()

            for i, cropped_frame in enumerate(cropped_stack):
                # Generate output filename from the matching source file
                if i < len(included_files):
                    original_name = Path(included_files[i]).stem
                    output_file = output_path / f"{original_name}_cropped.tif"
                else:
                    output_file = output_path / f"frame_{i:04d}_cropped.tif"

                # Convert frame to uint8 if needed
                if cropped_frame.dtype != np.uint8:
                    # Normalize to 0-255 range
                    frame_min = np.min(cropped_frame)
                    frame_max = np.max(cropped_frame)

                    if frame_max > frame_min:
                        # Scale to 0-255
                        normalized = (cropped_frame - frame_min) / (frame_max - frame_min)
                        cropped_frame = (normalized * 255).astype(np.uint8)
                    else:
                        # Constant frame - convert to uint8 with proper scaling
                        if np.issubdtype(cropped_frame.dtype, np.floating):
                            # For float dtypes, scale to 0-255 range
                            uint8_value = np.clip(np.round(frame_min * 255), 0, 255).astype(np.uint8)
                        else:
                            # For integer dtypes, just clip to 0-255
                            uint8_value = np.clip(frame_min, 0, 255).astype(np.uint8)
                        cropped_frame = np.full_like(cropped_frame, uint8_value, dtype=np.uint8)

                # Save frame
                img = Image.fromarray(cropped_frame)
                img.save(str(output_file))

            QMessageBox.information(
                self,
                "Cropping Complete",
                f"Cropped {len(cropped_stack)} frames to {output_dir}",
            )

        except Exception as e:
            QMessageBox.critical(self, "Cropping Error", f"Failed to crop image stack:\n{str(e)}")

    def _align_images(self) -> None:
        """Align images in the stack using a background worker thread."""
        # Guard: prevent re-entry while a worker is already running
        if self._alignment_worker is not None and self._alignment_worker.isRunning():
            QMessageBox.warning(
                self,
                "Alignment In Progress",
                "An alignment operation is already running. Please wait for it to finish.",
            )
            return

        # Check if images are loaded
        num_frames = self.stack_handler.get_image_count()
        if num_frames == 0:
            QMessageBox.warning(
                self,
                "No Images",
                "No image stack loaded. Please load an image stack first.",
            )
            return

        if num_frames < 2:
            QMessageBox.warning(
                self,
                "Not Enough Images",
                "At least 2 images are required for alignment.",
            )
            return

        # Show alignment dialog
        dialog = AlignmentDialog(self, num_frames)
        if dialog.exec() != QDialog.Accepted:
            return

        self._alignment_params = dialog.get_parameters()

        # Load frames on the main thread (PIL is not thread-safe)
        frame_data = self.stack_handler.get_all_frames_as_array()
        if frame_data is None:
            QMessageBox.warning(self, "No Image Data", "Failed to load image stack.")
            return

        # Show progress dialog
        self._alignment_progress = AlignmentProgressDialog(self, num_frames)
        self._alignment_progress.show()

        # Create and wire up the worker
        worker = AlignmentWorker(
            frame_data,
            transform_type=self._alignment_params["transform_type"],
            reference=self._alignment_params["reference"],
            enable_multiprocessing=get_enable_alignment_multiprocessing(),
            parent=self,
        )
        self._alignment_worker = worker

        worker.progress.connect(self._alignment_progress.update_progress)
        worker.finished.connect(self._on_alignment_finished)
        worker.error.connect(self._on_alignment_error)
        worker.cancelled.connect(self._on_alignment_cancelled)
        self._alignment_progress.rejected.connect(worker.request_cancel)

        worker.start()

    # ------------------------------------------------------------------
    # Alignment worker handler slots
    # ------------------------------------------------------------------

    def _on_alignment_finished(self, aligned_stack, tmats, confidence_scores) -> None:
        """Handle successful alignment completion."""
        self._alignment_progress.close()
        self._alignment_worker = None

        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
        low_confidence_frames = [i for i, conf in enumerate(confidence_scores) if conf < 0.5]

        result_message = (
            f"Alignment complete!\n\n"
            f"Average confidence: {avg_confidence:.2%}\n"
            f"Frames with low confidence (<50%): {len(low_confidence_frames)}"
        )

        if low_confidence_frames:
            result_message += f"\n\nLow confidence frames: {low_confidence_frames[:10]}"
            if len(low_confidence_frames) > 10:
                result_message += f" (+{len(low_confidence_frames) - 10} more)"

        reply = QMessageBox.question(
            self,
            "Alignment Complete",
            result_message + "\n\nWould you like to save the aligned images?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )

        if reply == QMessageBox.Yes:
            output_dir = QFileDialog.getExistingDirectory(
                self,
                "Select Output Directory for Aligned Stack",
                "",
            )
            if not output_dir:
                return

            self._save_aligned_stack(
                aligned_stack,
                tmats,
                confidence_scores,
                output_dir,
                self._alignment_params,
            )

            load_reply = QMessageBox.question(
                self,
                "Load Aligned Images?",
                "Would you like to load the aligned images into the viewer?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )

            if load_reply == QMessageBox.Yes:
                self.viewer.set_stack(output_dir)

        # Alignment finishing satisfies the alignment workflow step
        self.workflow_manager.complete_step_if_current(WorkflowStep.ALIGN_IMAGES)

    def _on_alignment_error(self, error_message: str) -> None:
        """Handle alignment errors from the worker thread."""
        self._alignment_progress.close()
        self._alignment_worker = None
        QMessageBox.critical(
            self,
            "Alignment Error",
            f"Failed to align images:\n{error_message}",
        )

    def _on_alignment_cancelled(self) -> None:
        """Handle alignment cancellation."""
        self._alignment_progress.close()
        self._alignment_worker = None
        QMessageBox.information(
            self,
            "Alignment Cancelled",
            "Image alignment was cancelled. No changes were saved.",
        )

    def _apply_exposure_contrast(self, arr: np.ndarray, exposure: int, contrast: int) -> np.ndarray:
        """
        Apply exposure and contrast adjustments to an image array.
        Uses the same logic as ImageViewer._apply_adjustments.

        Args:
            arr: Input image array
            exposure: Exposure value (-100 to 100)
            contrast: Contrast value (-100 to 100)

        Returns:
            Adjusted image array (normalized to 0-1 range)
        """
        # Store original dtype
        orig_dtype = arr.dtype
        # Convert to float32
        new_arr = arr.astype(np.float32, copy=True)
        min_pixel = float(np.min(new_arr))
        max_pixel = float(np.max(new_arr))
        pixel_range = max_pixel - min_pixel

        # Normalize to 0-1 range
        if pixel_range != 0:
            new_arr = (new_arr - min_pixel) / pixel_range
        else:
            # If all pixels are equal
            if np.issubdtype(orig_dtype, np.integer):
                max_possible = float(np.iinfo(orig_dtype).max)
                new_arr = new_arr / max_possible
            else:
                new_arr = np.clip(new_arr, 0, 1)

        # Apply exposure and contrast
        ev = exposure
        cv = contrast
        exposure_factor = 2 ** (ev / 50)
        contrast_factor = 1 + (cv / 100)
        # 0.5 to preserve greyscale
        new_arr = ((new_arr - 0.5) * contrast_factor + 0.5) * exposure_factor
        new_arr = np.clip(new_arr, 0, 1)

        return new_arr

    def _apply_exposure_contrast_global(
        self,
        arr: np.ndarray,
        exposure: int,
        contrast: int,
        global_min: float,
        global_range: float,
    ) -> np.ndarray:
        """
        Apply exposure and contrast adjustments to an image array using global normalization.
        This ensures consistent adjustments across all frames in a stack.

        Args:
            arr: Input image array
            exposure: Exposure value (-100 to 100)
            contrast: Contrast value (-100 to 100)
            global_min: Global minimum value across all frames
            global_range: Global range (global_max - global_min)

        Returns:
            Adjusted image array (normalized to 0-1 range)
        """
        # Convert to float32
        new_arr = arr.astype(np.float32, copy=True)

        # Normalize to 0-1 range using global min/max for consistency
        if global_range != 0:
            new_arr = (new_arr - global_min) / global_range
        else:
            # If all pixels are equal across entire stack
            new_arr = np.clip(new_arr, 0, 1)

        # Apply exposure and contrast
        ev = exposure
        cv = contrast
        exposure_factor = 2 ** (ev / 50)
        contrast_factor = 1 + (cv / 100)
        # 0.5 to preserve greyscale
        new_arr = ((new_arr - 0.5) * contrast_factor + 0.5) * exposure_factor
        new_arr = np.clip(new_arr, 0, 1)

        return new_arr

    def _save_aligned_stack(
        self,
        aligned_stack: np.ndarray,
        transformation_matrices: np.ndarray,
        confidence_scores: list,
        output_dir: str,
        params: dict,
    ) -> None:
        """Save aligned image stack to disk.

        Raw aligned images, no exposure/contrast adjustments.
        """
        import tifffile
        from pystackreg.util import to_uint16

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Convert to uint16 for saving (preserve original data range)
        aligned_stack_uint16 = to_uint16(aligned_stack)

        # Save aligned images (raw, without exposure/contrast adjustments)
        # Users can adjust exposure/contrast in the viewer after loading
        included_files = self.stack_handler.get_included_files()
        for i, aligned_frame in enumerate(aligned_stack_uint16):
            # Generate output filename from the matching source file
            if i < len(included_files):
                original_name = Path(included_files[i]).stem
                output_file = output_path / f"{original_name}_aligned.tif"
            else:
                output_file = output_path / f"frame_{i:04d}_aligned.tif"

            # Save using tifffile (preserves 16-bit precision)
            tifffile.imwrite(str(output_file), aligned_frame)

        # Save transformation matrices as numpy array
        matrices_path = output_path / "transformation_matrices.npy"
        np.save(str(matrices_path), transformation_matrices)

        # Save alignment metadata as JSON
        import json

        transform_data = {
            "transform_type": params.get("transform_type", "rigid_body"),
            "reference": params.get("reference", "first"),
            "reference_index": params.get("reference_index", 0),
            "num_frames": len(aligned_stack),
            "confidence_scores": [float(conf) for conf in confidence_scores],
            "average_confidence": (
                float(sum(confidence_scores) / len(confidence_scores)) if confidence_scores else 0.0
            ),
        }

        metadata_path = output_path / "alignment_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(transform_data, f, indent=2)

        QMessageBox.information(
            self,
            "Save Complete",
            f"Aligned {len(aligned_stack)} images saved to {output_dir}\nTransformation matrices and metadata saved.",
        )

    def _export_experiment(self) -> None:
        """Export the current experiment to a .nexp file."""
        try:
            self._flush_pending_display_settings()
            self._sync_rois_to_experiment()
            self._capture_display_settings()

            # Get export location
            default_name = f"{self.experiment.name}_export.nexp"
            if self.current_experiment_path:
                # Suggest a location near the original file
                original_path = Path(self.current_experiment_path)
                default_name = str(original_path.parent / f"{self.experiment.name}_export.nexp")

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
            if self.manager.save_experiment(self.experiment, file_path):
                QMessageBox.information(self, "Export Successful", f"Experiment exported to:\n{file_path}")
            else:
                QMessageBox.warning(self, "Export Failed", "Failed to export experiment.")
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"Failed to export experiment:\n{str(e)}")
