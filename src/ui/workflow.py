from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Optional, Set

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from core.experiment_manager import Experiment


class WorkflowStep(Enum):
    LOAD_IMAGES = auto()
    EDIT_IMAGES = auto()
    ALIGN_IMAGES = auto()
    SELECT_ROI = auto()
    DETECT_NEURONS = auto()
    ANALYZE_GRAPHS = auto()


class StepStatus(Enum):
    LOCKED = auto()
    ACTIVE = auto()
    COMPLETED = auto()


@dataclass(frozen=True)
class StepMeta:
    index: int
    short_label: str
    tooltip: str
    description: str


STEP_DEFINITIONS: Dict[WorkflowStep, StepMeta] = {
    WorkflowStep.LOAD_IMAGES: StepMeta(
        index=1,
        short_label="Load Image Stack",
        tooltip=(
            "Load the image stack to analyze. "
            "Use File → Open Image Stack or the Open Images button."
        ),
        description=(
            "Start by loading the image stack you want to analyze. "
            "You can use File → Open Image Stack, the Open Images "
            "button in the viewer, or drag-and-drop TIF/GIF files."
        ),
    ),
    WorkflowStep.EDIT_IMAGES: StepMeta(
        index=2,
        short_label="Edit Contrast & Exposure",
        tooltip=("Adjust exposure and contrast so structures of interest are clearly visible."),
        description=(
            "Use the Display options panel to adjust exposure and "
            "contrast until neurons and background are clearly separated."
        ),
    ),
    WorkflowStep.ALIGN_IMAGES: StepMeta(
        index=3,
        short_label="Align Images",
        tooltip="Align frames in the stack using PyStackReg to correct motion.",
        description=(
            "Run image alignment from the Tools → Align Images "
            "menu to correct for motion across the stack."
        ),
    ),
    WorkflowStep.SELECT_ROI: StepMeta(
        index=4,
        short_label="Select ROI",
        tooltip="Define the Region of Interest (ROI) where neurons will be detected.",
        description=(
            "Draw or adjust the ROI in the image viewer. The ROI "
            "defines the region used for intensity analysis and "
            "neuron detection."
        ),
    ),
    WorkflowStep.DETECT_NEURONS: StepMeta(
        index=5,
        short_label="Detect Neurons",
        tooltip="Run automated neuron detection inside the ROI.",
        description="Configure detection parameters, then run automated neuron detection. "
        "Results will be overlaid on the image and used for downstream analysis.",
    ),
    WorkflowStep.ANALYZE_GRAPHS: StepMeta(
        index=6,
        short_label="Analyze Graphs",
        tooltip="Inspect ROI intensity and neuron trajectory graphs.",
        description="Review ROI intensity traces and neuron trajectories in the analysis tabs. "
        "Use this step to interpret and validate the detected activity.",
    ),
}


class WorkflowManager(QObject):
    """
    Central controller for the guided workflow.

    Tracks the current step, completed steps, and persists state into the
    Experiment.settings["workflow"] structure so progress can be restored.
    """

    step_changed = Signal(WorkflowStep)
    state_changed = Signal()

    def __init__(self, experiment: Experiment, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._experiment = experiment
        self._steps: List[WorkflowStep] = list(WorkflowStep)

        self.current_step: WorkflowStep = WorkflowStep.LOAD_IMAGES
        self.completed_steps: Set[WorkflowStep] = set()
        self._ready_steps: Set[WorkflowStep] = set()

        self._load_or_initialize_state()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_step_status(self, step: WorkflowStep) -> StepStatus:
        if step in self.completed_steps:
            return StepStatus.COMPLETED
        if step == self.current_step:
            return StepStatus.ACTIVE
        return StepStatus.LOCKED

    def can_navigate_to_step(self, step: WorkflowStep) -> bool:
        """
        Users may navigate to the active step or any previously completed step,
        but not directly to locked future steps.
        """
        return step == self.current_step or step in self.completed_steps

    def complete_current_step(self) -> bool:
        """Advance to the next step only when the current step is marked ready."""
        if not self.is_step_ready(self.current_step):
            return False

        if self.current_step not in self.completed_steps:
            self.completed_steps.add(self.current_step)
        self._ready_steps.discard(self.current_step)

        next_step = self._get_next_step(self.current_step)
        if next_step is not None:
            self.current_step = next_step
            self.step_changed.emit(self.current_step)

        self._persist_state()
        self.state_changed.emit()
        return True

    def complete_step_if_current(self, step: WorkflowStep) -> None:
        """Convenience helper used by callers that know which step they satisfy."""
        if step == self.current_step:
            self.mark_step_ready(step)
            self.complete_current_step()

    def mark_step_ready(self, step: WorkflowStep) -> None:
        if step in self.completed_steps or step in self._ready_steps:
            return
        self._ready_steps.add(step)
        if step == self.current_step:
            self.state_changed.emit()

    def is_step_ready(self, step: WorkflowStep) -> bool:
        """Return True if the step has been completed or explicitly marked ready."""
        return step in self.completed_steps or step in self._ready_steps

    def attach_experiment(self, experiment: Experiment) -> None:
        self._experiment = experiment
        self.refresh_state()

    def refresh_state(self) -> None:
        """Reload workflow progress from the current experiment."""
        self._load_or_initialize_state()
        self.step_changed.emit(self.current_step)
        self.state_changed.emit()

    def reset_from_step(self, step: WorkflowStep) -> None:
        """
        Reset workflow from the given step onward.

        Used when upstream data changes (e.g., new image stack or new ROI),
        invalidating downstream results.
        """
        # Remove all completed steps at or after the given step
        indices = {s: STEP_DEFINITIONS[s].index for s in self.completed_steps}
        step_index = STEP_DEFINITIONS[step].index
        self.completed_steps = {s for s, idx in indices.items() if idx < step_index}
        self._ready_steps = {s for s in self._ready_steps if STEP_DEFINITIONS[s].index < step_index}

        # If current step is downstream of the reset point, move it back
        if STEP_DEFINITIONS[self.current_step].index >= step_index:
            self.current_step = step
            self.step_changed.emit(self.current_step)

        self._persist_state()
        self.state_changed.emit()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_next_step(self, step: WorkflowStep) -> Optional[WorkflowStep]:
        try:
            idx = self._steps.index(step)
        except ValueError:
            return None
        if idx + 1 < len(self._steps):
            return self._steps[idx + 1]
        return None

    def _load_or_initialize_state(self) -> None:
        """Load workflow state from experiment.settings or infer from data."""
        settings = self._experiment.settings
        workflow_raw = settings.get("workflow") or {}
        self.completed_steps.clear()
        self._ready_steps.clear()

        # Attempt to load explicit workflow state if present
        current_name = workflow_raw.get("current_step")
        completed_names = workflow_raw.get("completed_steps", [])

        if current_name:
            try:
                self.current_step = WorkflowStep[current_name]
            except KeyError:
                self.current_step = WorkflowStep.LOAD_IMAGES
        else:
            self.current_step = WorkflowStep.LOAD_IMAGES

        for name in completed_names:
            try:
                step = WorkflowStep[name]
            except KeyError:
                continue
            self.completed_steps.add(step)

        # If no explicit workflow state, try to infer reasonable defaults
        if not workflow_raw:
            self._infer_state_from_experiment()

        # Ensure state is persisted back into settings
        self._persist_state()

    def _infer_state_from_experiment(self) -> None:
        """
        Infer workflow progress from existing experiment data.

        This keeps existing experiments usable and roughly positions the user
        in the workflow when they open older .nexp files.
        """
        has_stack = bool(self._experiment.image_stack_path)
        has_roi = bool(self._experiment.roi)
        has_detection = self._experiment.get_neuron_detection_data() is not None

        if has_stack:
            self.completed_steps.add(WorkflowStep.LOAD_IMAGES)
            self.completed_steps.add(WorkflowStep.EDIT_IMAGES)

        if has_roi:
            self.completed_steps.add(WorkflowStep.SELECT_ROI)

        if has_detection:
            self.completed_steps.add(WorkflowStep.DETECT_NEURONS)
            self.completed_steps.add(WorkflowStep.ANALYZE_GRAPHS)

        # Set current step to the first not-completed step in order
        for step in self._steps:
            if step not in self.completed_steps:
                self.current_step = step
                break
        else:
            # All steps completed – keep current at final step
            self.current_step = WorkflowStep.ANALYZE_GRAPHS

    def _persist_state(self) -> None:
        """Persist current workflow state into the experiment.settings dict."""
        workflow_state = {
            "current_step": self.current_step.name,
            "completed_steps": [
                s.name
                for s in sorted(
                    self.completed_steps,
                    key=lambda s: STEP_DEFINITIONS[s].index,
                )
            ],
        }
        if "workflow" not in self._experiment.settings:
            self._experiment.settings["workflow"] = {}
        self._experiment.settings["workflow"].update(workflow_state)


class WorkflowStepper(QFrame):
    """
    Horizontal stepper widget that visually represents workflow progress.

    Shows numbered nodes, connecting lines, and a description for the
    current step. Nodes for completed steps are clickable so users can
    navigate backwards to make adjustments.
    """

    # Signals for step-specific actions that the main window can wire up
    requestAlignImages = Signal()
    requestSkipAlignment = Signal()

    def __init__(self, manager: WorkflowManager, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._manager = manager
        self._step_buttons: Dict[WorkflowStep, QToolButton] = {}

        self.setFrameShape(QFrame.NoFrame)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setObjectName("workflowStepper")

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(12, 8, 12, 8)
        root_layout.setSpacing(6)

        # Top row: step nodes
        steps_row = QHBoxLayout()
        steps_row.setSpacing(12)

        ordered_steps = sorted(
            STEP_DEFINITIONS.items(),
            key=lambda item: item[1].index,
        )

        total_steps = len(ordered_steps)
        for i, (step, meta) in enumerate(ordered_steps):
            btn = QToolButton()
            btn.setCheckable(True)
            btn.setAutoRaise(True)
            btn.setToolTip(meta.tooltip)
            btn.setText(f"{meta.index}\n{meta.short_label}")
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            btn.clicked.connect(self._make_step_clicked_handler(step))
            btn.setCursor(Qt.PointingHandCursor)
            self._step_buttons[step] = btn

            steps_row.addWidget(btn)
            if i < total_steps - 1:
                arrow = QLabel("\U000021e8")
                arrow.setAlignment(Qt.AlignCenter)
                arrow.setObjectName("workflowStepperArrow")
                arrow.setStyleSheet(
                    "color: #f97316; font-weight: 600; padding: 0 4px; font-size: 44px;"
                )
                arrow.setFixedWidth(28)
                steps_row.addWidget(arrow)

        root_layout.addLayout(steps_row)

        # Bottom row: description + step actions
        bottom_row = QHBoxLayout()
        bottom_row.setContentsMargins(0, 0, 0, 0)

        self._description_label = QLabel("")
        self._description_label.setWordWrap(True)
        self._description_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        bottom_row.addWidget(self._description_label, stretch=1)

        # Align Images button (shown only on ALIGN_IMAGES step)
        self._align_button = QPushButton("Align Images")
        self._align_button.setProperty("class", "primary")
        self._align_button.clicked.connect(self.requestAlignImages.emit)
        bottom_row.addWidget(self._align_button, stretch=0)

        # Skip alignment button (for already-aligned stacks)
        self._skip_align_button = QPushButton("Skip (already aligned)")
        self._skip_align_button.clicked.connect(self._on_skip_alignment_clicked)
        bottom_row.addWidget(self._skip_align_button, stretch=0)

        self._next_button = QPushButton("Next")
        self._next_button.setProperty("class", "primary")
        self._next_button.clicked.connect(self._on_next_clicked)
        bottom_row.addWidget(self._next_button, stretch=0)

        root_layout.addLayout(bottom_row)

        # Connect to manager signals
        self._manager.step_changed.connect(self._on_step_changed)
        self._manager.state_changed.connect(self._refresh)

        # Initial UI state
        self._refresh()

    # ------------------------------------------------------------------
    # Slots / handlers
    # ------------------------------------------------------------------

    def _make_step_clicked_handler(self, step: WorkflowStep):
        def handler() -> None:
            if not self._manager.can_navigate_to_step(step):
                return
            if step == self._manager.current_step:
                return
            # Navigate without changing completion state
            self._manager.current_step = step
            self._manager.step_changed.emit(step)
            self._manager._persist_state()  # type: ignore[attr-defined]
            self._manager.state_changed.emit()

        return handler

    def _on_step_changed(self, _step: WorkflowStep) -> None:
        self._refresh()

    def _on_next_clicked(self) -> None:
        self._manager.complete_current_step()

    def _on_skip_alignment_clicked(self) -> None:
        """
        Allow users to skip the alignment step explicitly when their stack
        is already aligned.
        """
        if self._manager.current_step == WorkflowStep.ALIGN_IMAGES:
            self._manager.mark_step_ready(WorkflowStep.ALIGN_IMAGES)
            self._manager.complete_current_step()

    # ------------------------------------------------------------------
    # UI refresh helpers
    # ------------------------------------------------------------------

    def _refresh(self) -> None:
        """Update button states and description to match manager state."""
        current = self._manager.current_step

        for step, button in self._step_buttons.items():
            status = self._manager.get_step_status(step)
            is_active = status == StepStatus.ACTIVE
            is_completed = status == StepStatus.COMPLETED

            button.setChecked(is_active)
            button.setEnabled(self._manager.can_navigate_to_step(step))
            self._apply_step_style(button, status)

            # Simple visual encoding via dynamic properties (picked up by stylesheet if desired)
            button.setProperty("workflowStatus", status.name.lower())
            button.style().unpolish(button)
            button.style().polish(button)

            # Prefix completed steps with a checkmark-like indicator in the text
            meta = STEP_DEFINITIONS[step]
            if is_completed:
                button.setText(f"✓ {meta.index}\n{meta.short_label}")
            else:
                button.setText(f"{meta.index}\n{meta.short_label}")

        # Description for current step
        current_meta = STEP_DEFINITIONS[current]
        self._description_label.setText(current_meta.description)

        # Next button only enabled when the active step is marked ready (and not the final step)
        can_advance = current != WorkflowStep.ANALYZE_GRAPHS and self._manager.is_step_ready(
            current
        )
        self._next_button.setEnabled(can_advance)

        # Show Align/Skip controls only on the alignment step
        is_align_step = current == WorkflowStep.ALIGN_IMAGES
        self._align_button.setVisible(is_align_step)
        self._skip_align_button.setVisible(is_align_step)

    def _apply_step_style(self, button: QToolButton, status: StepStatus) -> None:
        palette = {
            StepStatus.ACTIVE: ("#111827", "#f97316", "#ffffff", "600"),
            StepStatus.COMPLETED: ("#064e3b", "#10b981", "#e0f2f1", "500"),
            StepStatus.LOCKED: ("#1f2937", "#4b5563", "#94a3b8", "400"),
        }
        bg, border, text, weight = palette.get(status, ("#1f2937", "#4b5563", "#94a3b8", "400"))
        style = (
            "QToolButton {{"
            "border-radius: 10px;"
            "padding: 8px 6px;"
            "background-color: {bg};"
            "color: {fg};"
            "border: 2px solid {border};"
            "font-weight: {weight};"
            "}}"
        ).format(bg=bg, fg=text, border=border, weight=weight)
        button.setStyleSheet(style)
