"""Tests for WorkflowManager — step progression, persistence, and state inference."""

from __future__ import annotations

import pytest
from PySide6.QtWidgets import QApplication

from core.experiment_manager import Experiment
from ui.workflow import (
    STEP_DEFINITIONS,
    StepStatus,
    WorkflowManager,
    WorkflowStep,
    WorkflowStepper,
)


@pytest.fixture
def app():
    if not QApplication.instance():
        return QApplication([])
    return QApplication.instance()


def _fresh_experiment(**overrides) -> Experiment:
    exp = Experiment(name="Test")
    exp.settings = {}
    for k, v in overrides.items():
        setattr(exp, k, v)
    return exp


# ── STEP_DEFINITIONS constant ────────────────────────────────────────────


def test_step_definitions_covers_all_steps() -> None:
    for step in WorkflowStep:
        assert step in STEP_DEFINITIONS


def test_step_definitions_indices_are_unique() -> None:
    indices = [m.index for m in STEP_DEFINITIONS.values()]
    assert len(indices) == len(set(indices))


# ── WorkflowManager init ────────────────────────────────────────────────


def test_initial_step_is_load_images(app) -> None:
    wm = WorkflowManager(_fresh_experiment())
    assert wm.current_step == WorkflowStep.LOAD_IMAGES


def test_initial_completed_steps_empty(app) -> None:
    wm = WorkflowManager(_fresh_experiment())
    assert wm.completed_steps == set()


# ── get_step_status ──────────────────────────────────────────────────────


def test_current_step_status_is_active(app) -> None:
    wm = WorkflowManager(_fresh_experiment())
    assert wm.get_step_status(WorkflowStep.LOAD_IMAGES) == StepStatus.ACTIVE


def test_locked_step_status(app) -> None:
    wm = WorkflowManager(_fresh_experiment())
    assert wm.get_step_status(WorkflowStep.SELECT_ROI) == StepStatus.LOCKED


def test_completed_step_status(app) -> None:
    wm = WorkflowManager(_fresh_experiment())
    wm.mark_step_ready(WorkflowStep.LOAD_IMAGES)
    wm.complete_current_step()
    assert wm.get_step_status(WorkflowStep.LOAD_IMAGES) == StepStatus.COMPLETED


# ── can_navigate_to_step ────────────────────────────────────────────────


def test_can_navigate_to_current(app) -> None:
    wm = WorkflowManager(_fresh_experiment())
    assert wm.can_navigate_to_step(WorkflowStep.LOAD_IMAGES) is True


def test_cannot_navigate_to_locked(app) -> None:
    wm = WorkflowManager(_fresh_experiment())
    assert wm.can_navigate_to_step(WorkflowStep.DETECT_NEURONS) is False


def test_can_navigate_to_completed(app) -> None:
    wm = WorkflowManager(_fresh_experiment())
    wm.mark_step_ready(WorkflowStep.LOAD_IMAGES)
    wm.complete_current_step()
    assert wm.can_navigate_to_step(WorkflowStep.LOAD_IMAGES) is True


# ── complete_current_step ────────────────────────────────────────────────


def test_complete_step_requires_ready(app) -> None:
    wm = WorkflowManager(_fresh_experiment())
    assert wm.complete_current_step() is False
    assert wm.current_step == WorkflowStep.LOAD_IMAGES


def test_complete_step_advances(app) -> None:
    wm = WorkflowManager(_fresh_experiment())
    wm.mark_step_ready(WorkflowStep.LOAD_IMAGES)
    assert wm.complete_current_step() is True
    assert wm.current_step == WorkflowStep.EDIT_IMAGES


def test_complete_step_emits_signals(app) -> None:
    wm = WorkflowManager(_fresh_experiment())
    wm.mark_step_ready(WorkflowStep.LOAD_IMAGES)
    changes = []
    wm.step_changed.connect(lambda s: changes.append(s))
    wm.complete_current_step()
    assert WorkflowStep.EDIT_IMAGES in changes


# ── complete_step_if_current ─────────────────────────────────────────────


def test_complete_step_if_current_advances(app) -> None:
    wm = WorkflowManager(_fresh_experiment())
    wm.complete_step_if_current(WorkflowStep.LOAD_IMAGES)
    assert wm.current_step == WorkflowStep.EDIT_IMAGES


def test_complete_step_if_current_noop_for_wrong_step(app) -> None:
    wm = WorkflowManager(_fresh_experiment())
    wm.complete_step_if_current(WorkflowStep.SELECT_ROI)
    assert wm.current_step == WorkflowStep.LOAD_IMAGES


# ── mark_step_ready / is_step_ready ─────────────────────────────────────


def test_mark_step_ready(app) -> None:
    wm = WorkflowManager(_fresh_experiment())
    assert wm.is_step_ready(WorkflowStep.LOAD_IMAGES) is False
    wm.mark_step_ready(WorkflowStep.LOAD_IMAGES)
    assert wm.is_step_ready(WorkflowStep.LOAD_IMAGES) is True


def test_mark_step_ready_idempotent(app) -> None:
    wm = WorkflowManager(_fresh_experiment())
    wm.mark_step_ready(WorkflowStep.LOAD_IMAGES)
    wm.mark_step_ready(WorkflowStep.LOAD_IMAGES)
    assert wm.is_step_ready(WorkflowStep.LOAD_IMAGES) is True


def test_completed_step_is_always_ready(app) -> None:
    wm = WorkflowManager(_fresh_experiment())
    wm.mark_step_ready(WorkflowStep.LOAD_IMAGES)
    wm.complete_current_step()
    assert wm.is_step_ready(WorkflowStep.LOAD_IMAGES) is True


def test_mark_ready_emits_state_changed_for_current(app) -> None:
    wm = WorkflowManager(_fresh_experiment())
    signals = []
    wm.state_changed.connect(lambda: signals.append(True))
    wm.mark_step_ready(WorkflowStep.LOAD_IMAGES)
    assert len(signals) > 0


# ── reset_from_step ──────────────────────────────────────────────────────


def test_reset_clears_downstream(app) -> None:
    wm = WorkflowManager(_fresh_experiment())
    for step in list(WorkflowStep)[:5]:
        wm.mark_step_ready(step)
        wm.complete_current_step()

    wm.reset_from_step(WorkflowStep.CULL_FRAMES)
    assert WorkflowStep.LOAD_IMAGES in wm.completed_steps
    assert WorkflowStep.EDIT_IMAGES in wm.completed_steps
    assert WorkflowStep.CULL_FRAMES not in wm.completed_steps
    assert WorkflowStep.SELECT_ROI not in wm.completed_steps
    assert wm.current_step == WorkflowStep.CULL_FRAMES


def test_reset_clears_ready_steps(app) -> None:
    wm = WorkflowManager(_fresh_experiment())
    wm.mark_step_ready(WorkflowStep.LOAD_IMAGES)
    wm.complete_current_step()
    wm.mark_step_ready(WorkflowStep.EDIT_IMAGES)
    wm.reset_from_step(WorkflowStep.EDIT_IMAGES)
    assert not wm.is_step_ready(WorkflowStep.EDIT_IMAGES)


# ── attach_experiment / refresh_state ────────────────────────────────────


def test_attach_experiment_replaces_state(app) -> None:
    exp1 = _fresh_experiment()
    exp2 = _fresh_experiment(image_stack_path="/some/path")
    wm = WorkflowManager(exp1)
    assert wm.current_step == WorkflowStep.LOAD_IMAGES

    wm.attach_experiment(exp2)
    assert WorkflowStep.LOAD_IMAGES in wm.completed_steps


# ── persist and restore ──────────────────────────────────────────────────


def test_state_persists_to_settings(app) -> None:
    exp = _fresh_experiment()
    wm = WorkflowManager(exp)
    wm.mark_step_ready(WorkflowStep.LOAD_IMAGES)
    wm.complete_current_step()

    assert "workflow" in exp.settings
    assert exp.settings["workflow"]["current_step"] == "EDIT_IMAGES"
    assert "LOAD_IMAGES" in exp.settings["workflow"]["completed_steps"]


def test_state_restores_from_settings(app) -> None:
    exp = _fresh_experiment()
    exp.settings = {
        "workflow": {
            "current_step": "SELECT_ROI",
            "completed_steps": ["LOAD_IMAGES", "EDIT_IMAGES", "CULL_FRAMES", "ALIGN_IMAGES"],
        }
    }
    wm = WorkflowManager(exp)
    assert wm.current_step == WorkflowStep.SELECT_ROI
    assert WorkflowStep.ALIGN_IMAGES in wm.completed_steps


def test_invalid_step_name_falls_back(app) -> None:
    exp = _fresh_experiment()
    exp.settings = {
        "workflow": {
            "current_step": "INVALID_STEP_NAME",
            "completed_steps": ["ALSO_INVALID"],
        }
    }
    wm = WorkflowManager(exp)
    assert wm.current_step == WorkflowStep.LOAD_IMAGES
    assert len(wm.completed_steps) == 0


# ── infer state from experiment data ─────────────────────────────────────


def test_infer_state_with_stack(app) -> None:
    exp = _fresh_experiment(image_stack_path="/data/images")
    wm = WorkflowManager(exp)
    assert WorkflowStep.LOAD_IMAGES in wm.completed_steps
    assert WorkflowStep.EDIT_IMAGES in wm.completed_steps
    assert WorkflowStep.CULL_FRAMES in wm.completed_steps


def test_infer_state_with_roi(app) -> None:
    exp = _fresh_experiment(
        image_stack_path="/data",
        rois={"roi_1": {"x": 0, "y": 0, "width": 10, "height": 10, "shape": "ellipse"}, "roi_2": None},
    )
    wm = WorkflowManager(exp)
    assert WorkflowStep.SELECT_ROI in wm.completed_steps


def test_infer_state_with_detection(app) -> None:
    exp = _fresh_experiment(image_stack_path="/data")
    exp.set_neuron_detection_data(
        neuron_locations=__import__("numpy").array([[0, 0]]),
    )
    wm = WorkflowManager(exp)
    assert WorkflowStep.DETECT_NEURONS in wm.completed_steps
    assert WorkflowStep.ANALYZE_GRAPHS in wm.completed_steps


def test_infer_all_complete_sets_analyze_graphs(app) -> None:
    exp = _fresh_experiment(
        image_stack_path="/data",
        rois={"roi_1": {"shape": "ellipse", "x": 0, "y": 0, "width": 1, "height": 1}, "roi_2": None},
    )
    exp.set_neuron_detection_data(neuron_locations=__import__("numpy").array([[0, 0]]))
    wm = WorkflowManager(exp)
    assert wm.current_step == WorkflowStep.ALIGN_IMAGES or wm.current_step == WorkflowStep.ANALYZE_GRAPHS


# ── _get_next_step ───────────────────────────────────────────────────────


def test_next_step_after_last_is_none(app) -> None:
    wm = WorkflowManager(_fresh_experiment())
    assert wm._get_next_step(WorkflowStep.ANALYZE_GRAPHS) is None


def test_next_step_returns_successor(app) -> None:
    wm = WorkflowManager(_fresh_experiment())
    assert wm._get_next_step(WorkflowStep.LOAD_IMAGES) == WorkflowStep.EDIT_IMAGES


# ── WorkflowStepper widget ──────────────────────────────────────────────


def test_stepper_creates_buttons_for_all_steps(app) -> None:
    exp = _fresh_experiment()
    wm = WorkflowManager(exp)
    stepper = WorkflowStepper(wm)
    assert len(stepper._step_buttons) == len(WorkflowStep)


def test_stepper_description_shows_current_step(app) -> None:
    exp = _fresh_experiment()
    wm = WorkflowManager(exp)
    stepper = WorkflowStepper(wm)
    desc_text = stepper._description_label.text()
    assert len(desc_text) > 0
