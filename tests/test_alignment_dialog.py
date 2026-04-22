"""Tests for AlignmentDialog — parameter defaults, UI interactions, and accept validation."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from PySide6.QtWidgets import QApplication

from ui.alignment_dialog import AlignmentDialog


@pytest.fixture
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        return QApplication([])
    return app


# ---------------------------------------------------------------------------
# Default state
# ---------------------------------------------------------------------------


def test_default_num_frames(qapp: QApplication) -> None:
    dialog = AlignmentDialog(num_frames=5)
    assert dialog.num_frames == 5


def test_default_reference_index(qapp: QApplication) -> None:
    dialog = AlignmentDialog(num_frames=5)
    assert dialog.reference_index == 0


def test_default_transform_type(qapp: QApplication) -> None:
    dialog = AlignmentDialog(num_frames=5)
    assert dialog.transform_type == "rigid_body"


def test_default_reference_strategy(qapp: QApplication) -> None:
    dialog = AlignmentDialog(num_frames=5)
    assert dialog.reference_strategy == "first"


def test_get_parameters_returns_defaults(qapp: QApplication) -> None:
    dialog = AlignmentDialog(num_frames=5)
    assert dialog.get_parameters() == {
        "reference_index": 0,
        "transform_type": "rigid_body",
        "reference": "first",
    }


# ---------------------------------------------------------------------------
# Spinbox configuration
# ---------------------------------------------------------------------------


def test_spinbox_max_is_num_frames_minus_one(qapp: QApplication) -> None:
    dialog = AlignmentDialog(num_frames=10)
    assert dialog.reference_spinbox.maximum() == 9


def test_spinbox_max_is_zero_when_no_frames(qapp: QApplication) -> None:
    dialog = AlignmentDialog(num_frames=0)
    assert dialog.reference_spinbox.maximum() == 0


def test_spinbox_suffix_shows_frame_count(qapp: QApplication) -> None:
    dialog = AlignmentDialog(num_frames=8)
    assert "8 frames" in dialog.reference_spinbox.suffix()


def test_spinbox_suffix_empty_when_no_frames(qapp: QApplication) -> None:
    dialog = AlignmentDialog(num_frames=0)
    assert dialog.reference_spinbox.suffix() == ""


def test_spinbox_change_updates_reference_index(qapp: QApplication) -> None:
    dialog = AlignmentDialog(num_frames=5)
    dialog.reference_spinbox.setValue(3)
    assert dialog.reference_index == 3


# ---------------------------------------------------------------------------
# Transform type combo
# ---------------------------------------------------------------------------


def test_transform_combo_default_index(qapp: QApplication) -> None:
    dialog = AlignmentDialog(num_frames=5)
    assert dialog.transform_combo.currentIndex() == 0


def test_transform_combo_translation(qapp: QApplication) -> None:
    dialog = AlignmentDialog(num_frames=5)
    dialog.transform_combo.setCurrentIndex(1)
    assert dialog.transform_type == "translation"


def test_transform_combo_scaled_rotation(qapp: QApplication) -> None:
    dialog = AlignmentDialog(num_frames=5)
    dialog.transform_combo.setCurrentIndex(2)
    assert dialog.transform_type == "scaled_rotation"


def test_transform_combo_affine(qapp: QApplication) -> None:
    dialog = AlignmentDialog(num_frames=5)
    dialog.transform_combo.setCurrentIndex(3)
    assert dialog.transform_type == "affine"


def test_transform_combo_bilinear(qapp: QApplication) -> None:
    dialog = AlignmentDialog(num_frames=5)
    dialog.transform_combo.setCurrentIndex(4)
    assert dialog.transform_type == "bilinear"


# ---------------------------------------------------------------------------
# Reference strategy combo
# ---------------------------------------------------------------------------


def test_reference_combo_default_first(qapp: QApplication) -> None:
    dialog = AlignmentDialog(num_frames=5)
    assert dialog.reference_combo.currentIndex() == 0
    assert dialog.reference_strategy == "first"


def test_reference_combo_previous(qapp: QApplication) -> None:
    dialog = AlignmentDialog(num_frames=5)
    dialog.reference_combo.setCurrentIndex(1)
    assert dialog.reference_strategy == "previous"


def test_reference_combo_mean(qapp: QApplication) -> None:
    dialog = AlignmentDialog(num_frames=5)
    dialog.reference_combo.setCurrentIndex(2)
    assert dialog.reference_strategy == "mean"


# ---------------------------------------------------------------------------
# get_parameters reflects current UI state
# ---------------------------------------------------------------------------


def test_get_parameters_reflects_combo_changes(qapp: QApplication) -> None:
    dialog = AlignmentDialog(num_frames=5)
    dialog.transform_combo.setCurrentIndex(3)  # affine
    dialog.reference_combo.setCurrentIndex(2)  # mean
    dialog.reference_spinbox.setValue(2)

    params = dialog.get_parameters()
    assert params["transform_type"] == "affine"
    assert params["reference"] == "mean"
    assert params["reference_index"] == 2


# ---------------------------------------------------------------------------
# accept() validation
# ---------------------------------------------------------------------------


def test_accept_with_valid_frames_emits_accepted(qapp: QApplication) -> None:
    dialog = AlignmentDialog(num_frames=5)
    accepted_hits: list[bool] = []
    dialog.accepted.connect(lambda: accepted_hits.append(True))

    with patch("ui.alignment_dialog.QMessageBox.warning"):
        dialog.accept()

    assert accepted_hits == [True]


def test_accept_with_zero_frames_shows_warning(qapp: QApplication) -> None:
    dialog = AlignmentDialog(num_frames=0)
    with patch("ui.alignment_dialog.QMessageBox.warning") as mock_warn:
        dialog.accept()
    mock_warn.assert_called_once()


def test_accept_with_zero_frames_does_not_emit_accepted(qapp: QApplication) -> None:
    dialog = AlignmentDialog(num_frames=0)
    accepted_hits: list[bool] = []
    dialog.accepted.connect(lambda: accepted_hits.append(True))

    with patch("ui.alignment_dialog.QMessageBox.warning"):
        dialog.accept()

    assert accepted_hits == []


def test_accept_with_out_of_range_reference_shows_warning(qapp: QApplication) -> None:
    dialog = AlignmentDialog(num_frames=5)
    dialog.reference_index = 10  # Force an out-of-range value
    with patch("ui.alignment_dialog.QMessageBox.warning") as mock_warn:
        dialog.accept()
    mock_warn.assert_called_once()


def test_accept_with_out_of_range_reference_does_not_emit_accepted(qapp: QApplication) -> None:
    dialog = AlignmentDialog(num_frames=5)
    dialog.reference_index = 10
    accepted_hits: list[bool] = []
    dialog.accepted.connect(lambda: accepted_hits.append(True))

    with patch("ui.alignment_dialog.QMessageBox.warning"):
        dialog.accept()

    assert accepted_hits == []


def test_accept_with_boundary_reference_index_succeeds(qapp: QApplication) -> None:
    """Reference index equal to num_frames - 1 should be valid."""
    dialog = AlignmentDialog(num_frames=5)
    dialog.reference_spinbox.setValue(4)
    accepted_hits: list[bool] = []
    dialog.accepted.connect(lambda: accepted_hits.append(True))

    with patch("ui.alignment_dialog.QMessageBox.warning"):
        dialog.accept()

    assert accepted_hits == [True]
