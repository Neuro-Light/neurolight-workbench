from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from ui.rayleigh_plot import RayLeighPlotWidget


@pytest.fixture
def app():
    if not QApplication.instance():
        return QApplication([])
    return QApplication.instance()


def _sample_trajectories() -> np.ndarray:
    return np.array(
        [
            [2.0, 1.0, 0.0, 1.0, 6.0, 1.0, 0.0, 1.0, 6.0, 1.0, 0.0, 1.0, 2.0],
            [2.0, 1.0, 0.0, 5.0, 1.0, 0.5, 0.0, 5.0, 1.0, 0.5, 0.0, 1.0, 2.0],
            [2.0, 1.0, 0.0, 0.5, 1.0, 5.0, 0.0, 0.5, 1.0, 5.0, 0.0, 1.0, 2.0],
        ]
    )


def test_set_trajectory_data_populates_cycle_selector(app) -> None:
    widget = RayLeighPlotWidget()
    widget.interval_spin.setValue(60)
    widget.interval_unit_combo.setCurrentText("min")

    widget.set_trajectory_data(
        _sample_trajectories(),
        quality_mask=np.array([True, True, True]),
        roi_origin=np.array([0, 1, 0]),
    )

    assert widget.cycle_combo.isEnabled() is True
    assert widget.cycle_combo.count() == 2
    assert widget.cycle_combo.itemText(0) == "Day 1"
    assert widget.cycle_combo.itemText(1) == "Day 2"
    assert "First peak frame = 3" in widget.cycle_info_label.text()
    assert widget.export_png_btn.isEnabled() is True
    assert widget.export_csv_btn.isEnabled() is True
    assert widget.rayleigh_stats_label.text() != ""


def test_plot_without_complete_cycles_disables_exports(app) -> None:
    widget = RayLeighPlotWidget()
    widget.interval_spin.setValue(30)
    widget.interval_unit_combo.setCurrentText("min")
    trajectories = np.array(
        [
            [0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 3.0],
            [0.1, 1.1, 3.1, 1.1, 0.1, 1.1, 3.1],
        ]
    )

    widget.set_trajectory_data(trajectories, quality_mask=np.array([True, True]))

    assert widget.cycle_combo.isEnabled() is False
    assert widget.cycle_combo.count() == 0
    assert widget.export_png_btn.isEnabled() is False
    assert widget.export_csv_btn.isEnabled() is False
    assert widget.cycle_info_label.text() == "No complete trough-to-trough cycles were detected."
