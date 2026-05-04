from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
from matplotlib.collections import PathCollection
from PySide6.QtWidgets import QApplication

from core.rayleigh_cycles import RayleighCycleData
from ui.rayleigh_plot import RayLeighPlotWidget


@pytest.fixture
def app():
    if not QApplication.instance():
        return QApplication([])
    return QApplication.instance()


@pytest.fixture(autouse=True)
def _suppress_modal_message_boxes():
    with (
        patch("ui.rayleigh_plot.QMessageBox.warning"),
        patch("ui.rayleigh_plot.QMessageBox.information"),
    ):
        yield


def _sample_trajectories() -> np.ndarray:
    return np.array(
        [
            [2.0, 1.0, 0.0, 1.0, 6.0, 1.0, 0.0, 1.0, 6.0, 1.0, 0.0, 1.0, 2.0],
            [2.0, 1.0, 0.0, 5.0, 1.0, 0.5, 0.0, 5.0, 1.0, 0.5, 0.0, 1.0, 2.0],
            [2.0, 1.0, 0.0, 0.5, 1.0, 5.0, 0.0, 0.5, 1.0, 5.0, 0.0, 1.0, 2.0],
        ]
    )


def _first_cycle() -> RayleighCycleData:
    return RayleighCycleData(
        cycle_index=1,
        trough_start_frame=2,
        trough_end_frame=6,
        cycle_length_frames=4,
        cycle_length_minutes=240.0,
        first_peak_frame=3,
        neuron_indices=np.array([0, 1, 2]),
        peak_frames=np.array([4, 3, 5]),
        normalized_day_minutes=np.array([360.0, 0.0, 720.0]),
        theta=np.array([np.pi / 2, 0.0, np.pi]),
    )


def test_start_time_minutes_round_trip_and_wrap(app) -> None:
    widget = RayLeighPlotWidget()

    widget.set_experiment_start_time_minutes(1505)

    assert widget.get_experiment_start_time_minutes() == 65
    assert widget.start_time_edit.time().toString("HH:mm") == "01:05"


@pytest.mark.parametrize(
    ("unit_text", "expected_minutes"),
    [("sec", 2.0), ("min", 120.0), ("hr", 7200.0)],
)
def test_interval_conversion_and_display(app, unit_text: str, expected_minutes: float) -> None:
    widget = RayLeighPlotWidget()
    widget.interval_spin.setValue(120)
    widget.interval_unit_combo.setCurrentText(unit_text)

    assert widget._get_interval_minutes() == pytest.approx(expected_minutes)
    assert widget._get_interval_display() == f"120 {unit_text}"


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


def test_set_trajectory_data_empty_disables_plot_button(app) -> None:
    widget = RayLeighPlotWidget()

    widget.set_trajectory_data(np.empty((0, 4)))

    assert widget.plot_btn.isEnabled() is False
    assert widget.status_label.text() == "No neuron trajectories to display."


def test_set_trajectory_data_without_quality_mask_updates_status(app) -> None:
    widget = RayLeighPlotWidget()
    widget.canvas.draw_idle = Mock()

    widget.set_trajectory_data(_sample_trajectories(), quality_mask=None, roi_origin=None)

    assert widget.status_label.text() == "Ready to plot 3 neurons across 13 frames"


def test_plot_without_data_shows_warning(app) -> None:
    widget = RayLeighPlotWidget()

    with patch("ui.rayleigh_plot.QMessageBox.warning") as warning:
        widget._plot()

    warning.assert_called_once()


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


def test_plot_with_no_matching_filtered_neurons_clears_plot(app) -> None:
    widget = RayLeighPlotWidget()
    widget.neuron_trajectories = _sample_trajectories()
    widget.quality_mask = np.array([True, True, True])
    widget.roi_origin = np.array([0, 0, 0])
    widget.roi_view_combo.setCurrentText("ROI 2 only")
    widget.canvas.draw_idle = Mock()

    with patch("ui.rayleigh_plot.QMessageBox.information") as info:
        widget._plot()

    info.assert_called_once()
    assert widget._cycle_data == []
    assert widget.cycle_combo.isEnabled() is False
    assert widget.cycle_info_label.text() == "No neurons match the current ROI / quality filters."


def test_plot_selected_cycle_without_data_clears_plot(app) -> None:
    widget = RayLeighPlotWidget()
    widget.canvas.draw_idle = Mock()

    widget._plot_selected_cycle()

    assert widget.cycle_combo.isEnabled() is False
    assert widget.cycle_info_label.text() == "No complete trough-to-trough cycles available."


def test_plot_selected_cycle_recover_invalid_index_and_uses_fallback_scatter(app) -> None:
    widget = RayLeighPlotWidget()
    widget.canvas.draw_idle = Mock()
    widget._apply_theme = Mock()
    widget.roi_origin = np.array([0, 0, 0])
    widget._cycle_data = [_first_cycle()]
    widget._rebuild_cycle_combo()
    widget._set_cycle_controls_enabled(True)

    widget._plot_selected_cycle()
    widget.cycle_combo.setCurrentIndex(-1)
    widget._plot_selected_cycle()

    assert widget._current_cycle is widget._cycle_data[0]
    assert len(widget.figure.axes[0].collections) == 1


def test_rebuild_cycle_combo_restores_preferred_cycle(app) -> None:
    widget = RayLeighPlotWidget()
    widget._cycle_data = [
        _first_cycle(),
        RayleighCycleData(
            cycle_index=2,
            trough_start_frame=6,
            trough_end_frame=10,
            cycle_length_frames=4,
            cycle_length_minutes=240.0,
            first_peak_frame=7,
            neuron_indices=np.array([0, 1]),
            peak_frames=np.array([7, 8]),
            normalized_day_minutes=np.array([0.0, 360.0]),
            theta=np.array([0.0, np.pi / 2]),
        ),
    ]

    widget._rebuild_cycle_combo(preferred_cycle_index=2)

    assert widget.cycle_combo.currentIndex() == 1
    assert widget.cycle_combo.currentData() == 2


def test_plot_selected_cycle_splits_points_by_roi(app) -> None:
    widget = RayLeighPlotWidget()
    widget.canvas.draw_idle = Mock()
    widget._apply_theme = Mock()
    widget.roi_origin = np.array([0, 1, 0])
    widget._cycle_data = [_first_cycle()]
    widget._rebuild_cycle_combo()

    widget._plot_selected_cycle()

    assert len(widget.figure.axes[0].collections) == 2


def test_plot_selected_cycle_handles_stats_errors(app) -> None:
    widget = RayLeighPlotWidget()
    widget.canvas.draw_idle = Mock()
    widget._apply_theme = Mock()
    widget._cycle_data = [_first_cycle()]

    with (
        patch("ui.rayleigh_plot.rayleigh_test", side_effect=RuntimeError("boom")),
        patch("ui.rayleigh_plot.rao_spacing_test", side_effect=RuntimeError("boom")),
    ):
        widget._plot_selected_cycle()

    assert widget.rayleigh_stats_label.text() == ""
    assert widget.rao_stats_label.text() == ""


def test_refresh_theme_replots_only_with_data(app) -> None:
    widget = RayLeighPlotWidget()
    widget._plot = Mock()

    widget.refresh_theme()
    assert widget._plot.call_count == 0

    widget.neuron_trajectories = np.ones((1, 3), dtype=float)
    widget.refresh_theme()
    assert widget._plot.call_count == 1


def test_on_motion_updates_and_clears_cursor_text(app) -> None:
    widget = RayLeighPlotWidget()

    widget._on_motion(SimpleNamespace(inaxes=None, xdata=None, ydata=None))
    assert "Hover over the plot" in widget.cursor_label.text()

    widget._on_motion(SimpleNamespace(inaxes=object(), xdata=np.pi / 2, ydata=0.75))

    assert "Hover: θ = 90.0°" in widget.cursor_label.text()
    assert "06:00" in widget.cursor_label.text()
    assert "r = 0.750" in widget.cursor_label.text()


def test_on_pick_ignores_non_collection_and_missing_peak_minutes(app) -> None:
    widget = RayLeighPlotWidget()
    before = widget.cursor_label.text()

    widget._on_pick(SimpleNamespace(artist=object(), ind=[0]))
    assert widget.cursor_label.text() == before

    widget.figure.clear()
    ax = widget.figure.add_subplot(111, projection="polar")
    scatter = ax.scatter([0.0], [1.0], picker=5)
    widget._on_pick(SimpleNamespace(artist=scatter, ind=[0]))
    assert widget.cursor_label.text() == before


def test_on_pick_updates_selection_message(app) -> None:
    widget = RayLeighPlotWidget()
    widget.figure.clear()
    ax = widget.figure.add_subplot(111, projection="polar")
    scatter = ax.scatter([np.pi / 2], [0.8], picker=5)
    assert isinstance(scatter, PathCollection)
    scatter._rayleigh_peak_minutes = np.array([360.0])  # type: ignore[attr-defined]
    scatter._rayleigh_peak_frames = np.array([12])  # type: ignore[attr-defined]
    scatter._rayleigh_roi = "ROI 1"  # type: ignore[attr-defined]

    widget._on_pick(SimpleNamespace(artist=scatter, ind=[0]))

    assert "Selected (ROI 1): 1 neuron(s)" in widget.cursor_label.text()
    assert "normalized time = 06:00, frame = 12" in widget.cursor_label.text()
    assert "θ = 90.0°, r = 0.800" in widget.cursor_label.text()


def test_export_current_cycle_png_guard_and_save(app, tmp_path) -> None:
    widget = RayLeighPlotWidget()

    with patch("ui.rayleigh_plot.QMessageBox.information") as info:
        widget._export_current_cycle_png()
    info.assert_called_once()

    widget._current_cycle = _first_cycle()
    target = tmp_path / "cycle.png"
    widget.figure.savefig = Mock()

    with patch("ui.rayleigh_plot.QFileDialog.getSaveFileName", return_value=(str(target), "PNG Files (*.png)")):
        widget._export_current_cycle_png()

    widget.figure.savefig.assert_called_once_with(str(target), dpi=300, bbox_inches="tight")


def test_export_current_cycle_png_returns_when_cancelled(app) -> None:
    widget = RayLeighPlotWidget()
    widget._current_cycle = _first_cycle()
    widget.figure.savefig = Mock()

    with patch("ui.rayleigh_plot.QFileDialog.getSaveFileName", return_value=("", "")):
        widget._export_current_cycle_png()

    widget.figure.savefig.assert_not_called()


def test_export_current_cycle_csv_guard_and_cancel(app) -> None:
    widget = RayLeighPlotWidget()

    with patch("ui.rayleigh_plot.QMessageBox.information") as info:
        widget._export_current_cycle_csv()
    info.assert_called_once()

    widget._current_cycle = _first_cycle()
    with patch("ui.rayleigh_plot.QFileDialog.getSaveFileName", return_value=("", "")):
        widget._export_current_cycle_csv()


def test_export_current_cycle_csv_writes_expected_rows(app, tmp_path) -> None:
    widget = RayLeighPlotWidget()
    widget._current_cycle = _first_cycle()
    widget.roi_origin = np.array([0, 1, 2])
    target = tmp_path / "cycle.csv"
    captured: dict[str, object] = {}

    def _capture_savetxt(file_path, rows, **kwargs):
        captured["file_path"] = file_path
        captured["rows"] = rows
        captured["kwargs"] = kwargs

    with (
        patch("ui.rayleigh_plot.QFileDialog.getSaveFileName", return_value=(str(target), "CSV Files (*.csv)")),
        patch("ui.rayleigh_plot.np.savetxt", side_effect=_capture_savetxt),
    ):
        widget._export_current_cycle_csv()

    rows = captured["rows"]
    assert captured["file_path"] == str(target)
    assert isinstance(rows, np.ndarray)
    assert rows.shape == (3, 13)
    assert rows[:, 7].tolist() == ["ROI 1", "ROI 2", ""]
    assert rows[:, 8].tolist() == ["4", "3", "5"]
    assert "theta_degrees" in captured["kwargs"]["header"]


def test_export_current_cycle_csv_leaves_roi_labels_blank_when_origin_is_short(app, tmp_path) -> None:
    widget = RayLeighPlotWidget()
    widget._current_cycle = _first_cycle()
    widget.roi_origin = np.array([0, 1])
    target = tmp_path / "cycle.csv"
    captured: dict[str, object] = {}

    def _capture_savetxt(file_path, rows, **kwargs):
        captured["rows"] = rows

    with (
        patch("ui.rayleigh_plot.QFileDialog.getSaveFileName", return_value=(str(target), "CSV Files (*.csv)")),
        patch("ui.rayleigh_plot.np.savetxt", side_effect=_capture_savetxt),
    ):
        widget._export_current_cycle_csv()

    rows = captured["rows"]
    assert isinstance(rows, np.ndarray)
    assert rows[:, 7].tolist() == ["", "", ""]
