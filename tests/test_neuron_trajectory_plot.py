"""Tests for neuron_trajectory_plot module."""

import matplotlib

matplotlib.use("Agg")

from unittest.mock import Mock

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from ui.neuron_trajectory_plot import NeuronTrajectoryPlotWidget, _smooth_display


@pytest.fixture
def app():
    if not QApplication.instance():
        return QApplication([])
    return QApplication.instance()


class TestSmoothDisplay:
    """Tests for the _smooth_display pure function."""

    def test_returns_input_when_window_less_than_2(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        result = _smooth_display(y, 1)
        np.testing.assert_array_equal(result, y)

    def test_returns_input_when_window_is_zero(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        result = _smooth_display(y, 0)
        np.testing.assert_array_equal(result, y)

    def test_returns_input_when_array_shorter_than_window(self):
        y = np.array([1.0, 2.0], dtype=np.float32)
        result = _smooth_display(y, 5)
        np.testing.assert_array_equal(result, y)

    def test_applies_moving_average_with_valid_window(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        result = _smooth_display(y, 3)
        assert result.dtype == np.float32
        assert len(result) == len(y)
        # Middle value should be average of 2, 3, 4
        assert result[2] == pytest.approx(3.0, rel=1e-5)

    def test_output_dtype_is_float32(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        result = _smooth_display(y, 3)
        assert result.dtype == np.float32

    def test_with_integer_input(self):
        y = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        result = _smooth_display(y, 3)
        assert result.dtype == np.float32
        assert len(result) == 5


class TestNeuronTrajectoryPlotWidgetInit:
    """Tests for widget initialization."""

    def test_initial_state_has_no_data(self, app):
        w = NeuronTrajectoryPlotWidget()
        assert w.neuron_trajectories is None
        assert w.quality_mask is None
        assert w.neuron_locations is None
        assert w.roi_origin is None

    def test_export_buttons_disabled_initially(self, app):
        w = NeuronTrajectoryPlotWidget()
        assert w.export_btn.isEnabled() is False
        assert w.export_png_btn.isEnabled() is False

    def test_status_label_shows_initial_message(self, app):
        w = NeuronTrajectoryPlotWidget()
        assert "No neuron trajectories" in w.status_label.text()

    def test_show_peaks_checkbox_unchecked_initially(self, app):
        w = NeuronTrajectoryPlotWidget()
        assert w.show_peaks_checkbox.isChecked() is False

    def test_number_peaks_checkbox_hidden_initially(self, app):
        w = NeuronTrajectoryPlotWidget()
        assert w.number_peaks_checkbox.isVisible() is False


class TestNeuronTrajectoryPlotWidgetFindPeaks:
    """Tests for the _find_peaks_and_troughs method."""

    def test_returns_empty_for_short_data(self, app):
        w = NeuronTrajectoryPlotWidget()
        data = np.array([1.0, 2.0])
        peaks, troughs = w._find_peaks_and_troughs(data)
        assert len(peaks) == 0
        assert len(troughs) == 0

    def test_finds_single_peak(self, app):
        w = NeuronTrajectoryPlotWidget()
        # Create data with a clear peak in the middle
        data = np.array([0.0, 0.5, 1.0, 0.5, 0.0])
        peaks, troughs = w._find_peaks_and_troughs(data)
        assert 2 in peaks  # Peak at index 2

    def test_finds_single_trough(self, app):
        w = NeuronTrajectoryPlotWidget()
        # Create data with a clear trough in the middle
        data = np.array([1.0, 0.5, 0.0, 0.5, 1.0])
        peaks, troughs = w._find_peaks_and_troughs(data)
        assert 2 in troughs  # Trough at index 2

    def test_returns_numpy_arrays(self, app):
        w = NeuronTrajectoryPlotWidget()
        data = np.array([0.0, 1.0, 0.0, 1.0, 0.0])
        peaks, troughs = w._find_peaks_and_troughs(data)
        assert isinstance(peaks, np.ndarray)
        assert isinstance(troughs, np.ndarray)


class TestNeuronTrajectoryPlotWidgetPlotTrajectories:
    """Tests for plotting trajectories."""

    def test_plot_trajectories_enables_export_buttons(self, app):
        w = NeuronTrajectoryPlotWidget()
        w.canvas.draw_idle = Mock()
        trajectories = np.random.rand(5, 100)
        w.plot_trajectories(trajectories)
        assert w.export_btn.isEnabled() is True
        assert w.export_png_btn.isEnabled() is True

    def test_plot_trajectories_with_quality_mask(self, app):
        w = NeuronTrajectoryPlotWidget()
        w.canvas.draw_idle = Mock()
        trajectories = np.random.rand(5, 100)
        quality_mask = np.array([True, True, False, True, False])
        w.plot_trajectories(trajectories, quality_mask=quality_mask)
        assert w.quality_mask is not None
        # Status should mention trajectories count
        assert "trajectories" in w.status_label.text().lower()

    def test_plot_trajectories_with_empty_array_disables_export(self, app):
        w = NeuronTrajectoryPlotWidget()
        w.plot_trajectories(np.array([]))
        assert w.export_btn.isEnabled() is False

    def test_plot_trajectories_with_none_disables_export(self, app):
        w = NeuronTrajectoryPlotWidget()
        w.plot_trajectories(None)
        assert w.export_btn.isEnabled() is False


class TestNeuronTrajectoryPlotWidgetClearPlot:
    """Tests for clear_plot method."""

    def test_clear_plot_resets_state(self, app):
        w = NeuronTrajectoryPlotWidget()
        w.canvas.draw = Mock()
        w.canvas.draw_idle = Mock()

        # Set some state
        w.neuron_trajectories = np.random.rand(5, 100)
        w.quality_mask = np.array([True, True, False, True, False])
        w.export_btn.setEnabled(True)
        w.export_png_btn.setEnabled(True)

        w.clear_plot()

        assert w.neuron_trajectories is None
        assert w.quality_mask is None
        assert w.export_btn.isEnabled() is False
        assert w.export_png_btn.isEnabled() is False

    def test_clear_plot_resets_status_label(self, app):
        w = NeuronTrajectoryPlotWidget()
        w.canvas.draw = Mock()
        w.status_label.setText("Some other text")
        w.clear_plot()
        assert "No neuron trajectories" in w.status_label.text()


class TestNeuronTrajectoryPlotWidgetGetDisplayedIndices:
    """Tests for _get_displayed_neuron_indices method."""

    def test_returns_empty_list_when_no_data(self, app):
        w = NeuronTrajectoryPlotWidget()
        result = w._get_displayed_neuron_indices()
        assert result == []

    def test_returns_indices_up_to_max(self, app):
        w = NeuronTrajectoryPlotWidget()
        w.neuron_trajectories = np.random.rand(100, 50)
        w.max_neurons_spin.setValue(10)
        result = w._get_displayed_neuron_indices()
        assert len(result) <= 10

    def test_respects_quality_mask_show_good(self, app):
        w = NeuronTrajectoryPlotWidget()
        w.neuron_trajectories = np.random.rand(10, 50)
        w.quality_mask = np.array([True, False, True, False, True, False, True, False, True, False])
        w.show_good_checkbox.setChecked(True)
        w.show_bad_checkbox.setChecked(False)
        w.max_neurons_spin.setValue(100)
        result = w._get_displayed_neuron_indices()
        # Should only include good neurons (indices 0, 2, 4, 6, 8)
        for idx in result:
            assert w.quality_mask[idx] == True  # noqa: E712 - numpy bool comparison

    def test_respects_quality_mask_show_bad(self, app):
        w = NeuronTrajectoryPlotWidget()
        w.neuron_trajectories = np.random.rand(10, 50)
        w.quality_mask = np.array([True, False, True, False, True, False, True, False, True, False])
        w.show_good_checkbox.setChecked(False)
        w.show_bad_checkbox.setChecked(True)
        w.max_neurons_spin.setValue(100)
        result = w._get_displayed_neuron_indices()
        # Should only include bad neurons (indices 1, 3, 5, 7, 9)
        for idx in result:
            assert w.quality_mask[idx] == False  # noqa: E712 - numpy bool comparison


class TestNeuronTrajectoryPlotWidgetPeaksToggle:
    """Tests for peaks/troughs toggle behavior."""

    def test_show_peaks_toggle_enables_number_checkbox_visibility(self, app):
        w = NeuronTrajectoryPlotWidget()
        w.canvas.draw_idle = Mock()
        # Widget not shown, so use the internal visibility flag via show()
        w.show()
        assert w.number_peaks_checkbox.isVisible() is False
        w.show_peaks_checkbox.setChecked(True)
        # The checkbox visibility is set to True programmatically
        assert w._number_peaks_row_label.isVisible() is True

    def test_hide_peaks_toggle_hides_number_checkbox(self, app):
        w = NeuronTrajectoryPlotWidget()
        w.canvas.draw_idle = Mock()
        w.show_peaks_checkbox.setChecked(True)
        w.show_peaks_checkbox.setChecked(False)
        assert w.number_peaks_checkbox.isVisible() is False

    def test_hide_peaks_unchecks_number_checkbox(self, app):
        w = NeuronTrajectoryPlotWidget()
        w.canvas.draw_idle = Mock()
        w.show_peaks_checkbox.setChecked(True)
        w.number_peaks_checkbox.setChecked(True)
        w.show_peaks_checkbox.setChecked(False)
        assert w.number_peaks_checkbox.isChecked() is False


class TestNeuronTrajectoryPlotWidgetGetPreviousMarkerFrame:
    """Tests for _get_previous_marker_frame method."""

    def test_returns_none_for_first_marker(self, app):
        w = NeuronTrajectoryPlotWidget()
        w._peak_data = [(10, 1.0, "peak", 1), (20, 1.5, "peak", 2)]
        result = w._get_previous_marker_frame(10, "peak")
        assert result is None

    def test_returns_previous_frame(self, app):
        w = NeuronTrajectoryPlotWidget()
        w._peak_data = [(10, 1.0, "peak", 1), (20, 1.5, "peak", 2), (30, 1.2, "peak", 3)]
        result = w._get_previous_marker_frame(20, "peak")
        assert result == 10

    def test_handles_frame_zero(self, app):
        w = NeuronTrajectoryPlotWidget()
        w._peak_data = [(0, 1.0, "peak", 1), (20, 1.5, "peak", 2)]
        result = w._get_previous_marker_frame(20, "peak")
        assert result == 0

    def test_uses_correct_marker_type(self, app):
        w = NeuronTrajectoryPlotWidget()
        w._peak_data = [(10, 1.0, "peak", 1)]
        w._trough_data = [(5, 0.5, "trough", 1), (15, 0.3, "trough", 2)]
        result = w._get_previous_marker_frame(15, "trough")
        assert result == 5
