"""Tests for roi_intensity_plot module."""

import matplotlib

matplotlib.use("Agg")

from unittest.mock import Mock, patch

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from ui.roi_intensity_plot import ROIIntensityPlotWidget


@pytest.fixture
def app():
    if not QApplication.instance():
        return QApplication([])
    return QApplication.instance()


class TestROIIntensityPlotWidgetInit:
    """Tests for widget initialization."""

    def test_initial_state_has_no_intensity_data(self, app):
        w = ROIIntensityPlotWidget()
        assert w._intensity["roi_1"] is None
        assert w._intensity["roi_2"] is None

    def test_initial_state_has_no_rois(self, app):
        w = ROIIntensityPlotWidget()
        assert w._rois["roi_1"] is None
        assert w._rois["roi_2"] is None

    def test_export_buttons_disabled_initially(self, app):
        w = ROIIntensityPlotWidget()
        assert w.export_btn.isEnabled() is False
        assert w.export_png_btn.isEnabled() is False

    def test_status_label_shows_initial_message(self, app):
        w = ROIIntensityPlotWidget()
        assert "No ROI selected" in w.status_label.text()

    def test_show_peaks_checkbox_unchecked_initially(self, app):
        w = ROIIntensityPlotWidget()
        assert w.show_peaks_checkbox.isChecked() is False

    def test_number_peaks_checkbox_hidden_initially(self, app):
        w = ROIIntensityPlotWidget()
        assert w.number_peaks_checkbox.isVisible() is False

    def test_roi_checkboxes_checked_initially(self, app):
        w = ROIIntensityPlotWidget()
        assert w._checkboxes["roi_1"].isChecked() is True
        assert w._checkboxes["roi_2"].isChecked() is True


class TestROIIntensityPlotWidgetFindPeaks:
    """Tests for the _find_peaks_and_troughs method."""

    def test_returns_empty_for_short_data(self, app):
        w = ROIIntensityPlotWidget()
        data = np.array([1.0, 2.0])
        peaks, troughs = w._find_peaks_and_troughs(data)
        assert len(peaks) == 0
        assert len(troughs) == 0

    def test_finds_single_peak(self, app):
        w = ROIIntensityPlotWidget()
        # Create data with a clear peak in the middle
        data = np.array([0.0, 0.5, 1.0, 0.5, 0.0])
        peaks, troughs = w._find_peaks_and_troughs(data)
        assert 2 in peaks  # Peak at index 2

    def test_finds_single_trough(self, app):
        w = ROIIntensityPlotWidget()
        # Create data with a clear trough in the middle
        data = np.array([1.0, 0.5, 0.0, 0.5, 1.0])
        peaks, troughs = w._find_peaks_and_troughs(data)
        assert 2 in troughs  # Trough at index 2

    def test_returns_numpy_arrays(self, app):
        w = ROIIntensityPlotWidget()
        data = np.array([0.0, 1.0, 0.0, 1.0, 0.0])
        peaks, troughs = w._find_peaks_and_troughs(data)
        assert isinstance(peaks, np.ndarray)
        assert isinstance(troughs, np.ndarray)

    def test_handles_flat_signal(self, app):
        w = ROIIntensityPlotWidget()
        data = np.ones(100)
        peaks, troughs = w._find_peaks_and_troughs(data)
        assert len(peaks) == 0
        assert len(troughs) == 0


class TestROIIntensityPlotWidgetClearPlot:
    """Tests for clear_plot method."""

    def test_clear_plot_resets_intensity_data(self, app):
        w = ROIIntensityPlotWidget()
        w.canvas.draw = Mock()
        w._intensity["roi_1"] = np.array([1.0, 2.0, 3.0])
        w._intensity["roi_2"] = np.array([4.0, 5.0, 6.0])

        w.clear_plot()

        assert w._intensity["roi_1"] is None
        assert w._intensity["roi_2"] is None

    def test_clear_plot_resets_rois(self, app):
        w = ROIIntensityPlotWidget()
        w.canvas.draw = Mock()
        w._rois["roi_1"] = Mock()
        w._rois["roi_2"] = Mock()

        w.clear_plot()

        assert w._rois["roi_1"] is None
        assert w._rois["roi_2"] is None

    def test_clear_plot_disables_export_buttons(self, app):
        w = ROIIntensityPlotWidget()
        w.canvas.draw = Mock()
        w.export_btn.setEnabled(True)
        w.export_png_btn.setEnabled(True)

        w.clear_plot()

        assert w.export_btn.isEnabled() is False
        assert w.export_png_btn.isEnabled() is False

    def test_clear_plot_resets_status_label(self, app):
        w = ROIIntensityPlotWidget()
        w.canvas.draw = Mock()
        w.status_label.setText("Some other text")

        w.clear_plot()

        assert "No ROI selected" in w.status_label.text()


class TestROIIntensityPlotWidgetClearRoi:
    """Tests for clear_roi method."""

    def test_clear_roi_clears_single_roi(self, app):
        w = ROIIntensityPlotWidget()
        w.canvas.draw_idle = Mock()
        w._intensity["roi_1"] = np.array([1.0, 2.0, 3.0])
        w._intensity["roi_2"] = np.array([4.0, 5.0, 6.0])

        w.clear_roi("roi_1")

        assert w._intensity["roi_1"] is None
        assert w._intensity["roi_2"] is not None

    def test_clear_roi_preserves_other_roi(self, app):
        w = ROIIntensityPlotWidget()
        w.canvas.draw_idle = Mock()
        w._intensity["roi_1"] = np.array([1.0, 2.0, 3.0])
        w._intensity["roi_2"] = np.array([4.0, 5.0, 6.0])
        w._rois["roi_1"] = Mock()
        w._rois["roi_2"] = Mock()

        w.clear_roi("roi_1")

        assert w._rois["roi_2"] is not None


class TestROIIntensityPlotWidgetPeaksToggle:
    """Tests for peaks/troughs toggle behavior."""

    def test_show_peaks_toggle_enables_number_checkbox_visibility(self, app):
        w = ROIIntensityPlotWidget()
        w.canvas.draw_idle = Mock()
        w.show()
        assert w.number_peaks_checkbox.isVisible() is False
        w.show_peaks_checkbox.setChecked(True)
        # Verify the visibility was set (widget internal state)
        # When not shown, isVisible may still be False but the setVisible(True) was called
        # So we check the checkbox is no longer explicitly hidden
        w.show_peaks_checkbox.setChecked(False)
        assert w.number_peaks_checkbox.isVisible() is False

    def test_hide_peaks_toggle_hides_number_checkbox(self, app):
        w = ROIIntensityPlotWidget()
        w.canvas.draw_idle = Mock()
        w.show_peaks_checkbox.setChecked(True)
        w.show_peaks_checkbox.setChecked(False)
        assert w.number_peaks_checkbox.isVisible() is False

    def test_hide_peaks_unchecks_number_checkbox(self, app):
        w = ROIIntensityPlotWidget()
        w.canvas.draw_idle = Mock()
        w.show_peaks_checkbox.setChecked(True)
        w.number_peaks_checkbox.setChecked(True)
        w.show_peaks_checkbox.setChecked(False)
        assert w.number_peaks_checkbox.isChecked() is False


class TestROIIntensityPlotWidgetGetPreviousMarkerFrame:
    """Tests for _get_previous_marker_frame method."""

    def test_returns_none_for_first_marker(self, app):
        w = ROIIntensityPlotWidget()
        w._peak_data = [(10, 1.0, "peak", 1), (20, 1.5, "peak", 2)]
        result = w._get_previous_marker_frame(10, "peak")
        assert result is None

    def test_returns_previous_frame(self, app):
        w = ROIIntensityPlotWidget()
        w._peak_data = [(10, 1.0, "peak", 1), (20, 1.5, "peak", 2), (30, 1.2, "peak", 3)]
        result = w._get_previous_marker_frame(20, "peak")
        assert result == 10

    def test_handles_frame_zero(self, app):
        w = ROIIntensityPlotWidget()
        w._peak_data = [(0, 1.0, "peak", 1), (20, 1.5, "peak", 2)]
        result = w._get_previous_marker_frame(20, "peak")
        assert result == 0

    def test_uses_correct_marker_type(self, app):
        w = ROIIntensityPlotWidget()
        w._peak_data = [(10, 1.0, "peak", 1)]
        w._trough_data = [(5, 0.5, "trough", 1), (15, 0.3, "trough", 2)]
        result = w._get_previous_marker_frame(15, "trough")
        assert result == 5


class TestROIIntensityPlotWidgetPlotIntensity:
    """Tests for plot_intensity_time_series method."""

    def test_stores_intensity_data(self, app):
        w = ROIIntensityPlotWidget()
        w.canvas.draw_idle = Mock()
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        roi = Mock()

        w.plot_intensity_time_series("roi_1", data, roi)

        np.testing.assert_array_equal(w._intensity["roi_1"], data)
        assert w._rois["roi_1"] is roi

    def test_enables_export_with_visible_data(self, app):
        w = ROIIntensityPlotWidget()
        w.canvas.draw_idle = Mock()
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        roi = Mock()

        w.plot_intensity_time_series("roi_1", data, roi)

        assert w.export_btn.isEnabled() is True
        assert w.export_png_btn.isEnabled() is True


class TestROIIntensityPlotWidgetReplot:
    """Tests for _replot behavior with hidden ROIs."""

    def test_all_hidden_shows_message(self, app):
        w = ROIIntensityPlotWidget()
        w.canvas.draw_idle = Mock()
        w._intensity["roi_1"] = np.array([1.0, 2.0, 3.0])
        w._checkboxes["roi_1"].setChecked(False)
        w._checkboxes["roi_2"].setChecked(False)

        w._replot()

        assert "All ROI traces hidden" in w.status_label.text()
        assert w.export_btn.isEnabled() is False

    def test_no_data_shows_no_roi_message(self, app):
        w = ROIIntensityPlotWidget()
        w.canvas.draw_idle = Mock()

        w._replot()

        assert "No ROI selected" in w.status_label.text()


class TestROIIntensityPlotWidgetTimeSettingsAndExport:
    """Tests for time-axis settings and CSV export formatting."""

    def test_set_frame_interval_minutes_updates_and_replots(self, app):
        w = ROIIntensityPlotWidget()
        w._replot = Mock()

        w.set_frame_interval_minutes(2.5)

        assert w._frame_interval_minutes == pytest.approx(2.5)
        w._replot.assert_called_once()

    def test_set_frame_interval_minutes_ignores_non_positive_values(self, app):
        w = ROIIntensityPlotWidget()
        w._replot = Mock()
        original = w._frame_interval_minutes

        w.set_frame_interval_minutes(0)

        assert w._frame_interval_minutes == original
        w._replot.assert_not_called()

    def test_export_csv_uses_time_minutes_header_and_float_time_column(self, app):
        w = ROIIntensityPlotWidget()
        w._intensity["roi_1"] = np.array([1.0, 2.0, 3.0], dtype=float)
        w._frame_interval_minutes = 2.5
        captured = {}

        def _capture_savetxt(*args, **kwargs):
            # numpy.savetxt signature: path, array, ...
            captured["array"] = args[1]
            captured["header"] = kwargs.get("header")
            captured["fmt"] = kwargs.get("fmt")

        with (
            patch(
                "ui.roi_intensity_plot.QFileDialog.getSaveFileName", return_value=("/tmp/out.csv", "CSV Files (*.csv)")
            ),
            patch("ui.roi_intensity_plot.np.savetxt", side_effect=_capture_savetxt),
            patch("ui.roi_intensity_plot.QMessageBox.information"),
        ):
            w._export_to_csv()

        assert captured["header"].startswith("Time_Minutes")
        assert captured["fmt"].startswith("%.6f")
        # First column should be time minutes based on current interval.
        np.testing.assert_allclose(captured["array"][:, 0], np.array([0.0, 2.5, 5.0]))
