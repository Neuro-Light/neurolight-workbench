from unittest.mock import Mock, patch

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from ui.lomb_scargle_plot import LombScarglePlotWidget
from ui.constants import ROI_KEYS


@pytest.fixture
def app():
    if not QApplication.instance():
        return QApplication([])
    return QApplication.instance()


def _mock_ls_result(*, peak_frequency: float = 2.0, peak_power: float = 0.9, n_freq: int = 5, uneven: bool = False):
    freq = np.linspace(0.5, 2.5, n_freq)
    power = np.linspace(0.1, 1.0, n_freq)
    power[-1] = peak_power
    return {
        "frequency": freq,
        "power": power,
        "peak_frequency": peak_frequency,
        "peak_power": peak_power,
        "peak_period": float("inf") if peak_frequency == 0 else 1.0 / peak_frequency,
        "uneven_sampling": uneven,
        "n_samples": 10,
    }


def test_widget_initial_state_no_data_exports_disabled(app):
    w = LombScarglePlotWidget()
    assert w.export_png_btn.isEnabled() is False
    assert w.export_csv_btn.isEnabled() is False
    assert "No ROI intensity data" in w.status_label.text()
    assert w.get_all_peaks() == {}


def test_set_intensity_time_series_ignores_unknown_roi_key(app):
    w = LombScarglePlotWidget()
    before = dict(w._intensity)
    w.set_intensity_time_series("not_a_roi", np.arange(10))
    assert w._intensity == before


def test_update_plot_with_insufficient_points_does_not_enable_exports(app):
    w = LombScarglePlotWidget()
    # Ensure only ROI 1 is considered visible
    for k, cb in w._roi_checkboxes.items():
        cb.setChecked(k == "roi_1")

    w.set_intensity_time_series("roi_1", np.array([1.0, 2.0, 3.0]))  # < 4 points
    assert w.export_png_btn.isEnabled() is False
    assert w.export_csv_btn.isEnabled() is False
    assert w.get_peak_for_roi("roi_1") is None


def test_update_plot_computes_and_stores_peaks_enables_exports(app):
    w = LombScarglePlotWidget()

    # Keep test stable: avoid real theming + rendering side effects
    w.canvas.draw_idle = Mock()
    w._apply_theme = Mock()

    mock_ax = Mock()
    mock_ax.spines = {}
    w.figure.add_subplot = Mock(return_value=mock_ax)

    with patch("ui.lomb_scargle_plot.compute_lomb_scargle", side_effect=lambda t, y, **kw: _mock_ls_result()):
        # Only analyze ROI 1
        for k, cb in w._roi_checkboxes.items():
            cb.setChecked(k == "roi_1")

        w.set_intensity_time_series("roi_1", np.linspace(0, 1, 10))

    peak = w.get_peak_for_roi("roi_1")
    assert peak is not None
    assert peak["peak_frequency"] == pytest.approx(2.0)
    assert peak["peak_power"] == pytest.approx(0.9)
    assert peak["peak_period"] == pytest.approx(0.5)

    assert w.export_png_btn.isEnabled() is True
    assert w.export_csv_btn.isEnabled() is True
    assert "Lomb–Scargle computed" in w.status_label.text()


def test_axis_period_mode_annotates_using_period(app):
    w = LombScarglePlotWidget()
    w.canvas.draw_idle = Mock()
    w._apply_theme = Mock()

    mock_ax = Mock()
    mock_ax.spines = {}
    w.figure.add_subplot = Mock(return_value=mock_ax)

    with patch("ui.lomb_scargle_plot.compute_lomb_scargle", side_effect=lambda t, y, **kw: _mock_ls_result(peak_frequency=2.0)):
        # Select period mode
        w.axis_mode_combo.setCurrentIndex(1)  # "Period"
        # Only analyze ROI 1
        for k, cb in w._roi_checkboxes.items():
            cb.setChecked(k == "roi_1")
        w.set_intensity_time_series("roi_1", np.linspace(0, 1, 10))

    # In period mode, the global peak x-position should be 1 / f = 0.5
    mock_ax.axvline.assert_called()
    called_x = mock_ax.axvline.call_args[0][0]
    assert called_x == pytest.approx(0.5)


def test_clear_all_resets_state(app):
    w = LombScarglePlotWidget()
    w.canvas.draw = Mock()
    w.figure.clear = Mock()

    # Seed some state
    w._last_frequency = np.array([1.0, 2.0])
    w._results["roi_1"] = {"peak_frequency": 1.0, "peak_power": 2.0, "peak_period": 3.0}
    for k in ROI_KEYS:
        w._intensity[k] = np.arange(10, dtype=float)
    w.export_png_btn.setEnabled(True)
    w.export_csv_btn.setEnabled(True)

    w.clear_all()

    assert all(v is None for v in w._intensity.values())
    assert w._results == {}
    assert w._last_frequency is None
    assert w.export_png_btn.isEnabled() is False
    assert w.export_csv_btn.isEnabled() is False

