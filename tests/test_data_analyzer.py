import numpy as np
import pytest

from core.data_analyzer import DataAnalyzer
from core.experiment_manager import Experiment
from core.roi import ROI, ROIShape


def _make_analyzer() -> DataAnalyzer:
    return DataAnalyzer(Experiment(name="demo"))


def test_calculate_statistics_returns_expected_values():
    analyzer = _make_analyzer()
    data = np.array([1.0, 3.0, 5.0, 7.0])

    stats = analyzer.calculate_statistics(data)

    assert stats["mean"] == pytest.approx(4.0)
    assert stats["std"] == pytest.approx(np.std(data))
    assert stats["min"] == pytest.approx(1.0)
    assert stats["max"] == pytest.approx(7.0)


def test_extract_roi_intensity_time_series_uses_mask_pipeline():
    analyzer = _make_analyzer()
    frames = np.zeros((3, 5, 5), dtype=np.float32)
    frames[0, 0:2, 0:2] = 1.0
    frames[1, 0:2, 0:2] = 2.0
    frames[2, 0:2, 0:2] = 3.0
    roi = ROI(
        x=0,
        y=0,
        width=2,
        height=2,
        shape=ROIShape.POLYGON,
        points=[(0, 0), (1, 0), (1, 1), (0, 1)],
    )

    series = analyzer.extract_roi_intensity_time_series(frames, roi=roi)

    assert series.tolist() == [1.0, 2.0, 3.0]


def test_extract_roi_intensity_time_series_legacy_rectangle_path():
    analyzer = _make_analyzer()
    frames = np.arange(3 * 4 * 4, dtype=np.float32).reshape(3, 4, 4)

    series = analyzer.extract_roi_intensity_time_series(
        frames,
        roi=None,
        roi_x=1,
        roi_y=1,
        roi_width=2,
        roi_height=2,
    )

    expected = [
        np.mean(frames[0, 1:3, 1:3]),
        np.mean(frames[1, 1:3, 1:3]),
        np.mean(frames[2, 1:3, 1:3]),
    ]
    assert series == pytest.approx(expected)


def test_extract_roi_intensity_time_series_validates_dimensions():
    analyzer = _make_analyzer()
    with pytest.raises(ValueError, match="must be a 3D array"):
        analyzer.extract_roi_intensity_time_series(np.zeros((4, 4)))


def test_generate_plots_hist_and_line():
    analyzer = _make_analyzer()
    data = np.array([1.0, 2.0, 3.0])
    fig1 = analyzer.generate_plots(data, plot_type="hist")
    fig2 = analyzer.generate_plots(data, plot_type="line")
    assert fig1 is not None and fig2 is not None


def test_save_results_to_experiment_appends_run():
    analyzer = _make_analyzer()
    exp = Experiment(name="x")
    analyzer.save_results_to_experiment(exp)
    assert exp.analysis_results["runs"]


def test_time_series_and_correlation_stubs():
    analyzer = _make_analyzer()
    assert analyzer.time_series_analysis(np.array([1.0])) == {}
    assert analyzer.correlation_analysis(np.array([1.0]), np.array([2.0])) == {}


def test_extract_roi_intensity_legacy_missing_rect_returns_zeros():
    analyzer = _make_analyzer()
    frames = np.zeros((2, 4, 4), dtype=np.float32)
    out = analyzer.extract_roi_intensity_time_series(frames, roi=None)
    assert out.tolist() == [0.0, 0.0]


def test_extract_roi_intensity_legacy_empty_rect_returns_zeros():
    analyzer = _make_analyzer()
    frames = np.zeros((2, 4, 4), dtype=np.float32)
    out = analyzer.extract_roi_intensity_time_series(frames, roi=None, roi_x=2, roi_y=2, roi_width=0, roi_height=2)
    assert out.tolist() == [0.0, 0.0]
