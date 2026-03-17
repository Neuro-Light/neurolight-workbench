import numpy as np
import pytest

from core.lomb_scargle import _is_unevenly_sampled, compute_lomb_scargle


def test_is_unevenly_sampled_short_returns_false():
    assert _is_unevenly_sampled(np.array([0.0, 1.0])) is False


def test_is_unevenly_sampled_nonfinite_dt_returns_true():
    t = np.array([0.0, np.nan, 2.0])
    assert _is_unevenly_sampled(t) is True


def test_is_unevenly_sampled_zero_mean_dt_returns_true():
    # dt = [0, 0] so mean is 0 -> uneven
    t = np.array([1.0, 1.0, 1.0])
    assert _is_unevenly_sampled(t) is True


def test_is_unevenly_sampled_detects_variation():
    t = np.array([0.0, 1.0, 2.2, 3.2])
    assert _is_unevenly_sampled(t, rtol=1e-6, atol=1e-9) is True


def test_compute_lomb_scargle_rejects_too_few_finite_samples():
    t = np.array([0.0, 1.0, 2.0, np.nan])
    y = np.array([1.0, 2.0, np.inf, 4.0])
    with pytest.raises(ValueError, match="at least 4 finite samples"):
        compute_lomb_scargle(t, y)


def test_compute_lomb_scargle_rejects_zero_time_span():
    t = np.array([5.0, 5.0, 5.0, 5.0])
    y = np.array([1.0, 2.0, 3.0, 4.0])
    with pytest.raises(ValueError, match="non-zero range"):
        compute_lomb_scargle(t, y)


def test_compute_lomb_scargle_invalid_bounds_raises():
    t = np.linspace(0, 10, 50)
    y = np.sin(2 * np.pi * 1.0 * t)
    with pytest.raises(ValueError, match="Invalid frequency bounds"):
        compute_lomb_scargle(t, y, min_freq=2.0, max_freq=1.0)


def test_compute_lomb_scargle_num_freqs_floor_to_ten():
    t = np.linspace(0, 10, 50)
    y = np.sin(2 * np.pi * 1.0 * t)
    out = compute_lomb_scargle(t, y, num_freqs=3)
    assert out["frequency"].shape == (10,)
    assert out["power"].shape == (10,)


def test_compute_lomb_scargle_peak_frequency_close_to_true_for_sinusoid():
    t = np.linspace(0.0, 10.0, 300)
    true_f = 1.5
    y = np.sin(2.0 * np.pi * true_f * t)

    out = compute_lomb_scargle(t, y, num_freqs=2000)

    assert out["n_samples"] == t.size
    assert np.isfinite(out["peak_frequency"])
    assert abs(out["peak_frequency"] - true_f) < 0.05
    assert out["peak_power"] >= 0.0
    assert np.isfinite(out["peak_period"])


def test_compute_lomb_scargle_flags_uneven_sampling():
    # Uneven spacing (1, 1, 2, 1, ...)
    t = np.array([0.0, 1.0, 2.0, 4.0, 5.0, 7.0, 8.0, 10.0])
    y = np.sin(2.0 * np.pi * 0.8 * t)
    out = compute_lomb_scargle(t, y, num_freqs=500)
    assert out["uneven_sampling"] is True
