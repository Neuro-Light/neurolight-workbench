from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from scipy.signal import lombscargle


def _is_unevenly_sampled(t: np.ndarray, *, rtol: float = 1e-3, atol: float = 1e-6) -> bool:
    """
    Heuristic check for uneven sampling in the time vector.

    Returns True if the spacing between consecutive samples varies by more than
    the given relative / absolute tolerance.
    """
    if t.size < 3:
        return False
    dt = np.diff(t)
    if not np.all(np.isfinite(dt)):
        return True
    dt_mean = float(np.mean(dt))
    if dt_mean == 0.0:
        return True
    return bool(np.any(np.abs(dt - dt_mean) > (atol + rtol * dt_mean)))


def compute_lomb_scargle(
    t: np.ndarray,
    y: np.ndarray,
    *,
    min_freq: Optional[float] = None,
    max_freq: Optional[float] = None,
    num_freqs: int = 1000,
) -> Dict[str, Any]:
    """
    Compute a Lomb–Scargle periodogram for a single time series.

    Parameters
    ----------
    t:
        1D array of sample times (arbitrary units). Does not need to be evenly spaced.
    y:
        1D array of intensity values corresponding to ``t``.
    min_freq, max_freq:
        Optional frequency bounds. If omitted, they are estimated from the data
        span and approximate Nyquist frequency.
    num_freqs:
        Number of frequency samples between min_freq and max_freq.

    Returns
    -------
    dict with keys:
        - ``frequency``: 1D array of frequencies
        - ``power``: 1D array of Lomb–Scargle power values
        - ``peak_frequency``: frequency at maximum power (float)
        - ``peak_power``: maximum power value (float)
        - ``peak_period``: 1 / peak_frequency (float) or ``np.inf`` if peak_frequency == 0
        - ``uneven_sampling``: bool flag indicating whether t appears uneven
        - ``n_samples``: number of finite samples used
    """
    t_arr = np.asarray(t, dtype=float).ravel()
    y_arr = np.asarray(y, dtype=float).ravel()

    # Remove NaNs / infs
    mask = np.isfinite(t_arr) & np.isfinite(y_arr)
    t_valid = t_arr[mask]
    y_valid = y_arr[mask]
    n = int(t_valid.size)

    if n < 4:
        raise ValueError("Lomb–Scargle requires at least 4 finite samples.")

    # Normalize time to start at zero for numerical stability
    t0 = float(t_valid.min())
    t_rel = t_valid - t0
    t_span = float(t_rel.max() - t_rel.min())
    if t_span <= 0.0:
        raise ValueError("Time vector must span a non-zero range for Lomb–Scargle.")

    uneven = _is_unevenly_sampled(t_rel)

    # Estimate reasonable default frequency bounds if not provided.
    # - Lowest frequency corresponds to one cycle over the entire observation window.
    # - Highest frequency is based on the median sampling interval (approximate Nyquist).
    if min_freq is None:
        min_freq = 1.0 / t_span

    if max_freq is None:
        dt = np.diff(np.sort(t_rel))
        dt = dt[np.isfinite(dt) & (dt > 0)]
        if dt.size == 0:
            raise ValueError("Cannot estimate Nyquist frequency from time vector.")
        median_dt = float(np.median(dt))
        nyquist = 0.5 / median_dt
        max_freq = nyquist

    if min_freq <= 0 or max_freq <= 0 or max_freq <= min_freq:
        raise ValueError("Invalid frequency bounds for Lomb–Scargle.")

    if num_freqs < 10:
        num_freqs = 10

    frequency = np.linspace(min_freq, max_freq, num_freqs, dtype=float)
    angular_freq = 2.0 * np.pi * frequency

    # Pre-center the signal to improve numerical stability
    y_centered = y_valid - float(np.mean(y_valid))

    power = lombscargle(t_rel, y_centered, angular_freq, precenter=True, normalize=True)
    power = np.asarray(power, dtype=float)

    if not np.all(np.isfinite(power)) or power.size == 0:
        raise RuntimeError("Lomb–Scargle computation failed: non-finite power values.")

    peak_idx = int(np.argmax(power))
    peak_frequency = float(frequency[peak_idx])
    peak_power = float(power[peak_idx])
    peak_period = float(np.inf) if peak_frequency == 0.0 else float(1.0 / peak_frequency)

    return {
        "frequency": frequency,
        "power": power,
        "peak_frequency": peak_frequency,
        "peak_power": peak_power,
        "peak_period": peak_period,
        "uneven_sampling": uneven,
        "n_samples": n,
    }
