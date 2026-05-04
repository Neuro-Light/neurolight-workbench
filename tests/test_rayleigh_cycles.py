from __future__ import annotations

import numpy as np

from core.rayleigh_cycles import compute_cycle_rayleigh_data, find_signal_peaks_and_troughs


def test_find_signal_peaks_and_troughs_detects_repeating_cycles() -> None:
    signal = np.array([5.0, 3.0, 1.0, 3.0, 5.0, 3.0, 1.0, 3.0, 5.0, 3.0, 1.0, 3.0, 5.0])

    peaks, troughs = find_signal_peaks_and_troughs(signal)

    assert peaks.tolist() == [4, 8]
    assert troughs.tolist() == [2, 6, 10]


def test_compute_cycle_rayleigh_data_normalizes_each_cycle_from_first_peak() -> None:
    signal = np.array([5.0, 3.0, 1.0, 3.0, 5.0, 3.0, 1.0, 3.0, 5.0, 3.0, 1.0, 3.0, 5.0])
    trajectories = np.array(
        [
            [2.0, 1.0, 0.0, 1.0, 6.0, 1.0, 0.0, 1.0, 6.0, 1.0, 0.0, 1.0, 2.0],
            [2.0, 1.0, 0.0, 5.0, 1.0, 0.5, 0.0, 5.0, 1.0, 0.5, 0.0, 1.0, 2.0],
            [2.0, 1.0, 0.0, 0.5, 1.0, 5.0, 0.0, 0.5, 1.0, 5.0, 0.0, 1.0, 2.0],
        ]
    )

    cycles = compute_cycle_rayleigh_data(signal, trajectories, interval_minutes=60.0)

    assert len(cycles) == 2

    first_cycle = cycles[0]
    assert first_cycle.cycle_index == 1
    assert first_cycle.trough_start_frame == 2
    assert first_cycle.trough_end_frame == 6
    assert first_cycle.cycle_length_frames == 4
    assert first_cycle.cycle_length_minutes == 240.0
    assert first_cycle.first_peak_frame == 3
    assert first_cycle.peak_frames.tolist() == [4, 3, 5]
    assert first_cycle.normalized_day_minutes.tolist() == [360.0, 0.0, 720.0]

    second_cycle = cycles[1]
    assert second_cycle.cycle_index == 2
    assert second_cycle.first_peak_frame == 7
    assert second_cycle.peak_frames.tolist() == [8, 7, 9]
    assert second_cycle.normalized_day_minutes.tolist() == [360.0, 0.0, 720.0]


def test_compute_cycle_rayleigh_data_returns_empty_without_consecutive_troughs() -> None:
    signal = np.array([0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 3.0])
    trajectories = np.vstack([signal, signal + 0.1])

    cycles = compute_cycle_rayleigh_data(signal, trajectories, interval_minutes=30.0)

    assert cycles == []
