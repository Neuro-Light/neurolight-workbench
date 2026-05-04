from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from core.rayleigh_cycles import compute_cycle_rayleigh_data, find_signal_peaks_and_troughs


def test_find_signal_peaks_and_troughs_detects_repeating_cycles() -> None:
    signal = np.array([5.0, 3.0, 1.0, 3.0, 5.0, 3.0, 1.0, 3.0, 5.0, 3.0, 1.0, 3.0, 5.0])

    peaks, troughs = find_signal_peaks_and_troughs(signal)

    assert peaks.tolist() == [4, 8]
    assert troughs.tolist() == [2, 6, 10]


def test_find_signal_peaks_and_troughs_flattens_non_1d_input() -> None:
    signal = np.array([[0.0, 2.0, 0.0], [2.0, 0.0, 2.0]])

    peaks, troughs = find_signal_peaks_and_troughs(signal)

    assert peaks.tolist() == [1, 3]
    assert troughs.tolist() == [2, 4]


def test_find_signal_peaks_and_troughs_returns_empty_for_short_signal() -> None:
    peaks, troughs = find_signal_peaks_and_troughs(np.array([1.0, 2.0]))

    assert peaks.size == 0
    assert troughs.size == 0


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


def test_compute_cycle_rayleigh_data_respects_explicit_neuron_indices() -> None:
    signal = np.array([5.0, 3.0, 1.0, 3.0, 5.0, 3.0, 1.0, 3.0, 5.0, 3.0, 1.0, 3.0, 5.0])
    trajectories = np.array(
        [
            [2.0, 1.0, 0.0, 1.0, 6.0, 1.0, 0.0, 1.0, 6.0, 1.0, 0.0, 1.0, 2.0],
            [2.0, 1.0, 0.0, 5.0, 1.0, 0.5, 0.0, 5.0, 1.0, 0.5, 0.0, 1.0, 2.0],
        ]
    )

    cycles = compute_cycle_rayleigh_data(
        signal,
        trajectories,
        interval_minutes=30.0,
        neuron_indices=np.array([7, 11]),
    )

    assert [cycle.neuron_indices.tolist() for cycle in cycles] == [[7, 11], [7, 11]]


def test_compute_cycle_rayleigh_data_prefers_earliest_strongest_peak_in_cycle() -> None:
    signal = np.array([5.0, 3.0, 1.0, 3.0, 5.0, 3.0, 1.0, 3.0, 5.0, 3.0, 1.0, 3.0, 5.0])
    trajectories = np.array(
        [
            [0.0, 0.0, 0.0, 5.0, 1.0, 5.0, 0.0, 4.0, 1.0, 4.0, 0.0, 0.0, 0.0],
        ]
    )

    cycles = compute_cycle_rayleigh_data(signal, trajectories, interval_minutes=15.0)

    assert [cycle.peak_frames.tolist() for cycle in cycles] == [[3], [7]]


def test_compute_cycle_rayleigh_data_skips_cycles_without_neuron_peaks() -> None:
    signal = np.array([5.0, 3.0, 1.0, 3.0, 5.0, 3.0, 1.0, 3.0, 5.0, 3.0, 1.0, 3.0, 5.0])
    trajectories = np.zeros((2, signal.size), dtype=float)

    cycles = compute_cycle_rayleigh_data(signal, trajectories, interval_minutes=15.0)

    assert cycles == []


def test_compute_cycle_rayleigh_data_returns_empty_for_invalid_trajectory_shape() -> None:
    signal = np.array([0.0, 1.0, 0.0])

    assert compute_cycle_rayleigh_data(signal, np.array([0.0, 1.0, 0.0]), interval_minutes=10.0) == []
    assert compute_cycle_rayleigh_data(signal, np.empty((0, 3)), interval_minutes=10.0) == []


def test_compute_cycle_rayleigh_data_validates_signal_length() -> None:
    trajectories = np.ones((2, 5), dtype=float)

    with pytest.raises(ValueError, match="Signal length must match"):
        compute_cycle_rayleigh_data(np.ones((2, 2), dtype=float), trajectories, interval_minutes=10.0)


def test_compute_cycle_rayleigh_data_validates_positive_interval() -> None:
    signal = np.array([0.0, 1.0, 0.0, 1.0, 0.0])
    trajectories = np.ones((2, 5), dtype=float)

    with pytest.raises(ValueError, match="interval_minutes must be positive"):
        compute_cycle_rayleigh_data(signal, trajectories, interval_minutes=0.0)


def test_compute_cycle_rayleigh_data_validates_neuron_indices_length() -> None:
    signal = np.array([0.0, 1.0, 0.0, 1.0, 0.0])
    trajectories = np.ones((2, 5), dtype=float)

    with pytest.raises(ValueError, match="neuron_indices length must match"):
        compute_cycle_rayleigh_data(signal, trajectories, interval_minutes=10.0, neuron_indices=np.array([1]))


def test_compute_cycle_rayleigh_data_returns_empty_without_consecutive_troughs() -> None:
    signal = np.array([0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 3.0])
    trajectories = np.vstack([signal, signal + 0.1])

    cycles = compute_cycle_rayleigh_data(signal, trajectories, interval_minutes=30.0)

    assert cycles == []


def test_compute_cycle_rayleigh_data_skips_non_positive_cycle_windows() -> None:
    signal = np.array([0.0, 1.0, 0.0, 1.0, 0.0])
    trajectories = np.vstack([signal, signal + 0.5])

    with patch(
        "core.rayleigh_cycles.find_signal_peaks_and_troughs",
        return_value=(np.array([], dtype=int), np.array([3, 3])),
    ):
        cycles = compute_cycle_rayleigh_data(signal, trajectories, interval_minutes=10.0)

    assert cycles == []
