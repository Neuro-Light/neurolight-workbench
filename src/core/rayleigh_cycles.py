from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks


@dataclass(frozen=True)
class RayleighCycleData:
    cycle_index: int
    trough_start_frame: int
    trough_end_frame: int
    cycle_length_frames: int
    cycle_length_minutes: float
    first_peak_frame: int
    neuron_indices: np.ndarray
    peak_frames: np.ndarray
    normalized_day_minutes: np.ndarray
    theta: np.ndarray


def find_signal_peaks_and_troughs(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Find prominent local maxima and minima in a 1D signal."""
    values = np.asarray(data, dtype=float)
    if values.ndim != 1:
        values = values.ravel()
    if values.size < 3:
        return np.array([], dtype=int), np.array([], dtype=int)

    data_range = float(np.max(values) - np.min(values))
    prominence = data_range * 0.10 if data_range > 1e-6 else 1e-6
    distance = max(2, values.size // 100)
    peaks, _ = find_peaks(values, prominence=prominence, distance=distance)
    troughs, _ = find_peaks(-values, prominence=prominence, distance=distance)
    return peaks.astype(int), troughs.astype(int)


def _select_cycle_peak_frame(signal_window: np.ndarray) -> int | None:
    """Choose one representative peak from a trough-bounded cycle window."""
    peaks, _ = find_signal_peaks_and_troughs(signal_window)
    if peaks.size == 0:
        return None

    peak_values = signal_window[peaks]
    max_value = float(np.max(peak_values))
    strongest_peaks = peaks[np.isclose(peak_values, max_value)]
    return int(np.min(strongest_peaks))


def compute_cycle_rayleigh_data(
    signal: np.ndarray,
    neuron_trajectories: np.ndarray,
    interval_minutes: float,
    neuron_indices: np.ndarray | None = None,
) -> list[RayleighCycleData]:
    """
    Build one Rayleigh dataset per consecutive trough-to-trough interval.

    The cycle signal defines trough boundaries. Inside each cycle, each neuron
    contributes at most one peak: the strongest peak within that interval, with
    ties broken by taking the earliest one. The earliest neuron peak in the cycle
    is treated as time zero, while the full trough-to-trough span represents one day.
    """
    trajectories = np.asarray(neuron_trajectories, dtype=float)
    if trajectories.ndim != 2 or trajectories.size == 0:
        return []

    signal_values = np.asarray(signal, dtype=float)
    if signal_values.ndim != 1:
        signal_values = signal_values.ravel()
    if signal_values.size != trajectories.shape[1]:
        raise ValueError("Signal length must match the neuron trajectory frame count.")
    if interval_minutes <= 0:
        raise ValueError("interval_minutes must be positive.")

    if neuron_indices is None:
        neuron_indices = np.arange(trajectories.shape[0], dtype=int)
    else:
        neuron_indices = np.asarray(neuron_indices, dtype=int).ravel()
        if neuron_indices.size != trajectories.shape[0]:
            raise ValueError("neuron_indices length must match the number of trajectories.")

    _, troughs = find_signal_peaks_and_troughs(signal_values)
    if troughs.size < 2:
        return []

    cycles: list[RayleighCycleData] = []
    for cycle_number, (start_frame, end_frame) in enumerate(zip(troughs[:-1], troughs[1:]), start=1):
        cycle_length_frames = int(end_frame - start_frame)
        if cycle_length_frames <= 0:
            continue

        peak_frames: list[int] = []
        cycle_neuron_indices: list[int] = []

        for source_idx, neuron_idx in enumerate(neuron_indices):
            window = trajectories[source_idx, start_frame : end_frame + 1]
            peak_offset = _select_cycle_peak_frame(window)
            if peak_offset is None:
                continue

            peak_frames.append(start_frame + peak_offset)
            cycle_neuron_indices.append(int(neuron_idx))

        if not peak_frames:
            continue

        peak_frames_array = np.asarray(peak_frames, dtype=int)
        cycle_neuron_indices_array = np.asarray(cycle_neuron_indices, dtype=int)
        first_peak_frame = int(np.min(peak_frames_array))
        normalized_positions = (peak_frames_array - first_peak_frame) / float(cycle_length_frames)
        normalized_day_minutes = normalized_positions * (24.0 * 60.0)
        theta = normalized_positions * (2.0 * np.pi)

        cycles.append(
            RayleighCycleData(
                cycle_index=cycle_number,
                trough_start_frame=int(start_frame),
                trough_end_frame=int(end_frame),
                cycle_length_frames=cycle_length_frames,
                cycle_length_minutes=cycle_length_frames * float(interval_minutes),
                first_peak_frame=first_peak_frame,
                neuron_indices=cycle_neuron_indices_array,
                peak_frames=peak_frames_array,
                normalized_day_minutes=normalized_day_minutes,
                theta=theta,
            )
        )

    return cycles
