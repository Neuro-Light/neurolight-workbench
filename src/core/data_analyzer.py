from __future__ import annotations

from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np

from core.experiment_manager import Experiment
from core.roi import ROI

from scipy.signal import lombscargle


class DataAnalyzer:
    def __init__(self, experiment: Experiment) -> None:
        self.experiment = experiment

    def calculate_statistics(self, data: np.ndarray) -> Dict[str, float]:
        return {
            "mean": float(np.mean(data)),
            "std": float(np.std(data)),
            "min": float(np.min(data)),
            "max": float(np.max(data)),
        }

    def generate_plots(self, data: np.ndarray, plot_type: str = "hist"):
        fig, ax = plt.subplots()
        if plot_type == "hist":
            ax.hist(data.flatten(), bins=50)
            ax.set_title("Histogram")
        else:
            ax.plot(data)
        return fig

    def save_results_to_experiment(self, experiment: Experiment) -> None:
        experiment.analysis_results.setdefault("runs", []).append(
            {
                "summary": "Placeholder analysis",
            }
        )

    # Placeholders for future expansion
    def time_series_analysis(self, data: np.ndarray):
        return {}

    def correlation_analysis(self, data_a: np.ndarray, data_b: np.ndarray):
        return {}

    def extract_roi_intensity_time_series(
        self,
        frame_data: np.ndarray,
        roi: Optional[ROI] = None,
        roi_x: Optional[int] = None,
        roi_y: Optional[int] = None,
        roi_width: Optional[int] = None,
        roi_height: Optional[int] = None,
    ) -> np.ndarray:
        """
        Extract mean pixel intensity within an ROI across all frames.
        Uses mask-based extraction for polygon and ellipse ROIs.

        Args:
            frame_data: 3D numpy array (frames, height, width)
            roi: ROI object (preferred). Uses create_mask() for accurate extraction.
            roi_x, roi_y, roi_width, roi_height: Legacy rect params if roi is None.

        Returns:
            1D numpy array of mean intensities across frames
        """
        if frame_data.ndim != 3:
            raise ValueError("frame_data must be a 3D array (frames, height, width)")

        num_frames = frame_data.shape[0]
        frame_height = frame_data.shape[1]
        frame_width = frame_data.shape[2]

        if roi is not None:
            mask = roi.create_mask(frame_width, frame_height)
            roi_intensities = np.zeros(num_frames, dtype=np.float64)
            for t in range(num_frames):
                frame = frame_data[t]
                masked = frame[mask > 0]
                roi_intensities[t] = np.mean(masked) if len(masked) > 0 else 0.0
            return roi_intensities

        # Legacy: rectangular ROI
        if roi_x is None or roi_y is None or roi_width is None or roi_height is None:
            return np.zeros(num_frames)
        x1 = max(0, int(roi_x))
        y1 = max(0, int(roi_y))
        x2 = min(frame_width, x1 + int(roi_width))
        y2 = min(frame_height, y1 + int(roi_height))
        if x2 <= x1 or y2 <= y1:
            return np.zeros(num_frames)
        roi_intensities = np.zeros(num_frames)
        for t in range(num_frames):
            roi_region = frame_data[t, y1:y2, x1:x2]
            roi_intensities[t] = np.mean(roi_region)
        return roi_intensities

    def compute_lomb_scargle_periodogram(
        self,
        intensity: np.ndarray,
        dt: float = 1.0,
        min_period: Optional[float] = None,
        max_period: Optional[float] = None,
        oversampling: float = 5.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute Lomb–Scargle periodogram for a regularly sampled time series.

        Args:
            intensity: 1D array of values y(t).
            dt: Sampling interval in time units (e.g. seconds between frames).
            min_period: Minimum period to consider; defaults to 2 * dt.
            max_period: Maximum period to consider; defaults to len(intensity) * dt / 2.
            oversampling: Frequency grid oversampling factor.

        Returns:
            periods: 1D array of periods (same time units as dt).
            power: 1D array of Lomb–Scargle power values.
        """
        y = np.asarray(intensity, dtype=float)
        n = y.size
        if n < 3:
            raise ValueError("Need at least 3 points for Lomb–Scargle analysis.")

        t = np.arange(n, dtype=float) * dt

        if min_period is None:
            min_period = 2.0 * dt
        if max_period is None:
            max_period = (n * dt) / 2.0

        min_freq = 1.0 / max_period
        max_freq = 1.0 / min_period

        num_freqs = max(int(oversampling * n), 1)
        freqs = np.linspace(min_freq, max_freq, num_freqs)
        angular_freqs = 2.0 * np.pi * freqs

        y_centered = y - np.mean(y)
        power = lombscargle(t, y_centered, angular_freqs, normalize=True)

        with np.errstate(divide="ignore"):
            periods = 1.0 / freqs

        return periods, power
