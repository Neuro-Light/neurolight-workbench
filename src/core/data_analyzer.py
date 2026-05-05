from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from core.experiment_manager import Experiment
from core.roi import ROI


class DataAnalyzer:
    def __init__(self, experiment: Experiment) -> None:
        self.experiment = experiment

    def calculate_statistics(self, data: np.ndarray) -> dict[str, float]:
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
        roi: ROI | None = None,
        roi_x: int | None = None,
        roi_y: int | None = None,
        roi_width: int | None = None,
        roi_height: int | None = None,
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
