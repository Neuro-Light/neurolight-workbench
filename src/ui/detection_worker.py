"""Worker thread for neuron detection with progress reporting."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:
    from core.image_processor import ImageProcessor

import numpy as np
from PySide6.QtCore import QThread, Signal


class DetectionWorker(QThread):
    """Runs neuron detection in a background thread and emits progress (step, total, message)."""

    progress = Signal(int, int, str)  # (completed, total, message)
    finished = Signal(object, object, object)  # (neuron_locations, neuron_trajectories, quality_mask)
    error = Signal(str)

    def __init__(
        self,
        image_processor: "ImageProcessor",
        frame_data: np.ndarray,
        roi_mask: np.ndarray,
        params: Dict[str, Any],
        parent=None,
    ):
        super().__init__(parent)
        self._processor = image_processor
        self._frame_data = frame_data
        self._roi_mask = roi_mask
        self._params = params

    def run(self) -> None:
        try:

            def on_progress(completed: int, total: int, message: str) -> None:
                self.progress.emit(completed, total, message)

            locations, trajectories, quality_mask = self._processor.detect_neurons_in_roi(
                self._frame_data,
                self._roi_mask,
                cell_size=self._params.get("cell_size", 6),
                num_peaks=self._params.get("num_peaks", 800),
                correlation_threshold=self._params.get("correlation_threshold", 0.4),
                threshold_rel=self._params.get("threshold_rel", 0.03),
                apply_detrending=self._params.get("apply_detrending", True),
                use_max_projection=self._params.get("use_max_projection", True),
                preprocess_sigma=self._params.get("preprocess_sigma", 1.0),
                progress_callback=on_progress,
            )
            self.finished.emit(locations, trajectories, quality_mask)
        except Exception as e:
            self.error.emit(str(e))
