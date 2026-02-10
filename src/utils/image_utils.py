"""Shared image conversion utilities."""
from __future__ import annotations

import numpy as np
from PySide6.QtGui import QImage


def numpy_to_qimage(arr: np.ndarray) -> QImage:
    """Convert a NumPy array to a QImage.

    Supports:
    - 2-D arrays (grayscale, uint8 or uint16)
    - 3-D arrays with 3 channels (RGB) or 4 channels (RGBA)

    The row stride is taken from ``arr.strides[0]`` so that
    non-contiguous arrays (e.g. sliced views) are handled correctly.

    .. note::
        The returned QImage references the array's data buffer.
        Callers must keep the array alive for as long as the QImage
        (or any QPixmap derived from it) is in use.
    """
    if arr.ndim == 2:
        h, w = arr.shape
        if arr.dtype == np.uint8:
            fmt = QImage.Format_Grayscale8
        elif arr.dtype == np.uint16:
            fmt = QImage.Format_Grayscale16
        else:
            # Normalize non-uint8/uint16 data (e.g. float32, int16) to uint8
            arr_f = arr.astype(np.float64)
            lo, hi = float(arr_f.min()), float(arr_f.max())
            if hi - lo > 0:
                arr = ((arr_f - lo) / (hi - lo) * 255.0).astype(np.uint8)
            else:
                arr = np.clip(arr_f, 0, 255).astype(np.uint8)
            fmt = QImage.Format_Grayscale8
        bytes_per_line = w * arr.itemsize
        return QImage(arr.data, w, h, bytes_per_line, fmt)

    if arr.ndim == 3:
        h, w, c = arr.shape
        bytes_per_line = arr.strides[0]
        if c == 3:
            return QImage(arr.data, w, h, bytes_per_line, QImage.Format_RGB888)
        if c == 4:
            return QImage(arr.data, w, h, bytes_per_line, QImage.Format_RGBA8888)

    raise ValueError("Unsupported image shape")
