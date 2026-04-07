"""Tests for src/utils/image_utils.py — numpy_to_qimage.

PySide6 is available in this environment (QApplication is created by conftest.py),
so tests use the real QImage rather than a sys.modules-level mock.  QImage is a
data container in QtGui and does not require QApplication to instantiate.
"""

import numpy as np
import pytest
from PySide6.QtGui import QImage

from utils.image_utils import numpy_to_qimage


# ---------------------------------------------------------------------------
# 2-D grayscale — uint8
# ---------------------------------------------------------------------------


def test_2d_uint8_returns_qimage_instance():
    """2-D uint8 array should return a real QImage instance."""
    arr = np.zeros((4, 4), dtype=np.uint8)
    result = numpy_to_qimage(arr)
    assert isinstance(result, QImage)


def test_2d_uint8_uses_grayscale8_format():
    """2-D uint8 array must be encoded with Format_Grayscale8."""
    arr = np.zeros((4, 4), dtype=np.uint8)
    result = numpy_to_qimage(arr)
    assert result.format() == QImage.Format_Grayscale8


def test_2d_uint8_dimensions_match_array_shape():
    """QImage width and height must match the array's (rows, cols) shape."""
    arr = np.zeros((6, 8), dtype=np.uint8)
    result = numpy_to_qimage(arr)
    assert result.width() == 8
    assert result.height() == 6


def test_2d_uint8_single_pixel():
    """Single-pixel uint8 array must be handled without error."""
    arr = np.array([[42]], dtype=np.uint8)
    result = numpy_to_qimage(arr)
    assert isinstance(result, QImage)


# ---------------------------------------------------------------------------
# 2-D grayscale — uint16
# ---------------------------------------------------------------------------


def test_2d_uint16_returns_qimage_instance():
    """2-D uint16 array should return a real QImage instance."""
    arr = np.array([[0, 1000], [32768, 65535]], dtype=np.uint16)
    result = numpy_to_qimage(arr)
    assert isinstance(result, QImage)


def test_2d_uint16_uses_grayscale16_format():
    """2-D uint16 array must be encoded with Format_Grayscale16."""
    arr = np.zeros((3, 3), dtype=np.uint16)
    result = numpy_to_qimage(arr)
    assert result.format() == QImage.Format_Grayscale16


# ---------------------------------------------------------------------------
# 2-D grayscale — other dtypes (normalization path)
# ---------------------------------------------------------------------------


def test_2d_float32_with_range_uses_grayscale8():
    """float32 array with varying values is normalized and returned as Grayscale8."""
    arr = np.array([[0.0, 0.5], [0.25, 1.0]], dtype=np.float32)
    result = numpy_to_qimage(arr)
    assert result.format() == QImage.Format_Grayscale8


def test_2d_float32_flat_array_uses_grayscale8():
    """float32 array where all values are equal (hi==lo clip path) returns Grayscale8."""
    arr = np.full((4, 4), 0.7, dtype=np.float32)
    result = numpy_to_qimage(arr)
    assert result.format() == QImage.Format_Grayscale8


def test_2d_int16_normalized_to_grayscale8():
    """int16 array should be normalized and returned as Grayscale8."""
    arr = np.array([[-100, 0], [50, 200]], dtype=np.int16)
    result = numpy_to_qimage(arr)
    assert result.format() == QImage.Format_Grayscale8


def test_2d_float64_single_pixel_flat_clip_branch():
    """Single-pixel float64 (hi==lo) takes the flat-clip branch and returns QImage."""
    arr = np.array([[3.14]], dtype=np.float64)
    result = numpy_to_qimage(arr)
    assert isinstance(result, QImage)


# ---------------------------------------------------------------------------
# 3-D RGB / RGBA
# ---------------------------------------------------------------------------


def test_3d_rgb_returns_qimage_instance():
    """3-channel uint8 array should return a real QImage instance."""
    arr = np.zeros((10, 10, 3), dtype=np.uint8)
    result = numpy_to_qimage(arr)
    assert isinstance(result, QImage)


def test_3d_rgb_uses_rgb888_format():
    """3-channel array must be encoded with Format_RGB888."""
    arr = np.ones((5, 5, 3), dtype=np.uint8)
    result = numpy_to_qimage(arr)
    assert result.format() == QImage.Format_RGB888


def test_3d_rgb_dimensions_match_array_shape():
    """QImage width and height must match the 3-channel array's spatial dimensions."""
    arr = np.zeros((5, 7, 3), dtype=np.uint8)
    result = numpy_to_qimage(arr)
    assert result.width() == 7
    assert result.height() == 5


def test_3d_rgba_returns_qimage_instance():
    """4-channel uint8 array should return a real QImage instance."""
    arr = np.zeros((8, 8, 4), dtype=np.uint8)
    result = numpy_to_qimage(arr)
    assert isinstance(result, QImage)


def test_3d_rgba_uses_rgba8888_format():
    """4-channel array must be encoded with Format_RGBA8888."""
    arr = np.ones((6, 6, 4), dtype=np.uint8) * 128
    result = numpy_to_qimage(arr)
    assert result.format() == QImage.Format_RGBA8888


# ---------------------------------------------------------------------------
# Invalid / unsupported inputs → ValueError
# ---------------------------------------------------------------------------


def test_1d_array_raises_value_error():
    """1-D array is not a supported image shape and must raise ValueError."""
    arr = np.array([1, 2, 3], dtype=np.uint8)
    with pytest.raises(ValueError, match="Unsupported image shape"):
        numpy_to_qimage(arr)


def test_3d_two_channel_raises_value_error():
    """3-D array with 2 channels is unsupported and must raise ValueError."""
    arr = np.zeros((4, 4, 2), dtype=np.uint8)
    with pytest.raises(ValueError, match="Unsupported image shape"):
        numpy_to_qimage(arr)


def test_4d_array_raises_value_error():
    """4-D array is not a supported image shape and must raise ValueError."""
    arr = np.zeros((2, 4, 4, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="Unsupported image shape"):
        numpy_to_qimage(arr)


def test_3d_five_channel_raises_value_error():
    """3-D array with 5 channels falls through all branches and raises ValueError."""
    arr = np.zeros((4, 4, 5), dtype=np.uint8)
    with pytest.raises(ValueError, match="Unsupported image shape"):
        numpy_to_qimage(arr)


# ---------------------------------------------------------------------------
# Stride correctness — bytes_per_line must equal arr.strides[0]
# ---------------------------------------------------------------------------


def test_2d_uint8_bytes_per_line_matches_stride():
    """bytes_per_line passed to QImage must equal arr.strides[0] for a 2-D array."""
    arr = np.zeros((10, 20), dtype=np.uint8)
    result = numpy_to_qimage(arr)
    assert result.bytesPerLine() == arr.strides[0]


def test_3d_rgb_bytes_per_line_matches_stride():
    """bytes_per_line for an RGB array must equal arr.strides[0]."""
    arr = np.zeros((7, 9, 3), dtype=np.uint8)
    result = numpy_to_qimage(arr)
    assert result.bytesPerLine() == arr.strides[0]
