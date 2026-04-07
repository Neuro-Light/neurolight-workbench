"""Tests for src/utils/image_utils.py — numpy_to_qimage."""

import sys
import unittest.mock

# ---------------------------------------------------------------------------
# Mock PySide6 before importing anything from the package
# ---------------------------------------------------------------------------
_qimage_mock = unittest.mock.MagicMock()
_qimage_mock.Format_Grayscale8 = 1
_qimage_mock.Format_Grayscale16 = 2
_qimage_mock.Format_RGB888 = 3
_qimage_mock.Format_RGBA8888 = 4

_qtgui_mock = unittest.mock.MagicMock()
_qtgui_mock.QImage = _qimage_mock

sys.modules.setdefault("PySide6", unittest.mock.MagicMock())
sys.modules["PySide6.QtGui"] = _qtgui_mock

import importlib  # noqa: E402  (must come after sys.modules patching)
import pytest  # noqa: E402
import numpy as np  # noqa: E402

# Force a clean import so the module sees our mocked PySide6
if "utils.image_utils" in sys.modules:
    del sys.modules["utils.image_utils"]
from utils.image_utils import numpy_to_qimage  # noqa: E402


# ---------------------------------------------------------------------------
# 2-D grayscale — uint8
# ---------------------------------------------------------------------------

def test_2d_uint8_returns_qimage():
    """2-D uint8 array should produce a Grayscale8 QImage without error."""
    arr = np.array([[0, 128], [64, 255]], dtype=np.uint8)
    result = numpy_to_qimage(arr)
    _qimage_mock.assert_called()
    assert result is not None


def test_2d_uint8_uses_grayscale8_format():
    """2-D uint8 array must use Format_Grayscale8."""
    arr = np.zeros((4, 4), dtype=np.uint8)
    _qimage_mock.reset_mock()
    numpy_to_qimage(arr)
    _, kwargs = _qimage_mock.call_args_list[-1][0], _qimage_mock.call_args_list[-1][1]
    call_args = _qimage_mock.call_args_list[-1][0]
    # 5th positional arg is the format
    assert call_args[4] == _qimage_mock.Format_Grayscale8


def test_2d_uint8_single_pixel():
    """Single-pixel uint8 array must be handled without error."""
    arr = np.array([[42]], dtype=np.uint8)
    result = numpy_to_qimage(arr)
    assert result is not None


# ---------------------------------------------------------------------------
# 2-D grayscale — uint16
# ---------------------------------------------------------------------------

def test_2d_uint16_returns_qimage():
    """2-D uint16 array should produce a Grayscale16 QImage."""
    arr = np.array([[0, 1000], [32768, 65535]], dtype=np.uint16)
    result = numpy_to_qimage(arr)
    assert result is not None


def test_2d_uint16_uses_grayscale16_format():
    """2-D uint16 array must use Format_Grayscale16."""
    arr = np.zeros((3, 3), dtype=np.uint16)
    _qimage_mock.reset_mock()
    numpy_to_qimage(arr)
    call_args = _qimage_mock.call_args_list[-1][0]
    assert call_args[4] == _qimage_mock.Format_Grayscale16


# ---------------------------------------------------------------------------
# 2-D grayscale — other dtypes (normalization path)
# ---------------------------------------------------------------------------

def test_2d_float32_with_range_normalizes_to_uint8():
    """float32 array with varying values should be normalized to uint8 Grayscale8."""
    arr = np.array([[0.0, 0.5], [0.25, 1.0]], dtype=np.float32)
    _qimage_mock.reset_mock()
    numpy_to_qimage(arr)
    call_args = _qimage_mock.call_args_list[-1][0]
    assert call_args[4] == _qimage_mock.Format_Grayscale8


def test_2d_float32_flat_array_uses_clip_path():
    """float32 array where all values are equal should take the clip (hi==lo) path."""
    arr = np.full((4, 4), 0.7, dtype=np.float32)
    _qimage_mock.reset_mock()
    numpy_to_qimage(arr)
    call_args = _qimage_mock.call_args_list[-1][0]
    assert call_args[4] == _qimage_mock.Format_Grayscale8


def test_2d_int16_normalizes_to_uint8():
    """int16 array should be normalized and returned as Grayscale8."""
    arr = np.array([[-100, 0], [50, 200]], dtype=np.int16)
    _qimage_mock.reset_mock()
    numpy_to_qimage(arr)
    call_args = _qimage_mock.call_args_list[-1][0]
    assert call_args[4] == _qimage_mock.Format_Grayscale8


def test_2d_float64_single_pixel_flat():
    """Single-pixel float64 array (hi == lo) uses the clip/flat branch."""
    arr = np.array([[3.14]], dtype=np.float64)
    result = numpy_to_qimage(arr)
    assert result is not None


# ---------------------------------------------------------------------------
# 3-D RGB / RGBA
# ---------------------------------------------------------------------------

def test_3d_rgb_returns_qimage():
    """3-channel uint8 array should produce an RGB888 QImage."""
    arr = np.zeros((10, 10, 3), dtype=np.uint8)
    arr[0, 0] = [255, 0, 0]
    result = numpy_to_qimage(arr)
    assert result is not None


def test_3d_rgb_uses_rgb888_format():
    """3-channel array must use Format_RGB888."""
    arr = np.ones((5, 5, 3), dtype=np.uint8)
    _qimage_mock.reset_mock()
    numpy_to_qimage(arr)
    call_args = _qimage_mock.call_args_list[-1][0]
    assert call_args[4] == _qimage_mock.Format_RGB888


def test_3d_rgba_returns_qimage():
    """4-channel uint8 array should produce an RGBA8888 QImage."""
    arr = np.zeros((8, 8, 4), dtype=np.uint8)
    result = numpy_to_qimage(arr)
    assert result is not None


def test_3d_rgba_uses_rgba8888_format():
    """4-channel array must use Format_RGBA8888."""
    arr = np.ones((6, 6, 4), dtype=np.uint8) * 128
    _qimage_mock.reset_mock()
    numpy_to_qimage(arr)
    call_args = _qimage_mock.call_args_list[-1][0]
    assert call_args[4] == _qimage_mock.Format_RGBA8888


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
# Stride correctness (bytes_per_line is passed)
# ---------------------------------------------------------------------------

def test_2d_uint8_stride_passed_to_qimage():
    """bytes_per_line passed to QImage must equal arr.strides[0]."""
    arr = np.zeros((10, 20), dtype=np.uint8)
    _qimage_mock.reset_mock()
    numpy_to_qimage(arr)
    call_args = _qimage_mock.call_args_list[-1][0]
    # call: QImage(data, w, h, bytes_per_line, fmt)
    assert call_args[3] == arr.strides[0]


def test_3d_rgb_stride_passed_to_qimage():
    """bytes_per_line for RGB array must equal arr.strides[0]."""
    arr = np.zeros((7, 9, 3), dtype=np.uint8)
    _qimage_mock.reset_mock()
    numpy_to_qimage(arr)
    call_args = _qimage_mock.call_args_list[-1][0]
    assert call_args[3] == arr.strides[0]
