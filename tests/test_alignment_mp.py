"""Tests for multiprocessing-safe alignment helpers (``core.alignment_mp``)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from core.alignment_mp import register_pair, transform_frame


@patch("pystackreg.StackReg")
def test_register_pair_delegates_to_stackreg(mock_sr_cls: MagicMock) -> None:
    mock_inst = MagicMock()
    mock_inst.register.return_value = np.eye(3, dtype=np.float64)
    mock_sr_cls.return_value = mock_inst

    ref = np.zeros((4, 4), dtype=np.uint16)
    mov = np.ones((4, 4), dtype=np.uint16)
    transform_type = 42

    tmat = register_pair(ref, mov, transform_type)

    mock_sr_cls.assert_called_once_with(transform_type)
    mock_inst.register.assert_called_once()
    assert np.allclose(tmat, np.eye(3))


@patch("pystackreg.StackReg")
def test_transform_frame_clips_and_casts_uint16(mock_sr_cls: MagicMock) -> None:
    mock_inst = MagicMock()
    mock_inst.transform.return_value = np.array([70000.0, -100.0], dtype=np.float64)
    mock_sr_cls.return_value = mock_inst

    frame = np.zeros((2,), dtype=np.uint16)
    tmat = np.eye(2)
    transform_type = 1

    out = transform_frame(frame, tmat, transform_type)

    mock_inst.transform.assert_called_once()
    assert out.dtype == np.uint16
    assert out.tolist() == [65535, 0]
