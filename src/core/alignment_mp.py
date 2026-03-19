from __future__ import annotations

import numpy as np


def register_pair(ref_frame: np.ndarray, moving_frame: np.ndarray, transform_type: int) -> np.ndarray:
    """Register *moving_frame* to *ref_frame*.

    Kept in a non-Qt module so multiprocessing spawn workers don't import PySide.
    """
    from pystackreg import StackReg

    sr = StackReg(transform_type)
    return sr.register(ref_frame, moving_frame)


def transform_frame(frame: np.ndarray, tmat: np.ndarray, transform_type: int) -> np.ndarray:
    """Apply *tmat* to a single frame.

    Kept in a non-Qt module so multiprocessing spawn workers don't import PySide.
    """
    from pystackreg import StackReg

    sr = StackReg(transform_type)
    result = sr.transform(frame, tmat=tmat)
    np.clip(result, 0, 65535, out=result)
    return result.astype(np.uint16)

