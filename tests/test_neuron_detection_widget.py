from __future__ import annotations

import sys
import types

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication, QWidget

# The test environment does not always install matplotlib, but this widget only
# needs lightweight stubs for construction in this test.
matplotlib_module = types.ModuleType("matplotlib")
backends_module = types.ModuleType("matplotlib.backends")
backend_qtagg_module = types.ModuleType("matplotlib.backends.backend_qtagg")
figure_module = types.ModuleType("matplotlib.figure")


class _DummyCanvas(QWidget):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()


class _DummyFigure:
    def __init__(self, *args, **kwargs) -> None:
        pass


backend_qtagg_module.FigureCanvasQTAgg = _DummyCanvas
figure_module.Figure = _DummyFigure

sys.modules.setdefault("matplotlib", matplotlib_module)
sys.modules.setdefault("matplotlib.backends", backends_module)
sys.modules.setdefault("matplotlib.backends.backend_qtagg", backend_qtagg_module)
sys.modules.setdefault("matplotlib.figure", figure_module)

from ui.neuron_detection_widget import NeuronDetectionWidget


@pytest.fixture
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        return QApplication([])
    return app


def test_max_absent_frames_defaults_to_stack_length(qapp: QApplication) -> None:
    widget = NeuronDetectionWidget()
    widget.set_frame_data(np.zeros((7, 12, 12), dtype=np.float32))

    assert widget.max_absent_frames_spin.minimum() == 0
    assert widget.max_absent_frames_spin.maximum() == 7
    assert widget.max_absent_frames_spin.value() == 7
