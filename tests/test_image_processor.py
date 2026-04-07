import cv2
import numpy as np

from core.experiment_manager import Experiment
from core.image_processor import ImageProcessor
from core.roi import ROI, ROIShape


def _make_processor() -> ImageProcessor:
    return ImageProcessor(Experiment(name="demo"))


def test_crop_to_roi_polygon_masks_outside_pixels():
    processor = _make_processor()
    image = np.zeros((10, 10), dtype=np.uint8)
    image[2:7, 2:7] = 100
    roi = ROI(
        x=2,
        y=2,
        width=5,
        height=5,
        shape=ROIShape.POLYGON,
        points=[(2, 2), (7, 2), (4, 7)],
    )

    cropped = processor.crop_to_roi(image, roi, apply_mask=True)

    assert cropped.shape == (5, 5)
    assert cropped[1, 1] == 100
    assert cropped[4, 4] == 0  # clearly outside triangle (right of right edge at y=4)


def test_crop_to_roi_ellipse_on_rgb_image():
    processor = _make_processor()
    image = np.ones((12, 12, 3), dtype=np.uint8) * 50
    roi = ROI(x=2, y=3, width=6, height=4, shape=ROIShape.ELLIPSE)

    cropped = processor.crop_to_roi(image, roi, apply_mask=True)

    assert cropped.shape == (4, 6, 3)
    center = cropped.shape[0] // 2, cropped.shape[1] // 2
    assert np.all(cropped[center] == 50)
    assert np.all(cropped[0, 0] == 0)


def test_crop_stack_to_roi_applies_crop_to_each_frame():
    processor = _make_processor()
    stack = np.stack([np.full((8, 8), fill_value=i, dtype=np.uint8) for i in range(3)])
    roi = ROI(x=1, y=1, width=4, height=4, shape=ROIShape.ELLIPSE)

    cropped_stack = processor.crop_stack_to_roi(stack, roi, apply_mask=False)

    assert cropped_stack.shape == (3, 4, 4)
    assert np.array_equal(cropped_stack[0], processor.crop_to_roi(stack[0], roi, apply_mask=False))
    assert np.array_equal(cropped_stack[2], processor.crop_to_roi(stack[2], roi, apply_mask=False))


def test_detect_neurons_returns_bright_spot_centroids():
    processor = _make_processor()
    image = np.zeros((50, 50), dtype=np.uint8)
    cv2.circle(image, (10, 10), 1, 255, -1)
    cv2.circle(image, (35, 35), 1, 255, -1)

    neurons = processor.detect_neurons(image, threshold_percentile=95, min_area=1, max_area=50)

    assert len(neurons) == 2
    assert {(x, y) for x, y in neurons} == {(10, 10), (35, 35)}
