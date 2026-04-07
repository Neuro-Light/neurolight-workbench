import cv2
import numpy as np
import pytest
import tifffile
from PIL import Image

from core.experiment_manager import Experiment
from core.image_processor import ImageProcessor
from core.roi import ROI, ROIShape


def _make_processor() -> ImageProcessor:
    return ImageProcessor(Experiment(name="demo"))


# ── crop_to_roi ───────────────────────────────────────────────────────────────


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


def test_crop_to_roi_raises_for_4d_image():
    processor = _make_processor()
    image = np.zeros((5, 5, 3, 2), dtype=np.uint8)
    roi = ROI(x=0, y=0, width=3, height=3, shape=ROIShape.ELLIPSE)
    with pytest.raises(ValueError, match="Image must be 2D or 3D array"):
        processor.crop_to_roi(image, roi)


def test_crop_to_roi_no_mask_returns_full_slice():
    processor = _make_processor()
    image = np.ones((10, 10), dtype=np.uint8) * 5
    roi = ROI(x=2, y=2, width=4, height=4, shape=ROIShape.ELLIPSE)
    result = processor.crop_to_roi(image, roi, apply_mask=False)
    assert result.shape == (4, 4)
    assert np.all(result == 5)


def test_crop_to_roi_ellipse_zero_rx_skips_mask():
    processor = _make_processor()
    image = np.ones((10, 10), dtype=np.uint8) * 7
    roi = ROI(x=2, y=2, width=0, height=4, shape=ROIShape.ELLIPSE)
    result = processor.crop_to_roi(image, roi, apply_mask=True)
    assert result.shape[0] == 4


# ── crop_stack_to_roi ─────────────────────────────────────────────────────────


def test_crop_stack_to_roi_applies_crop_to_each_frame():
    processor = _make_processor()
    stack = np.stack([np.full((8, 8), fill_value=i, dtype=np.uint8) for i in range(3)])
    roi = ROI(x=1, y=1, width=4, height=4, shape=ROIShape.ELLIPSE)

    cropped_stack = processor.crop_stack_to_roi(stack, roi, apply_mask=False)

    assert cropped_stack.shape == (3, 4, 4)
    assert np.array_equal(cropped_stack[0], processor.crop_to_roi(stack[0], roi, apply_mask=False))
    assert np.array_equal(cropped_stack[2], processor.crop_to_roi(stack[2], roi, apply_mask=False))


def test_crop_stack_to_roi_raises_for_2d_input():
    processor = _make_processor()
    with pytest.raises(ValueError, match="Image stack must be 3D"):
        processor.crop_stack_to_roi(np.zeros((5, 5), dtype=np.uint8), ROI(x=0, y=0, width=3, height=3))


def test_crop_stack_to_roi_polygon_computes_mask_once():
    processor = _make_processor()
    stack = np.full((3, 10, 10), 100, dtype=np.uint8)
    roi = ROI(
        x=2,
        y=2,
        width=5,
        height=5,
        shape=ROIShape.POLYGON,
        points=[(2, 2), (7, 2), (4, 7)],
    )
    result = processor.crop_stack_to_roi(stack, roi, apply_mask=True)
    assert result.shape == (3, 5, 5)
    assert result[0, 1, 1] == 100
    assert result[1, 1, 1] == 100
    assert result[2, 1, 1] == 100
    assert result[0, 4, 4] == 0


def test_crop_stack_to_roi_ellipse_with_mask():
    processor = _make_processor()
    stack = np.full((2, 12, 12), 50, dtype=np.uint8)
    roi = ROI(x=2, y=2, width=6, height=6, shape=ROIShape.ELLIPSE)
    result = processor.crop_stack_to_roi(stack, roi, apply_mask=True)
    assert result.shape == (2, 6, 6)
    assert result[0, 3, 3] == 50


# ── load_image ────────────────────────────────────────────────────────────────


def test_load_image_returns_array(tmp_path):
    path = tmp_path / "frame.png"
    cv2.imwrite(str(path), np.zeros((10, 10), dtype=np.uint8))
    processor = _make_processor()
    img = processor.load_image(str(path))
    assert img.shape == (10, 10)


def test_load_image_raises_for_missing_file():
    processor = _make_processor()
    with pytest.raises(FileNotFoundError):
        processor.load_image("/nonexistent/path/image.png")


# ── preprocess_image ──────────────────────────────────────────────────────────


def test_preprocess_image_blurs_image():
    processor = _make_processor()
    image = np.random.randint(0, 255, (20, 20), dtype=np.uint8)
    result = processor.preprocess_image(image, {"ksize": 3})
    assert result.shape == image.shape


def test_preprocess_image_uses_default_ksize():
    processor = _make_processor()
    result = processor.preprocess_image(np.zeros((10, 10), dtype=np.uint8), {})
    assert result.shape == (10, 10)


# ── apply_opencv_filter ───────────────────────────────────────────────────────


def test_apply_opencv_filter_edges():
    processor = _make_processor()
    image = np.zeros((20, 20), dtype=np.uint8)
    result = processor.apply_opencv_filter(image, "edges")
    assert result.shape == (20, 20)


def test_apply_opencv_filter_unknown_returns_input():
    processor = _make_processor()
    image = np.ones((10, 10), dtype=np.uint8) * 42
    result = processor.apply_opencv_filter(image, "unknown")
    assert np.array_equal(result, image)


# ── detect_objects / extract_features placeholders ───────────────────────────


def test_detect_objects_returns_empty_list():
    assert _make_processor().detect_objects(np.zeros((10, 10), dtype=np.uint8)) == []


def test_extract_features_returns_empty_dict():
    assert _make_processor().extract_features(np.zeros((10, 10), dtype=np.uint8)) == {}


# ── detect_neurons ────────────────────────────────────────────────────────────


def test_detect_neurons_returns_bright_spot_centroids():
    processor = _make_processor()
    image = np.zeros((50, 50), dtype=np.uint8)
    cv2.circle(image, (10, 10), 1, 255, -1)
    cv2.circle(image, (35, 35), 1, 255, -1)

    neurons = processor.detect_neurons(image, threshold_percentile=95, min_area=1, max_area=50)

    assert len(neurons) == 2
    assert {(x, y) for x, y in neurons} == {(10, 10), (35, 35)}


def test_detect_neurons_normalizes_non_uint8():
    processor = _make_processor()
    image = np.zeros((50, 50), dtype=np.uint16)
    image[10, 10] = 60000
    neurons = processor.detect_neurons(image, threshold_percentile=95, min_area=1, max_area=50)
    assert isinstance(neurons, list)


# ── load_image_for_alignment ──────────────────────────────────────────────────


def test_load_image_for_alignment_2d_tiff(tmp_path):
    img = np.random.randint(0, 1000, (20, 20), dtype=np.uint16)
    path = tmp_path / "frame.tif"
    tifffile.imwrite(str(path), img)
    result = _make_processor().load_image_for_alignment(str(path))
    assert result.ndim == 2
    assert result.shape == (20, 20)


def test_load_image_for_alignment_single_page_3d_tiff(tmp_path):
    img = np.random.randint(0, 255, (1, 20, 20), dtype=np.uint8)
    path = tmp_path / "single.tif"
    tifffile.imwrite(str(path), img)
    result = _make_processor().load_image_for_alignment(str(path))
    assert result.ndim == 2
    assert result.shape == (20, 20)


def test_load_image_for_alignment_multipage_tiff_raises(tmp_path):
    img = np.random.randint(0, 255, (5, 20, 20), dtype=np.uint8)
    path = tmp_path / "multi.tif"
    tifffile.imwrite(str(path), img)
    with pytest.raises(ValueError, match="Multi-page TIFF detected"):
        _make_processor().load_image_for_alignment(str(path))


def test_load_image_for_alignment_rgb_tiff(tmp_path):
    img = (np.random.rand(20, 20, 3) * 255).astype(np.uint8)
    path = tmp_path / "rgb.tif"
    tifffile.imwrite(str(path), img)
    result = _make_processor().load_image_for_alignment(str(path))
    assert result.ndim == 2
    assert result.shape == (20, 20)


def test_load_image_for_alignment_rgba_tiff(tmp_path):
    img = (np.random.rand(20, 20, 4) * 255).astype(np.uint8)
    path = tmp_path / "rgba.tif"
    tifffile.imwrite(str(path), img)
    result = _make_processor().load_image_for_alignment(str(path))
    assert result.ndim == 2
    assert result.shape == (20, 20)


def test_load_image_for_alignment_multichannel_tiff_takes_first(tmp_path):
    img = (np.random.rand(20, 20, 5) * 255).astype(np.uint8)
    path = tmp_path / "multi_ch.tif"
    tifffile.imwrite(str(path), img)
    result = _make_processor().load_image_for_alignment(str(path))
    assert result.ndim == 2
    assert result.shape == (20, 20)


def test_load_image_for_alignment_png_via_pil(tmp_path):
    img = Image.fromarray(np.ones((20, 20), dtype=np.uint8) * 128, "L")
    path = tmp_path / "gray.png"
    img.save(str(path))
    result = _make_processor().load_image_for_alignment(str(path))
    assert result.ndim == 2


def test_load_image_for_alignment_bad_tiff_raises(tmp_path):
    with pytest.raises(ValueError, match="Failed to load"):
        _make_processor().load_image_for_alignment(str(tmp_path / "no_such.tif"))


def test_load_image_for_alignment_bad_png_raises(tmp_path):
    with pytest.raises(ValueError, match="Failed to load"):
        _make_processor().load_image_for_alignment(str(tmp_path / "no_such.png"))


def test_load_image_for_alignment_1d_raises(tmp_path, monkeypatch):
    import tifffile as tf

    monkeypatch.setattr(tf, "imread", lambda *a, **kw: np.array([1, 2, 3]))
    path = tmp_path / "fake.tif"
    path.touch()
    with pytest.raises(ValueError, match="1D array not supported"):
        _make_processor().load_image_for_alignment(str(path))


def test_load_image_for_alignment_4d_raises(tmp_path, monkeypatch):
    import tifffile as tf

    monkeypatch.setattr(tf, "imread", lambda *a, **kw: np.zeros((2, 10, 10, 3), dtype=np.uint8))
    path = tmp_path / "fake.tif"
    path.touch()
    with pytest.raises(ValueError, match="Unsupported image dimensions"):
        _make_processor().load_image_for_alignment(str(path))


# ── align_image_stack ─────────────────────────────────────────────────────────


def _small_stack(frames=3, h=16, w=16, dtype=np.uint8):
    rng = np.random.default_rng(0)
    return rng.integers(50, 200, (frames, h, w), dtype=dtype)


def test_align_image_stack_raises_for_non_3d():
    with pytest.raises(ValueError, match="Image stack must be 3D"):
        _make_processor().align_image_stack(np.zeros((16, 16), dtype=np.uint8))


def test_align_image_stack_first_reference():
    stack = _small_stack()
    aligned, tmats, scores = _make_processor().align_image_stack(stack, reference="first")
    assert aligned.shape == stack.shape
    assert len(scores) == 3
    assert scores[0] == 1.0


def test_align_image_stack_previous_reference():
    stack = _small_stack()
    aligned, tmats, scores = _make_processor().align_image_stack(stack, reference="previous")
    assert aligned.shape == stack.shape
    assert scores[0] == 1.0


def test_align_image_stack_mean_reference():
    stack = _small_stack()
    aligned, tmats, scores = _make_processor().align_image_stack(stack, reference="mean")
    assert aligned.shape == stack.shape
    assert len(scores) == 3


def test_align_image_stack_flat_stack_skips_normalization():
    stack = np.full((3, 16, 16), 100, dtype=np.uint8)
    aligned, tmats, scores = _make_processor().align_image_stack(stack, reference="first")
    assert aligned.shape == stack.shape


def test_align_image_stack_uint16_dtype():
    stack = _small_stack(dtype=np.uint16)
    aligned, tmats, scores = _make_processor().align_image_stack(stack)
    assert aligned.dtype == np.uint16


def test_align_image_stack_progress_callback_called():
    stack = _small_stack()
    calls = []

    def cb(done, total, msg):
        calls.append(msg)
        return True

    _make_processor().align_image_stack(stack, progress_callback=cb)
    assert len(calls) > 0


def test_align_image_stack_progress_callback_cancel_returns_early():
    stack = _small_stack()

    aligned, tmats, scores = _make_processor().align_image_stack(stack, progress_callback=lambda *_: False)
    assert aligned.shape == stack.shape
    assert scores == []


# ── detect_neurons_in_roi ─────────────────────────────────────────────────────


def _neuron_stack(frames=5, h=20, w=20, spots=None):
    """Return a synthetic image stack with bright spots at given (y, x) positions."""
    stack = np.zeros((frames, h, w), dtype=np.uint16)
    for y, x in spots or []:
        for f in range(frames):
            stack[f, y, x] = 1000
            if x + 1 < w:
                stack[f, y, x + 1] = 600
            if y + 1 < h:
                stack[f, y + 1, x] = 600
    return stack


def test_detect_neurons_in_roi_empty_mask_returns_empty():
    processor = _make_processor()
    stack = _neuron_stack(spots=[(5, 5)])
    roi_mask = np.zeros((20, 20), dtype=bool)
    locs, trajs, quality = processor.detect_neurons_in_roi(stack, roi_mask)
    assert locs.shape == (0, 2)
    assert trajs.shape[0] == 0
    assert quality.shape[0] == 0


def test_detect_neurons_in_roi_no_peaks_returns_empty():
    # Image too small for any peak to survive exclude_border
    processor = _make_processor()
    stack = np.random.randint(0, 255, (3, 4, 4), dtype=np.uint8)
    roi_mask = np.ones((4, 4), dtype=bool)
    locs, trajs, quality = processor.detect_neurons_in_roi(stack, roi_mask, cell_size=6, threshold_rel=0.01)
    assert locs.shape == (0, 2)


def test_detect_neurons_in_roi_single_neuron_quality_true():
    processor = _make_processor()
    stack = _neuron_stack(spots=[(10, 10)])
    roi_mask = np.ones((20, 20), dtype=bool)
    locs, trajs, quality = processor.detect_neurons_in_roi(
        stack,
        roi_mask,
        cell_size=3,
        num_peaks=5,
        correlation_threshold=0.0,
        threshold_rel=0.1,
        apply_detrending=False,
    )
    assert len(quality) == 1
    assert quality[0] is True or quality[0] == True  # noqa: E712


def test_detect_neurons_in_roi_basic_detection():
    processor = _make_processor()
    # Two bright spots far apart so both are detected
    stack = _neuron_stack(h=30, w=30, spots=[(5, 5), (22, 22)])
    roi_mask = np.ones((30, 30), dtype=bool)
    locs, trajs, quality = processor.detect_neurons_in_roi(
        stack,
        roi_mask,
        cell_size=3,
        num_peaks=10,
        correlation_threshold=-1.0,
        threshold_rel=0.1,
        apply_detrending=False,
    )
    assert locs.shape[1] == 2
    assert trajs.shape[1] == 5


def test_detect_neurons_in_roi_uniform_frame_zeros_projection():
    processor = _make_processor()
    # All-same-value frames → frame_max == frame_min → zeros_like branch
    stack = np.full((3, 15, 15), 50, dtype=np.uint8)
    roi_mask = np.ones((15, 15), dtype=bool)
    locs, trajs, quality = processor.detect_neurons_in_roi(stack, roi_mask, cell_size=3, threshold_rel=0.01)
    assert isinstance(locs, np.ndarray)


def test_detect_neurons_in_roi_with_detrending():
    processor = _make_processor()
    num_frames = 80
    stack = np.zeros((num_frames, 20, 20), dtype=np.uint16)
    stack[:, 10, 10] = 1000
    roi_mask = np.ones((20, 20), dtype=bool)
    locs, trajs, quality = processor.detect_neurons_in_roi(
        stack,
        roi_mask,
        cell_size=3,
        num_peaks=5,
        correlation_threshold=0.0,
        threshold_rel=0.1,
        apply_detrending=True,
    )
    assert isinstance(trajs, np.ndarray)


def test_detect_neurons_in_roi_raises_for_2d_frame_data():
    processor = _make_processor()
    with pytest.raises(ValueError, match="frame_data must be a 3D array"):
        processor.detect_neurons_in_roi(np.zeros((20, 20)), np.ones((20, 20), dtype=bool))


def test_detect_neurons_in_roi_raises_for_3d_roi_mask():
    processor = _make_processor()
    with pytest.raises(ValueError, match="roi_mask must be a 2D array"):
        processor.detect_neurons_in_roi(
            np.zeros((3, 20, 20), dtype=np.uint8),
            np.ones((3, 20, 20), dtype=bool),
        )


def test_detect_neurons_in_roi_raises_for_shape_mismatch():
    processor = _make_processor()
    with pytest.raises(ValueError, match="roi_mask shape"):
        processor.detect_neurons_in_roi(
            np.zeros((3, 20, 20), dtype=np.uint8),
            np.ones((15, 15), dtype=bool),
        )


def test_detect_neurons_in_roi_with_progress_callback():
    processor = _make_processor()
    stack = _neuron_stack(spots=[(10, 10)])
    roi_mask = np.ones((20, 20), dtype=bool)
    calls = []

    def cb(step, total, msg):
        calls.append(step)

    processor.detect_neurons_in_roi(
        stack,
        roi_mask,
        cell_size=3,
        num_peaks=5,
        correlation_threshold=0.0,
        threshold_rel=0.1,
        apply_detrending=False,
        progress_callback=cb,
    )
    assert len(calls) > 0


def test_detect_neurons_in_roi_mean_projection():
    processor = _make_processor()
    stack = _neuron_stack(spots=[(10, 10)])
    roi_mask = np.ones((20, 20), dtype=bool)
    locs, trajs, quality = processor.detect_neurons_in_roi(
        stack,
        roi_mask,
        cell_size=3,
        num_peaks=5,
        threshold_rel=0.1,
        use_max_projection=False,
        apply_detrending=False,
    )
    assert isinstance(locs, np.ndarray)
