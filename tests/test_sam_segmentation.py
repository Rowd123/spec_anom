import numpy as np
import pytest

from spectral_anomaly import (
    SAMStructureSegmenter,
    compute_dice,
    compute_iou,
    pixel_to_time_frequency,
    spectrogram_to_sam_image,
    time_frequency_to_pixel,
)


class FakePredictor:
    def set_image(self, image):
        self.image = image

    def predict(self, **kwargs):
        self.kwargs = kwargs
        masks = np.zeros((3, 4, 5), dtype=bool)
        masks[1, 1:3, 1:4] = True
        return masks, np.array([0.2, 0.9, 0.4]), np.zeros((3, 2, 2))


def test_spectrogram_conversion_is_rgb_uint8_without_colormap():
    stft = np.arange(20, dtype=float).reshape(4, 5).astype(complex)
    image = spectrogram_to_sam_image(stft)

    assert image.shape == (4, 5, 3)
    assert image.dtype == np.uint8
    np.testing.assert_array_equal(image[..., 0], image[..., 1])
    np.testing.assert_array_equal(image[..., 1], image[..., 2])
    assert image.min() == 0
    assert image.max() <= 255


def test_constant_spectrogram_is_well_defined():
    image = spectrogram_to_sam_image(np.ones((3, 4)))
    assert not image.any()


@pytest.mark.parametrize("bad", [np.ones(4), np.array([[np.nan]])])
def test_spectrogram_validation(bad):
    with pytest.raises(ValueError):
        spectrogram_to_sam_image(bad)


def test_physical_pixel_coordinates_round_trip_on_nonuniform_axes():
    times = np.array([12.0, 12.5, 14.0, 18.0])
    frequencies = np.array([100.0, 80.0, 30.0])
    pixel = time_frequency_to_pixel(13.25, 55.0, times, frequencies)
    physical = pixel_to_time_frequency(*pixel, times, frequencies)

    np.testing.assert_allclose(physical, (13.25, 55.0))


def test_coordinate_validation_rejects_outside_and_nonmonotonic_values():
    with pytest.raises(ValueError, match="outside"):
        time_frequency_to_pixel(99, 1, np.array([2, 3]), np.array([0, 1]))
    with pytest.raises(ValueError, match="strictly monotonic"):
        pixel_to_time_frequency(0, 0, np.array([0, 2, 1]), np.array([0, 1]))


def test_overlap_metrics_and_empty_mask_convention():
    predicted = np.array([[1, 1], [0, 0]], dtype=bool)
    target = np.array([[0, 1], [1, 0]], dtype=bool)
    assert compute_iou(predicted, target) == pytest.approx(1 / 3)
    assert compute_dice(predicted, target) == pytest.approx(0.5)
    assert compute_iou(np.zeros((2, 2)), np.zeros((2, 2))) == 1.0
    assert compute_dice(np.zeros((2, 2)), np.zeros((2, 2))) == 1.0
    with pytest.raises(ValueError):
        compute_iou(np.zeros((2, 2)), np.zeros((3, 2)))


def test_wrapper_supports_box_single_point_and_multiple_points_without_sam():
    predictor = FakePredictor()
    segmenter = SAMStructureSegmenter(predictor=predictor)
    image = np.zeros((4, 5, 3), dtype=np.uint8)
    segmenter.set_image(image)
    assert predictor.image is image

    box_result = segmenter.segment_box([0, 0, 4, 3])
    assert box_result.best_index == 1
    assert box_result.best_score == pytest.approx(0.9)
    assert box_result.best_mask.sum() == 6
    np.testing.assert_array_equal(predictor.kwargs["box"], [0, 0, 4, 3])

    segmenter.segment_point([2, 1])
    np.testing.assert_array_equal(predictor.kwargs["point_labels"], [1])
    segmenter.segment_points([[1, 1], [3, 2]], [1, 0])
    np.testing.assert_array_equal(predictor.kwargs["point_labels"], [1, 0])


def test_wrapper_validates_inputs_without_loading_sam():
    segmenter = SAMStructureSegmenter(predictor=FakePredictor())
    with pytest.raises(ValueError):
        segmenter.set_image(np.zeros((3, 4, 3), dtype=float))
    with pytest.raises(ValueError):
        segmenter.segment_box([3, 0, 1, 2])
    with pytest.raises(RuntimeError, match="set_image"):
        segmenter.segment_point([1, 2])
    segmenter.set_image(np.zeros((3, 4, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="outside"):
        segmenter.segment_point([4, 2])
    with pytest.raises(ValueError):
        segmenter.segment_points([[1, 2]], [2])
    with pytest.raises(ValueError, match="checkpoint"):
        SAMStructureSegmenter()
