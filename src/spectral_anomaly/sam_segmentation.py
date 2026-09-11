"""Optional SAM 2 experiment for segmenting objects in STFT spectrograms.

This module is deliberately independent from the threshold- and structure-tensor
pipeline.  Importing it does not import PyTorch or SAM 2; those dependencies are
loaded only when :class:`SAMStructureSegmenter` constructs a predictor.
"""

from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from typing import Sequence

import numpy as np


SAM2_INSTALL_HINT = (
    "SAM 2 is required for this experiment. Install it with "
    "`pip install 'git+https://github.com/facebookresearch/sam2.git'` and install "
    "a PyTorch build appropriate for your CPU or CUDA environment."
)


@dataclass(frozen=True)
class SAMSegmentationResult:
    """Masks and confidence estimates returned for one SAM prompt."""

    masks: np.ndarray
    scores: np.ndarray
    logits: np.ndarray

    @property
    def best_index(self) -> int:
        """Index of the highest-scoring proposed mask."""
        if self.scores.size == 0:
            raise ValueError("SAM returned no masks")
        return int(np.argmax(self.scores))

    @property
    def best_mask(self) -> np.ndarray:
        """Highest-scoring mask according to SAM's predicted IoU score."""
        return self.masks[self.best_index]

    @property
    def best_score(self) -> float:
        """Score associated with :attr:`best_mask`."""
        return float(self.scores[self.best_index])


def spectrogram_to_sam_image(
    stft: np.ndarray,
    *,
    percentiles: tuple[float, float] = (1.0, 99.0),
    epsilon: float = 1e-12,
) -> np.ndarray:
    """Convert a complex STFT (or its magnitude) to a robust grayscale RGB image.

    The conversion is ``log1p(abs(stft))``, percentile clipping, and linear
    scaling to uint8. No significance/coherence mask or color map is involved.
    """
    values = np.asarray(stft)
    if values.ndim != 2 or values.size == 0:
        raise ValueError("stft must be a non-empty two-dimensional array")
    magnitude = np.abs(values)
    if not np.all(np.isfinite(magnitude)):
        raise ValueError("stft must contain only finite values")
    if (
        len(percentiles) != 2
        or not 0 <= percentiles[0] < percentiles[1] <= 100
    ):
        raise ValueError("percentiles must be increasing values between 0 and 100")
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be positive and finite")

    image = np.log1p(magnitude)
    low, high = np.percentile(image, percentiles)
    image = np.clip(image, low, high)
    image = (image - low) / (high - low + epsilon)
    grayscale = (255 * image).astype(np.uint8)
    return np.repeat(grayscale[..., None], 3, axis=2)


def _validated_axis(axis: np.ndarray, name: str) -> tuple[np.ndarray, bool]:
    values = np.asarray(axis, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be a non-empty finite one-dimensional array")
    differences = np.diff(values)
    ascending = bool(np.all(differences > 0))
    descending = bool(np.all(differences < 0))
    if values.size > 1 and not (ascending or descending):
        raise ValueError(f"{name} must be strictly monotonic")
    return values, descending


def _coordinate_to_pixel(value: float, axis: np.ndarray, name: str) -> float:
    values, descending = _validated_axis(axis, name)
    if not np.isfinite(value):
        raise ValueError(f"{name} coordinate must be finite")
    ordered = values[::-1] if descending else values
    pixels = np.arange(values.size, dtype=float)
    ordered_pixels = pixels[::-1] if descending else pixels
    if value < ordered[0] or value > ordered[-1]:
        raise ValueError(f"{name} coordinate is outside the STFT axis")
    return float(np.interp(value, ordered, ordered_pixels))


def _pixel_to_coordinate(pixel: float, axis: np.ndarray, name: str) -> float:
    values, _ = _validated_axis(axis, name)
    if not np.isfinite(pixel) or pixel < 0 or pixel > values.size - 1:
        raise ValueError(f"{name} pixel is outside the STFT image")
    return float(np.interp(pixel, np.arange(values.size, dtype=float), values))


def time_frequency_to_pixel(
    time: float,
    frequency: float,
    spectral_time: np.ndarray,
    frequencies: np.ndarray,
) -> tuple[float, float]:
    """Map physical ``(time, frequency)`` to SAM image ``(x, y)``."""
    return (
        _coordinate_to_pixel(time, spectral_time, "spectral_time"),
        _coordinate_to_pixel(frequency, frequencies, "frequencies"),
    )


def pixel_to_time_frequency(
    x: float,
    y: float,
    spectral_time: np.ndarray,
    frequencies: np.ndarray,
) -> tuple[float, float]:
    """Map SAM image ``(x, y)`` to physical ``(time, frequency)``."""
    return (
        _pixel_to_coordinate(x, spectral_time, "spectral_time"),
        _pixel_to_coordinate(y, frequencies, "frequencies"),
    )


def compute_iou(pred_mask: np.ndarray, target_mask: np.ndarray) -> float:
    """Return intersection over union for two equally shaped binary masks."""
    predicted, target = _validate_masks(pred_mask, target_mask)
    union = np.count_nonzero(predicted | target)
    return 1.0 if union == 0 else float(np.count_nonzero(predicted & target) / union)


def compute_dice(pred_mask: np.ndarray, target_mask: np.ndarray) -> float:
    """Return the Sørensen-Dice coefficient for two binary masks."""
    predicted, target = _validate_masks(pred_mask, target_mask)
    denominator = np.count_nonzero(predicted) + np.count_nonzero(target)
    return 1.0 if denominator == 0 else float(
        2 * np.count_nonzero(predicted & target) / denominator
    )


def _validate_masks(
    pred_mask: np.ndarray, target_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    predicted = np.asarray(pred_mask)
    target = np.asarray(target_mask)
    if predicted.ndim != 2 or target.ndim != 2 or predicted.shape != target.shape:
        raise ValueError("pred_mask and target_mask must be equally shaped 2-D arrays")
    return predicted.astype(bool), target.astype(bool)


class SAMStructureSegmenter:
    """Small prompt-oriented wrapper around Meta's official SAM 2 predictor."""

    def __init__(
        self,
        checkpoint: str | PathLike[str] | None = None,
        *,
        model_config: str = "configs/sam2.1/sam2.1_hiera_s.yaml",
        device: str = "auto",
        predictor: object | None = None,
    ) -> None:
        self._image_shape: tuple[int, int] | None = None
        if predictor is not None:
            self.predictor = predictor
            self.device = device
            return
        if checkpoint is None:
            raise ValueError("checkpoint is required when predictor is not provided")
        try:
            import torch
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError as error:
            raise ImportError(SAM2_INSTALL_HINT) from error
        selected_device = (
            "cuda" if device == "auto" and torch.cuda.is_available() else
            "cpu" if device == "auto" else device
        )
        if selected_device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        model = build_sam2(model_config, str(checkpoint), device=selected_device)
        self.predictor = SAM2ImagePredictor(model)
        self.device = selected_device

    def set_image(self, image: np.ndarray) -> None:
        """Embed one uint8 RGB spectrogram for subsequent prompts."""
        values = np.asarray(image)
        if values.ndim != 3 or values.shape[2] != 3 or values.dtype != np.uint8:
            raise ValueError("image must be an H x W x 3 uint8 array")
        self.predictor.set_image(values)
        self._image_shape = values.shape[:2]

    def _validate_prompt_coordinates(self, coordinates: np.ndarray) -> None:
        if self._image_shape is None:
            raise RuntimeError("set_image must be called before requesting segmentation")
        height, width = self._image_shape
        if np.any(coordinates[..., 0] > width - 1) or np.any(
            coordinates[..., 1] > height - 1
        ):
            raise ValueError("prompt coordinates are outside the current image")

    def segment_box(
        self, box: Sequence[float], *, multimask_output: bool = True
    ) -> SAMSegmentationResult:
        """Segment an object prompted by ``[x_min, y_min, x_max, y_max]``."""
        values = np.asarray(box, dtype=np.float32)
        if values.shape != (4,) or not np.all(np.isfinite(values)):
            raise ValueError("box must contain four finite coordinates")
        if values[0] >= values[2] or values[1] >= values[3] or np.any(values < 0):
            raise ValueError("box must have ordered, non-negative coordinates")
        self._validate_prompt_coordinates(values.reshape(2, 2))
        return self._predict(box=values, multimask_output=multimask_output)

    def segment_point(
        self, point: Sequence[float], *, multimask_output: bool = True
    ) -> SAMSegmentationResult:
        """Segment the object containing one positive ``[x, y]`` point."""
        return self.segment_points([point], [1], multimask_output=multimask_output)

    def segment_points(
        self,
        points: Sequence[Sequence[float]],
        labels: Sequence[int],
        *,
        multimask_output: bool = True,
    ) -> SAMSegmentationResult:
        """Segment with positive (1) and/or negative (0) point prompts."""
        coordinates = np.asarray(points, dtype=np.float32)
        point_labels = np.asarray(labels, dtype=np.int32)
        if coordinates.ndim != 2 or coordinates.shape[1:] != (2,) or len(coordinates) == 0:
            raise ValueError("points must have shape (N, 2) with N >= 1")
        if point_labels.shape != (len(coordinates),) or not np.isin(point_labels, [0, 1]).all():
            raise ValueError("labels must contain one 0 or 1 for every point")
        if not np.all(np.isfinite(coordinates)) or np.any(coordinates < 0):
            raise ValueError("point coordinates must be finite and non-negative")
        self._validate_prompt_coordinates(coordinates)
        return self._predict(
            point_coords=coordinates,
            point_labels=point_labels,
            multimask_output=multimask_output,
        )

    def _predict(self, **kwargs: object) -> SAMSegmentationResult:
        masks, scores, logits = self.predictor.predict(**kwargs)
        masks_array = np.asarray(masks, dtype=bool)
        scores_array = np.asarray(scores, dtype=float).reshape(-1)
        logits_array = np.asarray(logits)
        if masks_array.ndim != 3 or len(masks_array) != len(scores_array):
            raise RuntimeError("SAM returned inconsistent masks and scores")
        return SAMSegmentationResult(masks_array, scores_array, logits_array)


def plot_sam_segmentation(
    stft: np.ndarray,
    sam_image: np.ndarray,
    result: SAMSegmentationResult,
    spectral_time: np.ndarray,
    frequencies: np.ndarray,
):
    """Plot original STFT, SAM input, best mask, and overlay using Plotly."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    magnitude = np.abs(np.asarray(stft))
    image = np.asarray(sam_image)
    if magnitude.shape != image.shape[:2] or result.best_mask.shape != magnitude.shape:
        raise ValueError("stft, sam_image, and selected mask shapes must agree")
    if len(spectral_time) != magnitude.shape[1] or len(frequencies) != magnitude.shape[0]:
        raise ValueError("physical axes must match the STFT shape")
    title = f"SAM best mask: {result.best_index} (score={result.best_score:.3f})"
    figure = make_subplots(rows=2, cols=2, subplot_titles=(
        "STFT magnitude (log1p)", "Image supplied to SAM (grayscale)", title,
        "Best mask over STFT",
    ))
    common = {"x": spectral_time, "y": frequencies, "showscale": False}
    figure.add_trace(go.Heatmap(z=np.log1p(magnitude), colorscale="Viridis", **common), row=1, col=1)
    figure.add_trace(go.Heatmap(z=image[..., 0], colorscale="Gray", **common), row=1, col=2)
    figure.add_trace(go.Heatmap(z=result.best_mask.astype(np.uint8), colorscale="Blues", **common), row=2, col=1)
    figure.add_trace(go.Heatmap(z=np.log1p(magnitude), colorscale="Viridis", **common), row=2, col=2)
    overlay = np.where(result.best_mask, 1.0, np.nan)
    figure.add_trace(go.Heatmap(z=overlay, colorscale=[[0, "red"], [1, "red"]], opacity=0.4, **common), row=2, col=2)
    for row in (1, 2):
        for column in (1, 2):
            figure.update_xaxes(title_text="time (s)", row=row, col=column)
            figure.update_yaxes(title_text="frequency (Hz)", row=row, col=column)
    figure.update_layout(
        title="SAM 2 spectrogram segmentation — scores: "
        + ", ".join(f"{score:.3f}" for score in result.scores),
        height=850,
        template="plotly_white",
    )
    return figure
