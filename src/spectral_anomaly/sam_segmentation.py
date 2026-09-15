"""Optional SAM 2 experiment for segmenting objects in spectral representations.

This module is deliberately independent from the threshold- and structure-tensor
pipeline.  Importing it does not import PyTorch or SAM 2; those dependencies are
loaded only when :class:`SAMStructureSegmenter` constructs a predictor.
"""

from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from typing import Any, Mapping, Sequence

import numpy as np


SAM2_INSTALL_HINT = (
    "SAM 2 is required for this experiment. Install it with "
    "`pip install 'git+https://github.com/facebookresearch/sam2.git'` and install "
    "a PyTorch build appropriate for your CPU or CUDA environment."
)


AUTOMATIC_MASK_DEFAULTS: dict[str, object] = {
    "points_per_side": 32,
    "points_per_batch": 64,
    "pred_iou_thresh": 0.8,
    "stability_score_thresh": 0.8,
    "stability_score_offset": 1.0,
    "box_nms_thresh": 0.7,
    "crop_n_layers": 0,
    "crop_nms_thresh": 0.7,
    "crop_overlap_ratio": 512 / 1500,
    "crop_n_points_downscale_factor": 1,
    "min_mask_region_area": 0,
    "use_m2m": False,
}


def validate_spectral_representation(representation: object) -> str:
    """Validate and return the spectral representation selected for SAM."""
    if representation not in {"stft", "msst"}:
        raise ValueError("representation must be either 'stft' or 'msst'")
    return str(representation)


def validate_automatic_mask_options(options: Mapping[str, object]) -> dict[str, object]:
    """Validate options accepted by Meta's ``SAM2AutomaticMaskGenerator``."""
    unknown = set(options) - set(AUTOMATIC_MASK_DEFAULTS)
    if unknown:
        raise ValueError(f"unknown automatic mask option(s): {sorted(unknown)}")
    result = {**AUTOMATIC_MASK_DEFAULTS, **options}
    for name in ("points_per_side", "points_per_batch"):
        value = result[name]
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    for name in ("crop_n_layers", "min_mask_region_area"):
        value = result[name]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    value = result["crop_n_points_downscale_factor"]
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError("crop_n_points_downscale_factor must be a positive integer")
    for name in (
        "pred_iou_thresh", "stability_score_thresh", "box_nms_thresh",
        "crop_nms_thresh", "crop_overlap_ratio",
    ):
        value = result[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be between 0 and 1")
    offset = result["stability_score_offset"]
    if isinstance(offset, bool) or not isinstance(offset, (int, float)) or offset < 0:
        raise ValueError("stability_score_offset must be non-negative")
    if not isinstance(result["use_m2m"], bool):
        raise ValueError("use_m2m must be a boolean")
    return result


@dataclass(frozen=True)
class SAMSegment:
    """One automatic SAM region and its segmentation—not anomaly—metadata."""

    segment_id: int
    mask: np.ndarray
    area: int
    bbox: tuple[float, float, float, float]
    predicted_iou: float | None
    stability_score: float | None
    point_coords: tuple[tuple[float, float], ...]
    crop_box: tuple[float, float, float, float] | None
    centroid: tuple[float, float] | None
    temporal_width: int
    frequency_width: int
    image_fraction: float
    metadata: Mapping[str, Any]

    @classmethod
    def from_sam_annotation(cls, segment_id: int, annotation: Mapping[str, Any]) -> "SAMSegment":
        """Convert the dictionary returned by the official automatic generator."""
        mask = np.asarray(annotation.get("segmentation"))
        if mask.ndim != 2:
            raise ValueError("SAM annotation segmentation must be a 2-D mask")
        mask = mask.astype(bool)
        area = int(annotation.get("area", np.count_nonzero(mask)))
        bbox_value = annotation.get("bbox")
        if not isinstance(bbox_value, (list, tuple)) or len(bbox_value) != 4:
            raise ValueError("SAM annotation bbox must contain [x, y, width, height]")
        bbox = tuple(float(value) for value in bbox_value)
        rows, columns = np.nonzero(mask)
        centroid = None if not len(rows) else (float(columns.mean()), float(rows.mean()))
        extras = {key: value for key, value in annotation.items() if key != "segmentation"}
        points = tuple(tuple(float(v) for v in point) for point in annotation.get("point_coords", ()))
        crop = annotation.get("crop_box")
        return cls(
            segment_id=segment_id, mask=mask, area=area, bbox=bbox,
            predicted_iou=_optional_float(annotation.get("predicted_iou")),
            stability_score=_optional_float(annotation.get("stability_score")),
            point_coords=points,
            crop_box=None if crop is None else tuple(float(v) for v in crop),
            centroid=centroid, temporal_width=int(round(bbox[2])),
            frequency_width=int(round(bbox[3])), image_fraction=area / mask.size,
            metadata=extras,
        )


def _optional_float(value: object) -> float | None:
    return None if value is None else float(value)


def annotations_to_segments(annotations: Sequence[Mapping[str, Any]]) -> list[SAMSegment]:
    """Return stable S1..SN regions, sorted from largest to smallest."""
    ordered = sorted(annotations, key=lambda item: int(item.get("area", 0)), reverse=True)
    return [SAMSegment.from_sam_annotation(index, item) for index, item in enumerate(ordered, 1)]


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


@dataclass(frozen=True)
class ContrastPoint:
    """One local energy-contrast maximum in physical and image coordinates."""

    time: float
    frequency: float
    contrast: float
    time_index: int
    frequency_index: int
    pixel: tuple[float, float]


@dataclass(frozen=True)
class GuidedSAMMask:
    """The selected SAM proposal for one independently prompted point."""

    point_index: int
    point: ContrastPoint
    selected_index: int
    mask: np.ndarray
    score: float
    proposals: SAMSegmentationResult


@dataclass(frozen=True)
class GuidedMaskDuplicate:
    """A guided mask suppressed because it overlaps a better-scored mask."""

    removed_point_index: int
    kept_point_index: int
    iou: float


@dataclass(frozen=True)
class GuidedSAMResult:
    """Raw per-point predictions and mask-IoU deduplication outcome."""

    raw_masks: tuple[GuidedSAMMask, ...]
    masks: tuple[GuidedSAMMask, ...]
    duplicates: tuple[GuidedMaskDuplicate, ...]


def spectrogram_to_sam_image(
    spectral_map: np.ndarray,
    *,
    transform: str = "log",
    percentiles: tuple[float, float] = (1.0, 99.0),
    epsilon: float = 1e-12,
) -> np.ndarray:
    """Convert a spectral representation (or its magnitude) to grayscale RGB.

    Normalisation is performed independently for each input window. It can make
    a low-energy window look visually strong and is not an absolute-energy
    calibration. ``transform="log"`` compresses the magnitude's dynamic range with
    ``log1p``; ``transform="linear"`` leaves it linear.  Both paths then use the
    same percentile clipping and uint8 scaling.  No significance/coherence mask
    or color map is involved.
    """
    values = np.asarray(spectral_map)
    if values.ndim != 2 or values.size == 0:
        raise ValueError("spectral_map must be a non-empty two-dimensional array")
    magnitude = np.abs(values)
    if not np.all(np.isfinite(magnitude)):
        raise ValueError("spectral_map must contain only finite values")
    if transform not in {"log", "linear"}:
        raise ValueError("transform must be either 'log' or 'linear'")
    if (
        len(percentiles) != 2
        or not 0 <= percentiles[0] < percentiles[1] <= 100
    ):
        raise ValueError("percentiles must be increasing values between 0 and 100")
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be positive and finite")

    image = np.log1p(magnitude) if transform == "log" else magnitude
    low, high = np.percentile(image, percentiles)
    image = np.clip(image, low, high)
    image = (image - low) / (high - low + epsilon)
    grayscale = (255 * image).astype(np.uint8)
    return np.repeat(grayscale[..., None], 3, axis=2)


def temporal_energy_contrast(
    stft: np.ndarray,
    spectral_time: np.ndarray,
    *,
    exclusion_seconds: float,
    neighborhood_seconds: float,
    epsilon: float,
    min_valid_references: int = 3,
) -> np.ndarray:
    """Compute same-frequency temporal energy contrast on a real time axis.

    Reference samples satisfy ``exclusion_seconds < abs(u-t) <=
    neighborhood_seconds``.  Only available samples are used at boundaries;
    there is no wrapping. Non-finite energy and insufficient backgrounds yield
    NaN, which deliberately cannot become a prompt.
    """
    values = np.asarray(stft)
    times, _ = _validated_axis(spectral_time, "spectral_time")
    if values.ndim != 2 or values.shape[1] != len(times):
        raise ValueError("stft must be 2-D with columns matching spectral_time")
    if (not np.isfinite(exclusion_seconds) or not np.isfinite(neighborhood_seconds)
            or exclusion_seconds < 0 or neighborhood_seconds <= exclusion_seconds):
        raise ValueError("durations must satisfy 0 <= exclusion_seconds < neighborhood_seconds")
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be positive and finite")
    if (not isinstance(min_valid_references, int) or isinstance(min_valid_references, bool)
            or min_valid_references < 1):
        raise ValueError("min_valid_references must be a positive integer")

    energy = np.abs(values) ** 2
    energy[~np.isfinite(energy)] = np.nan
    contrast = np.full(energy.shape, np.nan, dtype=float)
    for column, time in enumerate(times):
        distance = np.abs(times - time)
        references = (distance > exclusion_seconds) & (distance <= neighborhood_seconds)
        if not references.any():
            continue
        candidates = energy[:, references]
        counts = np.count_nonzero(np.isfinite(candidates), axis=1)
        background = np.full(energy.shape[0], np.nan)
        enough = counts >= min_valid_references
        if enough.any():
            background[enough] = np.nanmedian(candidates[enough], axis=1)
        valid = enough & np.isfinite(energy[:, column])
        contrast[valid, column] = energy[valid, column] / np.maximum(
            background[valid], epsilon
        )
    return contrast


def select_contrast_points(
    contrast: np.ndarray,
    spectral_time: np.ndarray,
    frequencies: np.ndarray,
    *,
    threshold: float,
    min_time_spacing_seconds: float,
    min_frequency_spacing_hz: float,
    max_points: int,
) -> list[ContrastPoint]:
    """Select 8-neighbourhood maxima, then greedily space them in physical units."""
    from scipy.ndimage import maximum_filter

    values = np.asarray(contrast, dtype=float)
    times, _ = _validated_axis(spectral_time, "spectral_time")
    frequency_values, _ = _validated_axis(frequencies, "frequencies")
    if values.shape != (len(frequency_values), len(times)):
        raise ValueError("contrast shape must match frequency and time axes")
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")
    if min_time_spacing_seconds < 0 or min_frequency_spacing_hz < 0:
        raise ValueError("point spacings must be non-negative")
    if not isinstance(max_points, int) or isinstance(max_points, bool) or max_points < 0:
        raise ValueError("max_points must be a non-negative integer")
    if max_points == 0:
        return []
    finite_values = np.where(np.isfinite(values), values, -np.inf)
    local = finite_values == maximum_filter(finite_values, size=3, mode="constant", cval=-np.inf)
    rows, columns = np.nonzero(local & (finite_values >= threshold))
    candidates = sorted(zip(rows, columns), key=lambda rc: (-values[rc], rc[1], rc[0]))
    selected: list[ContrastPoint] = []
    for row, column in candidates:
        time, frequency = float(times[column]), float(frequency_values[row])
        # Points are too close only when they are close along both dimensions.
        if any(abs(time - p.time) < min_time_spacing_seconds and
               abs(frequency - p.frequency) < min_frequency_spacing_hz for p in selected):
            continue
        pixel = time_frequency_to_pixel(time, frequency, times, frequency_values)
        if not np.allclose(pixel, (column, row), atol=1e-6):
            raise RuntimeError("physical-to-image conversion changed the STFT bin location")
        selected.append(ContrastPoint(
            time, frequency, float(values[row, column]), int(column), int(row), pixel,
        ))
        if len(selected) == max_points:
            break
    return selected


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
        raise ValueError(f"{name} coordinate is outside the spectral axis")
    return float(np.interp(value, ordered, ordered_pixels))


def _pixel_to_coordinate(pixel: float, axis: np.ndarray, name: str) -> float:
    values, _ = _validated_axis(axis, name)
    if not np.isfinite(pixel) or pixel < 0 or pixel > values.size - 1:
        raise ValueError(f"{name} pixel is outside the spectral image")
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
            self.device = str(getattr(predictor, "device", device))
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


class SAMAutomaticMaskSegmenter:
    """Wrapper around Meta's official ``SAM2AutomaticMaskGenerator``.

    The model and generator are constructed once. A call to :meth:`generate_masks`
    processes an image once; callers should not loop over manual prompts.
    """

    def __init__(
        self,
        checkpoint: str | PathLike[str] | None = None,
        *,
        model_config: str = "configs/sam2.1/sam2.1_hiera_s.yaml",
        device: str = "auto",
        generator: object | None = None,
        **automatic_mask_options: object,
    ) -> None:
        self.options = validate_automatic_mask_options(automatic_mask_options)
        if generator is not None:
            self.generator = generator
            self.device = str(getattr(generator, "device", device))
            return
        if checkpoint is None:
            raise ValueError("checkpoint is required when generator is not provided")
        try:
            import torch
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
            from sam2.build_sam import build_sam2
        except ImportError as error:
            raise ImportError(SAM2_INSTALL_HINT) from error
        selected_device = (
            "cuda" if device == "auto" and torch.cuda.is_available() else
            "cpu" if device == "auto" else device
        )
        if selected_device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        model = build_sam2(model_config, str(checkpoint), device=selected_device)
        self.generator = SAM2AutomaticMaskGenerator(model, **self.options)
        self.device = selected_device

    def generate_masks(self, image: np.ndarray) -> list[SAMSegment]:
        """Generate all automatic regions for one uint8 RGB image."""
        values = np.asarray(image)
        if values.ndim != 3 or values.shape[2] != 3 or values.dtype != np.uint8:
            raise ValueError("image must be an H x W x 3 uint8 array")
        annotations = self.generator.generate(values)
        if not isinstance(annotations, list):
            raise RuntimeError("SAM automatic mask generator returned a non-list result")
        segments = annotations_to_segments(annotations)
        if any(segment.mask.shape != values.shape[:2] for segment in segments):
            raise RuntimeError("SAM returned a mask whose shape differs from the input image")
        return segments


def deduplicate_guided_masks(
    masks: Sequence[GuidedSAMMask], *, iou_threshold: float
) -> tuple[list[GuidedSAMMask], list[GuidedMaskDuplicate]]:
    """Greedily retain quality-ranked masks using binary-mask (not box) IoU."""
    if not np.isfinite(iou_threshold) or not 0 <= iou_threshold <= 1:
        raise ValueError("iou_threshold must be between 0 and 1")
    ordered = sorted(masks, key=lambda item: (-item.score, item.point_index))
    kept: list[GuidedSAMMask] = []
    duplicates: list[GuidedMaskDuplicate] = []
    for candidate in ordered:
        matches = [(existing, compute_iou(candidate.mask, existing.mask)) for existing in kept]
        duplicate = next(((existing, overlap) for existing, overlap in matches
                          if overlap >= iou_threshold), None)
        if duplicate is None:
            kept.append(candidate)
        else:
            duplicates.append(GuidedMaskDuplicate(
                candidate.point_index, duplicate[0].point_index, duplicate[1]
            ))
    return kept, duplicates


def segment_contrast_points(
    segmenter: SAMStructureSegmenter,
    image: np.ndarray,
    points: Sequence[ContrastPoint],
    *,
    iou_threshold: float,
    multimask_output: bool = True,
) -> GuidedSAMResult:
    """Prompt SAM independently and retain its highest predicted-IoU proposal."""
    if not isinstance(multimask_output, bool):
        raise ValueError("multimask_output must be a boolean")
    segmenter.set_image(image)
    raw: list[GuidedSAMMask] = []
    for point_index, point in enumerate(points, 1):
        result = segmenter.segment_point(
            point.pixel, multimask_output=multimask_output
        )
        selected = result.best_index
        raw.append(GuidedSAMMask(
            point_index, point, selected, result.best_mask, result.best_score, result
        ))
    kept, duplicates = deduplicate_guided_masks(raw, iou_threshold=iou_threshold)
    return GuidedSAMResult(tuple(raw), tuple(kept), tuple(duplicates))


def plot_sam_guided_comparison(
    stft: np.ndarray,
    contrast: np.ndarray,
    sam_image: np.ndarray,
    spectral_time: np.ndarray,
    frequencies: np.ndarray,
    points: Sequence[ContrastPoint],
    guided: GuidedSAMResult,
    automatic: Sequence[SAMSegment],
):
    """Compare STFT, contrast, and individually inspectable automatic/guided masks."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    magnitude, image = np.abs(np.asarray(stft)), np.asarray(sam_image)
    if magnitude.shape != np.asarray(contrast).shape or magnitude.shape != image.shape[:2]:
        raise ValueError("stft, contrast, and sam_image shapes must agree")
    panels: list[tuple[str, np.ndarray | None]] = [
        ("STFT originale (log1p magnitude)", None), ("Contraste énergétique local", None)
    ]
    panels.extend((f"Auto S{s.segment_id} — score={_format_optional(s.predicted_iou)}", s.mask)
                  for s in automatic)
    for selected in guided.raw_masks:
        for variant_index, (mask, score) in enumerate(
                zip(selected.proposals.masks, selected.proposals.scores)):
            suffix = " — retenu" if variant_index == selected.selected_index else ""
            panels.append((f"Guidé P{selected.point_index} variante M{variant_index + 1} "
                           f"— score={score:.4f}{suffix}", mask))
    columns, rows = 3, (len(panels) + 2) // 3
    figure = make_subplots(rows=rows, cols=columns, subplot_titles=[title for title, _ in panels])
    common = {"x": spectral_time, "y": frequencies, "showscale": False}
    for index, (_, mask) in enumerate(panels):
        row, column = divmod(index, columns); row += 1; column += 1
        base = np.log1p(magnitude) if index != 1 else contrast
        figure.add_trace(go.Heatmap(z=base, colorscale="Viridis", **common), row, column)
        if mask is not None:
            removed = next((d for d in guided.duplicates
                            if (panels[index][0].startswith(f"Guidé P{d.removed_point_index} ")
                                and "— retenu" in panels[index][0])), None)
            custom = np.where(mask, "masque", "aucun masque")
            figure.add_trace(go.Heatmap(
                z=mask.astype(np.uint8), customdata=custom, **common,
                colorscale=[[0, "rgba(0,0,0,0)"], [.499, "rgba(0,0,0,0)"],
                            [.5, "rgba(255,0,0,.45)"], [1, "rgba(255,0,0,.45)"]],
                hovertemplate="%{customdata}<br>t=%{x}<br>f=%{y}<extra></extra>",
            ), row, column)
            if removed is not None:
                figure.add_annotation(text=(f"doublon de P{removed.kept_point_index}, "
                                            f"IoU={removed.iou:.3f}"), x=.5, y=.05,
                                      xref="x domain", yref="y domain", showarrow=False,
                                      bgcolor="white", row=row, col=column)
        if index == 1:
            figure.add_trace(go.Scatter(
                x=[p.time for p in points], y=[p.frequency for p in points], mode="markers+text",
                text=[f"P{i}" for i in range(1, len(points) + 1)], textposition="top center",
                customdata=[p.contrast for p in points], marker={"color": "cyan", "size": 8},
                hovertemplate="%{text}<br>t=%{x}<br>f=%{y}<br>C=%{customdata:.3g}<extra></extra>",
                showlegend=False,
            ), row, column)
    figure.update_layout(
        title=(f"SAM 2 automatique={len(automatic)} — guidé brut={len(guided.raw_masks)}, "
               f"après déduplication={len(guided.masks)}"),
        height=max(650, rows * 340), template="plotly_white",
    )
    return figure


def plot_sam_automatic_masks(
    spectral_map: np.ndarray,
    sam_image: np.ndarray,
    segments: Sequence[SAMSegment],
    spectral_time: np.ndarray,
    frequencies: np.ndarray,
    *,
    max_labels: int = 40,
    representation_name: str = "spectral representation",
    image_transform: str = "log",
):
    """Plot the exact SAM input, every region, an overlay, and metadata table."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    magnitude, image = np.abs(np.asarray(spectral_map)), np.asarray(sam_image)
    if magnitude.shape != image.shape[:2]:
        raise ValueError("spectral_map and sam_image shapes must agree")
    if len(spectral_time) != magnitude.shape[1] or len(frequencies) != magnitude.shape[0]:
        raise ValueError("physical axes must match the spectral map shape")
    if max_labels < 0:
        raise ValueError("max_labels must be non-negative")
    for segment in segments:
        if segment.mask.shape != magnitude.shape:
            raise ValueError("all segment masks must match the spectral map shape")
    common = {"x": spectral_time, "y": frequencies, "showscale": False}
    figure = make_subplots(
        rows=2, cols=3,
        specs=[[{}, {}, {}], [{}, {"type": "table", "colspan": 2}, None]],
        subplot_titles=(f"{representation_name} originale (log1p magnitude)", "Image exacte donnée à SAM",
                        f"Tous les masques ({len(segments)})", "Overlay des masques", "Résumé des segments"),
    )
    figure.add_trace(go.Heatmap(z=np.log1p(magnitude), colorscale="Viridis", **common), 1, 1)
    figure.add_trace(go.Heatmap(z=image[..., 0], colorscale="Gray", **common), 1, 2)
    figure.add_trace(go.Heatmap(z=np.log1p(magnitude), colorscale="Viridis", **common), 1, 3)
    _add_mask_contours(figure, segments, spectral_time, frequencies, 1, 3)
    figure.add_trace(go.Heatmap(z=np.log1p(magnitude), colorscale="Viridis", **common), 2, 1)
    _add_mask_contours(figure, segments, spectral_time, frequencies, 2, 1)
    for segment in segments[:max_labels]:
        x, y = segment.centroid or (segment.bbox[0], segment.bbox[1])
        figure.add_annotation(x=spectral_time[int(np.clip(round(x), 0, len(spectral_time)-1))],
                              y=frequencies[int(np.clip(round(y), 0, len(frequencies)-1))],
                              text=f"S{segment.segment_id}", showarrow=False,
                              font={"color": "white", "size": 9}, row=2, col=1)
    headers = ["segment_id", "area", "predicted_iou", "stability_score", "bbox", "centroid", "fraction"]
    columns = [
        [f"S{s.segment_id}" for s in segments], [s.area for s in segments],
        [_format_optional(s.predicted_iou) for s in segments],
        [_format_optional(s.stability_score) for s in segments],
        [str(tuple(round(v, 2) for v in s.bbox)) for s in segments],
        [str(tuple(round(v, 2) for v in s.centroid)) if s.centroid else "—" for s in segments],
        [f"{s.image_fraction:.4f}" for s in segments],
    ]
    figure.add_trace(go.Table(header={"values": headers}, cells={"values": columns}), 2, 2)
    for row, col in ((1, 1), (1, 2), (1, 3), (2, 1)):
        figure.update_xaxes(title_text="time (s)", row=row, col=col)
        figure.update_yaxes(title_text="frequency (Hz)", row=row, col=col)
    figure.update_layout(title=("SAM 2 automatic spectral regions (no anomaly decision)"
                                f"<br>Representation: {representation_name.upper()}"
                                f" — Image transform: {image_transform}"),
                         height=950, template="plotly_white")
    return figure


def _format_optional(value: float | None) -> str:
    return "—" if value is None else f"{value:.4f}"


def _add_mask_contours(figure, segments, spectral_time, frequencies, row, col):
    """Add independent mask boundaries; overlaps never create synthetic IDs."""
    import plotly.graph_objects as go
    from scipy.ndimage import binary_erosion

    palette = ["#ef4444", "#22c55e", "#3b82f6", "#f59e0b", "#a855f7", "#06b6d4"]
    for index, segment in enumerate(segments):
        boundary = segment.mask & ~binary_erosion(segment.mask)
        rows, columns = np.nonzero(boundary)
        figure.add_trace(go.Scattergl(
            x=np.asarray(spectral_time)[columns], y=np.asarray(frequencies)[rows],
            mode="markers", marker={"size": 3, "color": palette[index % len(palette)]},
            name=f"S{segment.segment_id}", legendgroup=f"S{segment.segment_id}",
            hovertemplate=f"segment S{segment.segment_id}<br>t=%{{x}}<br>f=%{{y}}<extra></extra>",
        ), row=row, col=col)


def plot_sam_diagnostic(
    signal: np.ndarray,
    signal_time: np.ndarray,
    spectral_map: np.ndarray,
    sam_image: np.ndarray,
    raw_segments,
    final_segments,
    spectral_time: np.ndarray,
    frequencies: np.ndarray,
    *,
    device: str,
    mode: str,
    contrast: np.ndarray | None = None,
    points: Sequence[ContrastPoint] = (),
    guided: GuidedSAMResult | None = None,
):
    """Plot exact SAM input, raw outputs, source metadata, and final masks."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    individual = [(f"Masque brut S{s.segment_id}", s) for s in raw_segments]
    if guided is not None:
        individual = []
        for prompted in guided.raw_masks:
            for proposal_index, (mask, score) in enumerate(
                    zip(prompted.proposals.masks, prompted.proposals.scores)):
                label = "retenu" if proposal_index == prompted.selected_index else "proposition"
                proxy = type("MaskView", (), {"segment_id": f"P{prompted.point_index}.M{proposal_index + 1}",
                                               "mask": np.asarray(mask, dtype=bool)})()
                individual.append((f"P{prompted.point_index} M{proposal_index + 1} — {label}, "
                                   f"predicted_iou={score:.3f}", proxy))
    base_rows = 3 + (1 if contrast is not None else 0)
    individual_rows = (len(individual) + 2) // 3
    table_row = base_rows + individual_rows + 1
    specs = [[{"colspan": 3}, None, None], [{}, {}, {}], [{}, {}, {}]]
    if contrast is not None:
        specs.append([{"colspan": 3}, None, None])
    specs.extend([[{}, {}, {}] for _ in range(individual_rows)])
    specs.append([{"type": "table", "colspan": 3}, None, None])
    titles = ["Signal temporel", "Représentation originale", "Image uint8 RGB exacte (canal gris)",
              "Masques SAM bruts — contours", "Overlay brut", "Après post-traitement"]
    if contrast is not None:
        titles.append("Contraste énergétique local et points positifs")
    titles.extend(title for title, _ in individual)
    titles.append("Métadonnées brutes et traçabilité finale")
    figure = make_subplots(rows=table_row, cols=3, specs=specs, subplot_titles=titles)
    common = {"x": spectral_time, "y": frequencies, "showscale": False}
    figure.add_trace(go.Scatter(x=signal_time, y=signal, name="signal"), 1, 1)
    magnitude = np.log1p(np.abs(spectral_map))
    figure.add_trace(go.Heatmap(z=magnitude, colorscale="Viridis", **common), 2, 1)
    figure.add_trace(go.Heatmap(z=sam_image[..., 0], colorscale="Gray", **common), 2, 2)
    _add_mask_contours(figure, raw_segments, spectral_time, frequencies, 2, 3)
    figure.add_trace(go.Heatmap(z=magnitude, colorscale="Viridis", **common), 3, 1)
    figure.add_trace(go.Heatmap(z=magnitude, colorscale="Viridis", **common), 3, 2)
    _add_mask_contours(figure, raw_segments, spectral_time, frequencies, 3, 2)
    figure.add_trace(go.Heatmap(z=magnitude, colorscale="Viridis", **common), 3, 3)
    _add_mask_contours(figure, final_segments, spectral_time, frequencies, 3, 3)
    next_row = 4
    if contrast is not None:
        figure.add_trace(go.Heatmap(z=contrast, colorscale="Magma", **common), next_row, 1)
        figure.add_trace(go.Scatter(
            x=[point.time for point in points], y=[point.frequency for point in points],
            mode="markers+text", text=[f"P{i}" for i in range(1, len(points) + 1)],
            customdata=[point.contrast for point in points], marker={"color": "cyan", "size": 9},
            hovertemplate="%{text}<br>t=%{x}<br>f=%{y}<br>contraste=%{customdata:.3g}<extra></extra>",
            name="points guidés",
        ), next_row, 1)
        next_row += 1
    for index, (_, segment) in enumerate(individual):
        row, col = next_row + index // 3, index % 3 + 1
        figure.add_trace(go.Heatmap(z=segment.mask.astype(np.uint8), colorscale="Blues", **common), row, col)
    source_records = [record for segment in raw_segments
                      for record in segment.metadata.get("source_segments", [])]
    headers = ["segment_id", "area", "predicted_iou", "stability_score", "bbox",
               "centroid", "image_fraction"]
    columns = [[record.get(key, "—") for record in source_records] for key in headers]
    final_trace = [f"F{s.segment_id}: sources={s.source_segment_ids}; "
                   f"links={s.metadata.get('overlap_links', [])}" for s in final_segments]
    columns[0] = [*columns[0], *final_trace]
    for column in columns[1:]:
        column.extend(["—"] * len(final_trace))
    figure.add_trace(go.Table(header={"values": headers}, cells={"values": columns}), table_row, 1)
    duplicate_text = ""
    if guided is not None and guided.duplicates:
        duplicate_text = " — déduplications guidées: " + str([
            {"removed_prompt": d.removed_point_index, "kept_prompt": d.kept_point_index,
             "iou": round(d.iou, 4)} for d in guided.duplicates])
    figure.update_layout(
        title=f"Diagnostic SAM 2 réel — mode={mode}, device={device}{duplicate_text}",
        height=max(1200, table_row * 320), template="plotly_white",
    )
    return figure


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
