"""Reusable SAM session shared by automatic and energy-contrast modes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .masks import Segment
from .sam_segmentation import (
    GuidedSAMResult,
    SAMAutomaticMaskSegmenter,
    SAMSegment,
    SAMStructureSegmenter,
    segment_contrast_points,
    select_contrast_points,
    spectrogram_to_sam_image,
    temporal_energy_contrast,
)


def _sam_metadata(segment: SAMSegment) -> dict[str, Any]:
    """Return all audit-relevant metadata without copying the binary mask."""
    return {
        "segment_id": segment.segment_id,
        "area": segment.area,
        "bbox": segment.bbox,
        "predicted_iou": segment.predicted_iou,
        "stability_score": segment.stability_score,
        "point_coords": segment.point_coords,
        "crop_box": segment.crop_box,
        "centroid": segment.centroid,
        "image_fraction": segment.image_fraction,
        "sam_metadata": dict(segment.metadata),
    }


@dataclass(frozen=True)
class SpectrumSegmentation:
    """Complete diagnostic output for one spectral image."""

    image: np.ndarray
    segments: tuple[Segment, ...]
    device: str
    mode: str
    raw_sam_segments: tuple[SAMSegment, ...] = ()
    raw_segments: tuple[Segment, ...] = ()
    contrast: np.ndarray | None = None
    points: tuple[Any, ...] = ()
    guided: GuidedSAMResult | None = None


class SAMSegmentationSession:
    """Construct SAM once, then segment any number of spectral windows."""

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        automatic_segmenter: SAMAutomaticMaskSegmenter | None = None,
        predictor: object | None = None,
    ) -> None:
        self.config = config
        self.mode = str(config["segmentation_mode"])
        model = config["model"]
        if self.mode == "automatic":
            self.segmenter = automatic_segmenter or SAMAutomaticMaskSegmenter(
                checkpoint=model["checkpoint"],
                model_config=model["config"],
                device=config["device"],
                **config["automatic"],
            )
        elif self.mode == "energy_contrast":
            if isinstance(predictor, SAMStructureSegmenter):
                self.segmenter = predictor
            else:
                self.segmenter = SAMStructureSegmenter(
                    checkpoint=None if predictor is not None else model["checkpoint"],
                    model_config=model["config"],
                    device=config["device"],
                    predictor=predictor,
                )
        else:
            raise ValueError("segmentation_mode must be 'automatic' or 'energy_contrast'")
        self.device = str(self.segmenter.device)

    def segment(self, spectral) -> SpectrumSegmentation:
        """Run inference without rebuilding the model or generator."""
        image_options = self.config["image"]
        image = spectrogram_to_sam_image(
            spectral.values,
            transform=image_options["transform"],
            percentiles=tuple(image_options["percentiles"]),
            epsilon=image_options["epsilon"],
        )
        if self.mode == "automatic":
            raw_sam = tuple(self.segmenter.generate_masks(image))
            segments = tuple(
                Segment(
                    item.segment_id,
                    item.mask,
                    metadata={"source_segments": [_sam_metadata(item)]},
                )
                for item in raw_sam
            )
            return SpectrumSegmentation(image, segments, self.device, self.mode,
                                        raw_sam_segments=raw_sam, raw_segments=segments)

        options = self.config["energy_contrast"]
        contrast = temporal_energy_contrast(
            spectral.stft,
            spectral.times,
            exclusion_seconds=options["exclusion_seconds"],
            neighborhood_seconds=options["neighborhood_seconds"],
            epsilon=options["epsilon"],
            min_valid_references=options["min_valid_references"],
        )
        points = tuple(select_contrast_points(
            contrast,
            spectral.times,
            spectral.frequencies,
            threshold=options["contrast_threshold"],
            min_time_spacing_seconds=options["min_time_spacing"],
            min_frequency_spacing_hz=options["min_frequency_spacing"],
            max_points=options["max_points"],
        ))
        guided = segment_contrast_points(
            self.segmenter,
            image,
            points,
            iou_threshold=options["mask_iou_threshold"],
            multimask_output=options["multimask_output"],
        )
        raw_segments = tuple(
            Segment(
                item.point_index,
                item.mask,
                metadata={"source_segments": [{
                    "segment_id": item.point_index,
                    "prompt_point_index": item.point_index,
                    "prompt": {"time": item.point.time, "frequency": item.point.frequency,
                               "contrast": item.point.contrast, "pixel": item.point.pixel},
                    "selected_proposal": item.selected_index,
                    "predicted_iou": item.score,
                    "proposal_scores": tuple(float(score) for score in item.proposals.scores),
                    "area": int(np.count_nonzero(item.mask)),
                }]},
            )
            for item in guided.raw_masks
        )
        kept_points = {item.point_index for item in guided.masks}
        segments = tuple(segment for segment in raw_segments
                         if segment.segment_id in kept_points)
        return SpectrumSegmentation(image, segments, self.device, self.mode,
                                    raw_segments=raw_segments, contrast=contrast,
                                    points=points, guided=guided)


def segment_spectrum(spectral, config, *, automatic_segmenter=None, predictor=None):
    """Compatibility helper for one image; use a session for multiple windows."""
    result = SAMSegmentationSession(
        config, automatic_segmenter=automatic_segmenter, predictor=predictor
    ).segment(spectral)
    return result.image, list(result.segments), result.points
