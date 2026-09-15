"""Common interface for automatic and energy-contrast SAM segmentation."""
from __future__ import annotations
import numpy as np
from .masks import Segment
from .sam_segmentation import (SAMAutomaticMaskSegmenter, SAMStructureSegmenter,
    select_contrast_points, segment_contrast_points, spectrogram_to_sam_image,
    temporal_energy_contrast)

def segment_spectrum(spectral, config, *, automatic_segmenter=None, predictor=None):
    image_cfg=config["image"]
    image=spectrogram_to_sam_image(spectral.values, transform=image_cfg.get("transform","log"),
                                  percentiles=tuple(image_cfg.get("percentiles",[1,99])), epsilon=image_cfg.get("epsilon",1e-12))
    if config["segmentation_mode"] == "automatic":
        runner=automatic_segmenter or SAMAutomaticMaskSegmenter(
            checkpoint=config["model"].get("checkpoint"), model_config=config["model"].get("config"),
            device=config.get("device","auto"), **config.get("automatic",{}))
        raw=runner.generate_masks(image)
        segments=[Segment(s.segment_id,s.mask,metadata={"predicted_iou":s.predicted_iou,"stability_score":s.stability_score}) for s in raw]
        return image, segments, ()
    opts=config["energy_contrast"]
    contrast=temporal_energy_contrast(spectral.stft,spectral.times,
        exclusion_seconds=opts["exclusion_seconds"], neighborhood_seconds=opts["neighborhood_seconds"],
        epsilon=opts["epsilon"], min_valid_references=opts["min_valid_references"])
    points=select_contrast_points(contrast,spectral.times,spectral.frequencies,
        threshold=opts["contrast_threshold"],min_time_spacing_seconds=opts["min_time_spacing"],
        min_frequency_spacing_hz=opts["min_frequency_spacing"],max_points=opts["max_points"])
    runner=predictor if isinstance(predictor,SAMStructureSegmenter) else SAMStructureSegmenter(predictor=predictor)
    guided=segment_contrast_points(runner,image,points,iou_threshold=opts.get("mask_iou_threshold",.85))
    segments=[Segment(i,g.mask,metadata={"predicted_iou":g.score,"prompt_point_index":g.point_index}) for i,g in enumerate(guided.masks,1)]
    return image,segments,points
