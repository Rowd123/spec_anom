"""Generate prompt-free SAM 2.1 regions for one MSST window."""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from spectral_anomaly import (
    AnalysisPeriod,
    SAMAutomaticMaskSegmenter,
    SAMStructureSegmenter,
    analyze_msst_periods,
    analyze_stft_periods,
    plot_sam_automatic_masks, plot_sam_guided_comparison,
    prepare_analysis_windows,
    spectrogram_to_sam_image,
    temporal_energy_contrast, select_contrast_points, segment_contrast_points,
    validate_automatic_mask_options,
    validate_spectral_representation,
)
from basic_usage import make_example_data

DEFAULT_CONFIG = Path(__file__).with_name("sam_structure_config.json")


def load_config(path: Path) -> dict[str, Any]:
    """Load and validate the automatic-mask experiment configuration."""
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"configuration file does not exist: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON configuration: {error}") from error
    required = {"analysis", "msst", "sam_image", "sam", "automatic_mask_generation", "output"}
    if not isinstance(config, dict) or not required.issubset(config):
        raise ValueError(f"configuration must contain {', '.join(sorted(required))}")
    if any(not isinstance(config[name], dict) for name in required):
        raise ValueError("every configuration section must be an object")
    config["representation"] = validate_spectral_representation(config.get("representation"))
    sam_image = config["sam_image"]
    if sam_image.get("transform") not in {"log", "linear"}:
        raise ValueError("sam_image.transform must be either 'log' or 'linear'")
    percentiles = sam_image.get("percentiles")
    if (not isinstance(percentiles, list) or len(percentiles) != 2
            or any(isinstance(value, bool) or not isinstance(value, (int, float))
                   for value in percentiles)
            or not 0 <= percentiles[0] < percentiles[1] <= 100):
        raise ValueError("sam_image.percentiles must be two increasing numbers from 0 to 100")
    sam = config["sam"]
    if not all(isinstance(sam.get(name), str) and sam[name] for name in
               ("checkpoint", "model_config", "device")):
        raise ValueError("sam checkpoint, model_config, and device must be non-empty strings")
    config["automatic_mask_generation"] = validate_automatic_mask_options(
        config["automatic_mask_generation"]
    )
    guided = config.get("guided_segmentation", {})
    if guided.get("enabled", False) and config["representation"] != "stft":
        raise ValueError("guided_segmentation requires representation='stft'")
    output = config["output"]
    if not isinstance(output.get("html"), str) or not isinstance(output.get("show"), bool):
        raise ValueError("output.html must be a string and output.show a boolean")
    return config


def main(config_path: Path) -> None:
    """Compute one spectral representation, run SAM once, and write Plotly HTML."""
    config = load_config(config_path)
    analysis = config["analysis"]
    metadata, windows = prepare_analysis_windows(
        make_example_data(), value_col="value", quality_col="quality",
        valid_quality_flags={"good"}, sampling_period=analysis["sampling_period"],
        window_size=analysis["window_size"], overlap=analysis["overlap"],
        min_valid_ratio=analysis["min_valid_ratio"],
        min_valid_samples=analysis["min_valid_samples"],
        max_interpolation_gap=analysis["max_interpolation_gap"],
    )
    window_id = analysis["window_id"]
    if window_id not in windows:
        accepted = list(windows)
        if not accepted:
            raise RuntimeError("no quality-valid analysis window is available")
        if window_id is not None:
            raise ValueError(f"window {window_id} is not quality-valid")
        window_id = accepted[0]
    window = windows[window_id]
    period = AnalysisPeriod(window.time, window.signal, window.observed_mask,
                            window.interpolated_mask, np.zeros(len(window.signal), dtype=bool),
                            (window_id,))
    representation = config["representation"]
    analyzer = analyze_msst_periods if representation == "msst" else analyze_stft_periods
    spectral_options = dict(config["msst"])
    if representation == "stft":
        spectral_options.pop("iteration_count", None)
    spectral = analyzer(
        {window_id: period}, sampling_frequency=analysis["sampling_frequency"],
        **spectral_options,
    )[window_id]
    spectral_map = spectral.msst if representation == "msst" else spectral.stft
    image_options = config["sam_image"]
    image = spectrogram_to_sam_image(
        spectral_map, transform=image_options["transform"],
        percentiles=tuple(image_options["percentiles"]),
    )
    print(metadata.loc[[window_id], ["start_time", "end_time"]].to_string())
    print(f"Spectral representation: {representation.upper()}")
    print(f"Image transform: {image_options['transform']}")
    print(f"Percentile clipping: {image_options['percentiles']}")
    print(f"Spectral map shape: {spectral_map.shape}")
    print(f"SAM image shape: {image.shape}")
    guided_options = config.get("guided_segmentation", {})
    contrast, points = None, []
    if guided_options.get("enabled", False):
        contrast = temporal_energy_contrast(
            spectral.stft, spectral.spectral_time,
            exclusion_seconds=guided_options["exclusion_seconds"],
            neighborhood_seconds=guided_options["neighborhood_seconds"],
            epsilon=guided_options["epsilon"],
            min_valid_references=guided_options["min_valid_references"],
        )
        points = select_contrast_points(
            contrast, spectral.spectral_time, spectral.frequencies,
            threshold=guided_options["contrast_threshold"],
            min_time_spacing_seconds=guided_options["min_time_spacing_seconds"],
            min_frequency_spacing_hz=guided_options["min_frequency_spacing_hz"],
            max_points=guided_options["max_points"],
        )
        print(f"Guided points selected before SAM: {len(points)}")
        for index, point in enumerate(points, 1):
            print(f"  P{index}: t={point.time:.6g}s, f={point.frequency:.6g}Hz, "
                  f"contrast={point.contrast:.6g}, pixel={point.pixel}")
    sam = config["sam"]
    segmenter = SAMAutomaticMaskSegmenter(
        sam["checkpoint"], model_config=sam["model_config"], device=sam["device"],
        **config["automatic_mask_generation"],
    )
    if segmenter.device == "cpu":
        warnings.warn("SAM automatic mask generation on CPU can be very slow", RuntimeWarning)
    started = time.perf_counter()
    segments = segmenter.generate_masks(image)
    elapsed = time.perf_counter() - started

    print(f"Generated masks: {len(segments)}")
    print(f"Retained masks: {len(segments)} (no post-generation filtering)")
    print("Filtered after generation: 0")
    print(f"Inference time: {elapsed:.3f} s")
    output = Path(config["output"]["html"])
    output.parent.mkdir(parents=True, exist_ok=True)
    if guided_options.get("enabled", False):
        assert contrast is not None
        prompted = SAMStructureSegmenter(
            sam["checkpoint"], model_config=sam["model_config"], device=sam["device"]
        )
        guided = segment_contrast_points(
            prompted, image, points, iou_threshold=guided_options["mask_iou_threshold"]
        )
        print(f"Guided points: {len(points)}")
        print(f"Guided masks: {len(guided.raw_masks)} raw, {len(guided.masks)} deduplicated")
        for duplicate in guided.duplicates:
            print(f"  P{duplicate.removed_point_index} duplicate of "
                  f"P{duplicate.kept_point_index}: IoU={duplicate.iou:.4f}")
        figure = plot_sam_guided_comparison(
            spectral.stft, contrast, image, spectral.spectral_time,
            spectral.frequencies, points, guided, segments,
        )
    else:
        figure = plot_sam_automatic_masks(
            spectral_map, image, segments, spectral.spectral_time, spectral.frequencies,
            representation_name=representation.upper(),
            image_transform=image_options["transform"],
        )
    figure.write_html(output, include_plotlyjs=True, full_html=True)
    print(f"Output written to {output}")
    if config["output"]["show"]:
        figure.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help=f"JSON configuration (default: {DEFAULT_CONFIG})")
    main(parser.parse_args().config)
