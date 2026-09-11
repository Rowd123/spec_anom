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
    analyze_msst_periods,
    plot_sam_automatic_masks,
    prepare_analysis_windows,
    spectrogram_to_sam_image,
    validate_automatic_mask_options,
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
    required = {"analysis", "msst", "sam", "automatic_mask_generation", "output"}
    if not isinstance(config, dict) or not required.issubset(config):
        raise ValueError(f"configuration must contain {', '.join(sorted(required))}")
    if any(not isinstance(config[name], dict) for name in required):
        raise ValueError("every configuration section must be an object")
    sam = config["sam"]
    if not all(isinstance(sam.get(name), str) and sam[name] for name in
               ("checkpoint", "model_config", "device")):
        raise ValueError("sam checkpoint, model_config, and device must be non-empty strings")
    config["automatic_mask_generation"] = validate_automatic_mask_options(
        config["automatic_mask_generation"]
    )
    output = config["output"]
    if not isinstance(output.get("html"), str) or not isinstance(output.get("show"), bool):
        raise ValueError("output.html must be a string and output.show a boolean")
    return config


def main(config_path: Path) -> None:
    """Compute one MSST, run automatic SAM once, and write Plotly HTML."""
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
    spectral = analyze_msst_periods(
        {window_id: period}, sampling_frequency=analysis["sampling_frequency"],
        **config["msst"],
    )[window_id]
    image = spectrogram_to_sam_image(spectral.msst)
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

    print(metadata.loc[[window_id], ["start_time", "end_time"]].to_string())
    print(f"MSST shape: {spectral.msst.shape}")
    print(f"SAM image shape: {image.shape}")
    print(f"Generated masks: {len(segments)}")
    print(f"Retained masks: {len(segments)} (no post-generation filtering)")
    print("Filtered after generation: 0")
    print(f"Inference time: {elapsed:.3f} s")
    output = Path(config["output"]["html"])
    output.parent.mkdir(parents=True, exist_ok=True)
    figure = plot_sam_automatic_masks(
        spectral.msst, image, segments, spectral.spectral_time, spectral.frequencies,
        representation_name="MSST",
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
