"""Run optional SAM 2.1 segmentation from a JSON experiment configuration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from spectral_anomaly import (
    AnalysisPeriod,
    SAMStructureSegmenter,
    analyze_stft_periods,
    plot_sam_segmentation,
    prepare_analysis_windows,
    spectrogram_to_sam_image,
    time_frequency_to_pixel,
)

from basic_usage import make_example_data


DEFAULT_CONFIG = Path(__file__).with_name("sam_structure_config.json")


def load_config(path: Path) -> dict[str, Any]:
    """Load and minimally validate the experiment JSON configuration."""
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"configuration file does not exist: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON configuration: {error}") from error
    required_sections = {"analysis", "stft", "sam", "prompt", "output"}
    if not isinstance(config, dict) or not required_sections.issubset(config):
        raise ValueError(
            "configuration must contain analysis, stft, sam, prompt, and output"
        )
    for section in required_sections:
        if not isinstance(config[section], dict):
            raise ValueError(f"configuration section {section!r} must be an object")
    prompt = config["prompt"]
    if prompt.get("type") not in {"box", "point"}:
        raise ValueError("prompt.type must be 'box' or 'point'")
    expected_coordinates = 4 if prompt["type"] == "box" else 2
    if not isinstance(prompt.get("coordinates"), list) or len(
        prompt["coordinates"]
    ) != expected_coordinates:
        raise ValueError(
            f"prompt.coordinates must contain {expected_coordinates} values"
        )
    return config


def main(config_path: Path) -> None:
    """Compute the existing STFT once, prompt SAM, and write a Plotly view."""
    config = load_config(config_path)
    analysis_options = config["analysis"]
    metadata, windows = prepare_analysis_windows(
        make_example_data(),
        value_col="value",
        quality_col="quality",
        valid_quality_flags={"good"},
        sampling_period=analysis_options["sampling_period"],
        window_size=analysis_options["window_size"],
        overlap=analysis_options["overlap"],
        min_valid_ratio=analysis_options["min_valid_ratio"],
        min_valid_samples=analysis_options["min_valid_samples"],
        max_interpolation_gap=analysis_options["max_interpolation_gap"],
    )
    window_id = analysis_options["window_id"]
    if window_id not in windows:
        accepted = list(windows)
        if not accepted:
            raise RuntimeError("no quality-valid analysis window is available")
        if window_id is None:
            window_id = accepted[0]
        else:
            raise ValueError(f"window {window_id} is not quality-valid")
    window = windows[window_id]
    period = AnalysisPeriod(
        time=window.time,
        signal=window.signal,
        observed_mask=window.observed_mask,
        interpolated_mask=window.interpolated_mask,
        anomaly_mask=np.zeros(len(window.signal), dtype=bool),
        source_window_ids=(window_id,),
    )
    spectral = analyze_stft_periods(
        {window_id: period},
        sampling_frequency=analysis_options["sampling_frequency"],
        **config["stft"],
    )[window_id]
    image = spectrogram_to_sam_image(spectral.stft)
    sam_options = config["sam"]
    segmenter = SAMStructureSegmenter(
        sam_options["checkpoint"],
        model_config=sam_options["model_config"],
        device=sam_options["device"],
    )
    segmenter.set_image(image)

    prompt_options = config["prompt"]
    coordinates = prompt_options["coordinates"]
    if prompt_options["type"] == "box":
        result = segmenter.segment_box(coordinates)
        prompt_description = f"pixel box {coordinates}"
    else:
        pixel_point = time_frequency_to_pixel(
            *coordinates, spectral.spectral_time, spectral.frequencies
        )
        result = segmenter.segment_point(pixel_point)
        prompt_description = (
            f"physical point (t, f)={coordinates}, pixel (x, y)={pixel_point}"
        )

    print(metadata.loc[[window_id], ["start_time", "end_time"]].to_string())
    print(f"Prompt: {prompt_description}")
    print(f"SAM scores: {result.scores}; selected mask: {result.best_index}")
    figure = plot_sam_segmentation(
        spectral.stft, image, result, spectral.spectral_time, spectral.frequencies
    )
    output_options = config["output"]
    output = Path(output_options["html"])
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(output, include_plotlyjs=True, full_html=True)
    print(f"Diagnostic written to {output}")
    if output_options["show"]:
        figure.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"JSON configuration (default: {DEFAULT_CONFIG})",
    )
    main(parser.parse_args().config)
