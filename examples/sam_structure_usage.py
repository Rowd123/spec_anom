"""Run optional SAM 2.1 segmentation on one quality-valid STFT window."""

from __future__ import annotations

import argparse
from pathlib import Path

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


def main(arguments: argparse.Namespace) -> None:
    """Compute the existing STFT once, prompt SAM, and write a Plotly view."""
    metadata, windows = prepare_analysis_windows(
        make_example_data(),
        value_col="value",
        quality_col="quality",
        valid_quality_flags={"good"},
        sampling_period="1s",
        window_size=256,
        overlap=128,
        min_valid_ratio=0.98,
        min_valid_samples=250,
        max_interpolation_gap=2,
    )
    window_id = arguments.window_id
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
        {window_id: period}, sampling_frequency=1.0, window="hann",
        n_fft=256, window_length=128, hop_length=4,
    )[window_id]
    image = spectrogram_to_sam_image(spectral.stft)
    segmenter = SAMStructureSegmenter(
        arguments.checkpoint, model_config=arguments.model_config,
        device=arguments.device,
    )
    segmenter.set_image(image)

    if arguments.box is not None:
        result = segmenter.segment_box(arguments.box)
        prompt = f"pixel box {arguments.box}"
    else:
        physical_point = arguments.point or (
            float(spectral.spectral_time[len(spectral.spectral_time) // 2]),
            float(spectral.frequencies[len(spectral.frequencies) // 2]),
        )
        pixel_point = time_frequency_to_pixel(
            *physical_point, spectral.spectral_time, spectral.frequencies
        )
        result = segmenter.segment_point(pixel_point)
        prompt = f"physical point (t, f)={physical_point}, pixel (x, y)={pixel_point}"

    print(metadata.loc[[window_id], ["start_time", "end_time"]].to_string())
    print(f"Prompt: {prompt}")
    print(f"SAM scores: {result.scores}; selected mask: {result.best_index}")
    figure = plot_sam_segmentation(
        spectral.stft, image, result, spectral.spectral_time, spectral.frequencies
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(arguments.output, include_plotlyjs=True, full_html=True)
    print(f"Diagnostic written to {arguments.output}")
    if arguments.show:
        figure.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--model-config", default="configs/sam2.1/sam2.1_hiera_s.yaml",
        help="SAM 2 config name (default: sam2.1_hiera_small)",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--window-id", type=int)
    prompts = parser.add_mutually_exclusive_group()
    prompts.add_argument(
        "--box", nargs=4, type=float, metavar=("X_MIN", "Y_MIN", "X_MAX", "Y_MAX"),
        help="bounding box in STFT image pixels",
    )
    prompts.add_argument(
        "--point", nargs=2, type=float, metavar=("TIME", "FREQUENCY"),
        help="positive point in physical seconds and hertz",
    )
    parser.add_argument("--output", type=Path, default=Path("sam_structure.html"))
    parser.add_argument("--show", action="store_true")
    main(parser.parse_args())
