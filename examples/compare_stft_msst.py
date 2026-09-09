"""Compare STFT and MSST morphology on the same quality-valid window."""

from __future__ import annotations

import argparse
from pathlib import Path

from spectral_anomaly import compare_structural_windows, plot_stft_msst_comparison

from basic_usage import make_example_data


def main(window_id: int, output: Path | None = None, show: bool = False) -> None:
    """Print comparison metrics and write the nine-panel Plotly diagnostic."""
    data = make_example_data()
    _, comparisons = compare_structural_windows(
        data,
        value_col="value",
        quality_col="quality",
        valid_quality_flags={"good"},
        sampling_period="1s",
        sampling_frequency=1.0,
        window_size=256,
        overlap=128,
        min_valid_ratio=0.98,
        min_valid_samples=250,
        max_interpolation_gap=2,
        window_ids=[window_id],
        msst_options={
            "iteration_count": 3,
            "window": "hann",
            "n_fft": 256,
            "window_length": 128,
            "hop_length": 4,
        },
        structure_options={
            "normalization_neighborhood": (9, 9),
            "tensor_sigma": (1.5, 1.5),
            "significance_threshold": 3.0,
            "coherence_threshold": 0.5,
            "minimum_component_area": 4,
            "small_component_area": 16,
        },
    )
    if window_id not in comparisons:
        raise ValueError(f"window {window_id} is not quality-valid")

    comparison = comparisons[window_id]
    print(comparison.metrics.to_string())
    destination = output or Path(f"stft_msst_comparison_window_{window_id}.html")
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure = plot_stft_msst_comparison(comparison)
    figure.write_html(destination, include_plotlyjs=True, full_html=True)
    print(f"Comparison written to {destination}")
    if show:
        figure.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window-id", type=int, default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--show", action="store_true")
    arguments = parser.parse_args()
    main(arguments.window_id, arguments.output, arguments.show)
