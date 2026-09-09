"""Analyze the morphology of every quality-valid monitoring window."""

from __future__ import annotations

import argparse
from pathlib import Path

from spectral_anomaly import analyze_structural_windows, plot_structural_window

from basic_usage import make_example_data


def main(output: Path, window_id: int | None = None, show: bool = False) -> None:
    """Run the morphology prototype and plot one quality-valid window."""
    data = make_example_data()
    metadata, analyses = analyze_structural_windows(
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
        },
    )
    accepted_ids = list(analyses)
    if not accepted_ids:
        raise RuntimeError("no quality-valid window is available")
    selected_id = accepted_ids[0] if window_id is None else window_id
    if selected_id not in analyses:
        raise ValueError(f"window {selected_id} is not quality-valid")

    feature_columns = list(analyses[selected_id].features)
    print(metadata.loc[metadata["accepted"], feature_columns].to_string())
    figure = plot_structural_window(analyses[selected_id])
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(output, include_plotlyjs=True, full_html=True)
    print(f"Six-panel diagnostic for window {selected_id} written to {output}")
    if show:
        figure.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window-id", type=int, help="quality-valid window to plot")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("msst_structure_window.html"),
        help="destination of the Plotly HTML diagnostic",
    )
    parser.add_argument("--show", action="store_true", help="also display the figure")
    arguments = parser.parse_args()
    main(arguments.output, arguments.window_id, arguments.show)
