"""Analyze the morphology of every quality-valid monitoring window."""

from __future__ import annotations

import argparse
from pathlib import Path

from spectral_anomaly import analyze_candidate_structures, plot_candidate_structures

from basic_usage import make_example_data


def main(output: Path, window_id: int | None = None, show: bool = False) -> None:
    """Run the morphology prototype and plot one quality-valid window."""
    data = make_example_data()
    metadata, results, components, candidates = analyze_candidate_structures(
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
        representation="stft",
        window_ids=None if window_id is None else [window_id],
        transform_options={
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
        association_options={
            "max_fragment_time_gap_seconds": 8.0,
            "max_fragment_frequency_gap_hz": 0.01,
            "max_fragment_frequency_centroid_difference_hz": 0.02,
            "max_fragment_orientation_difference_radians": 0.26,
        },
    )
    accepted_ids = list(results)
    if not accepted_ids:
        raise RuntimeError("no quality-valid window is available")
    selected_id = accepted_ids[0] if window_id is None else window_id
    if selected_id not in results:
        raise ValueError(f"window {selected_id} is not quality-valid")

    print(metadata.loc[[selected_id], ["start_time", "end_time"]].to_string())
    print("\nExtracted component population:")
    print(components[components["window_id"] == selected_id].to_string(index=False))
    print("\nAssociated candidate structures:")
    print(candidates[candidates["window_id"] == selected_id].to_string(index=False))
    figure = plot_candidate_structures(results[selected_id])
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
        default=Path("stft_candidate_structures.html"),
        help="destination of the Plotly HTML diagnostic",
    )
    parser.add_argument("--show", action="store_true", help="also display the figure")
    arguments = parser.parse_args()
    main(arguments.output, arguments.window_id, arguments.show)
