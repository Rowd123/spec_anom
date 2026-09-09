"""Preprocess a complete monitoring series, run MSST, and save its plot.

Run from the repository root after installing the package::

    python examples/full_monitoring_msst_usage.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from spectral_anomaly import analyze_msst_monitoring_data

from basic_usage import make_example_data


def main(output: Path, show: bool = False) -> None:
    """Run the complete raw-data-to-MSST pipeline."""
    frame = make_example_data()

    analysis, figure = analyze_msst_monitoring_data(
        frame,
        value_col="value",
        quality_col="quality",
        valid_quality_flags={"good"},
        sampling_period="1s",
        sampling_frequency=1.0,
        max_interpolation_gap=2,
        iteration_count=3,
        center=True,
        window="hann",
        n_fft=256,
        window_length=128,
        hop_length=4,
        output_html=output,
    )

    print(f"Processed samples: {len(analysis.processed_signal)}")
    print(f"STFT shape: {analysis.stft.shape}")
    print(f"MSST shape: {analysis.msst.shape}")
    print(f"Interactive figure written to {output}")
    if show:
        figure.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("msst_full_monitoring_period.html"),
        help="destination of the self-contained Plotly HTML figure",
    )
    parser.add_argument("--show", action="store_true", help="also display the figure")
    arguments = parser.parse_args()
    main(arguments.output, arguments.show)
