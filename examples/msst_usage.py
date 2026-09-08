"""Run energy detection, build fixed periods, and plot their STFT/MSST."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from spectral_anomaly import (
    analyze_msst_periods,
    detect_energy_anomalies,
    plot_msst_periods,
    prepare_analysis_periods,
)

from basic_usage import make_example_data


def main(output: Path, show: bool = False) -> None:
    """Execute both pipeline stages and save all studied periods."""
    data = make_example_data()
    energy_result, energy_windows = detect_energy_anomalies(
        data,
        value_col="value",
        quality_col="quality",
        valid_quality_flags={"good"},
        sampling_period="1s",
        window_size=256,
        window_name="hann",
        overlap=128,
        min_valid_ratio=0.98,
        min_valid_samples=250,
        max_interpolation_gap=2,
        history_size=12,
        min_history=6,
        k=5.0,
    )

    # Touching/overlapping anomalous windows are joined. Each group is extended
    # forward using subsequent accepted windows to exactly 1024 samples.
    period_metadata, periods = prepare_analysis_periods(
        energy_result,
        energy_windows,
        period_size=1024,
    )
    print(period_metadata.to_string())
    if not periods:
        print("No complete fixed period is available for MSST.")
        return

    analyses = analyze_msst_periods(
        periods,
        sampling_frequency=1.0,
        iteration_count=3,
        center=True,
        window="hann",
        n_fft=256,
        window_length=128,
        hop_length=4,
    )
    figure, _ = plot_msst_periods(analyses)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150, bbox_inches="tight")
    print(f"STFT/MSST figure written to {output}")
    if show:
        plt.show()
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("msst_analysis_periods.png"),
        help="path of the figure containing all analysed periods",
    )
    parser.add_argument("--show", action="store_true", help="also display the figure")
    arguments = parser.parse_args()
    main(arguments.output, arguments.show)
