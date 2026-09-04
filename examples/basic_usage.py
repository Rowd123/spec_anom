"""End-to-end example for the energy anomaly pre-filter.

Run from the repository root after installing the package::

    python examples/basic_usage.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from spectral_anomaly import detect_energy_anomalies, plot_suspicious_windows
from spectral_anomaly import detect_energy_anomalies, plot_window


def make_example_data(seed: int = 7) -> pd.DataFrame:
    """Build a realistic series containing bad data and one energy anomaly."""
    rng = np.random.default_rng(seed)
    sample_count = 30 * 256
    time = pd.date_range("2026-01-01", periods=sample_count, freq="1s", name="time")
    phase = np.arange(sample_count)

    # A stable oscillation, a small amount of noise, and a slowly drifting mean.
    signal = np.sin(2 * np.pi * phase / 32) + 0.08 * rng.standard_normal(sample_count)
    signal += 0.0002 * phase

    # The short high-amplitude episode is the anomaly that should be selected for
    # the future, more expensive MSST stage.
    anomaly_slice = slice(18 * 256, 19 * 256)
    signal[anomaly_slice] += 5 * np.sin(2 * np.pi * phase[anomaly_slice] / 16)

    quality = np.full(sample_count, "good", dtype=object)
    quality[3500] = "bad"
    data = pd.DataFrame({"value": signal, "quality": quality}, index=time)

    # Demonstrate all supported missing-data sources: NaNs, absent timestamps,
    # an invalid quality flag, and a duplicate timestamp. These gaps are short
    # enough to be interpolated, but still count as not genuinely observed.
    data.iloc[900:902, data.columns.get_loc("value")] = np.nan
    data = data.drop(data.index[2000])
    duplicate = data.iloc[[100]].assign(value=999.0)
    return pd.concat([data, duplicate])


def main(output: Path, show: bool = False) -> None:
    """Run the detector, display its result, and plot the first anomaly."""
    data = make_example_data()
    result, windows = detect_energy_anomalies(
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
        history_policy="exclude_suspicious",
    )

    columns = [
        "start_time",
        "end_time",
        "observed_samples",
        "interpolated_samples",
        "accepted",
        "energy",
        "baseline",
        "score",
        "suspicious",
    ]
    print(result.loc[:, columns].to_string())

    anomalies = result[result["suspicious"]]
    if anomalies.empty:
        print("No suspicious window found.")
        return

    first_anomaly_id = int(anomalies.index[0])
    window = windows[first_anomaly_id]
    print(
        f"\nFirst suspicious window: {first_anomaly_id}; "
        f"{window.observed_mask.sum()} genuinely observed samples and "
        f"{window.interpolated_mask.sum()} interpolated samples."
    )
    # One subplot is produced for every suspicious window. Use max_windows=N
    # here if a long recording creates too many subplots.
    fig, _ = plot_suspicious_windows(result, windows)
    fig, _ = plot_window(first_anomaly_id, result, windows)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Plot written to {output}")
    if show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("energy_anomaly_windows.png"),
        help="path of the figure containing all suspicious windows",
        default=Path("energy_anomaly_example.png"),
        help="path of the generated plot",
    )
    parser.add_argument("--show", action="store_true", help="also open the Matplotlib window")
    args = parser.parse_args()
    main(args.output, args.show)
