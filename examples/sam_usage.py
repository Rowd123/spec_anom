"""Run real SAM 2 on a rich time-frequency demonstration signal."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from spectral_anomaly import (
    SAMSegmentationSession,
    analyze_spectrum,
    load_config,
    plot_sam_diagnostic,
    postprocess_masks,
)


def demonstration_signal(size: int, sampling_frequency: float):
    """Persistent tone + local burst + rising chirp with a crossing."""
    time = np.arange(size) / sampling_frequency
    duration = max(time[-1], 1 / sampling_frequency)
    persistent = 0.55 * np.sin(2 * np.pi * 0.08 * time)
    envelope = np.exp(-0.5 * ((time - 0.48 * duration) / (0.055 * duration)) ** 2)
    burst = 1.5 * envelope * np.sin(2 * np.pi * 0.22 * time)
    start_frequency, end_frequency = 0.025, 0.18
    chirp_rate = (end_frequency - start_frequency) / duration
    chirp = 0.5 * np.sin(2 * np.pi * (
        start_frequency * time + 0.5 * chirp_rate * time ** 2
    ))
    return time, persistent + burst + chirp


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--spectral-config", default="configs/spectral_analysis.json")
    parser.add_argument("--sam-config", default="configs/sam.json")
    parser.add_argument("--output", default="sam_diagnostic.html")
    args = parser.parse_args(argv)
    spectral_config = load_config(args.spectral_config, "spectral")
    sam_config = load_config(args.sam_config, "sam")

    time, signal = demonstration_signal(
        spectral_config["windowing"]["size"], spectral_config["sampling_frequency"]
    )
    spectral = analyze_spectrum(signal, spectral_config)
    # Construction happens once here. The same session can segment further windows.
    session = SAMSegmentationSession(sam_config)
    segmentation = session.segment(spectral)
    final_segments = postprocess_masks(
        segmentation.segments, **sam_config["mask_postprocessing"]
    )
    figure = plot_sam_diagnostic(
        signal, time, spectral.values, segmentation.image,
        segmentation.raw_segments, final_segments, spectral.times, spectral.frequencies,
        device=session.device, mode=session.mode, contrast=segmentation.contrast,
        points=segmentation.points, guided=segmentation.guided,
    )
    figure.write_html(args.output)
    print(f"SAM mode={session.mode}; device={session.device}; "
          f"raw={len(segmentation.raw_segments)}; retained={len(segmentation.segments)}; "
          f"final={len(final_segments)}; output={args.output}")
    return segmentation, final_segments


if __name__ == "__main__":
    main()
