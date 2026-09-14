"""Train, calibrate, and run the automatic-SAM segment model branches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import numpy as np

from spectral_anomaly import (
    RobustFeaturePreprocessor, SegmentModels, chronological_window_split,
    SAMSegment, build_segment_table, stft_to_psd,
)

DEFAULT_CONFIG = Path(__file__).with_name("segment_pipeline_config.json")


def load(path: Path):
    config = json.loads(path.read_text())
    frame = pd.read_csv(config["input_csv"], parse_dates=["window_start", "window_end", "available_time"])
    frame["split"] = chronological_window_split(frame, **config["split"])
    return config, frame


def make_models(config):
    versions = config["versions"]
    isolation = config["isolation_forest"]
    clustering = config["hdbscan"]
    return SegmentModels(
        RobustFeaturePreprocessor(tuple(isolation["features"]), tuple(isolation["log1p_features"]),
                                  versions["isolation_preprocessing"]),
        RobustFeaturePreprocessor(tuple(clustering["features"]), tuple(clustering["log1p_features"]),
                                  versions["clustering_preprocessing"]),
        isolation_options=isolation["options"], hdbscan_options=clustering["options"],
        spot_options=config["spot"], model_version=versions["model"],
    )


def prepare_demo(config) -> None:
    """Create a sizeable deterministic feature corpus from varied STFT patterns.

    The masks emulate the output contract of automatic SAM so the downstream
    pipeline is runnable without weights; they are not a segmentation benchmark.
    """
    rng = np.random.default_rng(42)
    tables = []
    n_fft = config["stft_psd"]["n_fft"]
    frequency_count = n_fft // 2 + 1
    times = np.arange(32) * 2.0
    frequencies = np.linspace(0, config["stft_psd"]["sampling_frequency"] / 2,
                              frequency_count)
    for stream in ("demo-a", "demo-b"):
        for window_id in range(60):
            coefficients = rng.normal(0, .15, (frequency_count, 32)) + 1j * rng.normal(0, .15, (frequency_count, 32))
            coefficients[20, :] += 2.0                         # persistent tone
            coefficients[50:55, 14:18] += 2 + window_id / 80  # localized burst
            coefficients[10:100, 24:28] += rng.normal(0, .8, (90, 4))  # broadband
            tone_mask = np.zeros((frequency_count, 32), bool); tone_mask[20, :] = True
            burst_mask = np.zeros_like(tone_mask); burst_mask[50:55, 14:18] = True
            broadband_mask = np.zeros_like(tone_mask); broadband_mask[10:100, 24:28] = True
            masks = []
            for mask in (tone_mask, burst_mask, broadband_mask):
                annotation = {"segmentation": mask, "area": int(mask.sum()),
                              "bbox": [0, 0, 32, frequency_count], "predicted_iou": .9,
                              "stability_score": .9}
                masks.append(SAMSegment.from_sam_annotation(len(masks) + 1, annotation))
            psd = stft_to_psd(coefficients, **config["stft_psd"])
            start = pd.Timestamp("2026-01-01") + pd.Timedelta(minutes=window_id * 2)
            tables.append(build_segment_table(
                stream_id=stream, window_id=window_id, window_start=start,
                window_end=start + pd.Timedelta(seconds=64),
                available_time=start + pd.Timedelta(seconds=64), segments=masks,
                psd=psd, spectral_time=times, frequencies=frequencies,
                contrast_options=config["local_contrast"],
            ))
    pd.concat(tables, ignore_index=True).to_csv(config["input_csv"], index=False)
    print(f"demo corpus: {len(tables) * 3} segment rows; {config['input_csv']}")


def main(config_path: Path, stage: str) -> None:
    config = json.loads(config_path.read_text())
    if stage == "prepare-demo":
        prepare_demo(config)
        return
    config, frame = load(config_path)
    model_path = Path(config["model_file"])
    if stage == "train":
        model = make_models(config).fit(frame[frame.split == "train"])
        model.save(model_path)
        print(f"trained on {(frame.split == 'train').sum()} segment rows; {model_path}")
    elif stage == "calibrate":
        model = SegmentModels.load(model_path)
        model.calibrate_spot(frame[frame.split == "calibration"])
        model.save(model_path)
        print("SPOT states:", {stream: state.ready for stream, state in model.spot_states.items()})
    else:
        model = SegmentModels.load(model_path)
        result = model.predict(frame[frame.split == "evaluation"])
        result.to_csv(config["result_csv"], index=False)
        print(f"analysed {len(result)} segment rows; {config['result_csv']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare-demo", "train", "calibrate", "analyze"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    arguments = parser.parse_args()
    main(arguments.config, arguments.stage)
