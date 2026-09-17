"""Run the first real-data STFT/SAM/Isolation-Forest experiment."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from spectral_anomaly import (
    chronological_split,
    dataframe_to_segments,
    load_configs,
    score_atypicality_splits,
    train_atypicality,
)

LOGGER = logging.getLogger("train_atypicality")


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--time-col", required=True)
    parser.add_argument("--value-col", required=True)
    parser.add_argument("--source-id", help="stable source id used by configured exclusions")
    parser.add_argument("--quality-col")
    parser.add_argument("--valid-quality-flags", nargs="+")
    parser.add_argument("--window-overlap", type=int, default=0)
    parser.add_argument("--spectral-config", default="configs/spectral_analysis.json")
    parser.add_argument("--sam-config", default="configs/sam.json")
    parser.add_argument("--models-config", default="configs/models.json")
    parser.add_argument("--scores-output", required=True, type=Path)
    parser.add_argument(
        "--log-level", default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        help="niveau de verbosité (défaut: INFO)",
    )
    parser.add_argument("--log-file", type=Path, help="copie optionnelle des logs")
    parser.add_argument(
        "--log-every", type=int, default=100,
        help="fréquence des bilans de fenêtres (défaut: 100)",
    )
    return parser


def _configure_logging(level, log_file):
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf8"))
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )


def _log_score_statistics(result):
    for split in ("train", "calibration", "evaluation"):
        scores = result.loc[result["split"] == split, "atypicality_score"]
        LOGGER.info(
            "[final-scores] partition=%s count=%d min=%.6g median=%.6g mean=%.6g max=%.6g",
            split, scores.count(), scores.min(), scores.median(), scores.mean(), scores.max(),
        )


def main(argv=None):
    args = _parser().parse_args(argv)
    _configure_logging(args.log_level, args.log_file)
    total_started = perf_counter()
    stage_durations = {}
    if args.log_every < 1:
        raise ValueError("--log-every must be a positive integer")
    if args.quality_col and not args.valid_quality_flags:
        raise ValueError("--valid-quality-flags is required with --quality-col")
    if not args.quality_col and args.valid_quality_flags:
        raise ValueError("--quality-col is required with --valid-quality-flags")

    started = perf_counter()
    LOGGER.info("[input] reading %s", args.input)
    raw = pd.read_csv(args.input)
    if args.time_col not in raw:
        raise KeyError(f"unknown time column: {args.time_col!r}")
    timestamps = pd.to_datetime(raw.pop(args.time_col), errors="raise")
    raw.index = pd.DatetimeIndex(timestamps, name=args.time_col)
    stage_durations["input"] = perf_counter() - started
    period = "empty"
    if len(raw):
        period = f"{raw.index.min()} .. {raw.index.max()}"
    LOGGER.info(
        "[input] rows=%d period=%s selected_columns=%s quality_column=%s valid_quality_flags=%s",
        len(raw), period, [args.time_col, args.value_col], args.quality_col,
        args.valid_quality_flags,
    )

    started = perf_counter()
    spectral, sam, models = load_configs(
        args.spectral_config, args.sam_config, args.models_config
    )
    # Reflect command-line overrides in the effective configurations shown below.
    spectral["windowing"]["overlap"] = args.window_overlap
    spectral["quality"]["quality_column"] = args.quality_col
    spectral["quality"]["valid_flags"] = args.valid_quality_flags
    LOGGER.info("[configuration] spectral=%s", json.dumps(spectral, sort_keys=True))
    LOGGER.info("[configuration] sam=%s", json.dumps(sam, sort_keys=True))
    LOGGER.info(
        "[configuration] preprocessing=%s split=%s isolation_forest=%s",
        json.dumps(models["preprocessing"], sort_keys=True),
        json.dumps(models["split"], sort_keys=True),
        json.dumps(models["atypicality"], sort_keys=True),
    )
    LOGGER.info(
        "[configuration] SAM checkpoint=%s requested_device=%s",
        sam["model"]["checkpoint"], sam["device"],
    )
    stage_durations["configuration"] = perf_counter() - started
    started = perf_counter()
    segments, _, windows = dataframe_to_segments(
        raw,
        spectral,
        sam,
        value_col=args.value_col,
        quality_col=args.quality_col,
        valid_quality_flags=args.valid_quality_flags,
        window_overlap=args.window_overlap,
        return_window_metadata=True,
        log_every=args.log_every,
        source_id=args.source_id,
    )
    stage_durations["window_pipeline"] = perf_counter() - started
    accepted = int(windows["accepted"].sum())
    no_segment = accepted - (segments["window_id"].nunique() if not segments.empty else 0)
    LOGGER.info(
        "[segmentation-summary] windows_total=%d accepted=%d rejected=%d "
        "accepted_without_segment=%d segments=%d rejection_reasons=%s",
        len(windows), accepted, len(windows) - accepted, no_segment, len(segments),
        windows.loc[~windows["accepted"], "rejection_reason"].value_counts().to_dict(),
    )

    preview_splits = chronological_split(segments, models["split"])
    for name, split in zip(("train", "calibration", "evaluation"), preview_splits):
        LOGGER.info(
            "[split-preview] partition=%s windows=%d segments=%d",
            name, split["window_id"].nunique(), len(split),
        )
    started = perf_counter()
    pipeline, splits = train_atypicality(segments, models)
    stage_durations["training"] = perf_counter() - started
    started = perf_counter()
    result = score_atypicality_splits(pipeline, splits)
    stage_durations["scoring"] = perf_counter() - started
    _log_score_statistics(result)

    started = perf_counter()
    args.scores_output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.scores_output, index=False)
    stage_durations["saving"] = perf_counter() - started
    LOGGER.info("[output] scores_saved=%s", args.scores_output.resolve())
    if args.log_file:
        LOGGER.info("[output] log_saved=%s", args.log_file.resolve())
    LOGGER.info(
        "[final] duration=%.3fs stage_durations=%s windows_total=%d accepted=%d "
        "rejected=%d without_segment=%d segments=%d",
        perf_counter() - total_started,
        {name: round(value, 3) for name, value in stage_durations.items()},
        len(windows), accepted, len(windows) - accepted, no_segment, len(segments),
    )
    return result


def _run():
    try:
        return main()
    except Exception:
        LOGGER.exception("[fatal] unexpected error; execution context is in preceding logs")
        raise


if __name__ == "__main__":
    _run()
