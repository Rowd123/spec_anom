"""Run the first real-data STFT/SAM/Isolation-Forest experiment."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from spectral_anomaly import (
    chronological_split,
    dataframe_to_segments,
    load_configs,
    score_atypicality_splits,
    train_atypicality,
)


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--time-col", required=True)
    parser.add_argument("--value-col", required=True)
    parser.add_argument("--quality-col")
    parser.add_argument("--valid-quality-flags", nargs="+")
    parser.add_argument("--window-overlap", type=int, default=0)
    parser.add_argument("--spectral-config", default="configs/spectral_analysis.json")
    parser.add_argument("--sam-config", default="configs/sam.json")
    parser.add_argument("--models-config", default="configs/models.json")
    parser.add_argument("--scores-output", required=True, type=Path)
    return parser


def _print_score_statistics(result):
    for split in ("train", "calibration", "evaluation"):
        scores = result.loc[result["split"] == split, "atypicality_score"]
        print(
            f"scores {split}: count={scores.count()} min={scores.min():.6g} "
            f"median={scores.median():.6g} mean={scores.mean():.6g} "
            f"max={scores.max():.6g}"
        )


def main(argv=None):
    args = _parser().parse_args(argv)
    if args.quality_col and not args.valid_quality_flags:
        raise ValueError("--valid-quality-flags is required with --quality-col")
    if not args.quality_col and args.valid_quality_flags:
        raise ValueError("--quality-col is required with --valid-quality-flags")

    raw = pd.read_csv(args.input)
    if args.time_col not in raw:
        raise KeyError(f"unknown time column: {args.time_col!r}")
    timestamps = pd.to_datetime(raw.pop(args.time_col), errors="raise")
    raw.index = pd.DatetimeIndex(timestamps, name=args.time_col)

    spectral, sam, models = load_configs(
        args.spectral_config, args.sam_config, args.models_config
    )
    segments, _, windows = dataframe_to_segments(
        raw,
        spectral,
        sam,
        value_col=args.value_col,
        quality_col=args.quality_col,
        valid_quality_flags=args.valid_quality_flags,
        window_overlap=args.window_overlap,
        return_window_metadata=True,
    )
    accepted = int(windows["accepted"].sum())
    print(f"fenêtres produites: {len(windows)}")
    print(f"fenêtres acceptées: {accepted}")
    print(f"fenêtres rejetées: {len(windows) - accepted}")
    print(f"segments totaux: {len(segments)}")

    preview_splits = chronological_split(segments, models["split"])
    for name, split in zip(("train", "calibration", "evaluation"), preview_splits):
        print(f"segments {name}: {len(split)}")
    pipeline, splits = train_atypicality(segments, models)
    result = score_atypicality_splits(pipeline, splits)
    _print_score_statistics(result)

    args.scores_output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.scores_output, index=False)
    print(f"scores sauvegardés: {args.scores_output}")
    return result


if __name__ == "__main__":
    main()
