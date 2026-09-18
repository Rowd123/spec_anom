"""Command-line entry points for independently executable processing stages."""
from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path
import pandas as pd

from .config import load_config
from .workflows import (apply_spot, extract_segments, fit_hdbscan, predict_hdbscan,
                        score_isolation_forest, train_isolation_forest)


def _json(path):
    with Path(path).open(encoding="utf8") as stream: return json.load(stream)


def parser():
    root = argparse.ArgumentParser(prog="spectral-anomaly")
    root.add_argument("--log-level", default="INFO")
    commands = root.add_subparsers(dest="command", required=True)
    extract = commands.add_parser("extract")
    extract.add_argument("input"); extract.add_argument("output"); extract.add_argument("--spectral-config", required=True); extract.add_argument("--sam-config", required=True)
    extract.add_argument("--source-id", required=True); extract.add_argument("--channel-id", required=True); extract.add_argument("--value-col", default="value"); extract.add_argument("--index-col", default="time"); extract.add_argument("--save-arrays"); extract.add_argument("--resume", action="store_true")
    train = commands.add_parser("train-iforest"); train.add_argument("segments"); train.add_argument("model"); train.add_argument("--config", required=True)
    score = commands.add_parser("score-iforest"); score.add_argument("segments"); score.add_argument("model"); score.add_argument("output"); score.add_argument("--batch-size", type=int, default=65536); score.add_argument("--resume", action="store_true")
    fit = commands.add_parser("fit-hdbscan"); fit.add_argument("segments"); fit.add_argument("model"); fit.add_argument("output"); fit.add_argument("--config", required=True)
    predict = commands.add_parser("predict-hdbscan"); predict.add_argument("segments"); predict.add_argument("model"); predict.add_argument("output")
    spot = commands.add_parser("spot"); spot.add_argument("calibration"); spot.add_argument("scores"); spot.add_argument("output"); spot.add_argument("state"); spot.add_argument("--config", required=True); spot.add_argument("--resume", action="store_true")
    return root


def main(argv=None):
    args = parser().parse_args(argv); logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(asctime)s %(levelname)s %(name)s %(message)s")
    if args.command == "extract":
        frame = pd.read_csv(args.input); frame.index = pd.to_datetime(frame.pop(args.index_col), utc=True)
        extract_segments(frame, args.output, load_config(args.spectral_config, "spectral"), load_config(args.sam_config, "sam"), source_id=args.source_id, channel_id=args.channel_id, value_col=args.value_col, save_arrays=args.save_arrays, resume=args.resume)
        return None  # Console entry points must not pass a DataFrame to sys.exit.
    if args.command == "train-iforest": return train_isolation_forest(args.segments, args.model, load_config(args.config, "models"))
    if args.command == "score-iforest": return score_isolation_forest(args.segments, args.model, args.output, batch_size=args.batch_size, resume=args.resume)
    if args.command == "fit-hdbscan": return fit_hdbscan(args.segments, args.model, args.output, load_config(args.config, "models"))
    if args.command == "predict-hdbscan": return predict_hdbscan(args.segments, args.model, args.output)
    config = _json(args.config); return apply_spot(args.calibration, args.scores, args.output, args.state, config, resume=args.resume)
