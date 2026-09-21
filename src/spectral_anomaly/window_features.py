"""Independent persisted windows and explicit, causal window descriptors.

No detector is fitted here. Signal scaling constants must be supplied by the user.
"""
from __future__ import annotations

from collections import deque
from copy import deepcopy
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal, stats

from .artifacts import build_manifest, fingerprint, read_table, write_table
from .energy import prepare_analysis_windows
from .exclusions import excluded_observations, validate_exclusions
from .preprocessing import chronological_split

LOGGER = logging.getLogger(__name__)
STATISTICS = {"minimum", "maximum", "mean", "variance", "median", "iqr", "skewness", "kurtosis"}
DIFFERENCES = {"mean_absolute", "maximum_absolute", "variance"}


def _keys(value, expected, label):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(f"{label} must contain exactly {sorted(expected)}")


def _integer(value, minimum, label):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")


def validate_window_preprocessing(config):
    c = deepcopy(config)
    _keys(c, {"sampling_frequency", "sampling_period", "windowing", "quality", "exclusions", "normalization"}, "window preprocessing")
    fs = c["sampling_frequency"]
    if not np.isfinite(fs) or fs <= 0 or not np.isclose(pd.to_timedelta(c["sampling_period"]).total_seconds(), 1 / fs):
        raise ValueError("sampling_frequency must be positive and reciprocal of sampling_period")
    w = c["windowing"]
    _keys(w, {"size", "overlap"}, "windowing")
    _integer(w["size"], 2, "size"); _integer(w["overlap"], 0, "overlap")
    if w["overlap"] >= w["size"]:
        raise ValueError("overlap must be smaller than size")
    q = c["quality"]
    _keys(q, {"quality_column", "valid_flags", "min_valid_fraction", "max_interpolation_gap"}, "quality")
    if not 0 <= q["min_valid_fraction"] <= 1:
        raise ValueError("min_valid_fraction must be in [0, 1]")
    _integer(q["max_interpolation_gap"], 0, "max_interpolation_gap")
    c["exclusions"] = validate_exclusions(c["exclusions"])
    if not isinstance(c["normalization"], dict) or not c["normalization"]:
        raise ValueError("provide normalization constants keyed by channel_id")
    for name, constants in c["normalization"].items():
        _keys(constants, {"mu", "sigma"}, f"normalization.{name}")
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and np.isfinite(v) for v in constants.values()) or constants["sigma"] <= 0:
            raise ValueError(f"provide finite mu and positive sigma for {name}; no automatic estimation")
    return c


def validate_window_features(config, fs, size):
    c = deepcopy(config)
    _keys(c, {"statistics", "variance_ddof", "moment_bias", "undefined_policy", "variance_epsilon", "differences", "spectral", "temporal", "history", "log_every"}, "window features")
    for key, allowed in (("statistics", STATISTICS), ("differences", DIFFERENCES)):
        if not isinstance(c[key], list) or len(set(c[key])) != len(c[key]) or set(c[key]) - allowed:
            raise ValueError(f"invalid {key}; choices: {sorted(allowed)}")
    _integer(c["variance_ddof"], 0, "variance_ddof")
    if c["variance_ddof"] >= size - 1:
        raise ValueError("variance_ddof must be < window size - 1")
    if not isinstance(c["moment_bias"], bool) or c["undefined_policy"] not in {"zero", "error"}:
        raise ValueError("moment_bias must be boolean; undefined_policy must be zero or error")
    if not np.isfinite(c["variance_epsilon"]) or c["variance_epsilon"] < 0:
        raise ValueError("variance_epsilon must be finite and non-negative")
    _integer(c["log_every"], 1, "log_every")
    s = c["spectral"]
    _keys(s, {"enabled", "bands_hz", "window", "detrend", "nfft", "top_changes"}, "spectral")
    if not isinstance(s["enabled"], bool) or s["detrend"] not in (False, "constant", "linear"):
        raise ValueError("invalid spectral enabled/detrend")
    signal.get_window(s["window"], size)
    if s["nfft"] is not None:
        _integer(s["nfft"], size, "nfft")
    _integer(s["top_changes"], 0, "top_changes")
    if not s["enabled"] and s["top_changes"]:
        raise ValueError("top_changes requires spectral.enabled")
    previous = 0.0
    for band in s["bands_hz"]:
        if len(band) != 2 or not np.isfinite(band).all():
            raise ValueError("bands must contain finite [low, high] pairs")
        low, high = band
        if not previous <= low < high <= fs / 2:
            raise ValueError("bands must be ordered, nonoverlapping and within Nyquist")
        if s["enabled"] and high - low < fs / size - 1e-12:
            raise ValueError("band narrower than native FFT spacing; increase window size")
        previous = high
    if s["enabled"] and not s["bands_hz"] or s["top_changes"] > len(s["bands_hz"]):
        raise ValueError("spectral bands/top_changes are inconsistent")
    t = c["temporal"]
    _keys(t, {"lags_seconds", "changes"}, "temporal")
    if not isinstance(t["changes"], bool) or not isinstance(t["lags_seconds"], list):
        raise ValueError("invalid temporal configuration")
    lags = []
    for lag in t["lags_seconds"]:
        if not np.isfinite(lag) or not np.isclose(lag * fs, round(lag * fs)) or not 1 <= round(lag * fs) < size:
            raise ValueError("each lag must be an exact positive sample delay smaller than the window")
        lags.append(round(lag * fs))
    if len(set(lags)) != len(lags):
        raise ValueError("duplicate temporal lags")
    h = c["history"]
    _keys(h, {"size", "min_windows", "warmup", "reset_after_gap_seconds"}, "history")
    _integer(h["size"], 1, "history.size"); _integer(h["min_windows"], 1, "history.min_windows")
    if h["min_windows"] > h["size"] or h["warmup"] not in {"drop", "zero", "error"}:
        raise ValueError("invalid history size/min_windows/warmup")
    if h["reset_after_gap_seconds"] is not None and (not np.isfinite(h["reset_after_gap_seconds"]) or h["reset_after_gap_seconds"] <= 0):
        raise ValueError("reset_after_gap_seconds must be positive or null")
    if not c["statistics"] and not c["differences"] and not s["enabled"] and not lags:
        raise ValueError("enable at least one feature")
    return c


def prepare_windows(frame, output_path, config, *, source_id, channel_id, value_col="value"):
    """Reuse existing quality/window preparation, then apply fixed affine scaling."""
    c = validate_window_preprocessing(config)
    if channel_id not in c["normalization"]:
        raise ValueError(f"missing explicit normalization for channel_id={channel_id!r}")
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.hasnans or frame.empty:
        raise ValueError("provide nonempty input with a valid DatetimeIndex")
    q, w = c["quality"], c["windowing"]
    excluded = excluded_observations(frame, c["exclusions"], source_id=str(source_id))
    metadata, windows = prepare_analysis_windows(
        frame, value_col=value_col, sampling_period=c["sampling_period"],
        window_size=w["size"], overlap=w["overlap"], quality_col=q["quality_column"],
        valid_quality_flags=q["valid_flags"], min_valid_ratio=q["min_valid_fraction"],
        max_interpolation_gap=q["max_interpolation_gap"], excluded_times=frame.index[excluded])
    constants = c["normalization"][channel_id]
    rows = []
    for number, item in windows.items():
        values = (item.signal - constants["mu"]) / constants["sigma"]
        if not np.isfinite(values).all():
            raise ValueError(f"window {number} contains non-finite normalized values")
        start, end = metadata.loc[number, ["start_time", "end_time"]]
        rows.append(dict(source_id=str(source_id), channel_id=str(channel_id),
                         window_id=f"{source_id}:{channel_id}:{pd.Timestamp(start).isoformat()}",
                         window_number=number, window_start=start, window_end=end,
                         signal=values.tolist(), observed_mask=item.observed_mask.tolist(),
                         interpolated_mask=item.interpolated_mask.tolist()))
    if not rows:
        raise ValueError("no accepted windows; inspect input quality, exclusions and window size")
    table = pd.DataFrame(rows)
    manifest = build_manifest("prepared_windows", config=c,
                              provenance={"source_id": str(source_id), "channel_id": str(channel_id), "value_col": value_col})
    write_table(table, output_path, manifest)
    audit = metadata.reset_index()
    write_table(audit, str(output_path) + ".quality.parquet", build_manifest("window_quality", config=c))
    LOGGER.info("prepared windows=%d rejected=%d", len(table), len(metadata) - len(table))
    return table


def window_descriptors(values, config, fs):
    """Descriptors of one normalized window. PSD is a modified periodogram."""
    x = np.asarray(values, float)
    c = config
    if x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError("window must be a finite 1D array")
    centered = x - x.mean()
    variance = np.mean(centered ** 2)
    degenerate = variance <= c["variance_epsilon"]
    needs_moments = bool(set(c["statistics"]) & {"skewness", "kurtosis"}) or bool(c["temporal"]["lags_seconds"])
    if degenerate and needs_moments and c["undefined_policy"] == "error":
        raise ValueError("undefined standardized moment/autocorrelation in a constant window")
    functions = {
        "minimum": lambda: x.min(), "maximum": lambda: x.max(), "mean": lambda: x.mean(),
        "variance": lambda: x.var(ddof=c["variance_ddof"]), "median": lambda: np.median(x),
        "iqr": lambda: np.quantile(x, .75) - np.quantile(x, .25),
        "skewness": lambda: 0.0 if degenerate else stats.skew(x, bias=c["moment_bias"]),
        "kurtosis": lambda: 0.0 if degenerate else stats.kurtosis(x, fisher=True, bias=c["moment_bias"]),
    }
    result = {"wf_" + name: float(functions[name]()) for name in c["statistics"]}
    if needs_moments:
        result["wf_constant"] = float(degenerate)
    d = np.diff(x)
    functions = {"mean_absolute": lambda: np.mean(np.abs(d)), "maximum_absolute": lambda: np.max(np.abs(d)), "variance": lambda: d.var(ddof=c["variance_ddof"])}
    result.update({"wf_diff_" + name: float(functions[name]()) for name in c["differences"]})
    if c["spectral"]["enabled"]:
        s = c["spectral"]
        f, p = signal.periodogram(x, fs=fs, window=s["window"], detrend=s["detrend"], nfft=s["nfft"], scaling="density")
        df = f[1] - f[0]
        for i, (low, high) in enumerate(s["bands_hz"]):
            # Half-open bands, except Nyquist. Rectangular PSD quadrature.
            mask = (f >= low) & ((f < high) | ((high == fs / 2) & (f == high)))
            if not mask.any():
                raise ValueError(f"band {i} has no Fourier bins")
            result[f"wf_band_{i}_power"] = float(p[mask].sum() * df)
    for i, lag in enumerate(c["temporal"]["lags_seconds"]):
        k = round(lag * fs)
        result[f"wf_acf_{i}"] = 0.0 if degenerate else float(centered[:-k] @ centered[k:] / (centered @ centered))
    if not np.isfinite(list(result.values())).all():
        raise ValueError("nonfinite features; inspect data and moment settings")
    return result


def extract_window_features(windows_path, output_path, config):
    """Read only prepared windows; persist all enabled features before selection."""
    windows, source = read_table(windows_path, expected_type="prepared_windows")
    fs = source.config["sampling_frequency"]
    size = source.config["windowing"]["size"]
    c = validate_window_features(config, fs, size)
    h = c["history"]
    hop = (size - source.config["windowing"]["overlap"]) / fs
    max_gap = h["reset_after_gap_seconds"] or hop * 1.5
    use_history = c["spectral"]["top_changes"] > 0 or (c["temporal"]["changes"] and bool(c["temporal"]["lags_seconds"]))
    rows, dropped = [], 0
    if windows.duplicated(["source_id", "channel_id", "window_start"]).any():
        raise ValueError("duplicate windows")
    for _, group in windows.groupby(["source_id", "channel_id"], sort=False):
        history = deque(maxlen=h["size"])
        previous = None
        for row in group.sort_values("window_start", kind="stable").itertuples(index=False):
            if previous is not None and (row.window_start - previous).total_seconds() > max_gap:
                history.clear()
            previous = row.window_start
            if len(row.signal) != size:
                raise ValueError("window length differs from preprocessing manifest")
            current = window_descriptors(row.signal, c, fs)
            ready = len(history) >= h["min_windows"]
            features = current.copy()
            if use_history:
                features["wf_history_ready"] = float(ready)
                if not ready and h["warmup"] == "error":
                    raise ValueError("insufficient preceding windows")
                if c["temporal"]["changes"]:
                    for i in range(len(c["temporal"]["lags_seconds"])):
                        name = f"wf_acf_{i}"
                        features[name + "_change"] = current[name] - np.median([p[name] for p in history]) if ready else 0.0
                k = c["spectral"]["top_changes"]
                if k:
                    delta = np.array([current[f"wf_band_{i}_power"] - np.median([p[f"wf_band_{i}_power"] for p in history]) if ready else 0.0 for i in range(len(c["spectral"]["bands_hz"]))])
                    for rank, i in enumerate(np.argsort(-np.abs(delta), kind="stable")[:k]):
                        features[f"wf_spectral_change_{rank}"] = float(delta[i])
                        features[f"wf_spectral_change_{rank}_frequency"] = float(np.mean(c["spectral"]["bands_hz"][i])) if ready else 0.0
            history.append(current)
            if use_history and not ready and h["warmup"] == "drop":
                dropped += 1
                continue
            rows.append(dict(source_id=row.source_id, channel_id=row.channel_id,
                             window_id=row.window_id, segment_id=0, observation_kind="window",
                             window_start=row.window_start, window_end=row.window_end,
                             time_start=row.window_start, time_end=row.window_end,
                             frequency_min=0.0, frequency_max=fs / 2, **features))
            if len(rows) % c["log_every"] == 0:
                LOGGER.info("extracted window features=%d", len(rows))
    if not rows:
        raise ValueError("no feature rows; check history warmup and number of accepted windows")
    frame = pd.DataFrame(rows).sort_values(["window_start", "source_id", "channel_id"], kind="stable").reset_index(drop=True)
    features = tuple(name for name in frame if name.startswith("wf_"))
    effective = {"preprocessing": source.config, "features": c}
    frame["window_feature_signature"] = fingerprint(effective)
    manifest = build_manifest("window_features", config=effective, features=features,
                              provenance={"windows": str(windows_path), "warmup_dropped": dropped,
                                          "observation_unit": "whole_window", "segment_id": "0 denotes whole window, not a SAM mask"})
    write_table(frame, output_path, manifest)
    LOGGER.info("extracted rows=%d features=%d warmup_dropped=%d", len(frame), len(features), dropped)
    return frame


def split_window_features(features_path, output_dir, config):
    """Persist purged chronological splits for separate fitting/calibration/scoring."""
    frame, manifest = read_table(features_path, expected_type="window_features")
    config = {**config, "purge_overlap": True}
    parts = chronological_split(frame, config)
    output_dir = Path(output_dir)
    for name, part in zip(("train", "calibration", "evaluation"), parts):
        if part.empty:
            raise ValueError(f"empty {name} split after purging; increase data or change fractions")
    for name, part in zip(("train", "calibration", "evaluation"), parts):
        write_table(part, output_dir / f"{name}.parquet", build_manifest(
            "window_features", config=manifest.config, features=manifest.feature_columns,
            units=manifest.feature_units, provenance={**manifest.provenance, "split": name, "split_config": config,
                                                    "input": str(features_path)}))
    return parts
