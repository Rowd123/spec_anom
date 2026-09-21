"""Chronological splitting and independently configurable feature preprocessing."""
from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatchcase

import numpy as np
import pandas as pd


def chronological_split(frame, config):
    """Split chronologically while keeping every observation window intact.

    Fractions are applied to unique windows rather than segment rows.  ``window_id``
    is the preferred identity; the configured time column is used as a fallback.
    """
    fractions = [config[k] for k in ("train_fraction", "calibration_fraction", "evaluation_fraction")]
    if any(not np.isfinite(v) or v < 0 for v in fractions) or not np.isclose(sum(fractions), 1):
        raise ValueError("split fractions must be nonnegative and sum to one")
    time_column = config.get("time_column", "window_start")
    if time_column not in frame:
        raise KeyError(f"unknown split time column: {time_column!r}")
    group_columns = [time_column]
    if "window_id" in frame and "window_id" != time_column:
        group_columns.insert(0, "window_id")
    windows = (
        frame[group_columns]
        .drop_duplicates(subset=group_columns, keep="first")
        .sort_values(time_column, kind="stable")
    )
    n_windows = len(windows)
    train_stop = int(n_windows * config["train_fraction"])
    calibration_stop = train_stop + int(n_windows * config["calibration_fraction"])
    def keys(part):
        return set(map(tuple, part[group_columns].itertuples(index=False, name=None)))

    window_groups = (
        keys(windows.iloc[:train_stop]),
        keys(windows.iloc[train_stop:calibration_stop]),
        keys(windows.iloc[calibration_stop:]),
    )
    ordered = frame.sort_values(time_column, kind="stable")
    row_keys = pd.MultiIndex.from_frame(ordered[group_columns])
    parts = [ordered.loc[row_keys.isin(groups)].copy() for groups in window_groups]
    if config.get("purge_overlap", False):
        if "window_end" not in frame:
            raise ValueError("purge_overlap requires window_end")
        # Remove earlier windows whose inclusive end overlaps any later partition.
        for i in range(2):
            later = [p[time_column].min() for p in parts[i + 1:] if not p.empty]
            if later:
                parts[i] = parts[i].loc[parts[i]["window_end"] < min(later)].copy()
    return tuple(parts)


@dataclass
class FeaturePreprocessor:
    features: tuple[str, ...]
    log1p: tuple[str, ...] = ()
    method: str = "robust"
    center_: np.ndarray | None = None
    scale_: np.ndarray | None = None

    def fit(self, frame):
        if self.method not in {"none", "robust", "standard"}:
            raise ValueError("scaling must be none, robust, or standard")
        x = self._raw(frame)
        if not len(x):
            raise ValueError("cannot fit feature preprocessing on an empty table")
        if self.method == "none":
            self.center_ = np.zeros(x.shape[1])
            self.scale_ = np.ones(x.shape[1])
            return self
        self.center_ = np.median(x, axis=0) if self.method == "robust" else x.mean(0)
        self.scale_ = (
            1.4826 * np.median(np.abs(x - self.center_), axis=0)
            if self.method == "robust"
            else x.std(0)
        )
        self.scale_[self.scale_ == 0] = 1
        return self

    def _raw(self, frame):
        x = frame.loc[:, self.features].to_numpy(float).copy()
        for name in self.log1p:
            x[:, self.features.index(name)] = np.log1p(x[:, self.features.index(name)])
        if not np.isfinite(x).all():
            raise ValueError("selected/transformed features contain NaN or infinity")
        return x

    def transform(self, frame):
        if self.center_ is None:
            raise RuntimeError("preprocessor is not fitted")
        return (self._raw(frame) - self.center_) / self.scale_

    def fit_transform(self, frame):
        return self.fit(frame).transform(frame)


def selected_features(config, purpose, available):
    """Independent model feature allowlist plus optional exclusion globs."""
    key = f"{purpose}_features"
    names = config[key]
    if not isinstance(names, (list, tuple)) or not names or len(set(names)) != len(names):
        raise ValueError(f"{key} must be a nonempty list of distinct feature names")
    missing = set(names) - set(available)
    if missing:
        raise ValueError(f"unknown selected features: {sorted(missing)}")
    patterns = config.get(f"{purpose}_exclude", [])
    if not isinstance(patterns, list) or any(not isinstance(p, str) for p in patterns):
        raise ValueError("feature exclusions must be a list of glob patterns")
    for pattern in patterns:
        if not any(fnmatchcase(n, pattern) for n in available):
            raise ValueError(f"feature exclusion matches nothing: {pattern}")
    result = tuple(n for n in names if not any(fnmatchcase(n, p) for p in patterns))
    if not result:
        raise ValueError("feature selection excluded every feature")
    return result
