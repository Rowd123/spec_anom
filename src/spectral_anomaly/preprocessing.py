"""Chronological splitting and independently configurable feature preprocessing."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def chronological_split(frame, config):
    """Split chronologically while keeping every observation window intact.

    Fractions are applied to unique windows rather than segment rows.  ``window_id``
    is the preferred identity; the configured time column is used as a fallback.
    """
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
    return tuple(
        ordered.loc[row_keys.isin(groups)].copy() for groups in window_groups
    )


@dataclass
class FeaturePreprocessor:
    features: tuple[str, ...]
    log1p: tuple[str, ...] = ()
    method: str = "robust"
    center_: np.ndarray | None = None
    scale_: np.ndarray | None = None

    def fit(self, frame):
        x = self._raw(frame)
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
        return x

    def transform(self, frame):
        if self.center_ is None:
            raise RuntimeError("preprocessor is not fitted")
        return (self._raw(frame) - self.center_) / self.scale_

    def fit_transform(self, frame):
        return self.fit(frame).transform(frame)
