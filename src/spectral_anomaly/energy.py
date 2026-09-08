"""Windowed, locally normalised energy anomaly detection."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Hashable, Iterable, Literal, Mapping

import numpy as np
import pandas as pd
from scipy.signal import get_window


@dataclass(frozen=True)
class WindowData:
    """Arrays associated with one accepted analysis window."""

    time: pd.Index
    signal: np.ndarray
    centered: np.ndarray
    windowed: np.ndarray
    observed_mask: np.ndarray
    interpolated_mask: np.ndarray


def _validate_parameters(
    window_size: int,
    overlap: int,
    min_valid_ratio: float,
    min_valid_samples: int | None,
    max_interpolation_gap: int,
    history_size: int,
    min_history: int,
    k: float,
) -> None:
    if window_size < 2:
        raise ValueError("window_size must be at least 2")
    if not 0 <= overlap < window_size:
        raise ValueError("overlap must satisfy 0 <= overlap < window_size")
    if not 0 <= min_valid_ratio <= 1:
        raise ValueError("min_valid_ratio must be in [0, 1]")
    if min_valid_samples is not None and not 0 <= min_valid_samples <= window_size:
        raise ValueError("min_valid_samples must be in [0, window_size]")
    if max_interpolation_gap < 0:
        raise ValueError("max_interpolation_gap must be non-negative")
    if history_size < 1 or not 1 <= min_history <= history_size:
        raise ValueError("require 1 <= min_history <= history_size")
    if not np.isfinite(k) or k < 0:
        raise ValueError("k must be a finite non-negative number")


def _prepare_dataframe(
    data: pd.DataFrame,
    value_col: Hashable,
    quality_col: Hashable | None,
    valid_quality_flags: Iterable[object] | None,
) -> pd.DataFrame:
    """Sort input, keep the first duplicate, and mask invalid observations."""
    if value_col not in data.columns:
        raise KeyError(f"unknown value column: {value_col!r}")
    if quality_col is not None and quality_col not in data.columns:
        raise KeyError(f"unknown quality column: {quality_col!r}")
    if not isinstance(data.index, (pd.DatetimeIndex, pd.TimedeltaIndex)):
        if not pd.api.types.is_numeric_dtype(data.index.dtype):
            raise TypeError("index must be datetime-like or numeric")
        if not np.all(np.isfinite(data.index.to_numpy(dtype=float))):
            raise ValueError("numeric index must contain only finite values")

    # Duplicate policy is deliberately isolated here so it can later be replaced
    # by aggregation without changing regularisation.
    prepared = data.loc[~data.index.duplicated(keep="first")].sort_index().copy()
    values = pd.to_numeric(prepared[value_col], errors="coerce").astype(float)
    valid = values.notna()
    if quality_col is not None:
        if valid_quality_flags is None:
            raise ValueError("valid_quality_flags is required with quality_col")
        valid &= prepared[quality_col].isin(set(valid_quality_flags))
    prepared["__signal"] = values.where(valid)
    prepared["__observed"] = valid
    return prepared[["__signal", "__observed"]]


def _estimate_sampling_period(index: pd.Index) -> pd.Timedelta | float:
    """Estimate nominal period as the median positive adjacent timestamp gap.

    This robust estimate assumes that more than half of adjacent pairs represent
    one nominal interval.  Supply ``sampling_period`` when that is not true.
    """
    if len(index) < 2:
        raise ValueError("at least two distinct timestamps are required")
    if isinstance(index, (pd.DatetimeIndex, pd.TimedeltaIndex)):
        raw = np.diff(index.asi8)
        positive = raw[raw > 0]
        if not len(positive):
            raise ValueError("cannot estimate a positive sampling period")
        return pd.to_timedelta(int(np.median(positive)), unit="ns")
    raw = np.diff(index.to_numpy(dtype=float))
    positive = raw[np.isfinite(raw) & (raw > 0)]
    if not len(positive):
        raise ValueError("cannot estimate a positive sampling period")
    return float(np.median(positive))


def _regularize_signal(
    prepared: pd.DataFrame, sampling_period: pd.Timedelta | str | Real | None
) -> tuple[pd.DataFrame, pd.Timedelta | float]:
    """Reindex observations onto an exact regular grid (no nearest matching)."""
    datetime_like = isinstance(prepared.index, (pd.DatetimeIndex, pd.TimedeltaIndex))
    period = _estimate_sampling_period(prepared.index) if sampling_period is None else sampling_period
    if datetime_like:
        period = pd.to_timedelta(period)
        if period <= pd.Timedelta(0):
            raise ValueError("sampling_period must be positive")
        if isinstance(prepared.index, pd.DatetimeIndex):
            grid = pd.date_range(prepared.index[0], prepared.index[-1], freq=period)
        else:
            grid = pd.timedelta_range(prepared.index[0], prepared.index[-1], freq=period)
    else:
        if not isinstance(period, Real) or not np.isfinite(period) or period <= 0:
            raise ValueError("numeric sampling_period must be a positive finite number")
        period = float(period)
        count = int(np.floor((float(prepared.index[-1]) - float(prepared.index[0])) / period)) + 1
        grid_values = float(prepared.index[0]) + np.arange(count) * period
        grid = pd.Index(grid_values, name=prepared.index.name)

    off_grid = prepared.index.difference(grid)
    if len(off_grid):
        preview = list(off_grid[:3])
        raise ValueError(
            "timestamps are not exactly aligned to the inferred/provided grid; "
            f"examples: {preview}. Resample upstream or provide a suitable sampling_period."
        )
    regular = prepared.reindex(grid)
    regular["__observed"] = regular["__observed"].fillna(False).astype(bool)
    return regular, period


def _missing_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.r_[False, mask, False].astype(np.int8)
    changes = np.diff(padded)
    return list(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)))


def _interpolate_small_gaps(values: np.ndarray, maximum: int) -> tuple[np.ndarray, np.ndarray]:
    """Linearly fill only complete, internally bounded runs no longer than maximum."""
    filled = values.copy()
    interpolated = np.zeros(len(values), dtype=bool)
    if maximum == 0:
        return filled, interpolated
    for start, stop in _missing_runs(np.isnan(values)):
        if stop - start <= maximum and start > 0 and stop < len(values):
            filled[start:stop] = np.interp(
                np.arange(start, stop), [start - 1, stop], [filled[start - 1], filled[stop]]
            )
            interpolated[start:stop] = True
    return filled, interpolated


def _longest_missing_run(observed: np.ndarray) -> int:
    runs = _missing_runs(~observed)
    return max((stop - start for start, stop in runs), default=0)


def _compute_window_energy(signal: np.ndarray, taper: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    centered = signal - np.mean(signal)
    windowed = centered * taper
    return float(np.dot(windowed, windowed) / np.dot(taper, taper)), centered, windowed


def _compute_robust_score(energy: float, history: list[float]) -> tuple[float, float, float, float]:
    baseline = float(np.median(history))
    mad = float(np.median(np.abs(np.asarray(history) - baseline)))
    # The relative floor preserves a zero score for exact equality, while any
    # representable departure from a constant baseline remains detectable.
    floor = np.finfo(float).eps * max(1.0, abs(baseline))
    scale = max(1.4826 * mad, floor)
    return baseline, mad, scale, float((energy - baseline) / scale)


def detect_energy_anomalies(
    data: pd.DataFrame,
    *,
    value_col: Hashable,
    quality_col: Hashable | None = None,
    valid_quality_flags: Iterable[object] | None = None,
    sampling_period: pd.Timedelta | str | Real | None = None,
    window_size: int = 256,
    window_name: str | tuple = "hann",
    overlap: int = 0,
    min_valid_ratio: float = 0.9,
    min_valid_samples: int | None = None,
    max_interpolation_gap: int = 3,
    history_size: int = 20,
    min_history: int = 5,
    k: float = 4.0,
    history_policy: Literal["exclude_suspicious", "all"] = "exclude_suspicious",
) -> tuple[pd.DataFrame, dict[int, WindowData]]:
    """Detect unusual local energy on an exact, regular time grid.

    Rejected windows do not enter the baseline.  By default suspicious windows
    do not enter it either, preventing a detected event from raising subsequent
    thresholds.  ``windows`` contains arrays only for accepted windows and is
    keyed by the integer ``window_id`` used as the result index.
    """
    _validate_parameters(window_size, overlap, min_valid_ratio, min_valid_samples,
                         max_interpolation_gap, history_size, min_history, k)
    if history_policy not in {"exclude_suspicious", "all"}:
        raise ValueError("history_policy must be 'exclude_suspicious' or 'all'")
    prepared = _prepare_dataframe(data, value_col, quality_col, valid_quality_flags)
    regular, _ = _regularize_signal(prepared, sampling_period)
    raw = regular["__signal"].to_numpy(dtype=float)
    observed = regular["__observed"].to_numpy(dtype=bool)
    filled, interpolated = _interpolate_small_gaps(raw, max_interpolation_gap)
    taper = np.asarray(get_window(window_name, window_size, fftbins=False), dtype=float)
    if not np.any(taper) or np.dot(taper, taper) == 0:
        raise ValueError("Fourier window must have non-zero energy")

    rows: list[dict[str, object]] = []
    windows: dict[int, WindowData] = {}
    history: list[float] = []
    step = window_size - overlap
    required = min_valid_samples or 0
    for window_id, start in enumerate(range(0, len(regular) - window_size + 1, step)):
        stop = start + window_size
        obs = observed[start:stop]
        interp = interpolated[start:stop]
        signal = filled[start:stop]
        valid_count = int(obs.sum())
        ratio = valid_count / window_size
        longest = _longest_missing_run(obs)
        reasons: list[str] = []
        if ratio < min_valid_ratio:
            reasons.append("insufficient_valid_ratio")
        if valid_count < required:
            reasons.append("insufficient_valid_samples")
        if longest > max_interpolation_gap:
            reasons.append("gap_too_long")
        if np.isnan(signal).any():
            reasons.append("unfilled_missing_values")
        accepted = not reasons
        energy = baseline = mad = scale = score = np.nan
        suspicious = False
        if accepted:
            energy, centered, windowed = _compute_window_energy(signal, taper)
            if len(history) >= min_history:
                baseline, mad, scale, score = _compute_robust_score(energy, history[-history_size:])
                suspicious = bool(abs(score) > k)
            windows[window_id] = WindowData(
                time=regular.index[start:stop].copy(), signal=signal.copy(),
                centered=centered, windowed=windowed, observed_mask=obs.copy(),
                interpolated_mask=interp.copy(),
            )
            if history_policy == "all" or not suspicious:
                history.append(energy)
        rows.append({
            "start_time": regular.index[start], "end_time": regular.index[stop - 1],
            "grid_start": start, "grid_stop": stop, "expected_samples": window_size,
            "observed_samples": valid_count, "interpolated_samples": int(interp.sum()),
            "valid_ratio": ratio, "longest_missing_run": longest, "accepted": accepted,
            "rejection_reason": ";".join(reasons) if reasons else None, "energy": energy,
            "baseline": baseline, "mad": mad, "robust_scale": scale, "score": score,
            "suspicious": suspicious,
        })
    result = pd.DataFrame(rows)
    result.index = pd.RangeIndex(len(result), name="window_id")
    return result, windows


def plot_window(
    window_id: int,
    result: pd.DataFrame,
    windows: Mapping[int, WindowData],
    *,
    show_centered: bool = True,
):
    """Plot observed/interpolated points for one accepted window."""
    import matplotlib.pyplot as plt

    if window_id not in result.index:
        raise KeyError(f"unknown window_id: {window_id}")
    if window_id not in windows:
        raise ValueError("window was rejected and has no complete signal to plot")
    item = windows[window_id]
    fig, ax = plt.subplots()
    ax.plot(item.time, item.signal, color="0.65", label="regularized signal")
    ax.scatter(item.time[item.observed_mask], item.signal[item.observed_mask], s=18,
               color="C0", label="observed", zorder=3)
    ax.scatter(item.time[item.interpolated_mask], item.signal[item.interpolated_mask], s=28,
               marker="x", color="C3", label="interpolated", zorder=4)
    if show_centered:
        ax.plot(item.time, item.centered, color="C2", alpha=0.8, label="centered")
    state = "suspicious" if bool(result.loc[window_id, "suspicious"]) else "normal"
    ax.set(title=f"Window {window_id} ({state})", xlabel="time", ylabel="signal")
    ax.legend()
    fig.autofmt_xdate()
    return fig, ax


def plot_suspicious_windows(
    result: pd.DataFrame,
    windows: Mapping[int, WindowData],
    *,
    show_centered: bool = True,
    max_windows: int | None = None,
):
    """Plot every suspicious window in a single figure.

    Parameters
    ----------
    result, windows:
        Objects returned by :func:`detect_energy_anomalies`.
    show_centered:
        Also draw each locally centred signal.
    max_windows:
        Optionally limit the plot to the first N suspicious windows. This is
        useful when a long recording contains many detections.

    Returns
    -------
    tuple
        The Matplotlib figure and a one-dimensional array of axes.

    Raises
    ------
    ValueError
        If there are no suspicious windows or ``max_windows`` is invalid.
    """
    import matplotlib.pyplot as plt

    required_columns = {"suspicious", "score"}
    missing_columns = required_columns.difference(result.columns)
    if missing_columns:
        raise ValueError(f"result is missing columns: {sorted(missing_columns)}")
    if max_windows is not None and max_windows < 1:
        raise ValueError("max_windows must be positive or None")

    suspicious_ids = list(result.index[result["suspicious"].fillna(False).astype(bool)])
    if max_windows is not None:
        suspicious_ids = suspicious_ids[:max_windows]
    if not suspicious_ids:
        raise ValueError("no suspicious window to plot")

    absent = [window_id for window_id in suspicious_ids if window_id not in windows]
    if absent:
        raise ValueError(f"missing WindowData for suspicious windows: {absent}")

    fig, axes_grid = plt.subplots(
        len(suspicious_ids),
        1,
        figsize=(12, max(3.0, 2.8 * len(suspicious_ids))),
        squeeze=False,
        sharex=False,
    )
    axes = axes_grid[:, 0]
    for ax, window_id in zip(axes, suspicious_ids):
        item = windows[window_id]
        ax.plot(item.time, item.signal, color="0.65", label="regularized signal")
        ax.scatter(
            item.time[item.observed_mask],
            item.signal[item.observed_mask],
            s=14,
            color="C0",
            label="observed",
            zorder=3,
        )
        ax.scatter(
            item.time[item.interpolated_mask],
            item.signal[item.interpolated_mask],
            s=26,
            marker="x",
            color="C3",
            label="interpolated",
            zorder=4,
        )
        if show_centered:
            ax.plot(item.time, item.centered, color="C2", alpha=0.8, label="centered")
        score = result.loc[window_id, "score"]
        ax.set_title(f"Suspicious window {window_id} — score={score:.3g}")
        ax.set_xlabel("time")
        ax.set_ylabel("signal")
        ax.legend(loc="best")

    fig.suptitle(f"Energy anomalies ({len(suspicious_ids)} windows)")
    fig.tight_layout()
    fig.autofmt_xdate()
    return fig, axes
