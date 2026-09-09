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


def _compute_window_energy(
    signal: np.ndarray,
    taper: np.ndarray,
    *,
    exclude_dc_bin: bool,
) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    """Compute taper-normalised energy from the unchanged signal.

    The one-sided real FFT is weighted according to Parseval's identity. Thus,
    ``energy_including_dc`` is equivalent to the tapered time-domain energy, while
    ``dc_bin_energy`` isolates exactly the first frequency-bin contribution. The
    centered array is retained for inspection only; it is not used by the FFT.
    """
    centered = signal - np.mean(signal)
    windowed = signal * taper
    spectrum_power = np.abs(np.fft.rfft(windowed)) ** 2
    weights = np.full(len(spectrum_power), 2.0)
    weights[0] = 1.0
    if len(windowed) % 2 == 0:
        weights[-1] = 1.0
    normalizer = len(windowed) * np.dot(taper, taper)
    energy_including_dc = float(np.dot(weights, spectrum_power) / normalizer)
    dc_bin_energy = float(spectrum_power[0] / normalizer)
    energy = max(0.0, energy_including_dc - dc_bin_energy) if exclude_dc_bin else energy_including_dc
    return energy, energy_including_dc, dc_bin_energy, centered, windowed


def _compute_robust_score(energy: float, history: list[float]) -> tuple[float, float, float, float]:
    baseline = float(np.median(history))
    mad = float(np.median(np.abs(np.asarray(history) - baseline)))
    # The relative floor preserves a zero score for exact equality, while any
    # representable departure from a constant baseline remains detectable.
    floor = np.finfo(float).eps * max(1.0, abs(baseline))
    scale = max(1.4826 * mad, floor)
    return baseline, mad, scale, float((energy - baseline) / scale)


def prepare_analysis_windows(
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
) -> tuple[pd.DataFrame, dict[int, WindowData]]:
    """Prepare every quality-valid window without making an anomaly decision.

    This is the quality-control boundary shared by downstream detectors. In
    particular, acceptance depends only on observations and interpolation—not
    on energy—so every returned window is eligible for MSST analysis.
    """
    _validate_parameters(
        window_size,
        overlap,
        min_valid_ratio,
        min_valid_samples,
        max_interpolation_gap,
        history_size=1,
        min_history=1,
        k=0.0,
    )
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
        if accepted:
            centered = signal - np.mean(signal)
            windows[window_id] = WindowData(
                time=regular.index[start:stop].copy(),
                signal=signal.copy(),
                centered=centered,
                windowed=signal * taper,
                observed_mask=obs.copy(),
                interpolated_mask=interp.copy(),
            )
        rows.append(
            {
                "start_time": regular.index[start],
                "end_time": regular.index[stop - 1],
                "grid_start": start,
                "grid_stop": stop,
                "expected_samples": window_size,
                "observed_samples": valid_count,
                "interpolated_samples": int(interp.sum()),
                "valid_ratio": ratio,
                "longest_missing_run": longest,
                "accepted": accepted,
                "rejection_reason": ";".join(reasons) if reasons else None,
            }
        )
    result = pd.DataFrame(rows)
    result.index = pd.RangeIndex(len(result), name="window_id")
    return result, windows


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
    exclude_dc_bin: bool = True,
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
    if not isinstance(exclude_dc_bin, bool):
        raise TypeError("exclude_dc_bin must be a bool")
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
        energy = energy_including_dc = dc_bin_energy = np.nan
        baseline = mad = scale = score = np.nan
        suspicious = False
        if accepted:
            energy, energy_including_dc, dc_bin_energy, centered, windowed = (
                _compute_window_energy(
                    signal,
                    taper,
                    exclude_dc_bin=exclude_dc_bin,
                )
            )
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
            "rejection_reason": ";".join(reasons) if reasons else None,
            "energy": energy, "energy_including_dc": energy_including_dc,
            "dc_bin_energy": dc_bin_energy,
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
    """Return an interactive Plotly figure for one accepted window."""
    import plotly.graph_objects as go

    if window_id not in result.index:
        raise KeyError(f"unknown window_id: {window_id}")
    if window_id not in windows:
        raise ValueError("window was rejected and has no complete signal to plot")
    item = windows[window_id]
    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=item.time, y=item.signal, mode="lines", name="regularized signal",
        line={"color": "#9ca3af"},
    ))
    figure.add_trace(go.Scatter(
        x=item.time[item.observed_mask], y=item.signal[item.observed_mask],
        mode="markers", name="observed", marker={"size": 6, "color": "#2563eb"},
    ))
    figure.add_trace(go.Scatter(
        x=item.time[item.interpolated_mask], y=item.signal[item.interpolated_mask],
        mode="markers", name="interpolated",
        marker={"size": 9, "symbol": "x", "color": "#dc2626"},
    ))
    if show_centered:
        figure.add_trace(go.Scatter(
            x=item.time, y=item.centered, mode="lines", name="centered",
            line={"color": "#16a34a"}, opacity=0.8,
        ))
    state = "suspicious" if bool(result.loc[window_id, "suspicious"]) else "normal"
    figure.update_layout(
        title=f"Window {window_id} ({state})",
        xaxis_title="time", yaxis_title="signal", template="plotly_white",
        hovermode="x unified",
    )
    return figure


def plot_suspicious_windows(
    result: pd.DataFrame,
    windows: Mapping[int, WindowData],
    *,
    show_centered: bool = True,
    max_windows: int | None = None,
):
    """Return an interactive figure containing every suspicious window."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

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

    titles = [
        f"Suspicious window {window_id} — score={result.loc[window_id, 'score']:.3g}"
        for window_id in suspicious_ids
    ]
    figure = make_subplots(
        rows=len(suspicious_ids), cols=1, shared_xaxes=False,
        vertical_spacing=min(0.08, 0.3 / len(suspicious_ids)), subplot_titles=titles,
    )
    for row, window_id in enumerate(suspicious_ids, start=1):
        item = windows[window_id]
        show_legend = row == 1
        traces = [
            go.Scatter(
                x=item.time, y=item.signal, mode="lines", name="regularized signal",
                line={"color": "#9ca3af"}, legendgroup="regularized",
                showlegend=show_legend,
            ),
            go.Scatter(
                x=item.time[item.observed_mask], y=item.signal[item.observed_mask],
                mode="markers", name="observed",
                marker={"size": 5, "color": "#2563eb"},
                legendgroup="observed", showlegend=show_legend,
            ),
            go.Scatter(
                x=item.time[item.interpolated_mask], y=item.signal[item.interpolated_mask],
                mode="markers", name="interpolated",
                marker={"size": 8, "symbol": "x", "color": "#dc2626"},
                legendgroup="interpolated", showlegend=show_legend,
            ),
        ]
        if show_centered:
            traces.append(go.Scatter(
                x=item.time, y=item.centered, mode="lines", name="centered",
                line={"color": "#16a34a"}, opacity=0.8,
                legendgroup="centered", showlegend=show_legend,
            ))
        for trace in traces:
            figure.add_trace(trace, row=row, col=1)
        figure.update_xaxes(title_text="time", row=row, col=1)
        figure.update_yaxes(title_text="signal", row=row, col=1)

    figure.update_layout(
        title=f"Energy anomalies ({len(suspicious_ids)} windows)",
        height=max(420, 320 * len(suspicious_ids)), template="plotly_white",
        hovermode="x unified",
    )
    return figure
