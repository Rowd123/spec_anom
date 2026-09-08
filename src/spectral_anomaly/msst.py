"""Fixed-period STFT multisynchrosqueezing for suspicious energy windows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd
from numba import njit, prange
from ssqueezepy import phase_stft, stft

from .energy import WindowData


@dataclass(frozen=True)
class AnalysisPeriod:
    """A fixed-length, regularly sampled period selected for MSST."""

    time: pd.Index
    signal: np.ndarray
    observed_mask: np.ndarray
    interpolated_mask: np.ndarray
    anomaly_mask: np.ndarray
    source_window_ids: tuple[int, ...]


@dataclass(frozen=True)
class MSSTResult:
    """Time-frequency arrays computed for one analysis period."""

    period: AnalysisPeriod
    processed_signal: np.ndarray
    msst: np.ndarray
    stft: np.ndarray
    frequencies: np.ndarray
    spectral_time: np.ndarray


@njit(parallel=True, cache=True)
def _frequency_map_to_bins(
    frequencies: np.ndarray,
    first_frequency: float,
    frequency_step: float,
    frequency_count: int,
) -> np.ndarray:
    """Convert an instantaneous-frequency map to reassignment bin indices."""
    frequency_bins, time_bins = frequencies.shape
    bins = np.empty((frequency_bins, time_bins), dtype=np.int32)
    for time_index in prange(time_bins):
        for frequency_index in range(frequency_bins):
            value = frequencies[frequency_index, time_index]
            if not np.isfinite(value):
                bins[frequency_index, time_index] = -1
                continue
            target = int(round((value - first_frequency) / frequency_step))
            bins[frequency_index, time_index] = min(max(target, 0), frequency_count - 1)
    return bins


@njit(parallel=True, cache=True)
def _compose_frequency_map(base_map: np.ndarray, iteration_count: int) -> np.ndarray:
    """Compose the first-order reassignment map for higher MSST orders."""
    frequency_bins, time_bins = base_map.shape
    current = base_map.copy()
    for _ in range(1, iteration_count):
        new_map = np.empty_like(current)
        for time_index in prange(time_bins):
            for frequency_index in range(frequency_bins):
                target = current[frequency_index, time_index]
                new_map[frequency_index, time_index] = (
                    -1 if target < 0 else base_map[target, time_index]
                )
        current = new_map
    return current


@njit(parallel=True, cache=True)
def _squeeze_from_map(
    coefficients: np.ndarray, reassignment_map: np.ndarray, frequency_step: float
) -> np.ndarray:
    """Reassign the original STFT coefficients once using the composed map."""
    frequency_bins, time_bins = coefficients.shape
    squeezed = np.zeros_like(coefficients)
    # Each parallel worker owns a time column, so writes cannot conflict.
    for time_index in prange(time_bins):
        for frequency_index in range(frequency_bins):
            target = reassignment_map[frequency_index, time_index]
            if target >= 0:
                squeezed[target, time_index] += (
                    coefficients[frequency_index, time_index] * frequency_step
                )
    return squeezed


def msst_stft(
    signal: np.ndarray,
    sampling_frequency: float,
    iteration_count: int = 3,
    *,
    window: str | np.ndarray | None = None,
    n_fft: int | None = None,
    window_length: int | None = None,
    hop_length: int = 1,
    padtype: str = "reflect",
    dtype: str = "float32",
    gamma: float | None = None,
    return_map: bool = False,
):
    """Compute an iterative, STFT-based multisynchrosqueezing transform.

    The STFT and its derivative are computed once by ``ssqueezepy``. The first
    instantaneous-frequency map is composed ``iteration_count`` times, then the
    original STFT coefficients are reassigned once using that final map.
    """
    values = np.asarray(signal)
    if values.ndim != 1:
        raise ValueError("signal must be one-dimensional")
    if len(values) < 2 or not np.all(np.isfinite(values)):
        raise ValueError("signal must contain at least two finite samples")
    if not np.isfinite(sampling_frequency) or sampling_frequency <= 0:
        raise ValueError("sampling_frequency must be positive and finite")
    if iteration_count < 1:
        raise ValueError("iteration_count must be at least 1")
    if hop_length < 1:
        raise ValueError("hop_length must be at least 1")

    coefficients, derivative = stft(
        values,
        window=window,
        n_fft=n_fft,
        win_len=window_length,
        hop_len=hop_length,
        fs=sampling_frequency,
        padtype=padtype,
        modulated=True,
        derivative=True,
        dtype=dtype,
    )
    if not isinstance(coefficients, np.ndarray):
        raise RuntimeError("this implementation requires ssqueezepy CPU/NumPy output")
    if coefficients.shape[0] < 2:
        raise ValueError("the STFT must contain at least two frequency bins")

    real_dtype = np.float32 if coefficients.dtype == np.complex64 else np.float64
    frequency_axis = np.linspace(
        0.0, sampling_frequency / 2, coefficients.shape[0], dtype=real_dtype
    )
    frequency_step = float(frequency_axis[1] - frequency_axis[0])
    threshold = 10 * np.finfo(real_dtype).eps if gamma is None else gamma
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("gamma must be finite and non-negative")

    instantaneous_frequency = phase_stft(
        coefficients, derivative, frequency_axis, gamma=threshold
    )
    base_map = _frequency_map_to_bins(
        instantaneous_frequency,
        float(frequency_axis[0]),
        frequency_step,
        len(frequency_axis),
    )
    final_map = _compose_frequency_map(base_map, iteration_count)
    transformed = _squeeze_from_map(coefficients, final_map, frequency_step)
    if return_map:
        return transformed, coefficients, frequency_axis, final_map
    return transformed, coefficients, frequency_axis


def _group_suspicious_windows(result: pd.DataFrame) -> list[list[int]]:
    """Group suspicious windows whose half-open grid intervals touch or overlap."""
    required = {"suspicious", "accepted", "grid_start", "grid_stop"}
    missing = required.difference(result.columns)
    if missing:
        raise ValueError(f"result is missing columns: {sorted(missing)}")
    selected = result[result["suspicious"].fillna(False).astype(bool)].sort_values("grid_start")
    if not selected["accepted"].fillna(False).astype(bool).all():
        raise ValueError("a suspicious window cannot be rejected")

    groups: list[list[int]] = []
    current: list[int] = []
    current_stop = -1
    for window_id, row in selected.iterrows():
        start, stop = int(row["grid_start"]), int(row["grid_stop"])
        if current and start > current_stop:
            groups.append(current)
            current = []
        current.append(int(window_id))
        current_stop = max(current_stop, stop)
    if current:
        groups.append(current)
    return groups


def _index_accepted_samples(
    result: pd.DataFrame, windows: Mapping[int, WindowData]
) -> dict[int, tuple[object, float, bool, bool]]:
    """Create a global grid-position lookup from accepted overlapping windows."""
    samples: dict[int, tuple[object, float, bool, bool]] = {}
    for window_id, item in windows.items():
        if window_id not in result.index:
            raise ValueError(f"WindowData has unknown window id {window_id}")
        start = int(result.loc[window_id, "grid_start"])
        for offset, (time, value, observed, interpolated) in enumerate(
            zip(item.time, item.signal, item.observed_mask, item.interpolated_mask)
        ):
            position = start + offset
            previous = samples.get(position)
            candidate = (time, float(value), bool(observed), bool(interpolated))
            if previous is not None and (
                previous[0] != candidate[0]
                or not np.isclose(previous[1], candidate[1], rtol=1e-12, atol=1e-12)
            ):
                raise ValueError("overlapping WindowData objects are inconsistent")
            if previous is None:
                samples[position] = candidate
            else:
                samples[position] = (
                    previous[0], previous[1], previous[2] or candidate[2],
                    previous[3] or candidate[3],
                )
    return samples


def prepare_analysis_periods(
    result: pd.DataFrame,
    windows: Mapping[int, WindowData],
    *,
    period_size: int,
) -> tuple[pd.DataFrame, dict[int, AnalysisPeriod]]:
    """Join consecutive anomalies and extend them forward to fixed periods.

    A group begins at the first anomalous sample and is extended with subsequent
    accepted-window samples until ``period_size`` is reached. Groups already
    longer than the requested period, groups at the end of the recording, and
    groups crossing rejected/incomplete data are explicitly excluded.
    """
    if period_size < 2:
        raise ValueError("period_size must be at least 2")
    groups = _group_suspicious_windows(result)
    samples = _index_accepted_samples(result, windows)
    rows: list[dict[str, object]] = []
    periods: dict[int, AnalysisPeriod] = {}

    for period_id, group in enumerate(groups):
        anomaly_start = min(int(result.loc[item, "grid_start"]) for item in group)
        anomaly_stop = max(int(result.loc[item, "grid_stop"]) for item in group)
        anomaly_length = anomaly_stop - anomaly_start
        period_stop = anomaly_start + period_size
        reason: str | None = None
        if anomaly_length > period_size:
            reason = "anomaly_group_too_long"
        positions = range(anomaly_start, period_stop)
        if reason is None and any(position not in samples for position in positions):
            reason = "insufficient_following_valid_data"
        accepted = reason is None

        start_time = samples[anomaly_start][0] if anomaly_start in samples else None
        end_time = samples[period_stop - 1][0] if period_stop - 1 in samples else None
        if accepted:
            selected = [samples[position] for position in positions]
            periods[period_id] = AnalysisPeriod(
                time=pd.Index([item[0] for item in selected]),
                signal=np.asarray([item[1] for item in selected]),
                observed_mask=np.asarray([item[2] for item in selected], dtype=bool),
                interpolated_mask=np.asarray([item[3] for item in selected], dtype=bool),
                anomaly_mask=(
                    np.arange(anomaly_start, period_stop) < anomaly_stop
                ),
                source_window_ids=tuple(group),
            )
        rows.append(
            {
                "source_window_ids": tuple(group),
                "anomaly_grid_start": anomaly_start,
                "anomaly_grid_stop": anomaly_stop,
                "anomaly_samples": anomaly_length,
                "period_grid_start": anomaly_start,
                "period_grid_stop": period_stop,
                "period_samples": period_size,
                "start_time": start_time,
                "end_time": end_time,
                "accepted": accepted,
                "rejection_reason": reason,
            }
        )
    metadata = pd.DataFrame(rows)
    metadata.index = pd.RangeIndex(len(metadata), name="period_id")
    return metadata, periods


def analyze_msst_periods(
    periods: Mapping[int, AnalysisPeriod],
    *,
    sampling_frequency: float,
    iteration_count: int = 3,
    center: bool = True,
    **msst_options: object,
) -> dict[int, MSSTResult]:
    """Run MSST on all accepted fixed periods."""
    analyses: dict[int, MSSTResult] = {}
    for period_id, period in periods.items():
        processed = period.signal - np.mean(period.signal) if center else period.signal.copy()
        transformed, coefficients, frequencies = msst_stft(
            processed,
            sampling_frequency,
            iteration_count,
            **msst_options,
        )
        hop_length = int(msst_options.get("hop_length", 1))
        spectral_time = np.arange(coefficients.shape[1]) * hop_length / sampling_frequency
        analyses[period_id] = MSSTResult(
            period=period,
            processed_signal=processed,
            msst=transformed,
            stft=coefficients,
            frequencies=frequencies,
            spectral_time=spectral_time,
        )
    return analyses


def plot_msst_periods(
    analyses: Mapping[int, MSSTResult], *, max_periods: int | None = None
):
    """Plot the signal, STFT, and MSST for each studied period."""
    import matplotlib.pyplot as plt

    if max_periods is not None and max_periods < 1:
        raise ValueError("max_periods must be positive or None")
    selected = list(analyses.items())
    if max_periods is not None:
        selected = selected[:max_periods]
    if not selected:
        raise ValueError("no MSST analysis to plot")

    fig, axes = plt.subplots(
        len(selected), 3, figsize=(16, max(3.5, 3.5 * len(selected))), squeeze=False
    )
    for row, (period_id, analysis) in enumerate(selected):
        signal_axis, stft_axis, msst_axis = axes[row]
        signal_axis.plot(analysis.period.time, analysis.processed_signal, color="C0")
        anomaly = analysis.period.anomaly_mask
        signal_axis.scatter(
            analysis.period.time[anomaly], analysis.processed_signal[anomaly],
            s=8, color="C3", label="anomaly extent", zorder=3,
        )
        signal_axis.set(title=f"Period {period_id} — signal", ylabel="amplitude")
        signal_axis.legend(loc="best")

        for axis, coefficients, title in (
            (stft_axis, analysis.stft, "STFT"),
            (msst_axis, analysis.msst, "MSST"),
        ):
            mesh = axis.pcolormesh(
                analysis.spectral_time,
                analysis.frequencies,
                np.abs(coefficients),
                shading="auto",
                cmap="magma",
            )
            axis.set(title=f"Period {period_id} — {title}", xlabel="seconds", ylabel="Hz")
            fig.colorbar(mesh, ax=axis, label="magnitude")
    fig.tight_layout()
    fig.autofmt_xdate()
    return fig, axes
