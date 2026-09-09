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


def prepare_monitoring_period(
    result: pd.DataFrame,
    windows: Mapping[int, WindowData],
) -> AnalysisPeriod:
    """Build one continuous period spanning the complete monitoring window.

    The bounds are taken from the first ``grid_start`` and the last
    ``grid_stop`` in the energy-detection result. Overlapping accepted windows
    are de-duplicated on that global grid. Since an MSST cannot contain missing
    values, a descriptive error is raised if rejected windows leave any sample
    in the monitoring interval uncovered.
    """
    required = {"suspicious", "accepted", "grid_start", "grid_stop"}
    missing = required.difference(result.columns)
    if missing:
        raise ValueError(f"result is missing columns: {sorted(missing)}")
    if result.empty:
        raise ValueError("result must contain at least one monitoring window")

    starts = pd.to_numeric(result["grid_start"], errors="raise").astype(int)
    stops = pd.to_numeric(result["grid_stop"], errors="raise").astype(int)
    monitoring_start = int(starts.min())
    monitoring_stop = int(stops.max())
    if monitoring_stop - monitoring_start < 2:
        raise ValueError("the monitoring period must contain at least two samples")

    samples = _index_accepted_samples(result, windows)
    missing_positions = [
        position
        for position in range(monitoring_start, monitoring_stop)
        if position not in samples
    ]
    if missing_positions:
        raise ValueError(
            "the complete monitoring period is not covered by accepted data; "
            f"missing grid positions include {missing_positions[:3]}"
        )

    positions = np.arange(monitoring_start, monitoring_stop)
    selected = [samples[int(position)] for position in positions]
    anomaly_mask = np.zeros(len(positions), dtype=bool)
    suspicious = result[result["suspicious"].fillna(False).astype(bool)]
    for _, row in suspicious.iterrows():
        start = max(int(row["grid_start"]), monitoring_start) - monitoring_start
        stop = min(int(row["grid_stop"]), monitoring_stop) - monitoring_start
        anomaly_mask[start:stop] = True

    return AnalysisPeriod(
        time=pd.Index([item[0] for item in selected]),
        signal=np.asarray([item[1] for item in selected]),
        observed_mask=np.asarray([item[2] for item in selected], dtype=bool),
        interpolated_mask=np.asarray([item[3] for item in selected], dtype=bool),
        anomaly_mask=anomaly_mask,
        source_window_ids=tuple(int(window_id) for window_id in result.index),
    )


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


def analyze_msst_monitoring_period(
    result: pd.DataFrame,
    windows: Mapping[int, WindowData],
    *,
    sampling_frequency: float,
    iteration_count: int = 3,
    center: bool = True,
    **msst_options: object,
) -> MSSTResult:
    """Compute one MSST over the complete energy-monitoring period.

    Unlike :func:`analyze_msst_periods`, this function does not select or extend
    anomaly groups: it preserves the full time extent represented by ``result``.
    """
    period = prepare_monitoring_period(result, windows)
    return analyze_msst_periods(
        {0: period},
        sampling_frequency=sampling_frequency,
        iteration_count=iteration_count,
        center=center,
        **msst_options,
    )[0]


def plot_msst_periods(
    analyses: Mapping[int, MSSTResult], *, max_periods: int | None = None
):
    """Return an interactive Plotly view of signal, STFT, and MSST periods."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    if max_periods is not None and max_periods < 1:
        raise ValueError("max_periods must be positive or None")
    selected = list(analyses.items())
    if max_periods is not None:
        selected = selected[:max_periods]
    if not selected:
        raise ValueError("no MSST analysis to plot")

    titles = [
        title
        for period_id, _ in selected
        for title in (
            f"Period {period_id} — signal",
            f"Period {period_id} — STFT",
            f"Period {period_id} — MSST",
        )
    ]
    figure = make_subplots(
        rows=len(selected), cols=3, subplot_titles=titles,
        horizontal_spacing=0.06, vertical_spacing=min(0.1, 0.3 / len(selected)),
    )
    for row, (period_id, analysis) in enumerate(selected, start=1):
        figure.add_trace(go.Scatter(
            x=analysis.period.time, y=analysis.processed_signal,
            mode="lines", name="processed signal", line={"color": "#2563eb"},
            legendgroup="signal", showlegend=row == 1,
        ), row=row, col=1)
        anomaly = analysis.period.anomaly_mask
        figure.add_trace(go.Scatter(
            x=analysis.period.time[anomaly], y=analysis.processed_signal[anomaly],
            mode="markers", name="anomaly extent",
            marker={"size": 5, "color": "#dc2626"},
            legendgroup="anomaly", showlegend=row == 1,
        ), row=row, col=1)
        figure.add_trace(go.Heatmap(
            x=analysis.spectral_time, y=analysis.frequencies,
            z=np.abs(analysis.stft), colorscale="Viridis",
            colorbar={"title": "|STFT|", "x": 0.62},
            showscale=row == 1, hovertemplate="t=%{x:.4g}s<br>f=%{y:.4g}Hz<br>|S|=%{z:.4g}<extra></extra>",
        ), row=row, col=2)
        figure.add_trace(go.Heatmap(
            x=analysis.spectral_time, y=analysis.frequencies,
            z=np.abs(analysis.msst), colorscale="Turbo",
            colorbar={"title": "|MSST|", "x": 1.02},
            showscale=row == 1, hovertemplate="t=%{x:.4g}s<br>f=%{y:.4g}Hz<br>|T|=%{z:.4g}<extra></extra>",
        ), row=row, col=3)
        figure.update_xaxes(title_text="time", row=row, col=1)
        figure.update_yaxes(title_text="signal", row=row, col=1)
        for column in (2, 3):
            figure.update_xaxes(title_text="time (s)", row=row, col=column)
            figure.update_yaxes(title_text="frequency (Hz)", row=row, col=column)

    figure.update_layout(
        title=f"Fixed-period spectral analysis ({len(selected)} periods)",
        height=max(520, 420 * len(selected)), template="plotly_white",
        hovermode="closest",
    )
    return figure
