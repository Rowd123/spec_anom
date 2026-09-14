"""Feature and downstream models for automatic SAM STFT segments.

The monitored observation is one SAM segment, not one time sample.  Feature
extraction is independent for every binary mask, including overlapping masks.
Optional machine-learning dependencies are imported only when their branch is
used.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Hashable, Mapping, Sequence
import pickle

import numpy as np
import pandas as pd

from .sam_segmentation import SAMSegment


SEGMENT_FEATURES = (
    "integrated_energy", "physical_area", "mean_density", "duration",
    "frequency_width", "central_frequency", "frequency_dispersion",
    "temporal_variation_q95", "frequency_variation_q95", "local_contrast",
)


def _axis_edges(axis: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(axis, dtype=float)
    if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be a non-empty finite 1-D axis")
    if len(values) == 1:
        raise ValueError(f"{name} needs at least two values to define cell edges")
    differences = np.diff(values)
    if not (np.all(differences > 0) or np.all(differences < 0)):
        raise ValueError(f"{name} must be strictly monotonic")
    middle = (values[:-1] + values[1:]) / 2
    return np.r_[values[0] - differences[0] / 2, middle,
                 values[-1] + differences[-1] / 2]


def stft_to_psd(
    stft: np.ndarray,
    *,
    sampling_frequency: float,
    window: str | np.ndarray,
    window_length: int,
    n_fft: int,
    one_sided: bool = True,
) -> np.ndarray:
    """Convert raw, unnormalised FFT STFT coefficients to PSD (signal²/Hz).

    Scaling is ``abs(X)**2 / (fs * sum(window**2))``. For a real-signal
    one-sided spectrum, non-DC/non-Nyquist rows are doubled so their omitted
    negative-frequency energy is retained. This matches the raw FFT convention
    used by :func:`ssqueezepy.stft`; modulation affects phase, not magnitude.
    """
    from scipy.signal import get_window

    values = np.asarray(stft)
    if values.ndim != 2 or not np.all(np.isfinite(values)):
        raise ValueError("stft must be a finite 2-D array")
    if not np.isfinite(sampling_frequency) or sampling_frequency <= 0:
        raise ValueError("sampling_frequency must be positive and finite")
    if window_length < 1 or n_fft < window_length:
        raise ValueError("n_fft must be at least window_length >= 1")
    samples = (get_window(window, window_length, fftbins=True)
               if isinstance(window, str) else np.asarray(window, dtype=float))
    if samples.shape != (window_length,) or not np.all(np.isfinite(samples)):
        raise ValueError("window must contain window_length finite samples")
    window_energy = float(np.sum(samples ** 2))
    if window_energy <= 0:
        raise ValueError("window energy must be positive")
    psd = np.abs(values) ** 2 / (sampling_frequency * window_energy)
    if one_sided:
        expected = n_fft // 2 + 1
        if len(psd) != expected:
            raise ValueError("one-sided STFT row count must equal n_fft//2 + 1")
        stop = -1 if n_fft % 2 == 0 else None
        psd[1:stop] *= 2
    return psd


def _local_background_ratio(values: np.ndarray, times: np.ndarray, *,
                            exclusion_seconds: float, neighborhood_seconds: float,
                            epsilon: float, min_valid_references: int) -> np.ndarray:
    ratio = np.full(values.shape, np.nan)
    for column, time in enumerate(times):
        distance = np.abs(times - time)
        use = (distance > exclusion_seconds) & (distance <= neighborhood_seconds)
        candidates = values[:, use]
        counts = np.count_nonzero(np.isfinite(candidates), axis=1)
        valid_rows = counts >= min_valid_references
        background = np.full(len(values), np.nan)
        if valid_rows.any():
            background[valid_rows] = np.nanmedian(candidates[valid_rows], axis=1)
        valid = valid_rows & np.isfinite(values[:, column])
        ratio[valid, column] = values[valid, column] / np.maximum(
            background[valid], epsilon
        )
    return ratio


def extract_segment_features(
    psd: np.ndarray, mask: np.ndarray, spectral_time: np.ndarray,
    frequencies: np.ndarray, *, contrast_exclusion_seconds: float,
    contrast_neighborhood_seconds: float, contrast_epsilon: float,
    contrast_min_valid_references: int,
) -> dict[str, Any]:
    """Extract ten physical features from one individual SAM mask."""
    power, selected = np.asarray(psd, dtype=float), np.asarray(mask, dtype=bool)
    times, freqs = np.asarray(spectral_time, float), np.asarray(frequencies, float)
    time_edges, frequency_edges = _axis_edges(times, "spectral_time"), _axis_edges(freqs, "frequencies")
    if power.shape != selected.shape or power.shape != (len(freqs), len(times)):
        raise ValueError("psd, mask, and axes must have matching shapes")
    result: dict[str, Any] = {name: np.nan for name in SEGMENT_FEATURES}
    result.update(valid=False, invalid_reason="empty_mask" if not selected.any() else "")
    if not selected.any():
        return result
    if not np.all(np.isfinite(power[selected])) or np.any(power[selected] < 0):
        result["invalid_reason"] = "non_finite_or_negative_psd"
        return result
    dt, df = np.abs(np.diff(time_edges)), np.abs(np.diff(frequency_edges))
    cell_area = df[:, None] * dt[None, :]
    weights = power * cell_area
    energy = float(np.sum(weights[selected]))
    area = float(np.sum(cell_area[selected]))
    rows, columns = np.nonzero(selected)
    duration = float(abs(time_edges[columns.max() + 1] - time_edges[columns.min()]))
    width = float(abs(frequency_edges[rows.max() + 1] - frequency_edges[rows.min()]))
    total_weight = float(np.sum(weights[selected]))
    if total_weight > 0:
        frequency_grid = np.broadcast_to(freqs[:, None], power.shape)
        central = float(np.sum(frequency_grid[selected] * weights[selected]) / total_weight)
        dispersion = float(np.sqrt(np.sum(
            (frequency_grid[selected] - central) ** 2 * weights[selected]
        ) / total_weight))
    else:
        central = dispersion = np.nan
    # Derivatives are deliberately calculated on the complete numerical map.
    temporal_gradient = np.gradient(power, times, axis=1)
    frequency_gradient = np.gradient(power, freqs, axis=0)
    contrast = _local_background_ratio(
        power, times, exclusion_seconds=contrast_exclusion_seconds,
        neighborhood_seconds=contrast_neighborhood_seconds,
        epsilon=contrast_epsilon,
        min_valid_references=contrast_min_valid_references,
    )
    finite_contrast = contrast[selected & np.isfinite(contrast)]
    result.update(
        integrated_energy=energy, physical_area=area,
        mean_density=energy / area, duration=duration, frequency_width=width,
        central_frequency=central, frequency_dispersion=dispersion,
        temporal_variation_q95=float(np.percentile(np.abs(temporal_gradient[selected]), 95)),
        frequency_variation_q95=float(np.percentile(np.abs(frequency_gradient[selected]), 95)),
        local_contrast=(float(np.median(finite_contrast)) if len(finite_contrast) else np.nan),
    )
    if energy <= 0:
        result["invalid_reason"] = "zero_energy"
    elif not np.all(np.isfinite([result[name] for name in SEGMENT_FEATURES])):
        result["invalid_reason"] = "non_finite_feature_or_insufficient_contrast"
    else:
        result["valid"] = True
    return result


def build_segment_table(
    *, stream_id: Hashable, window_id: Hashable, window_start: Any,
    window_end: Any, available_time: Any, segments: Sequence[SAMSegment],
    psd: np.ndarray, spectral_time: np.ndarray, frequencies: np.ndarray,
    contrast_options: Mapping[str, Any],
) -> pd.DataFrame:
    """Build one model-ready row per automatic segment, retaining metadata."""
    rows = []
    time_values, frequency_values = np.asarray(spectral_time), np.asarray(frequencies)
    time_edges = _axis_edges(time_values, "spectral_time")
    frequency_edges = _axis_edges(frequency_values, "frequencies")
    for segment in segments:
        features = extract_segment_features(
            psd, segment.mask, spectral_time, frequencies,
            contrast_exclusion_seconds=contrast_options["exclusion_seconds"],
            contrast_neighborhood_seconds=contrast_options["neighborhood_seconds"],
            contrast_epsilon=contrast_options["epsilon"],
            contrast_min_valid_references=contrast_options["min_valid_references"],
        )
        mask_rows, mask_columns = np.nonzero(segment.mask)
        if len(mask_rows):
            time_bounds = sorted((time_edges[mask_columns.min()], time_edges[mask_columns.max() + 1]))
            frequency_bounds = sorted((frequency_edges[mask_rows.min()], frequency_edges[mask_rows.max() + 1]))
            centroid_time = float(np.mean(time_values[mask_columns]))
            centroid_frequency = float(np.mean(frequency_values[mask_rows]))
        else:
            time_bounds = frequency_bounds = [np.nan, np.nan]
            centroid_time = centroid_frequency = np.nan
        rows.append({
            "stream_id": stream_id, "window_id": window_id,
            "segment_id": segment.segment_id, "window_start": window_start,
            "window_end": window_end, "available_time": available_time,
            "bbox_x": segment.bbox[0], "bbox_y": segment.bbox[1],
            "bbox_width": segment.bbox[2], "bbox_height": segment.bbox[3],
            "centroid_x": None if segment.centroid is None else segment.centroid[0],
            "centroid_y": None if segment.centroid is None else segment.centroid[1],
            "time_min": time_bounds[0], "time_max": time_bounds[1],
            "frequency_min": frequency_bounds[0], "frequency_max": frequency_bounds[1],
            "centroid_time": centroid_time, "centroid_frequency": centroid_frequency,
            "sam_predicted_iou": segment.predicted_iou,
            "sam_stability_score": segment.stability_score, **features,
        })
    return pd.DataFrame(rows)


@dataclass
class RobustFeaturePreprocessor:
    """Training-only transforms followed by median/IQR standardisation."""

    feature_names: tuple[str, ...]
    log1p_features: tuple[str, ...] = ()
    version: str = "preprocess-v1"
    medians_: np.ndarray | None = None
    scales_: np.ndarray | None = None

    def _values(self, frame: pd.DataFrame) -> np.ndarray:
        missing = set(self.feature_names) - set(frame)
        if missing:
            raise ValueError(f"missing model features: {sorted(missing)}")
        values = frame.loc[:, self.feature_names].to_numpy(float, copy=True)
        for name in self.log1p_features:
            if name not in self.feature_names:
                raise ValueError(f"log1p feature is not selected: {name}")
            column = self.feature_names.index(name)
            if np.any(values[:, column] < 0):
                raise ValueError(f"log1p feature must be non-negative: {name}")
            values[:, column] = np.log1p(values[:, column])
        if not np.all(np.isfinite(values)):
            raise ValueError("model features must be finite")
        return values

    def fit(self, frame: pd.DataFrame) -> "RobustFeaturePreprocessor":
        values = self._values(frame)
        if not len(values):
            raise ValueError("cannot fit preprocessing on an empty table")
        self.medians_ = np.median(values, axis=0)
        q25, q75 = np.percentile(values, [25, 75], axis=0)
        self.scales_ = np.where(q75 > q25, q75 - q25, 1.0)
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if self.medians_ is None or self.scales_ is None:
            raise RuntimeError("preprocessor is not fitted")
        return (self._values(frame) - self.medians_) / self.scales_


def chronological_window_split(frame: pd.DataFrame, *, train_fraction: float,
                               calibration_fraction: float) -> pd.Series:
    """Split overlap-connected windows chronologically without splitting windows."""
    if train_fraction <= 0 or calibration_fraction <= 0 or train_fraction + calibration_fraction >= 1:
        raise ValueError("fractions must be positive and leave a non-empty evaluation fraction")
    windows = frame[["stream_id", "window_id", "window_start", "window_end"]].drop_duplicates()
    units = []
    for stream, group in windows.groupby("stream_id", sort=False):
        group = group.sort_values(["window_start", "window_end", "window_id"])
        block, block_end = [], None
        for row in group.itertuples(index=False):
            if block and row.window_start >= block_end:
                units.append(block); block = []
            block.append((stream, row.window_id)); block_end = row.window_end if block_end is None else max(block_end, row.window_end)
        if block: units.append(block)
    units.sort(key=lambda unit: min(windows[(windows.stream_id == unit[0][0]) &
                                            (windows.window_id == unit[0][1])].window_start))
    n = len(units)
    if n < 3:
        raise ValueError("at least three non-overlapping chronological window groups are required")
    train_stop = max(1, int(np.floor(n * train_fraction)))
    calibration_stop = max(train_stop + 1, int(np.floor(n * (train_fraction + calibration_fraction))))
    calibration_stop = min(calibration_stop, n)
    assignment = {}
    for index, unit in enumerate(units):
        label = "train" if index < train_stop else "calibration" if index < calibration_stop else "evaluation"
        assignment.update({key: label for key in unit})
    return pd.Series([assignment[(row.stream_id, row.window_id)] for row in frame.itertuples()], index=frame.index, name="split")


@dataclass
class SPOTState:
    """Upper-tail Peaks-Over-Threshold state for one stream."""

    risk: float = 1e-3
    initial_quantile: float = .98
    min_excesses: int = 20
    calibration_scores: list[float] = field(default_factory=list)
    base_threshold: float | None = None
    shape: float | None = None
    scale: float | None = None
    ready: bool = False

    def calibrate(self, scores: Sequence[float]) -> None:
        values = np.asarray(scores, float)
        self.calibration_scores = values.tolist()
        self.ready = False
        if (not 0 < self.risk < 1 - self.initial_quantile
                or not 0 < self.initial_quantile < 1
                or self.min_excesses < 3 or not np.all(np.isfinite(values))
                or len(np.unique(values)) < 3):
            return
        self.base_threshold = float(np.quantile(values, self.initial_quantile))
        self._fit_tail()

    def _fit_tail(self) -> None:
        """Refit the GPD while keeping the initial calibration threshold fixed."""
        from scipy.stats import genpareto
        if self.base_threshold is None:
            self.ready = False
            return
        values = np.asarray(self.calibration_scores, float)
        excesses = values[values > self.base_threshold] - self.base_threshold
        if len(excesses) < self.min_excesses or len(np.unique(excesses)) < 2:
            self.ready = False
            return
        shape, _, scale = genpareto.fit(excesses, floc=0)
        self.ready = bool(np.isfinite(shape) and np.isfinite(scale) and scale > 0)
        if self.ready:
            self.shape, self.scale = float(shape), float(scale)

    def threshold(self) -> float | None:
        if not self.ready or self.base_threshold is None or self.shape is None or self.scale is None:
            return None
        n, excess_count = len(self.calibration_scores), sum(s > self.base_threshold for s in self.calibration_scores)
        ratio = excess_count / (n * self.risk)
        return float(self.base_threshold + (self.scale * np.log(ratio) if abs(self.shape) < 1e-8
                     else self.scale / self.shape * (ratio ** self.shape - 1)))

    def process(self, score: float) -> tuple[float | None, bool | None]:
        threshold = self.threshold()
        if threshold is None or not np.isfinite(score):
            return threshold, None
        alert = bool(score > threshold)
        if not alert:  # SPOT policy: alerts never update the background tail.
            self.calibration_scores.append(float(score))
            self._fit_tail()
        return threshold, alert


class SegmentModels:
    """Independent Isolation-Forest/SPOT and HDBSCAN branches."""

    def __init__(self, isolation_preprocessor: RobustFeaturePreprocessor,
                 clustering_preprocessor: RobustFeaturePreprocessor, *,
                 isolation_options: Mapping[str, Any], hdbscan_options: Mapping[str, Any],
                 spot_options: Mapping[str, Any], model_version: str = "segments-v1"):
        self.isolation_preprocessor = isolation_preprocessor
        self.clustering_preprocessor = clustering_preprocessor
        self.isolation_options, self.hdbscan_options = dict(isolation_options), dict(hdbscan_options)
        self.spot_options, self.model_version = dict(spot_options), model_version
        self.isolation_forest = self.clusterer = None
        self.spot_states: dict[Hashable, SPOTState] = {}

    def fit(self, training: pd.DataFrame) -> "SegmentModels":
        from sklearn.ensemble import IsolationForest
        import hdbscan
        valid = training[training["valid"].astype(bool)]
        def fit_isolation():
            matrix = self.isolation_preprocessor.fit(valid).transform(valid)
            return IsolationForest(**self.isolation_options).fit(matrix)
        def fit_clusters():
            matrix = self.clustering_preprocessor.fit(valid).transform(valid)
            return hdbscan.HDBSCAN(prediction_data=True, **self.hdbscan_options).fit(matrix)
        with ThreadPoolExecutor(max_workers=2) as executor:
            isolation_future, cluster_future = executor.submit(fit_isolation), executor.submit(fit_clusters)
            self.isolation_forest, self.clusterer = isolation_future.result(), cluster_future.result()
        self.spot_states.clear()  # a changed model/preprocessing invalidates calibration
        return self

    def calibrate_spot(self, calibration: pd.DataFrame) -> None:
        if self.isolation_forest is None:
            raise RuntimeError("models are not fitted")
        self.spot_states.clear()
        for stream, rows in calibration[calibration["valid"].astype(bool)].groupby("stream_id", sort=False):
            scores = isolation_anomaly_scores(
                self.isolation_forest, self.isolation_preprocessor.transform(rows)
            )
            state = SPOTState(**self.spot_options); state.calibrate(scores)
            self.spot_states[stream] = state

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.isolation_forest is None or self.clusterer is None:
            raise RuntimeError("models are not fitted")
        import hdbscan
        output = frame.copy()
        for name, default in (("isolation_score", np.nan), ("spot_threshold", np.nan),
                              ("spot_ready", False), ("spot_state", "missing_stream"),
                              ("spot_alert", pd.NA),
                              ("cluster_id", -1), ("cluster_strength", 0.0)):
            output[name] = default
        valid = output["valid"].astype(bool)
        def isolation_branch():
            matrix = self.isolation_preprocessor.transform(output.loc[valid])
            return isolation_anomaly_scores(self.isolation_forest, matrix)
        def clustering_branch():
            matrix = self.clustering_preprocessor.transform(output.loc[valid])
            return hdbscan.approximate_predict(self.clusterer, matrix)
        with ThreadPoolExecutor(max_workers=2) as executor:
            isolation_future = executor.submit(isolation_branch)
            cluster_future = executor.submit(clustering_branch)
            isolation_scores, (labels, strengths) = isolation_future.result(), cluster_future.result()
        output.loc[valid, "isolation_score"] = isolation_scores
        output.loc[valid, "cluster_id"], output.loc[valid, "cluster_strength"] = labels, strengths
        ordered = output.loc[valid].sort_values(["available_time", "stream_id", "window_id", "segment_id"])
        for index, row in ordered.iterrows():
            state = self.spot_states.get(row.stream_id)
            if state is None:
                continue
            threshold, alert = state.process(float(row.isolation_score))
            output.at[index, "spot_threshold"] = np.nan if threshold is None else threshold
            output.at[index, "spot_ready"] = threshold is not None
            output.at[index, "spot_state"] = "ready" if threshold is not None else "not_ready"
            output.at[index, "spot_alert"] = alert
        output["model_version"] = self.model_version
        output["isolation_preprocessing_version"] = self.isolation_preprocessor.version
        output["clustering_preprocessing_version"] = self.clustering_preprocessor.version
        return output

    def save(self, path: str | Path) -> None:
        with Path(path).open("wb") as handle: pickle.dump(self, handle)

    @staticmethod
    def load(path: str | Path) -> "SegmentModels":
        with Path(path).open("rb") as handle: return pickle.load(handle)


def isolation_anomaly_scores(model: object, values: np.ndarray) -> np.ndarray:
    """Return ``-score_samples``; larger values are more atypical, not probabilities."""
    scores = -np.asarray(model.score_samples(values), dtype=float)
    if scores.ndim != 1 or not np.all(np.isfinite(scores)):
        raise ValueError("Isolation Forest returned invalid scores")
    return scores


def plot_segment_pipeline(stft: np.ndarray, spectral_time: np.ndarray,
                          frequencies: np.ndarray, segments: Sequence[SAMSegment],
                          results: pd.DataFrame):
    """Plot segment masks, chronological IF/SPOT scores, and HDBSCAN groups."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    figure = make_subplots(rows=1, cols=3, subplot_titles=("Segments sur STFT", "Isolation Forest / SPOT", "Groupes HDBSCAN"))
    figure.add_trace(go.Heatmap(z=np.log1p(np.abs(stft)), x=spectral_time, y=frequencies, colorscale="Viridis", showscale=False), 1, 1)
    for segment in segments:
        row = results[results.segment_id == segment.segment_id]
        alert = bool(row.spot_alert.iloc[0]) if len(row) and pd.notna(row.spot_alert.iloc[0]) else False
        figure.add_trace(go.Contour(z=segment.mask.astype(int), x=spectral_time, y=frequencies,
                                    contours={"start": .5, "end": .5}, showscale=False,
                                    line={"color": "red" if alert else "white"}, name=f"S{segment.segment_id}"), 1, 1)
    ordered = results.sort_values(["available_time", "segment_id"])
    figure.add_trace(go.Scatter(x=ordered.available_time, y=ordered.isolation_score, mode="markers+lines", name="score IF",
                                marker={"color": np.where(ordered.spot_alert.fillna(False), "red", "blue")}), 1, 2)
    figure.add_trace(go.Scatter(x=ordered.available_time, y=ordered.spot_threshold, mode="lines", name="seuil SPOT"), 1, 2)
    figure.add_trace(go.Scatter(x=ordered.central_frequency, y=ordered.duration, mode="markers", name="groupes",
                                marker={"color": ordered.cluster_id, "colorscale": "Turbo", "size": 8},
                                customdata=np.c_[ordered.segment_id, ordered.cluster_id, ordered.cluster_strength],
                                hovertemplate="segment=%{customdata[0]}<br>groupe=%{customdata[1]}<br>force=%{customdata[2]:.3f}<extra></extra>"), 1, 3)
    figure.update_layout(template="plotly_white", title="Segments : alertes SPOT et groupes HDBSCAN")
    return figure
