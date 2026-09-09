"""Morphological feature extraction from MSST maps."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Hashable, Iterable, Mapping

import numpy as np
import pandas as pd
from scipy import ndimage

from .energy import WindowData, prepare_analysis_windows
from .msst import AnalysisPeriod, MSSTResult, analyze_msst_periods


@dataclass(frozen=True)
class SpectralComponent:
    """Geometry of one connected significant region in physical coordinates."""

    label: int
    area_pixels: int
    integrated_significance: float
    duration_seconds: float
    bandwidth_hz: float
    bounding_box: tuple[int, int, int, int]
    normalized_aspect_ratio: float
    orientation_radians: float
    linearity: float


@dataclass(frozen=True)
class StructuralMSSTResult:
    """MSST maps, morphology, and window-level prototype features."""

    spectral: MSSTResult
    normalized_msst: np.ndarray
    coherence: np.ndarray
    orientation: np.ndarray
    significant_mask: np.ndarray
    component_labels: np.ndarray
    components: tuple[SpectralComponent, ...]
    features: Mapping[str, float]


def normalize_msst_local(
    msst: np.ndarray,
    *,
    neighborhood: tuple[int, int] = (9, 9),
    clip: float | None = 12.0,
) -> np.ndarray:
    """Return a robust local significance map of ``abs(msst)``.

    A two-dimensional running median estimates the local background and a
    running MAD estimates its scale. Only excess above the background is kept.
    This deliberately makes no global white/stationary-noise assumption.
    """
    magnitude = np.abs(np.asarray(msst))
    if magnitude.ndim != 2 or not np.all(np.isfinite(magnitude)):
        raise ValueError("msst must be a finite two-dimensional array")
    if len(neighborhood) != 2 or any(size < 3 or size % 2 == 0 for size in neighborhood):
        raise ValueError("neighborhood sizes must be odd integers of at least 3")
    if clip is not None and (not np.isfinite(clip) or clip <= 0):
        raise ValueError("clip must be positive and finite or None")

    background = ndimage.median_filter(magnitude, size=neighborhood, mode="reflect")
    deviation = np.abs(magnitude - background)
    mad = ndimage.median_filter(deviation, size=neighborhood, mode="reflect")
    positive_mad = mad[mad > 0]
    reference = float(np.median(positive_mad)) if positive_mad.size else 0.0
    floor = max(
        np.finfo(float).eps * max(1.0, float(magnitude.max(initial=0.0))),
        reference * 1e-3,
    )
    significance = np.maximum(magnitude - background, 0.0) / np.maximum(
        1.4826 * mad, floor
    )
    return np.minimum(significance, clip) if clip is not None else significance


def compute_structure_tensor(
    normalized_msst: np.ndarray,
    *,
    sigma: tuple[float, float] = (1.5, 1.5),
    epsilon: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute eigenvalues, orientation, and orientation-invariant coherence.

    Axes are differentiated in units of their Gaussian scale, making time and
    frequency dimensionless before combining their gradients. ``orientation``
    describes the local structure tangent in this normalized coordinate system.
    """
    values = np.asarray(normalized_msst, dtype=float)
    if values.ndim != 2 or not np.all(np.isfinite(values)):
        raise ValueError("normalized_msst must be a finite two-dimensional array")
    if len(sigma) != 2 or any(not np.isfinite(item) or item <= 0 for item in sigma):
        raise ValueError("sigma must contain two positive finite values")
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be positive and finite")

    frequency_gradient, time_gradient = np.gradient(
        values, float(sigma[0]), float(sigma[1])
    )
    j_tt = ndimage.gaussian_filter(time_gradient**2, sigma=sigma, mode="reflect")
    j_ff = ndimage.gaussian_filter(frequency_gradient**2, sigma=sigma, mode="reflect")
    j_tf = ndimage.gaussian_filter(
        time_gradient * frequency_gradient, sigma=sigma, mode="reflect"
    )
    discriminant = np.sqrt(np.maximum((j_tt - j_ff) ** 2 + 4 * j_tf**2, 0.0))
    lambda1 = 0.5 * (j_tt + j_ff + discriminant)
    lambda2 = 0.5 * (j_tt + j_ff - discriminant)
    coherence = discriminant / (lambda1 + lambda2 + epsilon)
    gradient_orientation = 0.5 * np.arctan2(2 * j_tf, j_tt - j_ff)
    orientation = (gradient_orientation + np.pi / 2 + np.pi / 2) % np.pi - np.pi / 2
    return coherence, orientation, lambda1, lambda2


def _component_geometry(
    label: int,
    component_mask: np.ndarray,
    significance: np.ndarray,
    time_step: float,
    frequency_step: float,
    total_duration: float,
    total_bandwidth: float,
) -> SpectralComponent:
    frequency_indices, time_indices = np.nonzero(component_mask)
    weights = significance[component_mask]
    f_min, f_max = int(frequency_indices.min()), int(frequency_indices.max())
    t_min, t_max = int(time_indices.min()), int(time_indices.max())
    duration = (t_max - t_min + 1) * time_step
    bandwidth = (f_max - f_min + 1) * frequency_step

    # PCA uses physical coordinates normalized by the complete map extent;
    # seconds and hertz therefore become comparable dimensionless fractions.
    coordinates = np.column_stack(
        (
            time_indices * time_step / max(total_duration, np.finfo(float).eps),
            frequency_indices
            * frequency_step
            / max(total_bandwidth, np.finfo(float).eps),
        )
    )
    covariance = np.cov(coordinates, rowvar=False, aweights=weights)
    covariance = np.atleast_2d(covariance)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    principal = eigenvectors[:, -1]
    orientation = float(np.arctan2(principal[1], principal[0]))
    linearity = float(
        (eigenvalues[-1] - eigenvalues[0]) / (eigenvalues.sum() + np.finfo(float).eps)
    )
    return SpectralComponent(
        label=label,
        area_pixels=int(component_mask.sum()),
        integrated_significance=float(weights.sum()),
        duration_seconds=float(duration),
        bandwidth_hz=float(bandwidth),
        bounding_box=(f_min, t_min, f_max + 1, t_max + 1),
        normalized_aspect_ratio=float(
            (duration / max(total_duration, np.finfo(float).eps))
            / (bandwidth / max(total_bandwidth, np.finfo(float).eps))
        ),
        orientation_radians=orientation,
        linearity=linearity,
    )


def extract_msst_structure(
    spectral: MSSTResult,
    *,
    normalization_neighborhood: tuple[int, int] = (9, 9),
    tensor_sigma: tuple[float, float] = (1.5, 1.5),
    significance_threshold: float = 3.0,
    coherence_threshold: float = 0.5,
    minimum_component_area: int = 4,
) -> StructuralMSSTResult:
    """Extract a small orientation-neutral morphology feature set from an MSST."""
    if significance_threshold < 0 or not np.isfinite(significance_threshold):
        raise ValueError("significance_threshold must be finite and non-negative")
    if not 0 <= coherence_threshold <= 1:
        raise ValueError("coherence_threshold must be in [0, 1]")
    if minimum_component_area < 1:
        raise ValueError("minimum_component_area must be positive")

    normalized = normalize_msst_local(
        spectral.msst, neighborhood=normalization_neighborhood
    )
    coherence, orientation, _, _ = compute_structure_tensor(
        normalized, sigma=tensor_sigma
    )
    significant = (normalized >= significance_threshold) & (
        coherence >= coherence_threshold
    )
    labels, _ = ndimage.label(significant, structure=np.ones((3, 3), dtype=np.uint8))
    counts = np.bincount(labels.ravel())
    keep = np.flatnonzero(counts >= minimum_component_area)
    keep = keep[keep != 0]
    retained = np.isin(labels, keep)
    labels, component_count = ndimage.label(
        retained, structure=np.ones((3, 3), dtype=np.uint8)
    )

    time_step = (
        float(np.median(np.diff(spectral.spectral_time)))
        if len(spectral.spectral_time) > 1
        else 0.0
    )
    frequency_step = (
        float(np.median(np.diff(spectral.frequencies)))
        if len(spectral.frequencies) > 1
        else 0.0
    )
    total_duration = max(time_step * labels.shape[1], time_step)
    total_bandwidth = max(frequency_step * labels.shape[0], frequency_step)
    components = tuple(
        _component_geometry(
            label,
            labels == label,
            normalized,
            time_step,
            frequency_step,
            total_duration,
            total_bandwidth,
        )
        for label in range(1, component_count + 1)
    )

    amplitude = normalized
    amplitude_sum = float(amplitude.sum())
    weighted_coherence = float(
        np.sum(amplitude * coherence) / max(amplitude_sum, np.finfo(float).eps)
    )
    active = amplitude >= significance_threshold
    active_coherence = coherence[active]
    weights = (amplitude * coherence)[active]
    angles = orientation[active]
    orientation_resultant = (
        float(np.abs(np.sum(weights * np.exp(2j * angles))) / weights.sum())
        if weights.size and weights.sum() > 0
        else 0.0
    )
    areas = [component.area_pixels for component in components]
    integrated = [component.integrated_significance for component in components]
    features = {
        "weighted_mean_coherence": weighted_coherence,
        "coherence_q90": float(np.quantile(active_coherence, 0.9))
        if active_coherence.size
        else 0.0,
        "coherent_pixel_fraction": float(retained.mean()),
        "orientation_dispersion": 1.0 - orientation_resultant,
        "component_count": float(component_count),
        "largest_component_area_fraction": float(max(areas, default=0) / labels.size),
        "largest_component_significance_fraction": float(
            max(integrated, default=0.0) / max(amplitude_sum, np.finfo(float).eps)
        ),
    }
    return StructuralMSSTResult(
        spectral=spectral,
        normalized_msst=normalized,
        coherence=coherence,
        orientation=orientation,
        significant_mask=retained,
        component_labels=labels,
        components=components,
        features=features,
    )


def analyze_structural_windows(
    data: pd.DataFrame,
    *,
    value_col: Hashable,
    sampling_frequency: float,
    quality_col: Hashable | None = None,
    valid_quality_flags: Iterable[object] | None = None,
    sampling_period: pd.Timedelta | str | Real | None = None,
    window_size: int = 256,
    overlap: int = 0,
    min_valid_ratio: float = 0.9,
    min_valid_samples: int | None = None,
    max_interpolation_gap: int = 3,
    msst_options: Mapping[str, object] | None = None,
    structure_options: Mapping[str, object] | None = None,
) -> tuple[pd.DataFrame, dict[int, StructuralMSSTResult]]:
    """Analyze every quality-valid window, independently of energy or score."""
    metadata, windows = prepare_analysis_windows(
        data,
        value_col=value_col,
        quality_col=quality_col,
        valid_quality_flags=valid_quality_flags,
        sampling_period=sampling_period,
        window_size=window_size,
        overlap=overlap,
        min_valid_ratio=min_valid_ratio,
        min_valid_samples=min_valid_samples,
        max_interpolation_gap=max_interpolation_gap,
    )
    periods = {
        window_id: AnalysisPeriod(
            time=window.time,
            signal=window.signal,
            observed_mask=window.observed_mask,
            interpolated_mask=window.interpolated_mask,
            anomaly_mask=np.zeros(len(window.signal), dtype=bool),
            source_window_ids=(window_id,),
        )
        for window_id, window in windows.items()
    }
    spectral = analyze_msst_periods(
        periods,
        sampling_frequency=sampling_frequency,
        **dict(msst_options or {}),
    )
    analyses = {
        window_id: extract_msst_structure(item, **dict(structure_options or {}))
        for window_id, item in spectral.items()
    }
    feature_rows = pd.DataFrame(
        {window_id: item.features for window_id, item in analyses.items()}
    ).T
    for column in feature_rows:
        metadata.loc[feature_rows.index, column] = feature_rows[column]
    return metadata, analyses


def plot_structural_window(analysis: StructuralMSSTResult):
    """Plot signal, spectral maps, coherence, and retained components."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    spectral = analysis.spectral
    figure = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=(
            "Signal",
            "STFT",
            "MSST",
            "MSST normalized",
            "Structure coherence",
            "Retained components",
        ),
        horizontal_spacing=0.07,
        vertical_spacing=0.14,
    )
    figure.add_trace(
        go.Scatter(
            x=spectral.period.time,
            y=spectral.processed_signal,
            mode="lines",
            name="processed signal",
        ),
        row=1,
        col=1,
    )
    maps = (
        (np.abs(spectral.stft), "Viridis", "|STFT|", 1, 2),
        (np.abs(spectral.msst), "Turbo", "|MSST|", 1, 3),
        (analysis.normalized_msst, "Magma", "local significance", 2, 1),
        (analysis.coherence, "Cividis", "coherence", 2, 2),
    )
    for values, colorscale, name, row, column in maps:
        figure.add_trace(
            go.Heatmap(
                x=spectral.spectral_time,
                y=spectral.frequencies,
                z=values,
                colorscale=colorscale,
                name=name,
                showscale=False,
                hovertemplate="t=%{x:.4g}s<br>f=%{y:.4g}Hz<br>value=%{z:.4g}<extra></extra>",
            ),
            row=row,
            col=column,
        )
    figure.add_trace(
        go.Heatmap(
            x=spectral.spectral_time,
            y=spectral.frequencies,
            z=np.abs(spectral.msst),
            colorscale="Greys",
            showscale=False,
            name="MSST background",
        ),
        row=2,
        col=3,
    )
    masked_labels = np.where(analysis.significant_mask, analysis.component_labels, np.nan)
    figure.add_trace(
        go.Heatmap(
            x=spectral.spectral_time,
            y=spectral.frequencies,
            z=masked_labels,
            colorscale="Rainbow",
            opacity=0.65,
            showscale=False,
            name="components",
            hovertemplate="t=%{x:.4g}s<br>f=%{y:.4g}Hz<br>component=%{z}<extra></extra>",
        ),
        row=2,
        col=3,
    )
    for row in (1, 2):
        for column in (1, 2, 3):
            figure.update_xaxes(
                title_text="time" if (row, column) == (1, 1) else "time (s)",
                row=row,
                col=column,
            )
            figure.update_yaxes(
                title_text="signal" if (row, column) == (1, 1) else "frequency (Hz)",
                row=row,
                col=column,
            )
    figure.update_layout(
        title="MSST morphology prototype",
        height=850,
        template="plotly_white",
        hovermode="closest",
    )
    return figure
