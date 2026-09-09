"""Morphological feature extraction from MSST maps."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Hashable, Iterable, Literal, Mapping

import numpy as np
import pandas as pd
from scipy import ndimage

from .energy import WindowData, prepare_analysis_windows
from .msst import (
    AnalysisPeriod,
    MSSTResult,
    analyze_msst_periods,
    analyze_stft_periods,
)


@dataclass(frozen=True)
class SpectralComponent:
    """Geometry of one connected significant region in physical coordinates."""

    label: int
    area_pixels: int
    integrated_significance: float
    duration_seconds: float
    bandwidth_hz: float
    time_span_seconds: float
    active_duration_seconds: float
    frequency_span_hz: float
    support_bandwidth_hz: float
    frequency_centroid: float
    frequency_min: float
    frequency_max: float
    relative_time_centroid: float
    mean_significance: float
    max_significance: float
    mean_coherence: float
    median_coherence: float
    coherence_q90: float
    bounding_box: tuple[int, int, int, int]
    normalized_aspect_ratio: float
    orientation_radians: float
    orientation_cos2: float
    orientation_sin2: float
    linearity: float


@dataclass(frozen=True)
class CandidateStructure:
    """One physical candidate assembled from compatible local fragments."""

    candidate_id: int
    component_ids: tuple[int, ...]
    component_count: int
    fragment_count: int
    area_pixels: int
    integrated_significance: float
    mean_significance: float
    max_significance: float
    duration_seconds: float
    time_span_seconds: float
    active_duration_seconds: float
    total_gap_duration_seconds: float
    maximum_gap_duration_seconds: float
    gap_fraction: float
    frequency_centroid: float
    frequency_min: float
    frequency_max: float
    frequency_span_hz: float
    support_bandwidth_hz: float
    bandwidth_hz: float
    mean_coherence: float
    median_coherence: float
    coherence_q90: float
    relative_time_centroid: float
    bounding_box: tuple[int, int, int, int]
    normalized_aspect_ratio: float
    orientation_radians: float
    orientation_cos2: float
    orientation_sin2: float
    linearity: float


@dataclass(frozen=True)
class StructuralSpectralResult:
    """Generic spectral map, morphology, and window-level prototype features."""

    spectral: MSSTResult
    representation: Literal["stft", "msst"]
    normalized_representation: np.ndarray
    coherence: np.ndarray
    orientation: np.ndarray
    significant_mask: np.ndarray
    component_labels: np.ndarray
    components: tuple[SpectralComponent, ...]
    features: Mapping[str, float]

    @property
    def raw_representation(self) -> np.ndarray:
        """Return the complex map on which morphology was computed."""
        return self.spectral.stft if self.representation == "stft" else self.spectral.msst

    @property
    def normalized_msst(self) -> np.ndarray:
        """Backward-compatible name for the normalized spectral representation."""
        return self.normalized_representation


# Backward-compatible public name from the MSST-only prototype.
StructuralMSSTResult = StructuralSpectralResult


@dataclass(frozen=True)
class StructuralComparison:
    """Morphology results for STFT and MSST of the exact same window."""

    stft: StructuralSpectralResult
    msst: StructuralSpectralResult
    metrics: pd.DataFrame


@dataclass(frozen=True)
class CandidateStructureResult:
    """Original component result plus graph-associated candidate structures."""

    structural: StructuralSpectralResult
    candidates: tuple[CandidateStructure, ...]
    candidate_labels: np.ndarray


@dataclass(frozen=True)
class RobustFeatureScaler:
    """Reusable median/MAD parameters fitted on component feature columns."""

    columns: tuple[str, ...]
    log1p_columns: tuple[str, ...]
    medians: pd.Series
    scales: pd.Series


COMPONENT_FEATURE_COLUMNS = (
    "frequency_centroid",
    "frequency_min",
    "frequency_max",
    "bandwidth_hz",
    "duration_seconds",
    "frequency_span_hz",
    "support_bandwidth_hz",
    "time_span_seconds",
    "active_duration_seconds",
    "relative_time_centroid",
    "area_pixels",
    "mean_significance",
    "max_significance",
    "integrated_significance",
    "mean_coherence",
    "median_coherence",
    "coherence_q90",
    "normalized_aspect_ratio",
    "linearity",
    "orientation_cos2",
    "orientation_sin2",
)

DEFAULT_LOG1P_FEATURES = (
    "bandwidth_hz",
    "duration_seconds",
    "area_pixels",
    "mean_significance",
    "max_significance",
    "integrated_significance",
    "normalized_aspect_ratio",
)


def normalize_spectral_map_local(
    spectral_map: np.ndarray,
    *,
    neighborhood: tuple[int, int] = (9, 9),
    clip: float | None = 12.0,
) -> np.ndarray:
    """Return a robust local significance map of a complex or magnitude map.

    A two-dimensional running median estimates the local background and a
    running MAD estimates its scale. Only excess above the background is kept.
    This deliberately makes no global white/stationary-noise assumption.
    """
    magnitude = np.abs(np.asarray(spectral_map))
    if magnitude.ndim != 2 or not np.all(np.isfinite(magnitude)):
        raise ValueError("spectral_map must be a finite two-dimensional array")
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


def normalize_msst_local(
    msst: np.ndarray,
    *,
    neighborhood: tuple[int, int] = (9, 9),
    clip: float | None = 12.0,
) -> np.ndarray:
    """Backward-compatible alias for local spectral-map normalization."""
    return normalize_spectral_map_local(msst, neighborhood=neighborhood, clip=clip)


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
    coherence: np.ndarray,
    frequency_axis: np.ndarray,
    spectral_time: np.ndarray,
    time_step: float,
    frequency_step: float,
    total_duration: float,
    total_bandwidth: float,
) -> SpectralComponent:
    frequency_indices, time_indices = np.nonzero(component_mask)
    weights = significance[component_mask]
    coherence_values = coherence[component_mask]
    f_min, f_max = int(frequency_indices.min()), int(frequency_indices.max())
    t_min, t_max = int(time_indices.min()), int(time_indices.max())
    duration = (t_max - t_min + 1) * time_step
    active_duration = len(np.unique(time_indices)) * time_step
    time_span = (t_max - t_min) * time_step
    frequency_span = float(frequency_axis[f_max] - frequency_axis[f_min])
    support_bandwidth = (f_max - f_min + 1) * frequency_step
    weight_sum = max(float(weights.sum()), np.finfo(float).eps)
    frequency_centroid = float(
        np.sum(frequency_axis[frequency_indices] * weights) / weight_sum
    )
    if len(spectral_time) > 1:
        relative_times = (spectral_time - spectral_time[0]) / (
            spectral_time[-1] - spectral_time[0]
        )
    else:
        relative_times = np.zeros_like(spectral_time, dtype=float)
    relative_time_centroid = float(
        np.sum(relative_times[time_indices] * weights) / weight_sum
    )

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
    if len(coordinates) < 2:
        orientation = 0.0
        linearity = 0.0
    else:
        covariance = np.cov(coordinates, rowvar=False, aweights=weights)
        covariance = np.atleast_2d(covariance)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        eigenvalues = np.maximum(eigenvalues, 0.0)
        principal = eigenvectors[:, -1]
        orientation = float(np.arctan2(principal[1], principal[0]))
        linearity = float(
            (eigenvalues[-1] - eigenvalues[0])
            / (eigenvalues.sum() + np.finfo(float).eps)
        )
    return SpectralComponent(
        label=label,
        area_pixels=int(component_mask.sum()),
        integrated_significance=float(weights.sum()),
        duration_seconds=float(duration),
        bandwidth_hz=float(support_bandwidth),
        time_span_seconds=float(time_span),
        active_duration_seconds=float(active_duration),
        frequency_span_hz=frequency_span,
        support_bandwidth_hz=float(support_bandwidth),
        frequency_centroid=frequency_centroid,
        frequency_min=float(frequency_axis[f_min]),
        frequency_max=float(frequency_axis[f_max]),
        relative_time_centroid=relative_time_centroid,
        mean_significance=float(weights.mean()),
        max_significance=float(weights.max()),
        mean_coherence=float(coherence_values.mean()),
        median_coherence=float(np.median(coherence_values)),
        coherence_q90=float(np.quantile(coherence_values, 0.9)),
        bounding_box=(f_min, t_min, f_max + 1, t_max + 1),
        normalized_aspect_ratio=float(
            (duration / max(total_duration, np.finfo(float).eps))
            / (support_bandwidth / max(total_bandwidth, np.finfo(float).eps))
        ),
        orientation_radians=orientation,
        orientation_cos2=float(np.cos(2 * orientation)),
        orientation_sin2=float(np.sin(2 * orientation)),
        linearity=linearity,
    )


def extract_spectral_structure(
    spectral: MSSTResult,
    *,
    representation: Literal["stft", "msst"] = "stft",
    normalization_neighborhood: tuple[int, int] = (9, 9),
    tensor_sigma: tuple[float, float] = (1.5, 1.5),
    significance_threshold: float = 3.0,
    coherence_threshold: float = 0.5,
    minimum_component_area: int = 1,
    small_component_area: int = 16,
) -> StructuralSpectralResult:
    """Apply the same morphology chain to either STFT or MSST coefficients."""
    if representation not in {"stft", "msst"}:
        raise ValueError("representation must be 'stft' or 'msst'")
    if significance_threshold < 0 or not np.isfinite(significance_threshold):
        raise ValueError("significance_threshold must be finite and non-negative")
    if not 0 <= coherence_threshold <= 1:
        raise ValueError("coherence_threshold must be in [0, 1]")
    if minimum_component_area < 1:
        raise ValueError("minimum_component_area must be positive")
    if small_component_area < minimum_component_area:
        raise ValueError("small_component_area must be at least minimum_component_area")

    raw_representation = (
        spectral.stft if representation == "stft" else spectral.msst
    )
    if raw_representation.size == 0:
        raise ValueError(f"{representation} coefficients are not available")
    normalized = normalize_spectral_map_local(
        raw_representation, neighborhood=normalization_neighborhood
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
            coherence,
            spectral.frequencies,
            spectral.spectral_time,
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
    median_area = float(np.median(areas)) if areas else 0.0
    maximum_area = float(max(areas, default=0))
    small_fraction = (
        float(np.mean(np.asarray(areas) <= small_component_area)) if areas else 0.0
    )
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
        "median_component_area": median_area,
        "max_component_area": maximum_area,
        "small_component_fraction": small_fraction,
    }
    return StructuralSpectralResult(
        spectral=spectral,
        representation=representation,
        normalized_representation=normalized,
        coherence=coherence,
        orientation=orientation,
        significant_mask=retained,
        component_labels=labels,
        components=components,
        features=features,
    )


def extract_msst_structure(
    spectral: MSSTResult, **options: object
) -> StructuralSpectralResult:
    """Backward-compatible MSST-specific morphology wrapper."""
    return extract_spectral_structure(spectral, representation="msst", **options)


def compare_stft_msst_structure(
    spectral: MSSTResult, **structure_options: object
) -> StructuralComparison:
    """Compare STFT and MSST morphology from one shared spectral transform."""
    stft_result = extract_spectral_structure(
        spectral, representation="stft", **structure_options
    )
    msst_result = extract_spectral_structure(
        spectral, representation="msst", **structure_options
    )
    metrics = pd.DataFrame(
        {"stft": stft_result.features, "msst": msst_result.features}
    )
    metrics["msst_minus_stft"] = metrics["msst"] - metrics["stft"]
    return StructuralComparison(stft=stft_result, msst=msst_result, metrics=metrics)


def axial_orientation_distance(first: float, second: float) -> float:
    """Return the smallest angular distance for orientations modulo pi."""
    return float(0.5 * abs(np.angle(np.exp(2j * (first - second)))))


def _interval_gap(
    first_min: float, first_max: float, second_min: float, second_max: float
) -> float:
    return max(0.0, max(first_min, second_min) - min(first_max, second_max))


def _component_time_bounds(
    component: SpectralComponent, spectral_time: np.ndarray, time_step: float
) -> tuple[float, float]:
    _, start, _, stop = component.bounding_box
    return float(spectral_time[start]), float(spectral_time[stop - 1] + time_step)


def associate_component_fragments(
    analysis: StructuralSpectralResult,
    *,
    max_fragment_time_gap_seconds: float = 0.0,
    max_fragment_frequency_gap_hz: float = 0.0,
    max_fragment_frequency_centroid_difference_hz: float = 0.0,
    max_fragment_orientation_difference_radians: float = np.deg2rad(10.0),
) -> CandidateStructureResult:
    """Associate compatible fragments through connected components of a graph.

    Pairwise edges require temporal, frequency-interval, centroid, and axial
    orientation compatibility. Graph connectivity is deliberately transitive;
    conservative thresholds are important because chains can otherwise bridge
    two occurrences whose endpoints would not be directly compatible.
    """
    thresholds = {
        "max_fragment_time_gap_seconds": max_fragment_time_gap_seconds,
        "max_fragment_frequency_gap_hz": max_fragment_frequency_gap_hz,
        "max_fragment_frequency_centroid_difference_hz": (
            max_fragment_frequency_centroid_difference_hz
        ),
        "max_fragment_orientation_difference_radians": (
            max_fragment_orientation_difference_radians
        ),
    }
    if any(not np.isfinite(value) or value < 0 for value in thresholds.values()):
        raise ValueError("fragment-association thresholds must be finite and non-negative")

    components = analysis.components
    labels = np.zeros_like(analysis.component_labels, dtype=np.int32)
    if not components:
        return CandidateStructureResult(analysis, (), labels)
    spectral = analysis.spectral
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
    parents = np.arange(len(components))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = int(parents[index])
        return index

    def union(first: int, second: int) -> None:
        first_root, second_root = find(first), find(second)
        if first_root != second_root:
            parents[second_root] = first_root

    for first_index, first in enumerate(components):
        first_start, first_stop = _component_time_bounds(
            first, spectral.spectral_time, time_step
        )
        for second_index in range(first_index + 1, len(components)):
            second = components[second_index]
            second_start, second_stop = _component_time_bounds(
                second, spectral.spectral_time, time_step
            )
            time_gap = _interval_gap(
                first_start, first_stop, second_start, second_stop
            )
            frequency_gap = _interval_gap(
                first.frequency_min,
                first.frequency_max,
                second.frequency_min,
                second.frequency_max,
            )
            centroid_difference = abs(
                first.frequency_centroid - second.frequency_centroid
            )
            orientation_difference = axial_orientation_distance(
                first.orientation_radians, second.orientation_radians
            )
            if (
                time_gap <= max_fragment_time_gap_seconds
                and frequency_gap <= max_fragment_frequency_gap_hz
                and centroid_difference
                <= max_fragment_frequency_centroid_difference_hz
                and orientation_difference
                <= max_fragment_orientation_difference_radians
            ):
                union(first_index, second_index)

    groups: dict[int, list[SpectralComponent]] = {}
    for index, component in enumerate(components):
        groups.setdefault(find(index), []).append(component)

    candidates: list[CandidateStructure] = []
    total_duration = max(time_step * labels.shape[1], time_step)
    total_bandwidth = max(frequency_step * labels.shape[0], frequency_step)
    for candidate_id, group in enumerate(groups.values(), start=1):
        component_ids = tuple(component.label for component in group)
        candidate_mask = np.isin(analysis.component_labels, component_ids)
        labels[candidate_mask] = candidate_id
        geometry = _component_geometry(
            candidate_id,
            candidate_mask,
            analysis.normalized_representation,
            analysis.coherence,
            spectral.frequencies,
            spectral.spectral_time,
            time_step,
            frequency_step,
            total_duration,
            total_bandwidth,
        )
        _, time_indices = np.nonzero(candidate_mask)
        active_columns = np.unique(time_indices)
        temporal_bin_count = int(active_columns[-1] - active_columns[0] + 1)
        active_bin_count = len(active_columns)
        missing_columns = np.setdiff1d(
            np.arange(active_columns[0], active_columns[-1] + 1), active_columns
        )
        if missing_columns.size:
            split_points = np.flatnonzero(np.diff(missing_columns) > 1) + 1
            maximum_gap_bins = max(
                len(run) for run in np.split(missing_columns, split_points)
            )
        else:
            maximum_gap_bins = 0
        temporal_span = temporal_bin_count * time_step
        active_duration = active_bin_count * time_step
        candidates.append(
            CandidateStructure(
                candidate_id=candidate_id,
                component_ids=component_ids,
                component_count=len(group),
                fragment_count=len(group),
                area_pixels=geometry.area_pixels,
                integrated_significance=geometry.integrated_significance,
                mean_significance=geometry.mean_significance,
                max_significance=geometry.max_significance,
                duration_seconds=temporal_span,
                time_span_seconds=temporal_span,
                active_duration_seconds=active_duration,
                total_gap_duration_seconds=temporal_span - active_duration,
                maximum_gap_duration_seconds=maximum_gap_bins * time_step,
                gap_fraction=1.0 - active_bin_count / temporal_bin_count,
                frequency_centroid=geometry.frequency_centroid,
                frequency_min=geometry.frequency_min,
                frequency_max=geometry.frequency_max,
                frequency_span_hz=geometry.frequency_span_hz,
                support_bandwidth_hz=geometry.support_bandwidth_hz,
                bandwidth_hz=geometry.support_bandwidth_hz,
                mean_coherence=geometry.mean_coherence,
                median_coherence=geometry.median_coherence,
                coherence_q90=geometry.coherence_q90,
                relative_time_centroid=geometry.relative_time_centroid,
                bounding_box=geometry.bounding_box,
                normalized_aspect_ratio=geometry.normalized_aspect_ratio,
                orientation_radians=geometry.orientation_radians,
                orientation_cos2=geometry.orientation_cos2,
                orientation_sin2=geometry.orientation_sin2,
                linearity=geometry.linearity,
            )
        )
    return CandidateStructureResult(analysis, tuple(candidates), labels)


def components_to_dataframe(
    analyses: Mapping[int, StructuralSpectralResult],
) -> pd.DataFrame:
    """Return one row per component while preserving its parent window ID."""
    rows: list[dict[str, object]] = []
    for window_id, analysis in analyses.items():
        for component in analysis.components:
            f_start, t_start, f_stop, t_stop = component.bounding_box
            rows.append(
                {
                    "window_id": window_id,
                    "component_id": component.label,
                    "representation": analysis.representation,
                    "area_pixels": component.area_pixels,
                    "integrated_significance": component.integrated_significance,
                    "duration_seconds": component.duration_seconds,
                    "bandwidth_hz": component.bandwidth_hz,
                    "time_span_seconds": component.time_span_seconds,
                    "active_duration_seconds": component.active_duration_seconds,
                    "frequency_span_hz": component.frequency_span_hz,
                    "support_bandwidth_hz": component.support_bandwidth_hz,
                    "frequency_centroid": component.frequency_centroid,
                    "frequency_min": component.frequency_min,
                    "frequency_max": component.frequency_max,
                    "relative_time_centroid": component.relative_time_centroid,
                    "mean_significance": component.mean_significance,
                    "max_significance": component.max_significance,
                    "mean_coherence": component.mean_coherence,
                    "median_coherence": component.median_coherence,
                    "coherence_q90": component.coherence_q90,
                    "normalized_aspect_ratio": component.normalized_aspect_ratio,
                    "orientation_radians": component.orientation_radians,
                    "orientation_cos2": component.orientation_cos2,
                    "orientation_sin2": component.orientation_sin2,
                    "linearity": component.linearity,
                    "frequency_bin_start": f_start,
                    "time_bin_start": t_start,
                    "frequency_bin_stop": f_stop,
                    "time_bin_stop": t_stop,
                }
            )
    columns = (
        "window_id",
        "component_id",
        "representation",
        *COMPONENT_FEATURE_COLUMNS,
        "orientation_radians",
        "frequency_bin_start",
        "time_bin_start",
        "frequency_bin_stop",
        "time_bin_stop",
    )
    return pd.DataFrame(rows, columns=columns)


def candidates_to_dataframe(
    results: Mapping[int, CandidateStructureResult],
) -> pd.DataFrame:
    """Return one row per associated candidate without discarding fragment IDs."""
    rows: list[dict[str, object]] = []
    for window_id, result in results.items():
        for candidate in result.candidates:
            f_start, t_start, f_stop, t_stop = candidate.bounding_box
            rows.append(
                {
                    "window_id": window_id,
                    "candidate_id": candidate.candidate_id,
                    "representation": result.structural.representation,
                    "component_ids": candidate.component_ids,
                    "component_count": candidate.component_count,
                    "fragment_count": candidate.fragment_count,
                    "area_pixels": candidate.area_pixels,
                    "integrated_significance": candidate.integrated_significance,
                    "mean_significance": candidate.mean_significance,
                    "max_significance": candidate.max_significance,
                    "duration_seconds": candidate.duration_seconds,
                    "time_span_seconds": candidate.time_span_seconds,
                    "active_duration_seconds": candidate.active_duration_seconds,
                    "total_gap_duration_seconds": candidate.total_gap_duration_seconds,
                    "maximum_gap_duration_seconds": candidate.maximum_gap_duration_seconds,
                    "gap_fraction": candidate.gap_fraction,
                    "frequency_centroid": candidate.frequency_centroid,
                    "frequency_min": candidate.frequency_min,
                    "frequency_max": candidate.frequency_max,
                    "frequency_span_hz": candidate.frequency_span_hz,
                    "support_bandwidth_hz": candidate.support_bandwidth_hz,
                    "bandwidth_hz": candidate.bandwidth_hz,
                    "mean_coherence": candidate.mean_coherence,
                    "median_coherence": candidate.median_coherence,
                    "coherence_q90": candidate.coherence_q90,
                    "relative_time_centroid": candidate.relative_time_centroid,
                    "normalized_aspect_ratio": candidate.normalized_aspect_ratio,
                    "orientation_radians": candidate.orientation_radians,
                    "orientation_cos2": candidate.orientation_cos2,
                    "orientation_sin2": candidate.orientation_sin2,
                    "linearity": candidate.linearity,
                    "frequency_bin_start": f_start,
                    "time_bin_start": t_start,
                    "frequency_bin_stop": f_stop,
                    "time_bin_stop": t_stop,
                }
            )
    columns = (
        "window_id",
        "candidate_id",
        "representation",
        "component_ids",
        "component_count",
        "fragment_count",
        "area_pixels",
        "integrated_significance",
        "mean_significance",
        "max_significance",
        "duration_seconds",
        "time_span_seconds",
        "active_duration_seconds",
        "total_gap_duration_seconds",
        "maximum_gap_duration_seconds",
        "gap_fraction",
        "frequency_centroid",
        "frequency_min",
        "frequency_max",
        "frequency_span_hz",
        "support_bandwidth_hz",
        "bandwidth_hz",
        "mean_coherence",
        "median_coherence",
        "coherence_q90",
        "relative_time_centroid",
        "normalized_aspect_ratio",
        "orientation_radians",
        "orientation_cos2",
        "orientation_sin2",
        "linearity",
        "frequency_bin_start",
        "time_bin_start",
        "frequency_bin_stop",
        "time_bin_stop",
    )
    return pd.DataFrame(rows, columns=columns)


def component_features(components: pd.DataFrame) -> pd.DataFrame:
    """Select numeric clustering candidates, excluding identifiers and raw angle."""
    missing = set(COMPONENT_FEATURE_COLUMNS).difference(components.columns)
    if missing:
        raise ValueError(f"components is missing feature columns: {sorted(missing)}")
    return components.loc[:, COMPONENT_FEATURE_COLUMNS].astype(float).copy()


def fit_component_feature_scaler(
    components: pd.DataFrame,
    *,
    log1p_columns: Iterable[str] = DEFAULT_LOG1P_FEATURES,
) -> RobustFeatureScaler:
    """Fit log1p plus median/MAD scaling without assigning anomaly labels."""
    values = component_features(components)
    if values.empty:
        raise ValueError("at least one component is required to fit feature scaling")
    selected_log = tuple(log1p_columns)
    unknown = set(selected_log).difference(values.columns)
    if unknown:
        raise ValueError(f"unknown log1p feature columns: {sorted(unknown)}")
    if (values.loc[:, selected_log] < 0).any().any():
        raise ValueError("log1p feature columns must be non-negative")
    values.loc[:, selected_log] = np.log1p(values.loc[:, selected_log])
    medians = values.median(axis=0)
    mad = (values - medians).abs().median(axis=0)
    scales = 1.4826 * mad
    scales = scales.mask(scales <= np.finfo(float).eps, 1.0)
    return RobustFeatureScaler(
        columns=tuple(values.columns),
        log1p_columns=selected_log,
        medians=medians,
        scales=scales,
    )


def transform_component_features(
    components: pd.DataFrame, scaler: RobustFeatureScaler
) -> pd.DataFrame:
    """Apply a previously fitted component-feature transformation."""
    values = component_features(components).loc[:, scaler.columns]
    if (values.loc[:, scaler.log1p_columns] < 0).any().any():
        raise ValueError("log1p feature columns must be non-negative")
    values.loc[:, scaler.log1p_columns] = np.log1p(
        values.loc[:, scaler.log1p_columns]
    )
    return (values - scaler.medians) / scaler.scales


def _resolve_transform_options(
    transform_options: Mapping[str, object] | None,
    legacy_msst_options: Mapping[str, object] | None,
) -> dict[str, object]:
    if transform_options is not None and legacy_msst_options is not None:
        raise ValueError("use transform_options or msst_options, not both")
    return dict(transform_options or legacy_msst_options or {})


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
    representation: Literal["stft", "msst"] = "stft",
    window_ids: Iterable[int] | None = None,
    transform_options: Mapping[str, object] | None = None,
    msst_options: Mapping[str, object] | None = None,
    structure_options: Mapping[str, object] | None = None,
) -> tuple[pd.DataFrame, dict[int, StructuralSpectralResult]]:
    """Analyze every quality-valid window with the selected representation."""
    if representation not in {"stft", "msst"}:
        raise ValueError("representation must be 'stft' or 'msst'")
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
    selected_ids = None if window_ids is None else set(window_ids)
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
        if selected_ids is None or window_id in selected_ids
    }
    selected_transform_options = _resolve_transform_options(
        transform_options, msst_options
    )
    if representation == "stft":
        selected_transform_options.pop("iteration_count", None)
        selected_transform_options.pop("gamma", None)
        selected_transform_options.pop("return_map", None)
        spectral = analyze_stft_periods(
            periods,
            sampling_frequency=sampling_frequency,
            **selected_transform_options,
        )
    else:
        spectral = analyze_msst_periods(
            periods,
            sampling_frequency=sampling_frequency,
            **selected_transform_options,
        )
    analyses = {
        window_id: extract_spectral_structure(
            item,
            representation=representation,
            **dict(structure_options or {}),
        )
        for window_id, item in spectral.items()
    }
    feature_rows = pd.DataFrame(
        {window_id: item.features for window_id, item in analyses.items()}
    ).T
    for column in feature_rows:
        metadata.loc[feature_rows.index, column] = feature_rows[column]
    return metadata, analyses


def analyze_structural_components(
    data: pd.DataFrame,
    **options: object,
) -> tuple[
    pd.DataFrame,
    dict[int, StructuralSpectralResult],
    pd.DataFrame,
]:
    """Run structural analysis and return its component population table."""
    metadata, analyses = analyze_structural_windows(data, **options)
    return metadata, analyses, components_to_dataframe(analyses)


def analyze_candidate_structures(
    data: pd.DataFrame,
    *,
    association_options: Mapping[str, float] | None = None,
    **analysis_options: object,
) -> tuple[
    pd.DataFrame,
    dict[int, CandidateStructureResult],
    pd.DataFrame,
    pd.DataFrame,
]:
    """Extract components, associate local fragments, and return both tables."""
    metadata, analyses = analyze_structural_windows(data, **analysis_options)
    results = {
        window_id: associate_component_fragments(
            analysis, **dict(association_options or {})
        )
        for window_id, analysis in analyses.items()
    }
    components = components_to_dataframe(analyses)
    candidates = candidates_to_dataframe(results)
    return metadata, results, components, candidates


def compare_structural_windows(
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
    window_ids: Iterable[int] | None = None,
    transform_options: Mapping[str, object] | None = None,
    msst_options: Mapping[str, object] | None = None,
    structure_options: Mapping[str, object] | None = None,
) -> tuple[pd.DataFrame, dict[int, StructuralComparison]]:
    """Compare STFT/MSST morphology on every identical quality-valid window."""
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
    selected_ids = None if window_ids is None else set(window_ids)
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
        if selected_ids is None or window_id in selected_ids
    }
    spectral = analyze_msst_periods(
        periods,
        sampling_frequency=sampling_frequency,
        **_resolve_transform_options(transform_options, msst_options),
    )
    comparisons = {
        window_id: compare_stft_msst_structure(
            item, **dict(structure_options or {})
        )
        for window_id, item in spectral.items()
    }
    for window_id, comparison in comparisons.items():
        for representation in ("stft", "msst"):
            for feature, value in comparison.metrics[representation].items():
                metadata.loc[window_id, f"{representation}_{feature}"] = value
    return metadata, comparisons


def plot_structural_window(analysis: StructuralSpectralResult):
    """Plot one representation, component IDs, and per-component features."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    spectral = analysis.spectral
    figure = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=(
            "Signal",
            f"{analysis.representation.upper()} raw",
            f"{analysis.representation.upper()} normalized",
            "Structure coherence (descriptive)",
            "Components and IDs",
            "Component features",
        ),
        horizontal_spacing=0.07,
        vertical_spacing=0.14,
        specs=[[{}, {}, {}], [{}, {}, {"type": "table"}]],
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
        (np.abs(analysis.raw_representation), "Viridis", 1, 2, None),
        (analysis.normalized_representation, "Magma", 1, 3, (0.0, 12.0)),
        (analysis.coherence, "Cividis", 2, 1, (0.0, 1.0)),
    )
    for values, colorscale, row, column, limits in maps:
        figure.add_trace(
            go.Heatmap(
                x=spectral.spectral_time,
                y=spectral.frequencies,
                z=values,
                colorscale=colorscale,
                zmin=None if limits is None else limits[0],
                zmax=None if limits is None else limits[1],
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
            z=np.abs(analysis.raw_representation),
            colorscale="Greys",
            showscale=False,
            name="MSST background",
        ),
        row=2,
        col=2,
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
        col=2,
    )
    if analysis.components:
        component_x = [
            spectral.spectral_time[0]
            + component.relative_time_centroid
            * (spectral.spectral_time[-1] - spectral.spectral_time[0])
            for component in analysis.components
        ]
        figure.add_trace(
            go.Scatter(
                x=component_x,
                y=[component.frequency_centroid for component in analysis.components],
                mode="markers+text",
                text=[str(component.label) for component in analysis.components],
                textposition="top center",
                marker={"size": 8, "color": "white", "line": {"color": "black"}},
                name="component ID",
                hovertemplate="component=%{text}<br>t=%{x:.4g}s<br>f=%{y:.4g}Hz<extra></extra>",
            ),
            row=2,
            col=2,
        )
    component_table = components_to_dataframe({0: analysis})
    table_columns = (
        "component_id",
        "frequency_centroid",
        "duration_seconds",
        "bandwidth_hz",
        "area_pixels",
        "mean_significance",
        "mean_coherence",
    )
    figure.add_trace(
        go.Table(
            header={"values": list(table_columns), "align": "left"},
            cells={
                "values": [component_table[column] for column in table_columns],
                "format": [None, ".4g", ".4g", ".4g", None, ".4g", ".4g"],
                "align": "left",
            },
        ),
        row=2,
        col=3,
    )
    for row, columns in ((1, (1, 2, 3)), (2, (1, 2))):
        for column in columns:
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
        title=(
            f"{analysis.representation.upper()} structure extraction "
            "(coherence is not an anomaly score)"
        ),
        height=850,
        template="plotly_white",
        hovermode="closest",
    )
    return figure


def plot_candidate_structures(result: CandidateStructureResult):
    """Compare original connected fragments with their associated candidates."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    analysis = result.structural
    spectral = analysis.spectral
    figure = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=(
            "Signal",
            f"{analysis.representation.upper()} raw",
            f"{analysis.representation.upper()} normalized",
            "Connected components",
            "Candidate structures",
            "Candidate features",
        ),
        specs=[[{}, {}, {}], [{}, {}, {"type": "table"}]],
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
    for values, colorscale, column, limits in (
        (np.abs(analysis.raw_representation), "Viridis", 2, None),
        (analysis.normalized_representation, "Magma", 3, (0.0, 12.0)),
    ):
        figure.add_trace(
            go.Heatmap(
                x=spectral.spectral_time,
                y=spectral.frequencies,
                z=values,
                colorscale=colorscale,
                zmin=None if limits is None else limits[0],
                zmax=None if limits is None else limits[1],
                showscale=False,
            ),
            row=1,
            col=column,
        )
    overlays = (
        (analysis.component_labels, analysis.components, "C", 1),
        (result.candidate_labels, result.candidates, "S", 2),
    )
    for label_map, objects, prefix, column in overlays:
        figure.add_trace(
            go.Heatmap(
                x=spectral.spectral_time,
                y=spectral.frequencies,
                z=np.abs(analysis.raw_representation),
                colorscale="Greys",
                showscale=False,
                hoverinfo="skip",
            ),
            row=2,
            col=column,
        )
        figure.add_trace(
            go.Heatmap(
                x=spectral.spectral_time,
                y=spectral.frequencies,
                z=np.where(label_map > 0, label_map, np.nan),
                colorscale="Rainbow",
                opacity=0.65,
                showscale=False,
            ),
            row=2,
            col=column,
        )
        if objects:
            x_values = [
                spectral.spectral_time[0]
                + item.relative_time_centroid
                * (spectral.spectral_time[-1] - spectral.spectral_time[0])
                for item in objects
            ]
            ids = [
                item.label if prefix == "C" else item.candidate_id for item in objects
            ]
            figure.add_trace(
                go.Scatter(
                    x=x_values,
                    y=[item.frequency_centroid for item in objects],
                    mode="markers+text",
                    text=[f"{prefix}{identifier}" for identifier in ids],
                    textposition="top center",
                    marker={
                        "size": 8,
                        "color": "white",
                        "line": {"color": "black"},
                    },
                    name=f"{prefix} ID",
                ),
                row=2,
                col=column,
            )
    candidate_table = candidates_to_dataframe({0: result})
    table_columns = (
        "candidate_id",
        "component_ids",
        "frequency_centroid",
        "duration_seconds",
        "active_duration_seconds",
        "gap_fraction",
        "linearity",
    )
    figure.add_trace(
        go.Table(
            header={"values": list(table_columns), "align": "left"},
            cells={
                "values": [candidate_table[column] for column in table_columns],
                "align": "left",
            },
        ),
        row=2,
        col=3,
    )
    for row, columns in ((1, (1, 2, 3)), (2, (1, 2))):
        for column in columns:
            is_signal = (row, column) == (1, 1)
            figure.update_xaxes(
                title_text="time" if is_signal else "time (s)", row=row, col=column
            )
            figure.update_yaxes(
                title_text="signal" if is_signal else "frequency (Hz)",
                row=row,
                col=column,
            )
    figure.update_layout(
        title="Connected fragments and candidate structures (no anomaly decision)",
        height=850,
        template="plotly_white",
        hovermode="closest",
    )
    return figure


def plot_stft_msst_comparison(comparison: StructuralComparison):
    """Plot nine aligned panels for direct STFT/MSST morphology inspection."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    stft_result, msst_result = comparison.stft, comparison.msst
    spectral = msst_result.spectral
    figure = make_subplots(
        rows=3,
        cols=3,
        subplot_titles=(
            "Signal",
            "STFT raw",
            "MSST raw",
            "STFT normalized",
            "MSST normalized",
            "STFT coherence",
            "MSST coherence",
            "STFT components",
            "MSST components",
        ),
        horizontal_spacing=0.06,
        vertical_spacing=0.1,
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
    heatmaps = (
        (np.abs(stft_result.raw_representation), "Viridis", 1, 2, None),
        (np.abs(msst_result.raw_representation), "Viridis", 1, 3, None),
        (stft_result.normalized_msst, "Magma", 2, 1, (0.0, 12.0)),
        (msst_result.normalized_msst, "Magma", 2, 2, (0.0, 12.0)),
        (stft_result.coherence, "Cividis", 2, 3, (0.0, 1.0)),
        (msst_result.coherence, "Cividis", 3, 1, (0.0, 1.0)),
    )
    for values, colorscale, row, column, limits in heatmaps:
        figure.add_trace(
            go.Heatmap(
                x=spectral.spectral_time,
                y=spectral.frequencies,
                z=values,
                colorscale=colorscale,
                zmin=None if limits is None else limits[0],
                zmax=None if limits is None else limits[1],
                showscale=False,
                hovertemplate="t=%{x:.4g}s<br>f=%{y:.4g}Hz<br>value=%{z:.4g}<extra></extra>",
            ),
            row=row,
            col=column,
        )
    for result, column in ((stft_result, 2), (msst_result, 3)):
        figure.add_trace(
            go.Heatmap(
                x=spectral.spectral_time,
                y=spectral.frequencies,
                z=np.abs(result.raw_representation),
                colorscale="Greys",
                showscale=False,
                hoverinfo="skip",
            ),
            row=3,
            col=column,
        )
        labels = np.where(
            result.significant_mask, result.component_labels, np.nan
        )
        figure.add_trace(
            go.Heatmap(
                x=spectral.spectral_time,
                y=spectral.frequencies,
                z=labels,
                colorscale="Rainbow",
                opacity=0.65,
                showscale=False,
                hovertemplate="t=%{x:.4g}s<br>f=%{y:.4g}Hz<br>component=%{z}<extra></extra>",
            ),
            row=3,
            col=column,
        )
    for row in range(1, 4):
        for column in range(1, 4):
            is_signal = (row, column) == (1, 1)
            figure.update_xaxes(
                title_text="time" if is_signal else "time (s)",
                row=row,
                col=column,
            )
            figure.update_yaxes(
                title_text="signal" if is_signal else "frequency (Hz)",
                row=row,
                col=column,
            )
    figure.update_layout(
        title="STFT versus MSST morphology",
        height=1150,
        template="plotly_white",
        hovermode="closest",
    )
    return figure
