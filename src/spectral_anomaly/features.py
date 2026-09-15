"""Physical and power-based descriptors extracted from final binary masks."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation

from .spectral import SpectralResult


SEGMENT_FEATURE_COLUMNS = (
    "segment_id",
    "time_frequency_area",
    "duration",
    "frequency_width",
    "central_frequency",
    "frequency_dispersion",
    "integrated_spectral_power",
    "mean_spectral_power_density",
    "temporal_variation",
    "frequency_variation",
    "local_energy_contrast",
)

FEATURE_MEANING = {
    "time_frequency_area": "geometry_seconds_hertz",
    "duration": "geometry_seconds",
    "frequency_width": "geometry_hertz",
    "central_frequency": "stft_psd_weighted_hertz",
    "frequency_dispersion": "stft_psd_weighted_hertz",
    "integrated_spectral_power": "integral_of_stft_psd_over_mask",
    "mean_spectral_power_density": "mean_stft_psd_inside_mask",
    "temporal_variation": "std_of_masked_power_profile_over_time",
    "frequency_variation": "std_of_masked_power_profile_over_frequency",
    "local_energy_contrast": "mean_mask_psd_over_median_local_ring_psd",
    # Compatibility aliases retained for downstream configurations.
    "integrated_energy": "alias_of_integrated_spectral_power",
    "mean_energy_density": "alias_of_mean_spectral_power_density",
}


def _axis_step(axis: np.ndarray, fallback: float) -> float:
    return float(np.median(np.diff(axis))) if len(axis) > 1 else fallback


def extract_segment_features(
    segment,
    spectral: SpectralResult,
    *,
    window_id=0,
    window_start=None,
):
    """Calculate descriptors from the STFT PSD selected by a SAM mask.

    The uint8 SAM image is deliberately unavailable to this function. SAM
    defines the support only; every power-weighted descriptor uses
    ``spectral.psd`` from the original complex STFT.
    """
    mask = np.asarray(segment.mask, dtype=bool)
    if mask.shape != spectral.stft.shape or mask.shape != spectral.psd.shape:
        raise ValueError("mask, STFT, and PSD shapes must agree")
    rows, columns = np.nonzero(mask)
    base = {
        "window_id": window_id,
        "window_start": window_start,
        "segment_id": segment.segment_id,
        "source_segment_ids": segment.source_segment_ids,
        "merge_count": segment.merge_count,
    }
    names = SEGMENT_FEATURE_COLUMNS[1:]
    if not len(rows):
        empty = {name: 0.0 for name in names}
        return {**base, **empty, "integrated_energy": 0.0, "mean_energy_density": 0.0}

    dt = _axis_step(spectral.times, 1 / spectral.sampling_frequency)
    df = _axis_step(spectral.frequencies, spectral.sampling_frequency / 2)
    pixel_area = dt * df
    masked_psd = np.asarray(spectral.psd[mask], dtype=float)
    integrated_power = float(masked_psd.sum() * pixel_area)
    mean_density = float(masked_psd.mean())

    frequencies = spectral.frequencies[rows]
    weights = np.maximum(masked_psd, 0)
    total_weight = float(weights.sum())
    central_frequency = (
        float(np.average(frequencies, weights=weights))
        if total_weight > 0
        else float(frequencies.mean())
    )
    frequency_dispersion = (
        float(np.sqrt(np.average((frequencies - central_frequency) ** 2, weights=weights)))
        if total_weight > 0
        else float(frequencies.std())
    )

    masked_map = np.where(mask, spectral.psd, 0.0)
    time_slice = slice(int(columns.min()), int(columns.max()) + 1)
    frequency_slice = slice(int(rows.min()), int(rows.max()) + 1)
    temporal_profile = masked_map[:, time_slice].sum(axis=0) * df
    frequency_profile = masked_map[frequency_slice, :].sum(axis=1) * dt

    local_ring = binary_dilation(mask, iterations=1) & ~mask
    local_background = np.asarray(spectral.psd[local_ring], dtype=float)
    background = float(np.median(local_background)) if local_background.size else 0.0
    contrast = mean_density / max(background, np.finfo(float).tiny)

    features = {
        "time_frequency_area": float(mask.sum() * pixel_area),
        # Pixel support convention: an occupied bin represents one full dt/df cell.
        "duration": float((columns.max() - columns.min() + 1) * dt),
        "frequency_width": float((rows.max() - rows.min() + 1) * df),
        "central_frequency": central_frequency,
        "frequency_dispersion": frequency_dispersion,
        "integrated_spectral_power": integrated_power,
        "mean_spectral_power_density": mean_density,
        "temporal_variation": float(np.std(temporal_profile)),
        "frequency_variation": float(np.std(frequency_profile)),
        "local_energy_contrast": float(contrast),
    }
    return {
        **base,
        **features,
        "integrated_energy": integrated_power,
        "mean_energy_density": mean_density,
    }


def segments_to_dataframe(segments, spectral, **metadata):
    """Return one feature row per supplied segment."""
    return pd.DataFrame([
        extract_segment_features(segment, spectral, **metadata)
        for segment in segments
    ])
