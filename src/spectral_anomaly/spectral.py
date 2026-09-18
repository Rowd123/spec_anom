"""Device-independent STFT geometry and aligned MSST/PSD products.

Framing and padding are owned by this module rather than delegated to FFT
libraries.  Consequently NumPy and Torch receive exactly the same windowed
frames and differ only in their FFT implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
import importlib.util
import json
import hashlib
import logging
import os
from time import perf_counter
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.signal import get_window

from .devices import DeviceSelection, resolve_device
from .energy import WindowData, prepare_analysis_windows
from .msst import msst_stft
from .exclusions import excluded_observations

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SpectralResult:
    """Aligned products for one signal window.

    ``values`` is the representation intended for geometry. ``stft`` is always
    the original STFT and ``psd`` is always derived from it, including in MSST
    mode. values contains raw complex STFT coefficients, ssqueezepy SST
    coefficients (configured squeezing), or MSST reassigned coefficients
    (including its frequency-step factor), never the normalized SAM image.
    All three arrays share the frequency/time grid exactly.
    """

    representation: str
    values: np.ndarray
    stft: np.ndarray
    psd: np.ndarray
    frequencies: np.ndarray
    times: np.ndarray
    device: DeviceSelection
    sampling_frequency: float
    processed_signal: np.ndarray


@dataclass(frozen=True)
class DataFrameSpectralWindow:
    """One quality-prepared DataFrame window and its absolute spectral axis."""

    window_id: int
    window: WindowData
    spectral: SpectralResult
    absolute_times: pd.Index


@dataclass(frozen=True)
class _TransformPlan:
    sampling_frequency: float
    window: np.ndarray
    window_length: int
    n_fft: int
    hop_length: int
    center: bool
    padtype: str
    dtype: np.dtype
    frame_starts: np.ndarray
    times: np.ndarray
    frequencies: np.ndarray


_PAD_MODES = {"reflect": "reflect", "constant": "constant", "zero": "constant", "edge": "edge"}
_TRANSFORM_KEYS = {"window", "window_length", "n_fft", "hop_length", "center", "padtype", "dtype", "frequency_min", "frequency_max"}
_FROM_CONFIG = object()


def _build_plan(signal_size: int, sampling_frequency: float, options: Mapping[str, Any]) -> _TransformPlan:
    unknown = set(options) - _TRANSFORM_KEYS
    if unknown:
        raise ValueError(f"unknown transform option(s): {sorted(unknown)}")
    window_length = int(options["window_length"])
    n_fft = int(options["n_fft"])
    hop_length = int(options["hop_length"])
    center = options["center"]
    padtype = options["padtype"]
    dtype_name = options["dtype"]
    if window_length < 2 or window_length > signal_size:
        raise ValueError("window_length must be between 2 and the signal length")
    if n_fft < window_length:
        raise ValueError("n_fft must be greater than or equal to window_length")
    if hop_length < 1:
        raise ValueError("hop_length must be positive")
    if not isinstance(center, bool):
        raise ValueError("transform.center must be a boolean")
    if padtype not in _PAD_MODES:
        raise ValueError(f"padtype must be one of {sorted(_PAD_MODES)}")
    if dtype_name not in {"float32", "float64"}:
        raise ValueError("dtype must be 'float32' or 'float64'")

    dtype = np.dtype(dtype_name)
    window = np.asarray(get_window(options["window"], window_length, fftbins=True), dtype=dtype)
    if not np.any(window):
        raise ValueError("the Fourier window must have non-zero energy")

    if center:
        # Centers are real input sample positions 0, hop, ... strictly below N.
        frame_starts = np.arange(0, signal_size, hop_length, dtype=int)
        times = frame_starts.astype(float) / sampling_frequency
    else:
        frame_starts = np.arange(0, signal_size - window_length + 1, hop_length, dtype=int)
        # For an even window its geometric center lies between two samples.
        times = (frame_starts + (window_length - 1) / 2) / sampling_frequency
    frequencies = np.fft.rfftfreq(n_fft, d=1 / sampling_frequency)
    return _TransformPlan(sampling_frequency, window, window_length, n_fft, hop_length,
                          center, str(padtype), dtype, frame_starts, times, frequencies)


def spectral_signature(config: Mapping[str, Any]) -> str:
    """Fingerprint every option that changes spectra or the study population."""
    selected = {name: config[name] for name in (
        "sampling_frequency", "windowing", "signal_preprocessing", "representation",
        "transform", "ssq_stft", "msst", "psd", "exclusions",
    )}
    payload = json.dumps(selected, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def _ssq_gpu_available() -> bool:
    return (importlib.util.find_spec("torch") is not None and
            importlib.util.find_spec("cupy") is not None)


@contextmanager
def _ssq_device(selection: DeviceSelection):
    """Set ssqueezepy's documented process flag for exactly one call."""
    previous = os.environ.get("SSQ_GPU")
    os.environ["SSQ_GPU"] = "1" if selection.accelerated else "0"
    context = None
    if selection.accelerated and ":" in selection.resolved:
        import torch
        context = torch.cuda.device(selection.resolved)
        context.__enter__()
    try:
        yield
    finally:
        if context is not None: context.__exit__(None, None, None)
        if previous is None: os.environ.pop("SSQ_GPU", None)
        else: os.environ["SSQ_GPU"] = previous


def _ssq_transform(signal: np.ndarray, plan: _TransformPlan, options: Mapping[str, Any],
                   selection: DeviceSelection) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from ssqueezepy import ssq_stft
    if not plan.center:
        raise ValueError("ssqueezepy ssq_stft uses centered padding; transform.center must be true")
    started = perf_counter()
    with _ssq_device(selection):
        transformed, coefficients, frequencies, _ = ssq_stft(
            signal, window=plan.window, n_fft=plan.n_fft, win_len=plan.window_length,
            hop_len=plan.hop_length, fs=plan.sampling_frequency,
            padtype=_PAD_MODES[plan.padtype], modulated=True,
            squeezing=options["squeezing"], gamma=options["gamma"],
            dtype=plan.dtype.name, astensor=False,
        )
    LOGGER.info("ssq_stft device=%s dtype=%s duration=%.3fs", selection.resolved,
                plan.dtype.name, perf_counter() - started)
    return np.asarray(transformed), np.asarray(coefficients), np.asarray(frequencies)


def analyze_spectra(signals, config: Mapping[str, Any]) -> list[SpectralResult]:
    """Analyze equal-sized windows, using ssqueezepy's genuine batch API for SST."""
    values = [np.asarray(signal) for signal in signals]
    if not values: return []
    if config["representation"] != "ssq_stft":
        return [analyze_spectrum(signal, config) for signal in values]
    if any(signal.ndim != 1 or len(signal) != len(values[0]) or
           not np.all(np.isfinite(signal)) for signal in values):
        raise ValueError("spectral batches require equal-sized finite 1-D signals")
    options = config["signal_preprocessing"]
    processed = np.stack([signal.astype(float, copy=True) for signal in values])
    if options["remove_mean"]: processed -= processed.mean(axis=1, keepdims=True)
    plan = _build_plan(processed.shape[1], float(config["sampling_frequency"]), config["transform"])
    selection = resolve_device(config.get("device", "auto"), gpu_supported=_ssq_gpu_available())
    batch_size = config["ssq_stft"]["batch_size"]
    output = []
    for start in range(0, len(processed), batch_size):
        batch = processed[start:start + batch_size].astype(plan.dtype, copy=False)
        transformed_batch, coefficient_batch, frequencies = _ssq_transform(
            batch, plan, config["ssq_stft"], selection
        )
        if transformed_batch.ndim == 2:
            transformed_batch, coefficient_batch = transformed_batch[None], coefficient_batch[None]
        for offset, (transformed, coefficients) in enumerate(zip(transformed_batch, coefficient_batch)):
            psd = stft_psd(coefficients, plan, config["psd"])
            low = float(config["transform"]["frequency_min"])
            high_value = config["transform"]["frequency_max"]
            high = plan.sampling_frequency / 2 if high_value is None else float(high_value)
            selected = (frequencies >= low) & (frequencies <= high)
            output.append(SpectralResult(
                "ssq_stft", transformed[selected], coefficients[selected], psd[selected],
                frequencies[selected], plan.times, selection, plan.sampling_frequency,
                processed[start + offset],
            ))
        if selection.accelerated:
            import torch
            LOGGER.info("ssq_stft batch=%d cuda_allocated=%d cuda_reserved=%d",
                        len(batch), torch.cuda.memory_allocated(selection.resolved),
                        torch.cuda.memory_reserved(selection.resolved))
    return output


def _frames(signal: np.ndarray, plan: _TransformPlan) -> np.ndarray:
    if plan.center:
        left = plan.window_length // 2
        right = plan.window_length - 1 - left
        padded = np.pad(signal, (left, right), mode=_PAD_MODES[plan.padtype])
        starts = plan.frame_starts
    else:
        padded = signal
        starts = plan.frame_starts
    offsets = np.arange(plan.window_length)
    return padded[starts[:, None] + offsets[None, :]] * plan.window[None, :]


def _cpu_stft(frames: np.ndarray, plan: _TransformPlan) -> np.ndarray:
    dtype = np.complex64 if plan.dtype == np.dtype("float32") else np.complex128
    return np.fft.rfft(frames, n=plan.n_fft, axis=1).T.astype(dtype, copy=False)


def _gpu_stft(frames: np.ndarray, plan: _TransformPlan, device: str) -> np.ndarray:
    import torch

    torch_dtype = torch.float32 if plan.dtype == np.dtype("float32") else torch.float64
    tensor = torch.as_tensor(frames, dtype=torch_dtype, device=device)
    coefficients = torch.fft.rfft(tensor, n=plan.n_fft, dim=1).transpose(0, 1)
    return coefficients.detach().cpu().numpy()


def stft_psd(coefficients: np.ndarray, plan: _TransformPlan, config: Mapping[str, Any]) -> np.ndarray:
    """Return a one-sided PSD in signal-unit²/Hz from raw, unnormalised FFTs.

    DC and Nyquist (when present) retain their power. Strictly positive interior
    bins are doubled to account for the omitted negative-frequency half.
    """
    unknown = set(config) - {"scaling", "one_sided"}
    if unknown:
        raise ValueError(f"unknown PSD option(s): {sorted(unknown)}")
    if config.get("scaling") != "density":
        raise ValueError("only psd.scaling='density' is supported")
    if config.get("one_sided") is not True:
        raise ValueError("real-signal STFT currently requires psd.one_sided=true")
    psd = np.abs(coefficients) ** 2 / (
        plan.sampling_frequency * np.sum(np.square(plan.window), dtype=float)
    )
    if plan.n_fft % 2 == 0:
        psd[1:-1] *= 2
    else:
        psd[1:] *= 2
    return psd


def analyze_spectrum(signal: np.ndarray, config: Mapping[str, Any]) -> SpectralResult:
    """Compute an aligned STFT or MSST and a physical STFT PSD."""
    raw = np.asarray(signal)
    if raw.ndim != 1 or raw.size < 2 or not np.all(np.isfinite(raw)):
        raise ValueError("signal must be a one-dimensional finite array with at least two samples")
    preprocessing = config["signal_preprocessing"]
    if set(preprocessing) != {"remove_mean"} or not isinstance(preprocessing["remove_mean"], bool):
        raise ValueError("signal_preprocessing must contain only boolean remove_mean")
    processed = raw.astype(float, copy=True)
    if preprocessing["remove_mean"]:
        processed -= processed.mean()

    representation = config["representation"]
    plan = _build_plan(len(processed), float(config["sampling_frequency"]), config["transform"])
    gpu_supported = representation == "stft" or (representation == "ssq_stft" and _ssq_gpu_available())
    selection = resolve_device(config.get("device", "auto"), gpu_supported=gpu_supported)
    if representation == "ssq_stft":
        transformed, coefficients, ssq_frequencies = _ssq_transform(
            processed.astype(plan.dtype, copy=False), plan, config["ssq_stft"], selection
        )
        if coefficients.shape != transformed.shape:
            raise RuntimeError("ssqueezepy STFT and SST grids are not aligned")
        plan_frequencies = ssq_frequencies
        if coefficients.shape[1] != len(plan.times):
            raise RuntimeError("ssqueezepy time grid differs from configured framing")
    elif representation == "msst":
        if not plan.center:
            raise ValueError("MSST uses ssqueezepy centered padding; transform.center must be true")
        msst_options = config["msst"]
        transformed, coefficients, plan_frequencies = msst_stft(
            processed.astype(plan.dtype, copy=False), plan.sampling_frequency,
            iteration_count=msst_options["iteration_count"], window=plan.window,
            n_fft=plan.n_fft, window_length=plan.window_length,
            hop_length=plan.hop_length, padtype=_PAD_MODES[plan.padtype],
            dtype=plan.dtype.name, gamma=msst_options["gamma"],
        )
        if coefficients.shape[1] != len(plan.times):
            raise RuntimeError("MSST time grid differs from configured framing")
    else:
        framed = _frames(processed.astype(plan.dtype, copy=False), plan)
        coefficients = (_gpu_stft(framed, plan, selection.resolved)
                        if selection.accelerated else _cpu_stft(framed, plan))
        plan_frequencies = plan.frequencies
    psd = stft_psd(coefficients, plan, config["psd"])

    if representation == "stft":
        transformed = coefficients
    elif representation in {"ssq_stft", "msst"}:
        pass
    else:
        raise ValueError("representation must be 'stft', 'ssq_stft', or 'msst'")
    if transformed.shape != coefficients.shape or psd.shape != coefficients.shape:
        raise RuntimeError("STFT, MSST, and PSD grids are not aligned")
    low = float(config["transform"]["frequency_min"])
    configured_high = config["transform"]["frequency_max"]
    high = plan.sampling_frequency / 2 if configured_high is None else float(configured_high)
    selected = (plan_frequencies >= low) & (plan_frequencies <= high)
    if not selected.any(): raise ValueError("configured frequency range contains no bins")
    transformed, coefficients, psd = transformed[selected], coefficients[selected], psd[selected]
    return SpectralResult(representation, transformed, coefficients, psd,
                          plan_frequencies[selected], plan.times, selection,
                          plan.sampling_frequency, processed)


def _absolute_spectral_times(
    window_time: pd.Index, offsets: np.ndarray, *, name=None
) -> pd.Index:
    """Apply frame-center offsets in seconds to the original time coordinate."""
    if isinstance(window_time, (pd.DatetimeIndex, pd.TimedeltaIndex)):
        return pd.Index(window_time[0] + pd.to_timedelta(offsets, unit="s"),
                        name=name if name is not None else window_time.name)
    return pd.Index(float(window_time[0]) + np.asarray(offsets, dtype=float),
                    name=name if name is not None else window_time.name)


def analyze_dataframe_windows(
    data: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    value_col,
    quality_col=_FROM_CONFIG,
    valid_quality_flags=_FROM_CONFIG,
    source_id=None,
) -> tuple[pd.DataFrame, dict[int, DataFrameSpectralWindow]]:
    """Clean, window and transform a timestamped DataFrame.

    Invalid quality flags and missing timestamps are handled by
    :func:`prepare_analysis_windows`. The relative STFT offsets remain in
    ``item.spectral.times`` for physical durations, while ``absolute_times``
    retains the coordinate system and name of the supplied DataFrame index.
    Omitted quality arguments use the config; explicitly passing
    ``quality_col=None`` disables quality-flag filtering.
    """
    windowing = config["windowing"]
    quality = config["quality"]
    selected_quality_col = (
        quality.get("quality_column") if quality_col is _FROM_CONFIG else quality_col
    )
    selected_flags = (
        quality.get("valid_flags")
        if valid_quality_flags is _FROM_CONFIG else valid_quality_flags
    )
    sampling_period = config["sampling_period"]
    excluded = excluded_observations(data, config["exclusions"], source_id=source_id)
    excluded_times = data.index[excluded]
    if pd.api.types.is_numeric_dtype(data.index.dtype) and not isinstance(sampling_period, (int, float)):
        sampling_period = pd.to_timedelta(sampling_period).total_seconds()
    metadata, prepared = prepare_analysis_windows(
        data,
        value_col=value_col,
        quality_col=selected_quality_col,
        valid_quality_flags=selected_flags,
        sampling_period=sampling_period,
        window_size=windowing["size"],
        overlap=windowing["overlap"],
        min_valid_ratio=quality["min_valid_fraction"],
        max_interpolation_gap=quality["max_interpolation_gap"],
        excluded_times=excluded_times,
    )
    analyses = {}
    for window_id, window in prepared.items():
        spectral = analyze_spectrum(window.signal, config)
        analyses[window_id] = DataFrameSpectralWindow(
            window_id=window_id,
            window=window,
            spectral=spectral,
            absolute_times=_absolute_spectral_times(
                window.time, spectral.times, name=data.index.name
            ),
        )
    return metadata, analyses


GEOMETRIC_FEATURES = frozenset({"time_frequency_area", "duration", "frequency_width", "temporal_variation", "frequency_variation"})
PHYSICAL_STFT_FEATURES = frozenset({"integrated_energy", "mean_energy_density", "central_frequency", "frequency_dispersion", "local_energy_contrast"})


def validate_features(features, representation):
    from .features import FEATURE_MEANING
    unknown = set(features) - FEATURE_MEANING.keys()
    if unknown:
        raise ValueError(f"unknown feature(s): {sorted(unknown)}")
    return tuple(features)
