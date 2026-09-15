"""Device-independent STFT geometry and aligned MSST/PSD products.

Framing and padding are owned by this module rather than delegated to FFT
libraries.  Consequently NumPy and Torch receive exactly the same windowed
frames and differ only in their FFT implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from scipy.signal import get_window

from .devices import DeviceSelection, resolve_device
from .msst import multisynchrosqueeze_stft


@dataclass(frozen=True)
class SpectralResult:
    """Aligned products for one signal window.

    ``values`` is the representation intended for geometry. ``stft`` is always
    the original STFT and ``psd`` is always derived from it, including in MSST
    mode. All three arrays share the frequency/time grid exactly.
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
_TRANSFORM_KEYS = {"window", "window_length", "n_fft", "hop_length", "center", "padtype", "dtype"}


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
    selection = resolve_device(config.get("device", "auto"), gpu_supported=representation == "stft")
    framed = _frames(processed.astype(plan.dtype, copy=False), plan)
    coefficients = (_gpu_stft(framed, plan, selection.resolved)
                    if selection.accelerated else _cpu_stft(framed, plan))
    psd = stft_psd(coefficients, plan, config["psd"])

    if representation == "msst":
        msst_options = config["msst"]
        if set(msst_options) != {"iteration_count", "gamma"}:
            raise ValueError("msst must contain only iteration_count and gamma")
        transformed = multisynchrosqueeze_stft(
            coefficients, plan.frequencies, plan.sampling_frequency, plan.hop_length,
            iteration_count=msst_options["iteration_count"], gamma=msst_options["gamma"]
        )
    elif representation == "stft":
        transformed = coefficients
    else:
        raise ValueError("representation must be 'stft' or 'msst'")
    if transformed.shape != coefficients.shape or psd.shape != coefficients.shape:
        raise RuntimeError("STFT, MSST, and PSD grids are not aligned")
    return SpectralResult(representation, transformed, coefficients, psd,
                          plan.frequencies, plan.times, selection,
                          plan.sampling_frequency, processed)


GEOMETRIC_FEATURES = frozenset({"time_frequency_area", "duration", "frequency_width", "temporal_variation", "frequency_variation"})
PHYSICAL_STFT_FEATURES = frozenset({"integrated_energy", "mean_energy_density", "central_frequency", "frequency_dispersion", "local_energy_contrast"})


def validate_features(features, representation):
    unknown = set(features) - GEOMETRIC_FEATURES - PHYSICAL_STFT_FEATURES
    if unknown:
        raise ValueError(f"unknown feature(s): {sorted(unknown)}")
    return tuple(features)
