"""Common STFT/MSST contract, retaining STFT PSD for physical measurements."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.signal import get_window
from .devices import DeviceSelection, resolve_device
from .msst import msst_stft, stft_only

@dataclass(frozen=True)
class SpectralResult:
    representation: str
    values: np.ndarray
    stft: np.ndarray
    psd: np.ndarray
    frequencies: np.ndarray
    times: np.ndarray
    device: DeviceSelection
    sampling_frequency: float

def _torch_stft(signal, fs, options, selection):
    import torch
    win_len = int(options.get("window_length", 128)); n_fft = int(options.get("n_fft", win_len)); hop = int(options.get("hop_length", 1))
    dtype = torch.float32 if options.get("dtype", "float32") == "float32" else torch.float64
    x = torch.as_tensor(signal, dtype=dtype, device=selection.resolved)
    window_name = options.get("window", "hann")
    if window_name not in {"hann", "hanning"}: raise ValueError("GPU STFT currently supports the Hann window")
    window = torch.hann_window(win_len, dtype=dtype, device=selection.resolved)
    z = torch.stft(x, n_fft=n_fft, hop_length=hop, win_length=win_len, window=window,
                   center=bool(options.get("center", True)), return_complex=True)
    return z.detach().cpu().numpy(), np.fft.rfftfreq(n_fft, 1/fs)

def analyze_spectrum(signal: np.ndarray, config: dict) -> SpectralResult:
    """Transform one regular signal; MSST is CPU-only, STFT has a real torch GPU path."""
    values = np.asarray(signal, dtype=float)
    fs = float(config["sampling_frequency"]); rep = config["representation"]; opts = dict(config["transform"])
    selection = resolve_device(config.get("device", "auto"), gpu_supported=rep == "stft")
    if selection.accelerated:
        coefficients, frequencies = _torch_stft(values, fs, opts, selection)
    elif rep == "msst":
        msst_opts = {**opts, **config.get("msst", {})}; msst_opts.pop("center", None)
        transformed, coefficients, frequencies = msst_stft(values, fs, **msst_opts)
    else:
        cpu_opts = {k:v for k,v in opts.items() if k in {"window","n_fft","window_length","hop_length","padtype","dtype"}}
        coefficients, frequencies = stft_only(values, fs, **cpu_opts)
    if rep == "stft": transformed = coefficients
    hop = int(opts.get("hop_length", 1)); times = np.arange(coefficients.shape[1]) * hop / fs
    # scipy/torch coefficient scaling differs; this density convention is explicit and stable within a backend.
    win_len = int(opts.get("window_length") or opts.get("n_fft") or min(128, len(values)))
    window = get_window(opts.get("window", "hann"), win_len)
    psd = np.abs(coefficients) ** 2 / (fs * np.sum(window ** 2))
    return SpectralResult(rep, transformed, coefficients, psd, frequencies, times, selection, fs)

GEOMETRIC_FEATURES = frozenset({"time_frequency_area", "duration", "frequency_width", "temporal_variation", "frequency_variation"})
PHYSICAL_STFT_FEATURES = frozenset({"integrated_energy", "mean_energy_density", "central_frequency", "frequency_dispersion", "local_energy_contrast"})

def validate_features(features, representation):
    unknown = set(features) - GEOMETRIC_FEATURES - PHYSICAL_STFT_FEATURES
    if unknown: raise ValueError(f"unknown feature(s): {sorted(unknown)}")
    # Physical features remain available with MSST because SpectralResult always carries aligned STFT PSD.
    return tuple(features)
