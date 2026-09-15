"""Capability-aware device and backend selection."""
from __future__ import annotations

from dataclasses import dataclass
import importlib.util


@dataclass(frozen=True)
class DeviceSelection:
    requested: str
    resolved: str
    accelerated: bool
    reason: str


def cuda_available() -> bool:
    if importlib.util.find_spec("torch") is None:
        return False
    import torch
    return bool(torch.cuda.is_available())


def resolve_device(device: str = "auto", *, gpu_supported: bool = True) -> DeviceSelection:
    if device not in {"auto", "cpu", "cuda"} and not (
        isinstance(device, str) and device.startswith("cuda:") and device[5:].isdigit()
    ):
        raise ValueError("device must be 'auto', 'cpu', 'cuda', or 'cuda:N'")
    available = gpu_supported and cuda_available()
    if device == "cpu":
        return DeviceSelection(device, "cpu", False, "CPU explicitly selected")
    if device == "auto":
        return DeviceSelection(device, "cuda" if available else "cpu", available,
                               "compatible CUDA implementation available" if available else
                               "CUDA unavailable or unsupported by this implementation")
    if not gpu_supported:
        raise RuntimeError("the selected implementation has no genuine GPU backend")
    if not cuda_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return DeviceSelection(device, device, True, "CUDA explicitly selected")


def resolve_backend(backend: str, *, gpu_package: str = "cuml") -> str:
    if backend not in {"auto", "cpu", "gpu"}:
        raise ValueError("backend must be 'auto', 'cpu', or 'gpu'")
    has_gpu = importlib.util.find_spec(gpu_package) is not None and cuda_available()
    if backend == "auto":
        return "gpu" if has_gpu else "cpu"
    if backend == "gpu" and not has_gpu:
        raise RuntimeError(f"GPU backend requires CUDA and {gpu_package}")
    return backend
