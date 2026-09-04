"""First-stage spectral anomaly detection."""

from .energy import WindowData, detect_energy_anomalies, plot_window

__all__ = ["WindowData", "detect_energy_anomalies", "plot_window"]
