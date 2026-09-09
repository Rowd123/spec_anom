"""First-stage spectral anomaly detection."""

from .energy import (
    WindowData,
    detect_energy_anomalies,
    prepare_analysis_windows,
    plot_suspicious_windows,
    plot_window,
)
from .msst import (
    AnalysisPeriod,
    MSSTResult,
    analyze_msst_monitoring_data,
    analyze_msst_monitoring_period,
    analyze_msst_periods,
    msst_stft,
    plot_msst_periods,
    prepare_analysis_periods,
    prepare_monitoring_period,
)
from .structure import (
    SpectralComponent,
    StructuralMSSTResult,
    analyze_structural_windows,
    compute_structure_tensor,
    extract_msst_structure,
    normalize_msst_local,
    plot_structural_window,
)

__all__ = [
    "WindowData",
    "detect_energy_anomalies",
    "prepare_analysis_windows",
    "plot_suspicious_windows",
    "plot_window",
    "AnalysisPeriod",
    "MSSTResult",
    "analyze_msst_monitoring_data",
    "analyze_msst_monitoring_period",
    "analyze_msst_periods",
    "msst_stft",
    "plot_msst_periods",
    "prepare_analysis_periods",
    "prepare_monitoring_period",
    "SpectralComponent",
    "StructuralMSSTResult",
    "analyze_structural_windows",
    "compute_structure_tensor",
    "extract_msst_structure",
    "normalize_msst_local",
    "plot_structural_window",
]
