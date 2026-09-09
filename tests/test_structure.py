import numpy as np
import pandas as pd
import pytest

from spectral_anomaly import (
    AnalysisPeriod,
    MSSTResult,
    analyze_structural_windows,
    compare_stft_msst_structure,
    compare_structural_windows,
    extract_msst_structure,
    plot_structural_window,
    plot_stft_msst_comparison,
    prepare_analysis_windows,
)
from spectral_anomaly import structure as structure_module


def make_spectral_result(msst: np.ndarray) -> MSSTResult:
    frequency_count, time_count = msst.shape
    period = AnalysisPeriod(
        time=pd.RangeIndex(time_count),
        signal=np.zeros(time_count),
        observed_mask=np.ones(time_count, dtype=bool),
        interpolated_mask=np.zeros(time_count, dtype=bool),
        anomaly_mask=np.zeros(time_count, dtype=bool),
        source_window_ids=(),
    )
    return MSSTResult(
        period=period,
        processed_signal=period.signal,
        msst=msst.astype(complex),
        stft=msst.astype(complex),
        frequencies=np.arange(frequency_count, dtype=float),
        spectral_time=np.arange(time_count, dtype=float),
    )


def synthetic_signals(sample_count: int = 256) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(42)
    time = np.arange(sample_count) / sample_count
    noise = rng.normal(0, 1, sample_count)
    return {
        "noise": noise,
        "weak_tone": noise + 0.25 * np.sin(2 * np.pi * 20 * time),
        "strong_tone": noise + 3.0 * np.sin(2 * np.pi * 20 * time),
        "linear_chirp": noise + np.sin(2 * np.pi * (8 * time + 35 * time**2)),
        "polynomial_chirp": noise
        + np.sin(2 * np.pi * (8 * time + 25 * time**3)),
        "impulse": noise + 8.0 * (np.abs(time - 0.5) < 0.01),
        "high_energy_noise": 3.0 * rng.normal(0, 1, sample_count),
        "equal_energy_weak_structure": (
            noise + 0.35 * np.sin(2 * np.pi * 20 * time)
        )
        / np.std(noise + 0.35 * np.sin(2 * np.pi * 20 * time)),
        "multiple_structures": noise
        + np.sin(2 * np.pi * 12 * time)
        + np.sin(2 * np.pi * (25 * time + 20 * time**2)),
    }


@pytest.mark.parametrize("name", synthetic_signals())
def test_required_synthetic_signal_cases_are_finite(name):
    signal = synthetic_signals()[name]

    assert signal.shape == (256,)
    assert np.all(np.isfinite(signal))


def test_all_required_synthetic_signals_pass_through_the_real_pipeline():
    cases = synthetic_signals()
    values = np.concatenate(list(cases.values()))
    data = pd.DataFrame({"value": values}, index=np.arange(len(values)) / 256)

    metadata, comparisons = compare_structural_windows(
        data,
        value_col="value",
        sampling_frequency=256.0,
        sampling_period=1 / 256,
        window_size=256,
        overlap=0,
        msst_options={
            "iteration_count": 1,
            "window_length": 32,
            "n_fft": 64,
            "hop_length": 4,
        },
    )

    assert metadata["accepted"].all()
    assert len(comparisons) == len(cases)
    for case_name, comparison in zip(cases, comparisons.values()):
        print(f"\n{case_name}\n{comparison.metrics.to_string()}")
        for analysis in (comparison.stft, comparison.msst):
            assert analysis.raw_representation.shape == analysis.normalized_msst.shape
            assert analysis.normalized_msst.shape == analysis.coherence.shape
            assert np.all(np.isfinite(list(analysis.features.values())))
        assert set(comparison.metrics.columns) == {
            "stft",
            "msst",
            "msst_minus_stft",
        }


def test_quality_preparation_does_not_use_energy_to_select_windows():
    values = np.r_[np.tile([-1.0, 1.0], 8), np.tile([-10.0, 10.0], 8)]
    data = pd.DataFrame({"value": values}, index=np.arange(len(values)))

    metadata, windows = prepare_analysis_windows(
        data, value_col="value", window_size=16, overlap=0
    )

    assert metadata["accepted"].tolist() == [True, True]
    assert set(windows) == {0, 1}


def test_high_energy_diffuse_noise_does_not_beat_a_weak_coherent_structure():
    rng = np.random.default_rng(2)
    diffuse_high_energy = np.abs(rng.normal(0, 3, (64, 128)))
    weak_coherent = np.abs(rng.normal(0, 1, (64, 128)))
    weak_coherent[28:31, :] += 2.5
    options = {
        "normalization_neighborhood": (9, 9),
        "significance_threshold": 2.0,
        "coherence_threshold": 0.35,
        "minimum_component_area": 8,
    }

    noise_result = extract_msst_structure(
        make_spectral_result(diffuse_high_energy), **options
    )
    structure_result = extract_msst_structure(
        make_spectral_result(weak_coherent), **options
    )

    assert diffuse_high_energy.mean() > weak_coherent.mean()
    assert noise_result.features["component_count"] == 0
    assert structure_result.features["component_count"] > 0
    assert (
        structure_result.features["weighted_mean_coherence"]
        > noise_result.features["weighted_mean_coherence"]
    )


def test_pipeline_analyzes_every_quality_valid_window_and_plots_six_panels(
    monkeypatch,
):
    data = pd.DataFrame(
        {"value": synthetic_signals(64)["multiple_structures"]},
        index=np.arange(64),
    )

    def fake_analyze(periods, **options):
        return {
            window_id: make_spectral_result(np.ones((16, 16)))
            for window_id in periods
        }

    monkeypatch.setattr(structure_module, "analyze_msst_periods", fake_analyze)
    metadata, analyses = analyze_structural_windows(
        data,
        value_col="value",
        sampling_frequency=1.0,
        window_size=32,
        overlap=0,
        structure_options={"significance_threshold": 0.0},
    )

    assert metadata["accepted"].tolist() == [True, True]
    assert set(analyses) == {0, 1}
    figure = plot_structural_window(analyses[0])
    assert len(figure.layout.annotations) == 6
    assert len(figure.data) == 7


def test_stft_representation_skips_msst_analysis(monkeypatch):
    data = pd.DataFrame({"value": np.arange(32.0)}, index=np.arange(32))

    def fail_if_called(*args, **kwargs):
        raise AssertionError("MSST must not run for representation='stft'")

    def fake_stft(periods, **options):
        return {
            window_id: make_spectral_result(np.ones((8, 8)))
            for window_id in periods
        }

    monkeypatch.setattr(structure_module, "analyze_msst_periods", fail_if_called)
    monkeypatch.setattr(structure_module, "analyze_stft_periods", fake_stft)

    _, analyses = analyze_structural_windows(
        data,
        value_col="value",
        sampling_frequency=1.0,
        window_size=32,
        representation="stft",
        structure_options={"significance_threshold": 0.0},
    )

    assert analyses[0].representation == "stft"


def test_comparison_exposes_fragmentation_metrics_and_nine_panels():
    raw = np.zeros((24, 32))
    raw[8, 2:12] = 5
    raw[15, 18:30] = 5
    spectral = make_spectral_result(raw)

    comparison = compare_stft_msst_structure(
        spectral,
        significance_threshold=1.0,
        coherence_threshold=0.2,
        minimum_component_area=2,
        small_component_area=8,
    )

    expected = {
        "component_count",
        "coherent_pixel_fraction",
        "weighted_mean_coherence",
        "coherence_q90",
        "largest_component_area_fraction",
        "largest_component_significance_fraction",
        "median_component_area",
        "max_component_area",
        "small_component_fraction",
    }
    assert expected.issubset(comparison.metrics.index)
    figure = plot_stft_msst_comparison(comparison)
    assert len(figure.layout.annotations) == 9
    assert len(figure.data) == 11
