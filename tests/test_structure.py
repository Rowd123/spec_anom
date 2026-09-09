import numpy as np
import pandas as pd
import pytest

from spectral_anomaly import (
    AnalysisPeriod,
    MSSTResult,
    associate_component_fragments,
    analyze_structural_windows,
    candidates_to_dataframe,
    compare_stft_msst_structure,
    compare_structural_windows,
    component_features,
    components_to_dataframe,
    extract_msst_structure,
    extract_spectral_structure,
    fit_component_feature_scaler,
    plot_structural_window,
    plot_stft_msst_comparison,
    plot_candidate_structures,
    prepare_analysis_windows,
    transform_component_features,
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

    monkeypatch.setattr(structure_module, "analyze_stft_periods", fake_analyze)
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
    assert len(figure.data) >= 7
    assert analyses[0].representation == "stft"


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


def controlled_structure(values: np.ndarray):
    return extract_spectral_structure(
        make_spectral_result(values),
        representation="stft",
        normalization_neighborhood=(9, 9),
        tensor_sigma=(1.0, 1.0),
        significance_threshold=2.0,
        coherence_threshold=0.2,
        minimum_component_area=3,
    )


def test_two_separate_structures_produce_two_component_rows_with_frequency_position():
    rng = np.random.default_rng(5)
    values = np.abs(rng.normal(1, 0.1, (40, 60)))
    values[8:10, 5:25] += 2
    values[27:29, 35:55] += 2

    analysis = controlled_structure(values)
    table = components_to_dataframe({7: analysis})

    assert len(analysis.components) == 2
    assert table["window_id"].tolist() == [7, 7]
    assert table["component_id"].tolist() == [1, 2]
    assert table.loc[0, "frequency_centroid"] < table.loc[1, "frequency_centroid"]
    assert (table["frequency_min"] <= table["frequency_centroid"]).all()
    assert (table["frequency_centroid"] <= table["frequency_max"]).all()
    assert (table["mean_coherence"] <= table["coherence_q90"]).all()
    assert (table["mean_significance"] <= table["max_significance"]).all()
    assert table["relative_time_centroid"].between(0, 1).all()


def test_same_geometry_at_different_frequencies_has_distinct_centroids():
    rng = np.random.default_rng(6)
    values = np.abs(rng.normal(1, 0.1, (40, 60)))
    values[8:10, 5:25] += 2
    values[27:29, 5:25] += 2

    selected = sorted(
        controlled_structure(values).components,
        key=lambda item: item.area_pixels,
        reverse=True,
    )[:2]
    low, high = sorted(selected, key=lambda item: item.frequency_centroid)

    assert low.area_pixels == pytest.approx(high.area_pixels, rel=0.1)
    assert low.linearity == pytest.approx(high.linearity, rel=0.02)
    assert high.frequency_centroid - low.frequency_centroid > 15


def test_horizontal_and_vertical_components_are_linear_with_axial_orientations():
    rng = np.random.default_rng(7)
    horizontal = np.abs(rng.normal(1, 0.1, (40, 60)))
    vertical = np.abs(rng.normal(1, 0.1, (40, 60)))
    horizontal[18:20, 10:50] += 2
    vertical[5:35, 29:31] += 2

    horizontal_component = max(
        controlled_structure(horizontal).components, key=lambda item: item.area_pixels
    )
    vertical_component = max(
        controlled_structure(vertical).components, key=lambda item: item.area_pixels
    )

    assert horizontal_component.linearity > 0.9
    assert vertical_component.linearity > 0.9
    assert horizontal_component.orientation_cos2 > 0.9
    assert vertical_component.orientation_cos2 < -0.9
    assert abs(horizontal_component.orientation_sin2) < 0.1
    assert abs(vertical_component.orientation_sin2) < 0.1


def test_amplitude_changes_significance_without_becoming_an_anomaly_label():
    def extract(amplitude):
        rng = np.random.default_rng(9)
        values = np.abs(rng.normal(1, 0.35, (40, 60)))
        values[18:20, 10:50] += amplitude
        result = extract_spectral_structure(
            make_spectral_result(values),
            representation="stft",
            normalization_neighborhood=(9, 9),
            tensor_sigma=(1.0, 1.0),
            significance_threshold=1.2,
            coherence_threshold=0.2,
            minimum_component_area=10,
        )
        return max(result.components, key=lambda item: item.area_pixels)

    weak, strong = extract(0.8), extract(1.6)

    assert weak.frequency_centroid == pytest.approx(strong.frequency_centroid, abs=1.0)
    assert weak.orientation_cos2 == pytest.approx(strong.orientation_cos2, abs=0.15)
    assert weak.mean_significance < strong.mean_significance
    assert not hasattr(weak, "anomaly")


def test_component_feature_scaling_excludes_ids_and_handles_axial_orientation():
    rng = np.random.default_rng(10)
    values = np.abs(rng.normal(1, 0.1, (40, 60)))
    values[8:10, 5:25] += 2
    values[27:29, 35:55] += 2
    table = components_to_dataframe({3: controlled_structure(values)})

    raw_features = component_features(table)
    scaler = fit_component_feature_scaler(table)
    scaled = transform_component_features(table, scaler)

    assert "window_id" not in raw_features
    assert "component_id" not in raw_features
    assert "orientation_radians" not in raw_features
    assert {"orientation_cos2", "orientation_sin2"}.issubset(raw_features)
    assert np.all(np.isfinite(scaled.to_numpy()))
    assert np.allclose(scaled.median(axis=0), 0.0)


def fragmented_map(
    fragments: list[tuple[slice, slice]], shape: tuple[int, int] = (30, 60)
):
    values = np.zeros(shape)
    for frequency_slice, time_slice in fragments:
        values[frequency_slice, time_slice] = 5.0
    return extract_spectral_structure(
        make_spectral_result(values),
        representation="stft",
        normalization_neighborhood=(9, 9),
        tensor_sigma=(1.0, 1.0),
        significance_threshold=2.0,
        coherence_threshold=0.2,
        minimum_component_area=1,
    )


def association(analysis, **overrides):
    options = {
        "max_fragment_time_gap_seconds": 2.0,
        "max_fragment_frequency_gap_hz": 1.0,
        "max_fragment_frequency_centroid_difference_hz": 1.0,
        "max_fragment_orientation_difference_radians": 0.2,
    }
    options.update(overrides)
    return associate_component_fragments(analysis, **options)


def test_interrupted_line_associates_three_components_into_one_candidate():
    analysis = fragmented_map(
        [
            (slice(10, 12), slice(2, 10)),
            (slice(10, 12), slice(12, 20)),
            (slice(10, 12), slice(22, 30)),
        ]
    )

    result = association(analysis)
    candidate = result.candidates[0]
    print("interrupted line:", [item.component_ids for item in result.candidates])

    assert len(analysis.components) == 3
    assert len(result.candidates) == 1
    assert candidate.component_ids == (1, 2, 3)
    assert candidate.area_pixels == 48
    assert candidate.active_duration_seconds == 24
    assert candidate.time_span_seconds == 28
    assert candidate.total_gap_duration_seconds == 4
    assert candidate.maximum_gap_duration_seconds == 2
    assert candidate.gap_fraction == pytest.approx(4 / 28)


def test_fragments_beyond_time_gap_remain_separate_candidates():
    analysis = fragmented_map(
        [
            (slice(10, 12), slice(2, 10)),
            (slice(10, 12), slice(16, 24)),
        ]
    )

    result = association(analysis, max_fragment_time_gap_seconds=5.0)
    print("large gap:", [item.component_ids for item in result.candidates])

    assert len(analysis.components) == 2
    assert [item.component_ids for item in result.candidates] == [(1,), (2,)]


def test_close_fragments_at_different_frequencies_remain_separate():
    analysis = fragmented_map(
        [
            (slice(5, 7), slice(2, 10)),
            (slice(18, 20), slice(12, 20)),
        ]
    )

    result = association(analysis)
    print("different frequencies:", [item.component_ids for item in result.candidates])

    assert len(result.candidates) == 2


def test_close_fragments_with_incompatible_orientations_remain_separate():
    analysis = fragmented_map(
        [
            (slice(10, 12), slice(2, 10)),
            (slice(5, 15), slice(12, 14)),
        ]
    )

    result = association(
        analysis,
        max_fragment_frequency_gap_hz=20.0,
        max_fragment_frequency_centroid_difference_hz=20.0,
    )
    print("different orientations:", [item.component_ids for item in result.candidates])

    assert len(analysis.components) == 2
    assert len(result.candidates) == 2


def test_fragment_graph_association_is_explicitly_transitive():
    analysis = fragmented_map(
        [
            (slice(10, 12), slice(2, 10)),
            (slice(10, 12), slice(12, 20)),
            (slice(10, 12), slice(22, 30)),
        ]
    )

    result = association(analysis)
    print("transitive chain:", [item.component_ids for item in result.candidates])

    # C1 and C3 are 12 seconds apart, but C1--C2--C3 is one graph component.
    assert len(result.candidates) == 1
    assert result.candidates[0].fragment_count == 3


def test_two_independent_fragmented_structures_make_two_candidates_and_tables():
    analysis = fragmented_map(
        [
            (slice(5, 7), slice(2, 10)),
            (slice(5, 7), slice(12, 20)),
            (slice(20, 22), slice(3, 11)),
            (slice(20, 22), slice(13, 21)),
        ]
    )

    result = association(analysis)
    print("two fragmented structures:", [item.component_ids for item in result.candidates])
    components = components_to_dataframe({4: analysis})
    candidates = candidates_to_dataframe({4: result})

    assert len(components) == 4
    assert len(candidates) == 2
    assert candidates["component_ids"].tolist() == [(1, 2), (3, 4)]
    assert (candidates["window_id"] == 4).all()
    assert (candidates["representation"] == "stft").all()
    assert np.allclose(
        candidates["frequency_span_hz"],
        candidates["frequency_max"] - candidates["frequency_min"],
    )
    figure = plot_candidate_structures(result)
    assert len(figure.layout.annotations) == 6
