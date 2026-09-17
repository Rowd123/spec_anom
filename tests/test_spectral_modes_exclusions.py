from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from spectral_anomaly import (analyze_dataframe_windows, analyze_spectra,
    analyze_spectrum, cuda_available, filter_excluded_segments, load_config)
from spectral_anomaly.spectral import _ssq_gpu_available


def configuration(representation="stft"):
    value = load_config("configs/spectral_analysis.json", "spectral")
    value["representation"] = representation
    value["device"] = "cpu"
    value["windowing"] = {"size": 32, "overlap": 16}
    value["transform"].update(window_length=16, n_fft=32, hop_length=4)
    return value


@pytest.mark.parametrize("representation", ["stft", "ssq_stft", "msst"])
def test_selected_representation_reaches_aligned_geometry(representation):
    config = configuration(representation)
    result = analyze_spectrum(np.sin(2 * np.pi * np.arange(64) / 8), config)
    assert result.representation == representation
    assert result.values.shape == result.stft.shape == result.psd.shape
    assert result.values.shape == (len(result.frequencies), len(result.times))
    assert np.all(np.diff(result.frequencies) > 0)
    assert np.all(np.diff(result.times) > 0)


def test_frequency_range_is_applied_to_all_aligned_products():
    config = configuration("ssq_stft")
    config["sampling_frequency"] = 32.0; config["sampling_period"] = "0.03125s"
    config["transform"].update(frequency_min=4.0, frequency_max=10.0)
    result = analyze_spectrum(np.sin(2 * np.pi * 6 * np.arange(64) / 32), config)
    assert result.frequencies.min() >= 4 and result.frequencies.max() <= 10
    assert result.values.shape == result.stft.shape == result.psd.shape


def test_ssq_batch_size_does_not_change_results():
    signals = [np.sin(np.arange(64) / scale) for scale in (3, 4, 5)]
    one = configuration("ssq_stft"); one["ssq_stft"]["batch_size"] = 1
    many = deepcopy(one); many["ssq_stft"]["batch_size"] = 3
    first, second = analyze_spectra(signals, one), analyze_spectra(signals, many)
    for left, right in zip(first, second):
        np.testing.assert_allclose(left.values, right.values, rtol=2e-5, atol=2e-6)
        np.testing.assert_array_equal(left.frequencies, right.frequencies)
        np.testing.assert_array_equal(left.times, right.times)


def test_excluded_event_rejects_every_intersecting_window_without_reconnecting():
    config = configuration()
    config["exclusions"] = {"events": [{"source_id": "A", "start": "2025-01-01 00:00:20", "end": "2025-01-01 00:00:21"}],
                            "event_id_column": None, "unknown_id_policy": "warning"}
    index = pd.date_range("2025-01-01", periods=80, freq="s")
    frame = pd.DataFrame({"value": np.arange(80.0)}, index=index)
    metadata, windows = analyze_dataframe_windows(frame, config, value_col="value", source_id="A")
    rejected = metadata.rejection_reason.fillna("").str.contains("excluded_event")
    assert rejected.any()
    assert not set(metadata.index[rejected]) & set(windows)
    for window_id in metadata.index[~rejected & metadata.accepted]:
        row = metadata.loc[window_id]
        assert row.end_time < index[20] or row.start_time > index[21]


def test_saved_features_can_be_filtered_by_window_support_and_source():
    frame = pd.DataFrame({"source_id": ["A", "A", "B"], "window_start": pd.to_datetime(["2025-01-01", "2025-01-03", "2025-01-01"]),
                          "window_end": pd.to_datetime(["2025-01-02", "2025-01-04", "2025-01-02"]), "segment_id": [1, 2, 3]})
    config = {"events": [{"source_id": "A", "start": "2025-01-02", "end": "2025-01-02"}],
              "event_id_column": None, "unknown_id_policy": "warning"}
    result = filter_excluded_segments(frame, config)
    assert result.segment_id.tolist() == [2, 3]


def test_explicit_ssq_cuda_fails_clearly_when_backend_is_unavailable(monkeypatch):
    config = configuration("ssq_stft"); config["device"] = "cuda"
    monkeypatch.setattr("spectral_anomaly.spectral._ssq_gpu_available", lambda: False)
    with pytest.raises(RuntimeError, match="no genuine GPU backend"):
        analyze_spectrum(np.ones(64), config)


@pytest.mark.skipif(not (cuda_available() and _ssq_gpu_available()),
                    reason="ssqueezepy CUDA backend (PyTorch + CuPy) is unavailable")
def test_ssq_cpu_and_cuda_have_matching_axes_and_values():
    signal = np.sin(2 * np.pi * np.arange(64) / 8)
    cpu = configuration("ssq_stft")
    gpu = deepcopy(cpu); gpu["device"] = "cuda"
    left, right = analyze_spectrum(signal, cpu), analyze_spectrum(signal, gpu)
    np.testing.assert_array_equal(left.times, right.times)
    np.testing.assert_array_equal(left.frequencies, right.frequencies)
    np.testing.assert_allclose(left.values, right.values, rtol=2e-4, atol=2e-5)
