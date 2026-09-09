import numpy as np
import pandas as pd
import pytest

from spectral_anomaly import (
    WindowData,
    analyze_msst_monitoring_period,
    analyze_msst_periods,
    plot_msst_periods,
    prepare_analysis_periods,
    prepare_monitoring_period,
)
from spectral_anomaly import msst as msst_module


def make_window(start: int, size: int = 4) -> WindowData:
    positions = np.arange(start, start + size)
    signal = positions.astype(float)
    return WindowData(
        time=pd.Index(positions),
        signal=signal,
        centered=signal - signal.mean(),
        windowed=signal.copy(),
        observed_mask=np.ones(size, dtype=bool),
        interpolated_mask=np.zeros(size, dtype=bool),
    )


def make_detection_result() -> tuple[pd.DataFrame, dict[int, WindowData]]:
    starts = [0, 2, 4, 6, 8]
    result = pd.DataFrame(
        {
            "grid_start": starts,
            "grid_stop": [start + 4 for start in starts],
            "accepted": True,
            "suspicious": [False, True, True, False, False],
        },
        index=pd.RangeIndex(5, name="window_id"),
    )
    return result, {window_id: make_window(start) for window_id, start in enumerate(starts)}


def test_consecutive_anomalies_are_joined_and_extended_forward():
    result, windows = make_detection_result()
    metadata, periods = prepare_analysis_periods(result, windows, period_size=8)

    assert len(metadata) == 1
    assert metadata.loc[0, "source_window_ids"] == (1, 2)
    assert metadata.loc[0, "anomaly_samples"] == 6
    assert metadata.loc[0, "accepted"]
    assert np.array_equal(periods[0].signal, np.arange(2.0, 10.0))
    assert np.array_equal(periods[0].anomaly_mask, [True] * 6 + [False] * 2)


def test_anomaly_group_longer_than_period_is_explicitly_excluded():
    result, windows = make_detection_result()
    metadata, periods = prepare_analysis_periods(result, windows, period_size=4)

    assert not metadata.loc[0, "accepted"]
    assert metadata.loc[0, "rejection_reason"] == "anomaly_group_too_long"
    assert periods == {}


def test_period_without_enough_following_data_is_excluded():
    result, windows = make_detection_result()
    result["suspicious"] = [False, False, False, False, True]
    metadata, periods = prepare_analysis_periods(result, windows, period_size=8)

    assert not metadata.loc[0, "accepted"]
    assert metadata.loc[0, "rejection_reason"] == "insufficient_following_valid_data"
    assert periods == {}


def test_analysis_centers_period_and_forwards_msst_options(monkeypatch):
    result, windows = make_detection_result()
    _, periods = prepare_analysis_periods(result, windows, period_size=8)
    calls = []

    def fake_msst(signal, sampling_frequency, iteration_count, **options):
        calls.append((signal, sampling_frequency, iteration_count, options))
        coefficients = np.ones((3, 4), dtype=np.complex64)
        return 2 * coefficients, coefficients, np.array([0.0, 0.5, 1.0])

    monkeypatch.setattr(msst_module, "msst_stft", fake_msst)
    analyses = analyze_msst_periods(
        periods,
        sampling_frequency=2.0,
        iteration_count=4,
        hop_length=2,
    )

    assert np.mean(calls[0][0]) == pytest.approx(0)
    assert calls[0][1:] == (2.0, 4, {"hop_length": 2})
    assert np.array_equal(analyses[0].spectral_time, [0.0, 1.0, 2.0, 3.0])

    figure = plot_msst_periods(analyses)
    assert len(figure.data) == 4
    assert [item.text for item in figure.layout.annotations] == [
        "Period 0 — signal",
        "Period 0 — STFT",
        "Period 0 — MSST",
    ]


def test_prepare_monitoring_period_covers_the_whole_detection_grid():
    result, windows = make_detection_result()

    period = prepare_monitoring_period(result, windows)

    assert np.array_equal(period.signal, np.arange(12.0))
    assert period.source_window_ids == (0, 1, 2, 3, 4)
    assert np.array_equal(
        period.anomaly_mask, [False, False] + [True] * 6 + [False] * 4
    )


def test_prepare_monitoring_period_rejects_uncovered_data():
    result, windows = make_detection_result()
    del windows[2]
    del windows[3]

    with pytest.raises(ValueError, match="not covered by accepted data"):
        prepare_monitoring_period(result, windows)


def test_monitoring_period_analysis_returns_one_full_transformation(monkeypatch):
    result, windows = make_detection_result()

    def fake_msst(signal, sampling_frequency, iteration_count, **options):
        coefficients = np.ones((3, len(signal)), dtype=np.complex64)
        return 2 * coefficients, coefficients, np.array([0.0, 0.5, 1.0])

    monkeypatch.setattr(msst_module, "msst_stft", fake_msst)
    analysis = analyze_msst_monitoring_period(
        result, windows, sampling_frequency=2.0, center=False
    )

    assert np.array_equal(analysis.processed_signal, np.arange(12.0))
    assert analysis.msst.shape == (3, 12)
    assert analysis.spectral_time[-1] == pytest.approx(5.5)
