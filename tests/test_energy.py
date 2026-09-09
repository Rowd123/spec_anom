import numpy as np
import pandas as pd
import pytest

from spectral_anomaly import detect_energy_anomalies, plot_suspicious_windows


def frame(values, *, index=None, quality=None):
    if index is None:
        index = pd.date_range("2025-01-01", periods=len(values), freq="1s")
    data = {"value": values}
    if quality is not None:
        data["quality"] = quality
    return pd.DataFrame(data, index=index)


def detect(data, **kwargs):
    defaults = dict(value_col="value", sampling_period="1s", window_size=8,
                    min_valid_ratio=0.5, max_interpolation_gap=2,
                    history_size=4, min_history=2, k=3)
    defaults.update(kwargs)
    return detect_energy_anomalies(data, **defaults)


def test_regular_signal_produces_complete_fixed_windows():
    result, windows = detect(frame(np.sin(np.arange(24))), min_valid_ratio=1)
    assert len(result) == 3
    assert result["accepted"].all()
    assert (result["observed_samples"] == 8).all()
    assert all(len(item.signal) == 8 for item in windows.values())


def test_duplicate_timestamp_keeps_first_occurrence():
    idx = pd.date_range("2025-01-01", periods=8, freq="1s")
    data = frame(np.arange(8.0), index=idx)
    duplicate = pd.DataFrame({"value": [999.0]}, index=idx[:1])
    result, windows = detect(pd.concat([data.iloc[:1], duplicate, data.iloc[1:]]),
                             min_valid_ratio=1)
    assert result.loc[0, "observed_samples"] == 8
    assert windows[0].signal[0] == 0.0


@pytest.mark.parametrize("kind", ["nan", "quality", "absent"])
def test_isolated_missing_sources_are_counted_as_unobserved_and_interpolated(kind):
    values = np.arange(8.0)
    idx = pd.date_range("2025-01-01", periods=8, freq="1s")
    quality = np.repeat("good", 8)
    if kind == "nan":
        values[3] = np.nan
    elif kind == "quality":
        quality[3] = "bad"
    else:
        values, idx, quality = np.delete(values, 3), idx.delete(3), np.delete(quality, 3)
    data = frame(values, index=idx, quality=quality)
    result, windows = detect(data, quality_col="quality", valid_quality_flags={"good"})
    assert result.loc[0, "observed_samples"] == 7
    assert result.loc[0, "interpolated_samples"] == 1
    assert result.loc[0, "valid_ratio"] == 7 / 8
    assert windows[0].interpolated_mask[3]
    assert not windows[0].observed_mask[3]


def test_small_gap_is_filled_but_large_gap_is_rejected_with_reasons():
    values = np.arange(24.0)
    values[3:5] = np.nan
    values[11:14] = np.nan
    result, _ = detect(frame(values), min_valid_ratio=0.5)
    assert result.loc[0, "accepted"]
    assert result.loc[0, "interpolated_samples"] == 2
    assert not result.loc[1, "accepted"]
    assert "gap_too_long" in result.loc[1, "rejection_reason"]
    assert "unfilled_missing_values" in result.loc[1, "rejection_reason"]


def test_energy_fft_uses_original_signal_without_centering():
    values = np.array([10.0, 14.0, 9.0, 12.0, 7.0, 11.0, 10.5, 8.0])
    result, windows = detect(frame(values), min_valid_ratio=1)
    window = windows[0]

    assert np.array_equal(window.signal, values)
    assert np.allclose(window.windowed, values * np.hanning(8))
    assert np.mean(window.centered) == pytest.approx(0)
    assert result.loc[0, "dc_bin_energy"] > 0


def test_dc_bin_is_removed_from_energy_by_default():
    values = np.array([0.0, 4.0, -1.0, 2.0, -3.0, 1.0, 0.5, -2.0])
    data = frame(values)
    without_dc, _ = detect(data, min_valid_ratio=1)
    with_dc, _ = detect(data, min_valid_ratio=1, exclude_dc_bin=False)

    row = without_dc.loc[0]
    assert row["dc_bin_energy"] > 0
    assert row["energy"] + row["dc_bin_energy"] == pytest.approx(
        row["energy_including_dc"]
    )
    assert with_dc.loc[0, "energy"] == pytest.approx(row["energy_including_dc"])


def test_dc_exclusion_uses_parseval_equivalent_full_energy():
    values = np.array([0.0, 4.0, -1.0, 2.0, -3.0, 1.0, 0.5, -2.0])
    result, windows = detect(frame(values), min_valid_ratio=1, exclude_dc_bin=False)
    window = windows[0]
    taper_energy = np.sum(np.hanning(8) ** 2)
    expected = np.sum((values * np.hanning(8)) ** 2) / taper_energy

    assert result.loc[0, "energy"] == pytest.approx(expected)


def test_slow_energy_drift_uses_a_local_not_global_baseline():
    blocks = []
    pattern = np.array([-1.0, 1.0] * 4)
    for amplitude in np.linspace(1, 1.35, 12):
        blocks.append(amplitude * pattern)
    result, _ = detect(frame(np.concatenate(blocks)), history_size=3, min_history=3, k=10)
    assert not result["suspicious"].any()
    assert result["baseline"].dropna().iloc[-1] > result["baseline"].dropna().iloc[0]


def test_artificial_energy_anomaly_is_detected_and_excluded_from_history():
    normal = np.array([-1.0, 1.0] * 4)
    values = np.concatenate([normal, normal, normal, 8 * normal, normal])
    result, _ = detect(frame(values), k=3)
    assert result.loc[3, "suspicious"]
    assert not result.loc[4, "suspicious"]
    assert result.loc[4, "baseline"] == pytest.approx(result.loc[2, "energy"])


def test_zero_mad_has_finite_scores_and_zero_threshold_detects_change():
    normal = np.array([-1.0, 1.0] * 4)
    values = np.concatenate([normal, normal, normal, 1.01 * normal])
    result, _ = detect(frame(values), k=0)
    assert result.loc[2, "mad"] == 0
    assert np.isfinite(result.loc[3, "score"])
    assert result.loc[3, "score"] != 0
    assert result.loc[3, "suspicious"]


def test_all_windows_rejected_when_observed_ratio_is_too_low():
    values = np.arange(24.0)
    values[np.arange(24) % 2 == 0] = np.nan
    result, windows = detect(frame(values), min_valid_ratio=0.75,
                             max_interpolation_gap=1)
    assert not result["accepted"].any()
    assert windows == {}
    assert result["rejection_reason"].str.contains("insufficient_valid_ratio").all()


def test_numeric_time_index_and_parameter_validation():
    data = frame(np.arange(8.0), index=pd.Index(np.arange(8) * 0.5))
    result, _ = detect_energy_anomalies(
        data, value_col="value", sampling_period=0.5, window_size=8,
        min_valid_ratio=1, max_interpolation_gap=0, history_size=2, min_history=1,
    )
    assert result.loc[0, "accepted"]
    with pytest.raises(ValueError, match="overlap"):
        detect(data, overlap=8)


def test_off_grid_timestamp_is_not_silently_snapped():
    idx = pd.to_datetime(
        ["2025-01-01 00:00:00", "2025-01-01 00:00:01.1"], format="mixed"
    )
    with pytest.raises(ValueError, match="not exactly aligned"):
        detect(frame([1.0, 2.0], index=idx), sampling_period="1s")


def test_plot_suspicious_windows_plots_every_anomaly():
    normal = np.array([-1.0, 1.0] * 4)
    values = np.concatenate([normal, normal, normal, 8 * normal, normal, 7 * normal])
    result, windows = detect(frame(values), k=3)

    figure = plot_suspicious_windows(result, windows, show_centered=False)

    assert len(figure.layout.annotations) == int(result["suspicious"].sum())
    assert all("Suspicious window" in item.text for item in figure.layout.annotations)


def test_plot_suspicious_windows_rejects_empty_selection():
    result, windows = detect(frame(np.tile([-1.0, 1.0], 12)), k=100)
    with pytest.raises(ValueError, match="no suspicious window"):
        plot_suspicious_windows(result, windows)
