from copy import deepcopy

import numpy as np
import pytest

from spectral_anomaly import analyze_spectrum, cuda_available, load_config


def config(**transform_updates):
    value = load_config("configs/spectral_analysis.json", "spectral")
    value["device"] = "cpu"
    value["transform"].update(transform_updates)
    return value


def test_centered_framing_uses_real_sample_positions_and_expected_shape():
    cfg = config(window_length=8, n_fft=16, hop_length=3, center=True)
    result = analyze_spectrum(np.arange(20.0), cfg)
    np.testing.assert_allclose(result.times, np.arange(0, 20, 3) / cfg["sampling_frequency"])
    assert result.stft.shape == (9, 7)


def test_uncentered_framing_reports_geometric_window_centers():
    cfg = config(window_length=8, n_fft=16, hop_length=3, center=False)
    result = analyze_spectrum(np.arange(20.0), cfg)
    np.testing.assert_allclose(result.times, (np.arange(0, 13, 3) + 3.5) / cfg["sampling_frequency"])
    assert result.stft.shape == (9, 5)


def test_padding_changes_boundary_values_but_not_geometry():
    signal = np.arange(20.0)
    reflected = analyze_spectrum(signal, config(window_length=8, n_fft=8, hop_length=2,
                                                 center=True, padtype="reflect"))
    zero = analyze_spectrum(signal, config(window_length=8, n_fft=8, hop_length=2,
                                            center=True, padtype="zero"))
    np.testing.assert_array_equal(reflected.times, zero.times)
    assert not np.allclose(reflected.stft[:, 0], zero.stft[:, 0])


def test_remove_mean_is_distinct_from_stft_centering():
    signal = 5 + np.sin(2 * np.pi * np.arange(64) / 8)
    cfg = config(window_length=64, n_fft=64, hop_length=64, center=False, window="boxcar")
    cfg["signal_preprocessing"]["remove_mean"] = False
    original = analyze_spectrum(signal, cfg)
    cfg["signal_preprocessing"]["remove_mean"] = True
    demeaned = analyze_spectrum(signal, cfg)
    assert original.processed_signal.mean() == pytest.approx(5)
    assert demeaned.processed_signal.mean() == pytest.approx(0, abs=1e-12)
    assert abs(demeaned.stft[0, 0]) < abs(original.stft[0, 0]) * 1e-10
    np.testing.assert_array_equal(original.times, demeaned.times)


def test_one_sided_density_psd_has_physical_parseval_normalization():
    fs = 64.0
    signal = np.sin(2 * np.pi * 8 * np.arange(64) / fs)
    cfg = config(window_length=64, n_fft=64, hop_length=64, center=False, window="boxcar")
    cfg["sampling_frequency"] = fs
    cfg["sampling_period"] = f"{1 / fs}s"
    cfg["signal_preprocessing"]["remove_mean"] = False
    result = analyze_spectrum(signal, cfg)
    df = result.frequencies[1] - result.frequencies[0]
    assert np.sum(result.psd[:, 0]) * df == pytest.approx(np.mean(signal ** 2), rel=1e-6)
    assert result.psd[8, 0] == pytest.approx(0.5, rel=1e-6)


def test_msst_stft_and_psd_are_exactly_grid_aligned():
    cfg = config(window_length=32, n_fft=64, hop_length=4, center=True)
    cfg["representation"] = "msst"
    result = analyze_spectrum(np.sin(2 * np.pi * np.arange(96) / 16), cfg)
    assert result.values.shape == result.stft.shape == result.psd.shape
    assert result.values.shape == (len(result.frequencies), len(result.times))


def test_explicit_cuda_for_msst_is_rejected():
    cfg = config()
    cfg["representation"] = "msst"
    cfg["device"] = "cuda"
    with pytest.raises(RuntimeError, match="no genuine GPU backend"):
        analyze_spectrum(np.ones(256), cfg)


@pytest.mark.skipif(not cuda_available(), reason="CUDA is unavailable")
def test_cpu_and_gpu_stft_share_geometry_and_numerical_convention():
    signal = np.random.default_rng(4).normal(size=256)
    cpu_cfg = config()
    gpu_cfg = deepcopy(cpu_cfg)
    gpu_cfg["device"] = "cuda"
    cpu = analyze_spectrum(signal, cpu_cfg)
    gpu = analyze_spectrum(signal, gpu_cfg)
    np.testing.assert_array_equal(cpu.times, gpu.times)
    np.testing.assert_array_equal(cpu.frequencies, gpu.frequencies)
    np.testing.assert_allclose(cpu.stft, gpu.stft, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(cpu.psd, gpu.psd, rtol=3e-5, atol=1e-7)
