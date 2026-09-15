import importlib.util
from pathlib import Path

import numpy as np

from spectral_anomaly import (
    SEGMENT_FEATURE_COLUMNS,
    Segment,
    analyze_spectrum,
    extract_segment_features,
    load_config,
    segments_to_dataframe,
)


SPEC = importlib.util.spec_from_file_location(
    "sam_spectral_usage", Path("examples/sam_spectral_usage.py")
)
EXAMPLE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(EXAMPLE)


def spectral_result():
    config = load_config("configs/spectral_analysis.json", "spectral")
    config["device"] = "cpu"
    config["representation"] = "stft"
    time, signal = EXAMPLE.demonstration_signal(
        config["windowing"]["size"], config["sampling_frequency"]
    )
    return time, signal, analyze_spectrum(signal, config)


def test_reference_feature_table_contains_only_the_decided_columns():
    _, _, spectral = spectral_result()
    mask = np.zeros(spectral.stft.shape, dtype=bool)
    mask[4:9, 8:15] = True
    frame = segments_to_dataframe([Segment(1, mask)], spectral)
    assert list(frame.loc[:, SEGMENT_FEATURE_COLUMNS].columns) == list(SEGMENT_FEATURE_COLUMNS)
    forbidden = {"orientation", "linearity", "coherence", "normalized_aspect_ratio",
                 "integrated_significance", "fragment_count", "component_count"}
    assert forbidden.isdisjoint(SEGMENT_FEATURE_COLUMNS)


def test_power_features_use_psd_and_not_sam_rgb_values():
    _, _, spectral = spectral_result()
    mask = np.zeros(spectral.stft.shape, dtype=bool)
    mask[3:8, 5:20] = True
    segment = Segment(1, mask)
    baseline = extract_segment_features(segment, spectral)
    # RGB does not enter the extractor API. Scaling the PSD must scale power
    # descriptors while leaving physical mask geometry unchanged.
    scaled = type(spectral)(
        spectral.representation, spectral.values, spectral.stft, spectral.psd * 4,
        spectral.frequencies, spectral.times, spectral.device,
        spectral.sampling_frequency, spectral.processed_signal,
    )
    changed = extract_segment_features(segment, scaled)
    assert changed["integrated_spectral_power"] == baseline["integrated_spectral_power"] * 4
    assert changed["mean_spectral_power_density"] == baseline["mean_spectral_power_density"] * 4
    assert np.isclose(changed["temporal_variation"], baseline["temporal_variation"] * 4)
    assert np.isclose(changed["frequency_variation"], baseline["frequency_variation"] * 4)
    assert np.isclose(changed["local_energy_contrast"], baseline["local_energy_contrast"])
    assert changed["time_frequency_area"] == baseline["time_frequency_area"]
    assert changed["duration"] == baseline["duration"]


def test_diagnostic_uses_contours_and_builds_feature_table():
    time, signal, spectral = spectral_result()
    first = np.zeros(spectral.stft.shape, dtype=bool)
    second = np.zeros_like(first)
    first[3:8, 5:20] = True
    second[6:11, 15:30] = True
    segments = (Segment(1, first), Segment(2, second))
    features = segments_to_dataframe(segments, spectral)
    image = np.repeat(np.zeros((*spectral.stft.shape, 1), dtype=np.uint8), 3, axis=2)
    figure = EXAMPLE.build_figure(
        time, signal, spectral, image, segments, segments, features,
        postprocessed=False, sam_device="cpu",
    )
    assert any(trace.type == "scattergl" and trace.name == "S1" for trace in figure.data)
    assert figure.data[-1].type == "table"
    assert list(figure.data[-1].header.values) == list(SEGMENT_FEATURE_COLUMNS)
