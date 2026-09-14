import numpy as np
import pandas as pd
import pytest
import sys
import types

from spectral_anomaly import (
    RobustFeaturePreprocessor, SPOTState, SegmentModels,
    SAMSegment, build_segment_table, chronological_window_split, extract_segment_features,
    isolation_anomaly_scores, stft_to_psd,
)


def test_one_sided_psd_scaling_doubles_only_interior_bins():
    coefficients = np.ones((5, 2), dtype=complex)
    psd = stft_to_psd(
        coefficients, sampling_frequency=8, window=np.ones(8),
        window_length=8, n_fft=8, one_sided=True,
    )
    np.testing.assert_allclose(psd[[0, -1]], 1 / 64)
    np.testing.assert_allclose(psd[1:-1], 2 / 64)


def test_segment_features_use_cell_edges_and_keep_overlapping_mask_independent():
    psd = np.ones((3, 4))
    psd[1, 2] = 5
    mask = np.zeros_like(psd, dtype=bool); mask[1, 2] = True
    features = extract_segment_features(
        psd, mask, np.array([0., 1., 2., 4.]), np.array([0., 2., 5.]),
        contrast_exclusion_seconds=0, contrast_neighborhood_seconds=4,
        contrast_epsilon=.1, contrast_min_valid_references=2,
    )
    assert features["valid"]
    assert features["physical_area"] == pytest.approx(3.75)
    assert features["duration"] == pytest.approx(1.5)
    assert features["frequency_width"] == pytest.approx(2.5)
    assert features["integrated_energy"] == pytest.approx(18.75)
    assert features["mean_density"] == pytest.approx(5)
    assert features["central_frequency"] == pytest.approx(2)
    assert features["local_contrast"] == pytest.approx(5)


@pytest.mark.parametrize("mask, reason", [
    (np.zeros((2, 3), bool), "empty_mask"),
    (np.array([[1, 0, 0], [0, 0, 0]], bool), "zero_energy"),
])
def test_invalid_empty_or_zero_energy_segments_are_explicit(mask, reason):
    result = extract_segment_features(
        np.zeros((2, 3)), mask, np.arange(3.), np.arange(2.),
        contrast_exclusion_seconds=0, contrast_neighborhood_seconds=2,
        contrast_epsilon=1, contrast_min_valid_references=1,
    )
    assert not result["valid"] and result["invalid_reason"] == reason


def test_segment_table_keeps_overlapping_masks_as_individual_physical_rows():
    first = np.zeros((2, 3), bool); first[0, :2] = True
    second = np.zeros((2, 3), bool); second[:, 1] = True
    segments = [SAMSegment.from_sam_annotation(i, {
        "segmentation": mask, "area": int(mask.sum()), "bbox": [0, 0, 2, 2],
        "predicted_iou": .8 + i / 100, "stability_score": .9,
    }) for i, mask in enumerate((first, second), 1)]
    table = build_segment_table(
        stream_id="stream", window_id=4, window_start=0, window_end=3,
        available_time=3, segments=segments, psd=np.ones((2, 3)),
        spectral_time=np.array([0., 1., 2.]), frequencies=np.array([0., 2.]),
        contrast_options={"exclusion_seconds": 0, "neighborhood_seconds": 2,
                          "epsilon": .1, "min_valid_references": 1},
    )
    assert table.segment_id.tolist() == [1, 2]
    assert table.valid.all()
    assert table.loc[0, "time_min"] == pytest.approx(-.5)
    assert table.loc[1, "frequency_max"] == pytest.approx(3)


def test_isolation_score_sign_makes_lower_score_samples_more_anomalous():
    class FakeForest:
        def score_samples(self, values):
            return np.array([-0.2, -0.9])
    scores = isolation_anomaly_scores(FakeForest(), np.zeros((2, 1)))
    np.testing.assert_array_equal(scores, [.2, .9])


def test_temporal_split_keeps_windows_and_overlap_components_together():
    frame = pd.DataFrame({
        "stream_id": ["a"] * 8,
        "window_id": [0, 0, 1, 1, 2, 2, 3, 3],
        "window_start": [0, 0, 5, 5, 20, 20, 30, 30],
        "window_end": [10, 10, 15, 15, 25, 25, 35, 35],
    })
    split = chronological_window_split(frame, train_fraction=.4, calibration_fraction=.3)
    assert split.groupby(frame.window_id).nunique().eq(1).all()
    assert split[frame.window_id == 0].iloc[0] == split[frame.window_id == 1].iloc[0]
    assert split.tolist() == sorted(split, key={"train": 0, "calibration": 1, "evaluation": 2}.get)


def test_spot_reports_not_ready_for_insufficient_or_constant_calibration():
    state = SPOTState(risk=.01, initial_quantile=.8, min_excesses=3)
    state.calibrate([1, 1, 1, 1, 1])
    assert not state.ready
    assert state.process(2) == (None, None)


def test_spot_uses_threshold_before_update_and_excludes_alerts():
    scores = np.r_[np.linspace(0, 1, 90), np.linspace(1.1, 3, 10)]
    state = SPOTState(risk=.01, initial_quantile=.85, min_excesses=5)
    state.calibrate(scores)
    assert state.ready and state.threshold() is not None
    count_before = len(state.calibration_scores)
    threshold, alert = state.process(100)
    assert threshold is not None and alert is True
    assert len(state.calibration_scores) == count_before


def test_hdbscan_noise_label_is_not_a_spot_decision(monkeypatch):
    module = types.SimpleNamespace(
        approximate_predict=lambda model, values: (np.full(len(values), -1), np.zeros(len(values)))
    )
    monkeypatch.setitem(sys.modules, "hdbscan", module)
    prep_if = RobustFeaturePreprocessor(("duration",)).fit(pd.DataFrame({"duration": [1, 2]}))
    prep_cluster = RobustFeaturePreprocessor(("frequency_width",)).fit(
        pd.DataFrame({"frequency_width": [1, 2]})
    )
    models = SegmentModels(prep_if, prep_cluster, isolation_options={},
                           hdbscan_options={}, spot_options={})
    models.isolation_forest = type("Forest", (), {
        "score_samples": lambda self, values: np.full(len(values), -.5)
    })()
    models.clusterer = object()
    frame = pd.DataFrame({
        "valid": [True], "duration": [1.], "frequency_width": [1.],
        "stream_id": ["a"], "window_id": [0], "segment_id": [1],
        "available_time": [1],
    })
    results = models.predict(frame)
    assert results.loc[0, "cluster_id"] == -1
    assert results.loc[0, "cluster_strength"] == 0
    assert pd.isna(results.loc[0, "spot_alert"])
