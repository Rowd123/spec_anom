from dataclasses import replace
import logging

import numpy as np
import pandas as pd
import pytest

from spectral_anomaly import Segment, analyze_spectrum, extract_segment_features, load_config, read_table
from spectral_anomaly.config import validate_config
from spectral_anomaly.features import FEATURE_MEANING
from spectral_anomaly.pipeline import dataframe_to_segments
from spectral_anomaly.selection import select_segments_for_analysis
from spectral_anomaly.sam_segmentation import annotations_to_segments
from spectral_anomaly.workflows import extract_segments, train_isolation_forest, score_isolation_forest, fit_hdbscan


def configs(rep="stft"):
    cfg = load_config("configs/spectral_analysis.json", "spectral")
    cfg.update(device="cpu", representation=rep)
    cfg["windowing"].update(size=32, overlap=0)
    cfg["transform"].update(window_length=16, n_fft=32, hop_length=4)
    sam = load_config("configs/sam.json", "sam")
    sam["mask_postprocessing"].update(iou_threshold=.4)
    return cfg, sam


@pytest.mark.parametrize("representation", ["stft", "ssq_stft", "msst"])
def test_representation_energy_and_unchanged_physical_features(representation):
    cfg, _ = configs(representation)
    spectrum = analyze_spectrum(np.sin(np.arange(32) / 3), cfg)
    mask = np.zeros(spectrum.values.shape, bool)
    mask[1:3, 1:4] = True
    segment = Segment(1, mask)
    actual = extract_segment_features(segment, spectrum)
    expected = np.abs(spectrum.values[mask].astype(complex)) ** 2
    assert actual["representation_energy"] == pytest.approx(expected.sum())
    assert actual["mean_representation_energy_density"] == pytest.approx(expected.mean())
    changed = extract_segment_features(segment, replace(spectrum, values=spectrum.values * 3))
    for name in FEATURE_MEANING:
        if "representation" not in name:
            assert changed[name] == actual[name]
    dt = np.median(np.diff(spectrum.times)); df = np.median(np.diff(spectrum.frequencies))
    assert actual["integrated_spectral_power"] == pytest.approx(spectrum.psd[mask].sum() * dt * df)
    assert actual["integrated_energy"] == actual["integrated_spectral_power"]
    assert actual["mean_energy_density"] == actual["mean_spectral_power_density"]


def test_known_complex_signed_and_empty_mask():
    cfg, _ = configs()
    spectrum = analyze_spectrum(np.arange(32), cfg)
    values = np.zeros_like(spectrum.values)
    values[1, 1:4] = [3+4j, -2, 1j]
    spectrum = replace(spectrum, values=values)
    mask = np.zeros(values.shape, bool); mask[1, 1:4] = True
    result = extract_segment_features(Segment(1, mask), spectrum)
    assert result["representation_energy"] == 30
    assert result["mean_representation_energy_density"] == 10
    empty = extract_segment_features(Segment(1, np.zeros_like(mask)), spectrum)
    assert empty["representation_energy"] == empty["mean_representation_energy_density"] == 0


def test_disabled_threshold_equality_and_nonfinite_audit():
    frame = pd.DataFrame({"segment_id": range(5), "representation_energy": [1, 2, 3, np.nan, np.inf]})
    kept, audit = select_segments_for_analysis(frame, {})
    pd.testing.assert_frame_equal(kept, frame)
    assert audit["number_of_segments_rejected_by_energy"] == 0
    kept, audit = select_segments_for_analysis(frame, {"enabled": True, "min_energy": 2})
    assert kept.segment_id.tolist() == [1, 2]
    assert [audit[k] for k in ("number_of_segments_before_selection", "number_of_segments_after_selection", "number_of_segments_rejected_by_energy")] == [5, 2, 3]
    assert [x["reason"] for x in audit["rejected_segments"]] == ["below_min_representation_energy", "non_finite_representation_energy", "non_finite_representation_energy"]


@pytest.mark.parametrize("options", [None, [], {"foo": 1}, {"enabled": "true"},
    {"energy_feature": "typo"}, {"energy_feature": []}, {"enabled": True},
    {"min_energy": -1}, {"min_energy": np.nan}, {"min_energy": np.inf},
    {"min_energy": "2"}, {"min_energy": True}])
def test_invalid_selection_configuration(options):
    cfg, _ = configs(); cfg["segment_selection"] = options
    with pytest.raises(ValueError, match="segment_selection"):
        validate_config(cfg, "spectral")


def test_legacy_config_defaults_and_alternative_feature():
    cfg, _ = configs(); del cfg["segment_selection"]
    assert validate_config(cfg, "spectral")["segment_selection"]["enabled"] is False
    frame = pd.DataFrame({"mean_representation_energy_density": [1., 2.]})
    selected, _ = select_segments_for_analysis(frame, {"enabled": True, "energy_feature": "mean_representation_energy_density", "min_energy": 2})
    assert len(selected) == 1
    with pytest.raises(ValueError, match="column is missing"):
        select_segments_for_analysis(frame, {"enabled": True, "min_energy": 2})


class AutomaticMasks:
    device = "cpu"

    def generate_masks(self, image):
        # Two overlapping masks must merge before energy computation/selection.
        masks = []
        for left, right in [(1, 4), (2, 5), (6, 7)]:
            mask = np.zeros(image.shape[:2], bool)
            mask[1:3, left:right] = True
            masks.append({"segmentation": mask, "area": int(mask.sum()), "bbox": [left, 1, right-left, 2]})
        return annotations_to_segments(masks)


def signal_frame(windows=12):
    rng = np.random.default_rng(12)
    return pd.DataFrame({"value": rng.normal(size=windows*32)}, index=pd.date_range("2025-01-01", periods=windows*32, freq="s"))


def test_selection_after_union_and_logging(monkeypatch, caplog):
    cfg, sam = configs("msst"); sam["mask_postprocessing"].update(iou_threshold=.4)
    original = analyze_spectrum(np.arange(32), cfg)
    spectrum = replace(original, values=np.ones_like(original.values))
    monkeypatch.setattr("spectral_anomaly.pipeline.analyze_spectra", lambda signals, config: [spectrum] * len(signals))
    cfg["segment_selection"].update(enabled=True, min_energy=7)
    with caplog.at_level(logging.INFO):
        frame, diagnostics = dataframe_to_segments(signal_frame(1), cfg, sam, automatic_segmenter=AutomaticMasks())
    assert frame.representation_energy.tolist() == [8]
    assert frame.merge_count.tolist() == [2]
    assert diagnostics[0]["number_of_segments_before_selection"] == 2
    assert diagnostics[0]["number_of_segments_after_selection"] == 1
    assert diagnostics[0]["number_of_segments_rejected_by_energy"] == 1
    assert "segments_before_selection=2 segments_after_selection=1 segments_rejected_energy=1" in caplog.text


@pytest.mark.parametrize("threshold", [0, 1e30])
def test_parquet_columns_manifest_and_empty_selection(tmp_path, threshold):
    cfg, sam = configs(); cfg["segment_selection"].update(enabled=True, min_energy=threshold)
    path = tmp_path / "segments.parquet"
    result = extract_segments(signal_frame(2), path, cfg, sam, source_id="source", channel_id="channel", automatic_segmenter=AutomaticMasks())
    saved, manifest = read_table(path)
    assert {"representation_energy", "mean_representation_energy_density"} <= set(saved)
    assert "representation_energy" in manifest.feature_columns
    assert manifest.config["spectral"]["segment_selection"] == cfg["segment_selection"]
    audit = manifest.provenance["segment_selection"]
    assert audit["number_of_segments_before_selection"] == 4
    assert audit["number_of_segments_after_selection"] == len(saved) == len(result)
    assert audit["number_of_segments_rejected_by_energy"] == 4-len(saved)
    assert bool(saved.empty) == (threshold > 0)


def test_iforest_and_hdbscan_read_filtered_parquet(tmp_path):
    cfg, sam = configs(); raw, _ = dataframe_to_segments(signal_frame(), cfg, sam, automatic_segmenter=AutomaticMasks())
    threshold = float(raw.representation_energy.quantile(.25))
    cfg["segment_selection"].update(enabled=True, min_energy=threshold)
    path = tmp_path / "segments.parquet"
    filtered = extract_segments(signal_frame(), path, cfg, sam, source_id="source", channel_id="channel", automatic_segmenter=AutomaticMasks())
    assert 0 < len(filtered) < len(raw)
    assert (filtered.representation_energy >= threshold).all()
    models = load_config("configs/models.json", "models")
    models["atypicality"].update(backend="cpu", n_estimators=10, n_jobs=1)
    models["hdbscan"].update(backend="cpu", min_cluster_size=2, min_samples=1)
    models["preprocessing"]["atypicality_features"].append("representation_energy")
    models["preprocessing"]["hdbscan_features"].append("mean_representation_energy_density")
    train_isolation_forest(path, tmp_path / "if.pkl", models)
    scores = score_isolation_forest(path, tmp_path / "if.pkl", tmp_path / "scores.parquet")
    clusters = fit_hdbscan(path, tmp_path / "hdb.pkl", tmp_path / "clusters.parquet", models)
    for output in [scores, clusters]:
        assert len(output) == len(filtered)
        assert list(zip(output.window_id, output.segment_id)) == list(zip(filtered.window_id, filtered.segment_id))


def test_extract_cli_success_returns_no_exit_error(tmp_path, monkeypatch):
    from spectral_anomaly.cli import main
    input_path = tmp_path / "signal.csv"
    signal_frame(1).rename_axis("time").to_csv(input_path)
    captured = {}
    def fake_extract(frame, output, spectral, sam, **kwargs):
        captured.update(kwargs)
        return pd.DataFrame({"representation_energy": [1.]})
    monkeypatch.setattr("spectral_anomaly.cli.extract_segments", fake_extract)
    assert main(["extract", str(input_path), str(tmp_path / "segments.parquet"),
                 "--spectral-config", "configs/spectral_analysis.json",
                 "--sam-config", "configs/sam.json", "--source-id", "source",
                 "--channel-id", "channel"]) is None
    assert captured["source_id"] == "source"
