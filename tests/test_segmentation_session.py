from copy import deepcopy
from types import SimpleNamespace

import numpy as np

from spectral_anomaly import (
    SAMSegmentationSession,
    SAMStructureSegmenter,
    dataframe_to_segments,
    load_config,
    postprocess_masks,
)
from spectral_anomaly.sam_segmentation import annotations_to_segments


class CountingAutomatic:
    device = "cpu"

    def __init__(self):
        self.calls = 0

    def generate_masks(self, image):
        self.calls += 1
        mask = np.zeros(image.shape[:2], dtype=bool)
        mask[1:3, 1:4] = True
        return annotations_to_segments([{
            "segmentation": mask, "area": 6, "bbox": [1, 1, 3, 2],
            "predicted_iou": 0.91, "stability_score": 0.87,
            "point_coords": [[2, 2]], "crop_box": [0, 0, image.shape[1], image.shape[0]],
        }])


class RecordingPredictor:
    def __init__(self):
        self.multimask = []
        self.image_calls = 0

    def set_image(self, image):
        self.image_calls += 1

    def predict(self, **kwargs):
        self.multimask.append(kwargs["multimask_output"])
        count = 3 if kwargs["multimask_output"] else 1
        masks = np.zeros((count, 4, 5), dtype=bool)
        masks[:, 1:3, 1:4] = True
        return masks, np.linspace(0.7, 0.9, count), np.zeros((count, 2, 2))


def spectrum():
    stft = np.ones((4, 5), dtype=complex)
    stft[2, 2] = 8
    return SimpleNamespace(values=stft, stft=stft, times=np.arange(5.0),
                           frequencies=np.arange(4.0))


def test_automatic_session_reuses_runner_and_preserves_all_sam_metadata():
    config = load_config("configs/sam.json", "sam")
    runner = CountingAutomatic()
    session = SAMSegmentationSession(config, automatic_segmenter=runner)
    first = session.segment(spectrum())
    second = session.segment(spectrum())
    assert runner.calls == 2
    assert first.device == "cpu" and second.mode == "automatic"
    source = first.segments[0].metadata["source_segments"][0]
    assert source["area"] == 6 and source["bbox"] == (1.0, 1.0, 3.0, 2.0)
    assert source["predicted_iou"] == 0.91 and source["stability_score"] == 0.87
    assert source["point_coords"] == ((2.0, 2.0),)


def test_merged_segment_keeps_each_sam_source_record_and_overlap_iou():
    config = load_config("configs/sam.json", "sam")
    runner = CountingAutomatic()
    result = SAMSegmentationSession(config, automatic_segmenter=runner).segment(spectrum())
    duplicate = deepcopy(result.segments[0])
    duplicate = type(duplicate)(2, duplicate.mask, metadata={"source_segments": [{
        "segment_id": 2, "area": 6, "predicted_iou": 0.8, "stability_score": 0.7,
    }]})
    merged = postprocess_masks([result.segments[0], duplicate], iou_threshold=0.8)
    assert len(merged) == 1
    assert [record["segment_id"] for record in merged[0].metadata["source_segments"]] == [1, 2]
    assert merged[0].metadata["overlap_links"][0]["iou"] == 1.0


def test_guided_session_honors_multimask_output_and_is_reusable():
    config = load_config("configs/sam.json", "sam")
    config["segmentation_mode"] = "energy_contrast"
    config["energy_contrast"].update({
        "exclusion_seconds": 0, "neighborhood_seconds": 2,
        "min_valid_references": 1, "contrast_threshold": 2,
        "min_time_spacing": 0, "min_frequency_spacing": 0,
        "multimask_output": False,
    })
    predictor = RecordingPredictor()
    session = SAMSegmentationSession(config, predictor=predictor)
    first = session.segment(spectrum())
    second = session.segment(spectrum())
    assert predictor.multimask and set(predictor.multimask) == {False}
    assert predictor.image_calls == 2
    assert first.guided is not None and second.guided is not None
    assert all(len(item.proposals.masks) == 1 for item in first.guided.raw_masks)


def test_guided_session_can_build_predictor_from_model_configuration(monkeypatch):
    config = load_config("configs/sam.json", "sam")
    config["segmentation_mode"] = "energy_contrast"
    captured = {}

    class FakeWrapper:
        device = "cuda:1"
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("spectral_anomaly.segmentation.SAMStructureSegmenter", FakeWrapper)
    session = SAMSegmentationSession(config)
    assert captured == {"checkpoint": config["model"]["checkpoint"],
                        "model_config": config["model"]["config"],
                        "device": config["device"], "predictor": None}
    assert session.device == "cuda:1"


def test_dataframe_pipeline_constructs_one_session_for_all_windows(monkeypatch):
    import pandas as pd

    spectral_config = load_config("configs/spectral_analysis.json", "spectral")
    spectral_config["device"] = "cpu"
    sam_config = load_config("configs/sam.json", "sam")
    calls = {"init": 0, "segment": 0}

    class FakeSession:
        device = "cpu"
        def __init__(self, config, **kwargs):
            calls["init"] += 1
        def segment(self, spectral):
            calls["segment"] += 1
            return SimpleNamespace(
                image=np.zeros((*spectral.values.shape, 3), dtype=np.uint8),
                segments=(), raw_segments=(), raw_sam_segments=(), points=(),
                contrast=None, guided=None,
            )

    monkeypatch.setattr("spectral_anomaly.pipeline.SAMSegmentationSession", FakeSession)
    frame = pd.DataFrame(
        {"value": np.sin(np.arange(384) / 10)},
        index=pd.date_range("2024-01-01", periods=384, freq="s"),
    )
    _, diagnostics = dataframe_to_segments(frame, spectral_config, sam_config)
    assert calls == {"init": 1, "segment": 2}
    assert len(diagnostics) == 2
