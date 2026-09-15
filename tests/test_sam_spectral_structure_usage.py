import importlib.util
from pathlib import Path

import numpy as np

from spectral_anomaly import Segment, load_config, segments_to_dataframe


PATH = Path("examples/sam_spectral_structure_usage.py")
SPEC = importlib.util.spec_from_file_location("sam_spectral_structure_usage", PATH)
EXAMPLE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(EXAMPLE)


def prepared_window():
    config = load_config("configs/spectral_analysis.json", "spectral")
    config["device"] = "cpu"
    frame, _ = EXAMPLE.demonstration_frame(config)
    _, prepared = EXAMPLE.prepare_example_window(frame, config, 0)
    return frame, prepared


def test_six_panel_layout_has_real_contours_and_feature_table():
    frame, prepared = prepared_window()
    shape = prepared.spectral.stft.shape
    first = np.zeros(shape, dtype=bool)
    second = np.zeros(shape, dtype=bool)
    first[4:9, 8:20] = True
    second[7:13, 18:32] = True
    segments = (Segment(1, first), Segment(2, second))
    features = segments_to_dataframe(segments, prepared.spectral)
    image = np.zeros((*shape, 3), dtype=np.uint8)
    figure = EXAMPLE.build_six_panel_figure(
        frame, prepared, image, segments, segments, features,
        sam_device="cpu", postprocessed=False,
    )
    titles = [annotation.text for annotation in figure.layout.annotations]
    assert titles[:6] == ["Signal", "STFT", "SAM input image", "Raw SAM segments",
                          "Selected segments (raw)", "Segment features"]
    assert sum(trace.type == "scattergl" for trace in figure.data) == 4
    assert figure.data[-1].type == "table"


def test_example_does_not_import_historical_morphology_pipeline():
    source = PATH.read_text(encoding="utf8")
    forbidden_calls = (
        "analyze_candidate_structures(", "extract_spectral_structure(",
        "compute_structure_tensor(", "associate_component_fragments(",
    )
    assert not any(call in source for call in forbidden_calls)
    assert "from spectral_anomaly.structure" not in source
