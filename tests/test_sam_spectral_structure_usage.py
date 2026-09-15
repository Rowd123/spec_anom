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
        sam_device="cpu", postprocessed=False, value_col="value",
        quality_col="quality", valid_quality_flags=("good",),
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


def test_real_dataframe_columns_and_flags_are_forwarded(tmp_path):
    import pandas as pd

    config = load_config("configs/spectral_analysis.json", "spectral")
    config["device"] = "cpu"
    index = pd.date_range("2025-03-01", periods=300, freq="s", name="recorded_at")
    frame = pd.DataFrame({"amplitude": np.sin(np.arange(300) / 10),
                          "status": "valid"}, index=index)
    frame.loc[index[20], "status"] = "bad"
    csv = tmp_path / "measurements.csv"
    frame.to_csv(csv)
    loaded = EXAMPLE.load_measurements(csv)
    metadata, prepared = EXAMPLE.prepare_example_window(
        loaded, config, 0, value_col="amplitude", quality_col="status",
        valid_quality_flags=("valid",),
    )
    assert prepared.absolute_times.name == "recorded_at"
    assert metadata.loc[0, "interpolated_samples"] == 1


def test_dataframe_without_quality_column_is_supported():
    import pandas as pd

    config = load_config("configs/spectral_analysis.json", "spectral")
    config["device"] = "cpu"
    frame = pd.DataFrame(
        {"amplitude": np.sin(np.arange(300) / 10)},
        index=pd.date_range("2025-03-01", periods=300, freq="s"),
    )
    _, prepared = EXAMPLE.prepare_example_window(
        frame, config, 0, value_col="amplitude", quality_col=None,
        valid_quality_flags=None,
    )
    assert prepared.window.observed_mask.all()
