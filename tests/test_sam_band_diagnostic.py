import importlib.util
import json
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "examples" / "sam_band_diagnostic.py"
SPEC = importlib.util.spec_from_file_location("sam_band_diagnostic", SCRIPT)
diagnostic = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(diagnostic)


def test_extracts_exact_numeric_plotly_heatmaps_and_axes(tmp_path):
    log_magnitude = np.log1p([[0.0, 3.0], [8.0, 15.0]]).tolist()
    image = [[0, 64], [128, 255]]
    axes = {"x": [45.0, 47.0], "y": [0.5, 0.0]}
    data = [
        {"type": "heatmap", "z": log_magnitude, **axes},
        {"type": "heatmap", "z": image, **axes},
    ]
    html = tmp_path / "source.html"
    html.write_text("<script>Plotly.newPlot(" + json.dumps("plot") + "," + json.dumps(data)
                    + ",{});</script>", encoding="utf-8")

    magnitude, rgb, time, frequency = diagnostic.load_exact_input(html, None)

    np.testing.assert_allclose(magnitude, [[0, 3], [8, 15]])
    np.testing.assert_array_equal(rgb[..., 0], image)
    np.testing.assert_array_equal(rgb[..., 0], rgb[..., 2])
    np.testing.assert_array_equal(time, [45, 47])
    np.testing.assert_array_equal(frequency, [0.5, 0.0])


def test_mask_hover_explicitly_describes_uncovered_pixels():
    hover = diagnostic._mask_hover(np.array([[True, False]]))
    assert hover.tolist() == [["masque", "aucun masque"]]
