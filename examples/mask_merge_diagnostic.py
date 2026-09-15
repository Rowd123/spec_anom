"""Synthetic diagnostic for transitive mask merging; this does not run SAM."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import plotly.graph_objects as go

from spectral_anomaly import Segment, postprocess_masks


def main(output="mask_merge_diagnostic.html"):
    masks = []
    for segment_id, start in enumerate((3, 7, 11), 1):
        mask = np.zeros((24, 30), dtype=bool)
        mask[6:18, start:start + 10] = True
        masks.append(Segment(segment_id, mask, metadata={"source_segments": [{
            "segment_id": segment_id, "area": int(mask.sum()), "origin": "synthetic"
        }]}))
    merged = postprocess_masks(masks, iou_threshold=0.4)
    figure = go.Figure()
    colors = ("red", "green", "blue")
    for segment, color in zip(masks, colors):
        rows, columns = np.nonzero(segment.mask)
        figure.add_trace(go.Scatter(x=columns, y=rows, mode="markers", opacity=0.25,
                                    marker={"color": color}, name=f"source S{segment.segment_id}"))
    figure.update_layout(title=f"Diagnostic synthétique uniquement — {merged[0].metadata}",
                         yaxis={"autorange": "reversed"}, template="plotly_white")
    figure.write_html(output)
    return Path(output)


if __name__ == "__main__":
    main()
