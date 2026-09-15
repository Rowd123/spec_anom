"""Six-panel STFT + real SAM 2 + segment-feature diagnostic.

This is the SAM counterpart of ``structure_usage.py`` in presentation only. It
does not import or execute the historical structure-tensor pipeline.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sam_spectral_usage import (
    add_mask_contours,
    demonstration_frame,
    prepare_example_window,
)
from spectral_anomaly import (
    SAMSegmentationSession,
    SEGMENT_FEATURE_COLUMNS,
    load_config,
    postprocess_masks,
    segments_to_dataframe,
)


def build_six_panel_figure(
    frame,
    prepared,
    sam_image,
    raw_segments,
    selected_segments,
    features,
    *,
    sam_device,
    postprocessed,
    value_col,
    quality_col,
    valid_quality_flags,
):
    """Return the compact 2 x 3 diagnostic requested for visual assessment."""
    window = prepared.window
    spectral = prepared.spectral
    times = prepared.absolute_times
    magnitude = np.log1p(np.abs(spectral.stft))
    common = {"x": times, "y": spectral.frequencies, "showscale": False}
    selected_title = "Postprocessed segments" if postprocessed else "Selected segments (raw)"
    figure = make_subplots(
        rows=2,
        cols=3,
        specs=[[{}, {}, {}], [{}, {}, {"type": "table"}]],
        subplot_titles=(
            "Signal",
            "STFT",
            "SAM input image",
            "Raw SAM segments",
            selected_title,
            "Segment features",
        ),
        horizontal_spacing=0.055,
        vertical_spacing=0.16,
    )

    figure.add_trace(go.Scatter(
        x=window.time, y=window.signal, mode="lines", name="cleaned signal",
        line={"color": "#111827"},
    ), 1, 1)
    source = frame.loc[(frame.index >= window.time[0]) & (frame.index <= window.time[-1])]
    if quality_col is not None:
        invalid = ~source[quality_col].isin(valid_quality_flags)
        figure.add_trace(go.Scatter(
            x=source.index[invalid], y=source.loc[invalid, value_col], mode="markers",
            name="invalid quality flag",
            marker={"symbol": "x", "size": 8, "color": "#dc2626"},
        ), 1, 1)
    figure.add_trace(go.Scatter(
        x=window.time[window.interpolated_mask], y=window.signal[window.interpolated_mask],
        mode="markers", name="interpolated", marker={"symbol": "diamond-open", "size": 8,
                                                       "color": "#f59e0b"},
    ), 1, 1)
    figure.add_trace(go.Heatmap(z=magnitude, colorscale="Viridis", **common), 1, 2)
    # SAM receives three identical channels; channel zero displays the exact
    # uint8 intensities while preserving the physical axes in Plotly.
    figure.add_trace(go.Heatmap(z=sam_image[..., 0], colorscale="Gray", **common), 1, 3)

    figure.add_trace(go.Heatmap(z=magnitude, colorscale="Viridis", **common), 2, 1)
    add_mask_contours(figure, raw_segments, times, spectral.frequencies, row=2, col=1)
    figure.add_trace(go.Heatmap(z=magnitude, colorscale="Viridis", **common), 2, 2)
    add_mask_contours(figure, selected_segments, times, spectral.frequencies, row=2, col=2)

    table_columns = []
    for name in SEGMENT_FEATURE_COLUMNS:
        values = features[name].tolist() if name in features else []
        table_columns.append(
            [f"S{value}" for value in values]
            if name == "segment_id"
            else [f"{float(value):.5g}" for value in values]
        )
    figure.add_trace(go.Table(
        header={"values": list(SEGMENT_FEATURE_COLUMNS), "fill_color": "#dbeafe",
                "align": "left", "font": {"size": 10}},
        cells={"values": table_columns, "align": "left", "height": 22,
               "font": {"size": 9}},
    ), 2, 3)

    figure.update_xaxes(title_text="time from DataFrame index", row=1, col=1)
    figure.update_yaxes(title_text="signal", row=1, col=1)
    for row, col in ((1, 2), (1, 3), (2, 1), (2, 2)):
        figure.update_xaxes(title_text="time from DataFrame index", row=row, col=col)
        figure.update_yaxes(title_text="frequency (Hz)", row=row, col=col)
    figure.update_layout(
        title=("STFT + SAM 2 automatic + segment features"
               f" — window {prepared.window_id}, SAM device={sam_device}"
               f" — postprocessing={'on' if postprocessed else 'off'}"),
        width=1900,
        height=920,
        template="plotly_white",
        hovermode="closest",
    )
    return figure


def _sam_diagnostics(segmentation):
    return [
        {
            "segment_id": f"S{segment.segment_id}",
            "predicted_iou": segment.predicted_iou,
            "stability_score": segment.stability_score,
        }
        for segment in segmentation.raw_sam_segments
    ]


def load_measurements(path: Path) -> pd.DataFrame:
    """Load a CSV whose first column is the time index.

    Numeric indexes are retained as seconds. Text indexes are parsed as
    datetimes so the original measurement times remain the pipeline axis.
    """
    frame = pd.read_csv(path, index_col=0)
    if frame.empty:
        raise ValueError(f"input CSV is empty: {path}")
    if not pd.api.types.is_numeric_dtype(frame.index.dtype):
        name = frame.index.name
        parsed = pd.to_datetime(frame.index, errors="raise", utc=True)
        frame.index = pd.DatetimeIndex(parsed, name=name)
    return frame


def _coerce_quality_flags(frame, quality_col, flags):
    """Match CLI strings to a numeric quality column when necessary."""
    if quality_col is None:
        if flags:
            raise ValueError("--valid-quality-flags requires --quality-col")
        return None
    if quality_col not in frame.columns:
        raise KeyError(f"unknown quality column: {quality_col!r}")
    if not flags:
        raise ValueError("--valid-quality-flags is required with --quality-col")
    if pd.api.types.is_numeric_dtype(frame[quality_col].dtype):
        return tuple(pd.to_numeric(pd.Series(flags), errors="raise").tolist())
    return tuple(flags)


def main(output: Path, spectral_config_path: Path, sam_config_path: Path,
         window_id: int = 0, postprocess: bool = False, show: bool = False,
         input_path: Path | None = None, value_col: str = "value",
         quality_col: str | None = None, valid_quality_flags=None):
    """Run real automatic SAM once and write a standalone six-panel HTML file."""
    spectral_config = deepcopy(load_config(spectral_config_path, "spectral"))
    sam_config = deepcopy(load_config(sam_config_path, "sam"))
    spectral_config["representation"] = "stft"
    sam_config["segmentation_mode"] = "automatic"

    if input_path is None:
        frame, _ = demonstration_frame(spectral_config)
        selected_quality_col = "quality" if quality_col is None else quality_col
        selected_flags = ("good",) if valid_quality_flags is None else tuple(valid_quality_flags)
    else:
        frame = load_measurements(input_path)
        selected_quality_col = quality_col
        selected_flags = _coerce_quality_flags(frame, selected_quality_col,
                                                valid_quality_flags)
    if value_col not in frame.columns:
        raise KeyError(f"unknown value column: {value_col!r}")
    print(f"Value column: {value_col}")
    print(f"Quality column: {selected_quality_col}")
    print(f"Accepted quality flags: {list(selected_flags) if selected_flags else []}")

    metadata, prepared = prepare_example_window(
        frame, spectral_config, window_id, value_col=value_col,
        quality_col=selected_quality_col, valid_quality_flags=selected_flags,
    )
    session = SAMSegmentationSession(sam_config)
    segmentation = session.segment(prepared.spectral)
    raw_segments = segmentation.raw_segments
    selected_segments = (
        tuple(postprocess_masks(segmentation.segments, **sam_config["mask_postprocessing"]))
        if postprocess else raw_segments
    )
    features = segments_to_dataframe(
        selected_segments,
        prepared.spectral,
        window_id=window_id,
        window_start=metadata.loc[window_id, "start_time"],
    )

    print(metadata.loc[[window_id], ["start_time", "end_time", "observed_samples",
                                     "interpolated_samples", "valid_ratio"]].to_string())
    print(f"\nDetected raw SAM segments: {len(raw_segments)}")
    print(f"Selected segments: {len(selected_segments)}")
    print("\nSegment features:")
    terminal_features = features.reindex(columns=SEGMENT_FEATURE_COLUMNS).copy()
    if not terminal_features.empty:
        terminal_features["segment_id"] = terminal_features["segment_id"].map(lambda value: f"S{value}")
    print(terminal_features.to_string(index=False))
    print("\nSAM diagnostics (not model features):")
    diagnostics = _sam_diagnostics(segmentation)
    print("none" if not diagnostics else "\n".join(str(item) for item in diagnostics))

    figure = build_six_panel_figure(
        frame, prepared, segmentation.image, raw_segments, selected_segments, features,
        sam_device=session.device, postprocessed=postprocess, value_col=value_col,
        quality_col=selected_quality_col, valid_quality_flags=selected_flags,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(output, include_plotlyjs=True, full_html=True)
    print(f"\nSix-panel SAM spectral diagnostic written to {output}")
    if show:
        figure.show()
    return features


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spectral-config", type=Path,
                        default=Path("configs/spectral_analysis.json"))
    parser.add_argument("--sam-config", type=Path, default=Path("configs/sam.json"))
    parser.add_argument("--output", type=Path, default=Path("sam_spectral_segments.html"))
    parser.add_argument("--input", type=Path,
                        help="CSV input; its first column is used as the time index")
    parser.add_argument("--value-col", default="value",
                        help="numeric signal column (default: value)")
    parser.add_argument("--quality-col",
                        help="quality column; omit for no quality filtering on CSV input")
    parser.add_argument("--valid-quality-flags", nargs="+",
                        help="values of --quality-col accepted as valid")
    parser.add_argument("--window-id", type=int, default=0)
    parser.add_argument("--postprocess", action="store_true")
    parser.add_argument("--show", action="store_true")
    arguments = parser.parse_args()
    main(arguments.output, arguments.spectral_config, arguments.sam_config,
         arguments.window_id, arguments.postprocess, arguments.show,
         arguments.input, arguments.value_col, arguments.quality_col,
         arguments.valid_quality_flags)
