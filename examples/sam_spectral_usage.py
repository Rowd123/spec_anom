"""Reference diagnostic: signal -> STFT -> real SAM 2 -> segment features."""

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
from scipy.ndimage import binary_erosion

from sam_usage import demonstration_signal
from spectral_anomaly import (
    SAMSegmentationSession,
    SEGMENT_FEATURE_COLUMNS,
    analyze_dataframe_windows,
    load_config,
    postprocess_masks,
    segments_to_dataframe,
)


_COLORS = ("#ef4444", "#22c55e", "#3b82f6", "#f59e0b", "#a855f7", "#06b6d4")


def demonstration_frame(config):
    """Build timestamped measurements with flags and one missing timestamp."""
    windowing = config["windowing"]
    size = windowing["size"] + (windowing["size"] - windowing["overlap"])
    relative_time, signal = demonstration_signal(size, config["sampling_frequency"])
    index = pd.date_range("2025-01-15T12:00:00Z", periods=size,
                          freq=config["sampling_period"], name="timestamp")
    quality = np.full(size, "good", dtype=object)
    quality[[38, 39, 205]] = "bad"
    frame = pd.DataFrame({"value": signal, "quality": quality}, index=index)
    # A missing row tests regularisation independently from explicit bad flags.
    frame = frame.drop(index[92])
    return frame, relative_time


def prepare_example_window(frame, config, window_id):
    """Run the production quality preparation and select one accepted window."""
    metadata, windows = analyze_dataframe_windows(
        frame,
        config,
        value_col="value",
        quality_col="quality",
        valid_quality_flags=("good",),
    )
    if window_id not in metadata.index:
        raise ValueError(f"window-id {window_id} is outside 0..{len(metadata) - 1}")
    if window_id not in windows:
        reason = metadata.loc[window_id, "rejection_reason"]
        raise ValueError(f"window {window_id} was rejected by quality control: {reason}")
    return metadata, windows[window_id]


def add_mask_contours(figure, segments, times, frequencies, *, row, col, labels=True):
    """Draw true mask boundaries independently, never by summing segment IDs."""
    for index, segment in enumerate(segments):
        boundary = segment.mask & ~binary_erosion(segment.mask)
        frequency_indices, time_indices = np.nonzero(boundary)
        color = _COLORS[index % len(_COLORS)]
        figure.add_trace(go.Scattergl(
            x=times[time_indices],
            y=frequencies[frequency_indices],
            mode="markers",
            marker={"size": 3, "color": color},
            name=f"S{segment.segment_id}",
            legendgroup=f"S{segment.segment_id}",
            hovertemplate=f"S{segment.segment_id}<br>t=%{{x}}"
                          "<br>f=%{y:.4f} Hz<extra></extra>",
        ), row=row, col=col)
        if labels:
            rows, columns = np.nonzero(segment.mask)
            if len(rows):
                figure.add_annotation(
                    x=times[int(round(columns.mean()))],
                    y=float(frequencies[int(round(rows.mean()))]),
                    text=f"S{segment.segment_id}", showarrow=True,
                    arrowcolor=color, font={"color": color, "size": 12},
                    bgcolor="rgba(255,255,255,0.75)", row=row, col=col,
                )


def build_figure(source_frame, window, spectral, spectral_times, sam_image,
                 raw_segments, displayed_segments, features, *, postprocessed,
                 sam_device, window_id):
    """Build the self-contained pedagogical HTML figure."""
    figure = make_subplots(
        rows=4, cols=2,
        specs=[[{"colspan": 2}, None], [{}, {}], [{}, {}],
               [{"type": "table", "colspan": 2}, None]],
        subplot_titles=(
            "A — Signal temporel",
            "B — STFT complexe (log1p magnitude)",
            "C — Image uint8 RGB exacte envoyée à SAM",
            f"D — Raw SAM masks ({len(raw_segments)})",
            "E — STFT avec contours " + ("post-traités" if postprocessed else "SAM bruts"),
            "F — Caractéristiques calculées depuis la PSD STFT",
        ),
        vertical_spacing=0.08,
    )
    common = {"x": spectral_times, "y": spectral.frequencies, "showscale": False}
    magnitude = np.log1p(np.abs(spectral.stft))
    figure.add_trace(go.Scatter(
        x=window.time, y=window.signal, name="signal préparé",
        line={"color": "#111827"},
    ), 1, 1)
    in_window = source_frame.loc[
        (source_frame.index >= window.time[0]) & (source_frame.index <= window.time[-1])
    ]
    valid = in_window["quality"].eq("good")
    figure.add_trace(go.Scatter(
        x=in_window.index[valid], y=in_window.loc[valid, "value"], mode="markers",
        name="observations valides", marker={"size": 4, "color": "#2563eb"},
    ), 1, 1)
    figure.add_trace(go.Scatter(
        x=in_window.index[~valid], y=in_window.loc[~valid, "value"], mode="markers",
        name="flags qualité rejetés", marker={"size": 9, "color": "#dc2626", "symbol": "x"},
    ), 1, 1)
    figure.add_trace(go.Scatter(
        x=window.time[window.interpolated_mask], y=window.signal[window.interpolated_mask],
        mode="markers", name="lacunes interpolées",
        marker={"size": 9, "color": "#f59e0b", "symbol": "diamond-open"},
    ), 1, 1)
    figure.add_trace(go.Heatmap(z=magnitude, colorscale="Viridis", **common), 2, 1)
    # The three RGB channels are identical; plotting channel 0 preserves the
    # exact uint8 values while retaining the physical datetime/frequency axes.
    figure.add_trace(go.Heatmap(z=sam_image[..., 0], colorscale="Gray", **common), 2, 2)
    figure.add_trace(go.Heatmap(z=magnitude, colorscale="Greys", opacity=0.35, **common), 3, 1)
    add_mask_contours(figure, raw_segments, spectral_times, spectral.frequencies,
                  row=3, col=1, labels=True)
    figure.add_trace(go.Heatmap(z=magnitude, colorscale="Viridis", **common), 3, 2)
    add_mask_contours(figure, displayed_segments, spectral_times, spectral.frequencies,
                  row=3, col=2, labels=True)

    headers = list(SEGMENT_FEATURE_COLUMNS)
    table_values = []
    for name in headers:
        values = features[name].tolist() if name in features else []
        if name == "segment_id":
            values = [f"S{value}" for value in values]
        else:
            values = [f"{float(value):.6g}" for value in values]
        table_values.append(values)
    figure.add_trace(go.Table(
        header={"values": headers, "fill_color": "#dbeafe", "align": "left"},
        cells={"values": table_values, "align": "left", "height": 24},
    ), 4, 1)
    for row, col in ((2, 1), (2, 2), (3, 1), (3, 2)):
        figure.update_xaxes(title_text="timestamp (UTC)", row=row, col=col)
        figure.update_yaxes(title_text="fréquence (Hz)", row=row, col=col)
    figure.update_xaxes(title_text="timestamp fourni par l'index du DataFrame (UTC)", row=1, col=1)
    figure.update_yaxes(title_text="amplitude", row=1, col=1)
    figure.update_layout(
        title=("STFT → image normalisée → SAM 2 automatique → contours → caractéristiques"
               f"<br>fenêtre={window_id}; device SAM={sam_device}; "
               f"post-traitement={'activé' if postprocessed else 'désactivé'}; "
               "les valeurs RGB ne servent à aucune caractéristique"),
        height=1450, template="plotly_white", hovermode="closest",
    )
    return figure


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--spectral-config", default="configs/spectral_analysis.json")
    parser.add_argument("--sam-config", default="configs/sam.json")
    parser.add_argument("--output", default="sam_spectral_diagnostic.html")
    parser.add_argument("--window-id", type=int, default=0,
                        help="Prepared quality-valid window to analyse (default: 0).")
    parser.add_argument("--postprocess", action="store_true",
                        help="Apply mask_postprocessing from sam.json (disabled by default here).")
    parser.add_argument("--show", action="store_true", help="Open the generated HTML in a browser.")
    args = parser.parse_args(argv)

    spectral_config = deepcopy(load_config(args.spectral_config, "spectral"))
    sam_config = deepcopy(load_config(args.sam_config, "sam"))
    spectral_config["representation"] = "stft"
    sam_config["segmentation_mode"] = "automatic"
    frame, _ = demonstration_frame(spectral_config)
    metadata, prepared = prepare_example_window(frame, spectral_config, args.window_id)
    window = prepared.window
    spectral = prepared.spectral
    spectral_times = prepared.absolute_times

    # Real SAM 2 is constructed exactly once; no demo generator or fabricated mask.
    session = SAMSegmentationSession(sam_config)
    segmentation = session.segment(spectral)
    raw_segments = segmentation.raw_segments
    displayed_segments = (
        tuple(postprocess_masks(segmentation.segments, **sam_config["mask_postprocessing"]))
        if args.postprocess else raw_segments
    )
    features = segments_to_dataframe(
        displayed_segments, spectral, window_id=args.window_id,
        window_start=metadata.loc[args.window_id, "start_time"],
    )
    figure = build_figure(
        frame, window, spectral, spectral_times, segmentation.image, raw_segments,
        displayed_segments, features, postprocessed=args.postprocess,
        sam_device=session.device, window_id=args.window_id,
    )
    figure.write_html(args.output)
    if args.show:
        figure.show()
    print(f"SAM automatic device={session.device}; raw={len(raw_segments)}; "
          f"displayed={len(displayed_segments)}; output={args.output}")
    return features


if __name__ == "__main__":
    main()
