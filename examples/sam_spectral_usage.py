"""Reference diagnostic: signal -> STFT -> real SAM 2 -> segment features."""

import argparse
from copy import deepcopy
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.ndimage import binary_erosion

from sam_usage import demonstration_signal
from spectral_anomaly import (
    SAMSegmentationSession,
    SEGMENT_FEATURE_COLUMNS,
    analyze_spectrum,
    load_config,
    postprocess_masks,
    segments_to_dataframe,
)


_COLORS = ("#ef4444", "#22c55e", "#3b82f6", "#f59e0b", "#a855f7", "#06b6d4")


def _add_contours(figure, segments, times, frequencies, *, row, col, labels):
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
            hovertemplate=f"S{segment.segment_id}<br>t=%{{x:.3f}} s"
                          "<br>f=%{y:.4f} Hz<extra></extra>",
        ), row=row, col=col)
        if labels:
            rows, columns = np.nonzero(segment.mask)
            if len(rows):
                figure.add_annotation(
                    x=float(times[int(round(columns.mean()))]),
                    y=float(frequencies[int(round(rows.mean()))]),
                    text=f"S{segment.segment_id}", showarrow=True,
                    arrowcolor=color, font={"color": color, "size": 12},
                    bgcolor="rgba(255,255,255,0.75)", row=row, col=col,
                )


def build_figure(time, signal, spectral, sam_image, raw_segments, displayed_segments,
                 features, *, postprocessed, sam_device):
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
    common = {"x": spectral.times, "y": spectral.frequencies, "showscale": False}
    magnitude = np.log1p(np.abs(spectral.stft))
    figure.add_trace(go.Scatter(x=time, y=signal, name="signal", line={"color": "#111827"}), 1, 1)
    figure.add_trace(go.Heatmap(z=magnitude, colorscale="Viridis", **common), 2, 1)
    figure.add_trace(go.Image(z=sam_image, x0=float(spectral.times[0]),
                              dx=float(np.median(np.diff(spectral.times))),
                              y0=float(spectral.frequencies[0]),
                              dy=float(np.median(np.diff(spectral.frequencies)))), 2, 2)
    figure.add_trace(go.Heatmap(z=magnitude, colorscale="Greys", opacity=0.35, **common), 3, 1)
    _add_contours(figure, raw_segments, spectral.times, spectral.frequencies,
                  row=3, col=1, labels=True)
    figure.add_trace(go.Heatmap(z=magnitude, colorscale="Viridis", **common), 3, 2)
    _add_contours(figure, displayed_segments, spectral.times, spectral.frequencies,
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
        figure.update_xaxes(title_text="temps (s)", row=row, col=col)
        figure.update_yaxes(title_text="fréquence (Hz)", row=row, col=col)
    figure.update_xaxes(title_text="temps (s)", row=1, col=1)
    figure.update_yaxes(title_text="amplitude", row=1, col=1)
    figure.update_layout(
        title=("STFT → image normalisée → SAM 2 automatique → contours → caractéristiques"
               f"<br>device SAM={sam_device}; post-traitement={'activé' if postprocessed else 'désactivé'}; "
               "les valeurs RGB ne servent à aucune caractéristique"),
        height=1450, template="plotly_white", hovermode="closest",
    )
    return figure


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--spectral-config", default="configs/spectral_analysis.json")
    parser.add_argument("--sam-config", default="configs/sam.json")
    parser.add_argument("--output", default="sam_spectral_diagnostic.html")
    parser.add_argument("--postprocess", action="store_true",
                        help="Apply mask_postprocessing from sam.json (disabled by default here).")
    parser.add_argument("--show", action="store_true", help="Open the generated HTML in a browser.")
    args = parser.parse_args(argv)

    spectral_config = deepcopy(load_config(args.spectral_config, "spectral"))
    sam_config = deepcopy(load_config(args.sam_config, "sam"))
    spectral_config["representation"] = "stft"
    sam_config["segmentation_mode"] = "automatic"
    time, signal = demonstration_signal(
        spectral_config["windowing"]["size"], spectral_config["sampling_frequency"]
    )
    spectral = analyze_spectrum(signal, spectral_config)

    # Real SAM 2 is constructed exactly once; no demo generator or fabricated mask.
    session = SAMSegmentationSession(sam_config)
    segmentation = session.segment(spectral)
    raw_segments = segmentation.raw_segments
    displayed_segments = (
        tuple(postprocess_masks(segmentation.segments, **sam_config["mask_postprocessing"]))
        if args.postprocess else raw_segments
    )
    features = segments_to_dataframe(displayed_segments, spectral)
    figure = build_figure(
        time, signal, spectral, segmentation.image, raw_segments, displayed_segments,
        features, postprocessed=args.postprocess, sam_device=session.device,
    )
    figure.write_html(args.output)
    if args.show:
        figure.show()
    print(f"SAM automatic device={session.device}; raw={len(raw_segments)}; "
          f"displayed={len(displayed_segments)}; output={args.output}")
    return features


if __name__ == "__main__":
    main()
