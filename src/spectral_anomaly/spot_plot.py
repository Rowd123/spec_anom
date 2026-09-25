"""Plot saved window decisions against the original signal, without refitting."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .artifacts import read_table


def plot_spot_windows(signal, decisions, *, source_id, channel_id, value_col="value"):
    """Shade inclusive saved window bounds; overlap means at least one alarm.

    Scores are placed at window_end (when the whole window is available).
    Unshaded samples are not necessarily normal: they may be unscored.
    """
    required = {"source_id", "channel_id", "window_start", "window_end",
                "window_id", "score", "threshold", "is_anomaly"}
    missing = required - set(decisions)
    if missing:
        raise ValueError(f"decisions missing columns: {sorted(missing)}")
    rows = decisions.loc[decisions.source_id.eq(source_id) &
                         decisions.channel_id.eq(channel_id)].copy()
    if rows.empty:
        raise ValueError(f"no decisions for {source_id}/{channel_id}")
    if rows.window_id.duplicated().any():
        raise ValueError("expected one decision per window; duplicate window_id")
    if rows.is_anomaly.isna().any() or not pd.api.types.is_bool_dtype(rows.is_anomaly):
        raise ValueError("is_anomaly must contain non-null booleans")
    for column in ("window_start", "window_end"):
        rows[column] = pd.to_datetime(rows[column], utc=True, errors="raise")
    if rows[["window_start", "window_end"]].isna().any().any() or (rows.window_end < rows.window_start).any():
        raise ValueError("invalid window bounds")
    if not np.isfinite(rows[["score", "threshold"]].to_numpy(float)).all():
        raise ValueError("scores and thresholds must be finite")
    rows = rows.sort_values("window_end", kind="stable")
    if not isinstance(signal.index, pd.DatetimeIndex):
        raise ValueError("signal must have a DatetimeIndex")
    raw = signal[[value_col]].copy()
    raw.index = pd.to_datetime(raw.index, utc=True)
    raw = raw.sort_index(kind="stable")
    if raw.empty or raw.index.hasnans:
        raise ValueError("signal is empty or has missing timestamps")
    if rows.window_end.max() < raw.index.min() or rows.window_start.min() > raw.index.max():
        raise ValueError("signal and decisions have no time overlap")

    # Merge only overlapping alarm intervals to avoid darker overlap shading and
    # thousands of redundant Plotly shapes. Individual alarms remain hoverable.
    intervals = []
    for row in rows.loc[rows.is_anomaly].sort_values("window_start").itertuples():
        if intervals and row.window_start <= intervals[-1][1]:
            intervals[-1] = (intervals[-1][0], max(intervals[-1][1], row.window_end))
        else:
            intervals.append((row.window_start, row.window_end))
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[.65, .35], vertical_spacing=.09,
                        subplot_titles=("Original signal — red: flagged windows", "Window score and SPOT threshold"))
    fig.add_trace(go.Scattergl(x=raw.index, y=raw[value_col], name=value_col,
                             mode="lines", connectgaps=False,
                             line=dict(color="#219ebc", width=1)), row=1, col=1)
    for start, end in intervals:
        fig.add_vrect(x0=start, x1=end, fillcolor="#e45756", opacity=.22,
                      line_width=0, layer="below", row=1, col=1)
    custom = np.column_stack([rows.window_id.astype(str), rows.window_start.astype(str),
                              rows.window_end.astype(str)])
    hover = ("Window: %{customdata[0]}<br>Start: %{customdata[1]}<br>"
             "End: %{customdata[2]}<br>Score: %{y:.6g}<extra>%{fullData.name}</extra>")
    fig.add_trace(go.Scattergl(x=rows.window_end, y=rows.score, mode="markers",
                             name="IF score", customdata=custom, hovertemplate=hover,
                             marker=dict(size=5, color="#219ebc")), row=2, col=1)
    fig.add_trace(go.Scatter(x=rows.window_end, y=rows.threshold, mode="lines",
                            name="SPOT threshold", line=dict(color="#ed9b40", dash="dash")), row=2, col=1)
    alarm = rows.is_anomaly.to_numpy()
    fig.add_trace(go.Scattergl(x=rows.loc[alarm, "window_end"], y=rows.loc[alarm, "score"],
                             name="Anomalous windows", mode="markers", customdata=custom[alarm],
                             hovertemplate=hover, marker=dict(color="#e45756", size=8)), row=2, col=1)
    fig.update_layout(template="plotly_white", height=750,
                      title=f"{source_id} / {channel_id} — {int(alarm.sum())}/{len(rows)} flagged windows",
                      legend=dict(orientation="h", y=-.16), margin=dict(b=120))
    fig.update_yaxes(title_text=value_col, row=1, col=1)
    fig.update_yaxes(title_text="Anomaly score", row=2, col=1)
    fig.update_xaxes(title_text="Time (UTC) — scores at window end", row=2, col=1)
    return fig


def export_spot_plot(input_path, decisions_path, output_path, *, source_id, channel_id,
                     value_col="value", index_col="time", sep=",", datetime_format="ISO8601"):
    frame = pd.read_csv(input_path, sep=sep)
    frame.index = pd.to_datetime(frame.pop(index_col), utc=True, format=datetime_format)
    decisions, _ = read_table(decisions_path, expected_type="spot_decisions")
    fig = plot_spot_windows(frame, decisions, source_id=source_id,
                            channel_id=channel_id, value_col=value_col)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output, include_plotlyjs=True, full_html=True,
                   config={"scrollZoom": True, "toImageButtonOptions": {"format": "svg"}})
    return output
