import numpy as np
import pandas as pd
import pytest

from spectral_anomaly.artifacts import build_manifest, write_table
from spectral_anomaly.cli import main
from spectral_anomaly.spot_plot import plot_spot_windows
from spectral_anomaly.workflows import apply_spot


def fixture_data():
    time = pd.date_range("2026-01-01", periods=12, freq="s", tz="UTC")
    signal = pd.DataFrame({"value": np.arange(12.)}, index=time)
    rows = pd.DataFrame(dict(source_id=["S"] * 3, channel_id=["C"] * 3,
                             window_id=[0, 1, 2], window_start=time[[1, 3, 8]],
                             window_end=time[[4, 6, 10]], score=[3., 4., .1],
                             threshold=[2.] * 3, is_anomaly=[True, True, False]))
    return signal, rows


def test_overlaps_merge_but_individual_decisions_remain():
    signal, rows = fixture_data()
    other = rows.assign(source_id="OTHER", window_id=[3, 4, 5])
    fig = plot_spot_windows(signal, pd.concat([rows, other]), source_id="S", channel_id="C")
    assert len(fig.layout.shapes) == 1
    assert fig.layout.shapes[0].x0 == rows.window_start.iloc[0]
    assert fig.layout.shapes[0].x1 == rows.window_end.iloc[1]
    assert len(fig.data[3].x) == 2
    assert list(pd.to_datetime(fig.data[1].x, utc=True)) == list(rows.window_end)
    np.testing.assert_array_equal(fig.data[0].y, signal.value)


def test_no_alarms_and_validation():
    signal, rows = fixture_data()
    fig = plot_spot_windows(signal, rows.assign(is_anomaly=False), source_id="S", channel_id="C")
    assert len(fig.layout.shapes) == 0
    with pytest.raises(ValueError, match="no decisions"):
        plot_spot_windows(signal, rows, source_id="wrong", channel_id="C")
    with pytest.raises(ValueError, match="duplicate"):
        plot_spot_windows(signal, pd.concat([rows, rows]), source_id="S", channel_id="C")
    with pytest.raises(ValueError, match="invalid window bounds"):
        plot_spot_windows(signal, rows.assign(window_end=rows.window_start - pd.Timedelta(seconds=1)), source_id="S", channel_id="C")


def test_spot_to_plot_cli(tmp_path):
    signal, rows = fixture_data()
    calibration = pd.DataFrame(dict(source_id="S", channel_id="C",
                                    score=np.random.default_rng(4).exponential(size=200)))
    scores = rows.drop(columns=["threshold", "is_anomaly"]).assign(score=[.1, 100., .1])
    manifest = build_manifest("scores", config={}, model_id="m", preprocessing_id="p", score_convention="higher")
    cal, score, decisions = [tmp_path / f"{name}.parquet" for name in ("cal", "scores", "decisions")]
    write_table(calibration, cal, manifest)
    write_table(scores, score, manifest)
    result = apply_spot(cal, score, decisions, tmp_path / "spot.pkl",
                        {"q": .01, "initial_quantile": .8, "min_excesses": 5})
    assert result.is_anomaly.tolist() == [False, True, False]
    csv = tmp_path / "signal.csv"
    signal.rename_axis("time").to_csv(csv, sep=";")
    html = tmp_path / "report.html"
    assert main(["plot-spot", str(csv), str(decisions), str(html),
                 "--source-id", "S", "--channel-id", "C", "--sep", ";"]) is None
    content = html.read_text()
    assert "plotly.js" in content and "Anomalous windows" in content
    assert "1/3 flagged windows" in content.replace(r"\u002f", "/")
