"""Visualise the selected spectral representation on two reference signals."""

import argparse
from copy import deepcopy
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from spectral_anomaly import analyze_spectrum, load_config


def reference_signals(size: int, sampling_frequency: float):
    """Return a frequency-check signal and a structural demonstration signal."""
    times = np.arange(size) / sampling_frequency
    simple = np.sin(2 * np.pi * 0.08 * times) + 0.35 * np.sin(2 * np.pi * 0.20 * times)

    duration = max(times[-1], 1 / sampling_frequency)
    persistent = 0.55 * np.sin(2 * np.pi * 0.08 * times)
    burst_envelope = np.exp(-0.5 * ((times - 0.48 * duration) / (0.06 * duration)) ** 2)
    burst = 1.4 * burst_envelope * np.sin(2 * np.pi * 0.22 * times)
    start_frequency, end_frequency = 0.025, 0.18
    chirp_rate = (end_frequency - start_frequency) / duration
    chirp_phase = 2 * np.pi * (start_frequency * times + 0.5 * chirp_rate * times ** 2)
    project = persistent + burst + 0.45 * np.sin(chirp_phase)
    return times, (("Contrôle à 0,08 et 0,20 Hz", simple),
                   ("Ton permanent, bouffée et chirp", project))


def _representations(signal, config):
    if not config["visualization"]["compare_representations"]:
        result = analyze_spectrum(signal, config)
        return ((result.representation.upper(), result),)
    results = []
    for representation in ("stft", "msst"):
        selected = deepcopy(config)
        selected["representation"] = representation
        results.append((representation.upper(), analyze_spectrum(signal, selected)))
    stft, msst = results[0][1], results[1][1]
    np.testing.assert_array_equal(stft.times, msst.times)
    np.testing.assert_array_equal(stft.frequencies, msst.frequencies)
    return tuple(results)


def build_figure(config):
    times, scenarios = reference_signals(config["windowing"]["size"], config["sampling_frequency"])
    representation_count = 2 if config["visualization"]["compare_representations"] else 1
    rows = len(scenarios) * (1 + representation_count)
    titles = []
    computed = []
    for scenario_name, signal in scenarios:
        products = _representations(signal, config)
        computed.append((scenario_name, signal, products))
        titles.extend((scenario_name, *(f"{scenario_name} — {name}" for name, _ in products)))
    figure = make_subplots(rows=rows, cols=1, shared_xaxes=False, subplot_titles=titles)
    row = 1
    for _, signal, products in computed:
        figure.add_trace(go.Scatter(x=times, y=signal, name="signal", showlegend=row == 1), row=row, col=1)
        figure.update_xaxes(title_text="temps (s)", row=row, col=1)
        figure.update_yaxes(title_text="amplitude", row=row, col=1)
        row += 1
        for name, result in products:
            figure.add_trace(go.Heatmap(
                x=result.times, y=result.frequencies, z=np.log1p(np.abs(result.values)),
                colorbar={"title": f"log1p |{name}|"}, name=name,
            ), row=row, col=1)
            figure.update_xaxes(title_text="centre temporel de trame (s)", row=row, col=1)
            figure.update_yaxes(title_text="fréquence (Hz)", row=row, col=1)
            row += 1
    figure.update_layout(height=330 * rows, template="plotly_white",
                         title="Analyse spectrale — aucune décision d'anomalie")
    return figure


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/spectral_analysis.json")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    config = load_config(args.config, "spectral")
    output = Path(args.output or config["visualization"]["output"])
    build_figure(config).write_html(output)
    return output


if __name__ == "__main__":
    main()
