"""Inspect the configured physical time/frequency axes without SAM."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from spectral_anomaly import analyze_spectrum, load_config

def main(argv=None):
    parser=argparse.ArgumentParser(); parser.add_argument("--config",default="configs/spectral_analysis.json"); parser.add_argument("--output")
    args=parser.parse_args(argv); cfg=load_config(args.config,"spectral")
    t=np.arange(cfg["windowing"]["size"])/cfg["sampling_frequency"]
    signal=np.sin(2*np.pi*.08*t)+.35*np.sin(2*np.pi*.2*t)
    result=analyze_spectrum(signal,cfg)
    fig=make_subplots(rows=2,cols=1,subplot_titles=("Signal temporel",result.representation.upper()))
    fig.add_trace(go.Scatter(x=t,y=signal,name="signal"),row=1,col=1)
    fig.add_trace(go.Heatmap(x=result.times,y=result.frequencies,z=np.log1p(np.abs(result.values)),name=result.representation),row=2,col=1)
    fig.update_xaxes(title_text="temps (s)",row=2,col=1); fig.update_yaxes(title_text="fréquence (Hz)",row=2,col=1)
    output=args.output or cfg["visualization"]["output"]; fig.write_html(output); return Path(output)
if __name__ == "__main__": main()
