"""Train/calibrate/analyze example using the three configuration files."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
from spectral_anomaly import load_configs, score_atypicality_splits, train_models

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument("--spectral-config",default="configs/spectral_analysis.json"); p.add_argument("--sam-config",default="configs/sam.json"); p.add_argument("--models-config",default="configs/models.json")
    a=p.parse_args(argv); _,_,cfg=load_configs(a.spectral_config,a.sam_config,a.models_config)
    rng=np.random.default_rng(cfg["reproducibility"]["random_seed"]); n=200
    frame=pd.DataFrame({"window_start":pd.date_range("2024-01-01",periods=n,freq="min")})
    for f in set(cfg["preprocessing"]["atypicality_features"]+cfg["preprocessing"]["hdbscan_features"]): frame[f]=np.abs(rng.normal(size=n))+.01
    pipeline,splits=train_models(frame,cfg); result=score_atypicality_splits(pipeline,splits); print(result[["split","atypicality_score"]].tail()); return result
if __name__ == "__main__": main()
