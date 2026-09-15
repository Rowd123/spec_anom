"""Composable DataFrame-to-segments and segments-to-models orchestration."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .energy import prepare_analysis_windows
from .features import segments_to_dataframe
from .masks import postprocess_masks
from .models import AtypicalityModel, HDBSCANModel
from .preprocessing import FeaturePreprocessor, chronological_split
from .segmentation import segment_spectrum
from .spectral import analyze_spectrum, validate_features
from .spot import SPOT

def dataframe_to_segments(frame, spectral_config, sam_config, *, value_col="value", automatic_segmenter=None, predictor=None):
    w=spectral_config["windowing"]; q=spectral_config["quality"]
    metadata, windows=prepare_analysis_windows(frame,value_col=value_col,
      sampling_period=spectral_config["sampling_period"], window_size=w["size"],overlap=w["overlap"],
      min_valid_ratio=q["min_valid_fraction"],max_interpolation_gap=q["max_interpolation_gap"],
      quality_col=q.get("quality_column"),valid_quality_flags=q.get("valid_flags"))
    rows=[]; diagnostics=[]
    for window_id,item in windows.items():
        spectrum=analyze_spectrum(item.centered,spectral_config)
        image,raw,points=segment_spectrum(spectrum,sam_config,automatic_segmenter=automatic_segmenter,predictor=predictor)
        final=postprocess_masks(raw,**sam_config["mask_postprocessing"])
        start=metadata.loc[window_id,"start_time"] if "start_time" in metadata else window_id
        rows.append(segments_to_dataframe(final,spectrum,window_id=window_id,window_start=start))
        diagnostics.append({"window_id":window_id,"spectral":spectrum,"sam_image":image,"raw_segments":raw,"segments":final,"points":points})
    return (pd.concat(rows,ignore_index=True) if rows else pd.DataFrame()), diagnostics

@dataclass
class ModelPipeline:
    anomaly_preprocessor: FeaturePreprocessor
    cluster_preprocessor: FeaturePreprocessor
    atypicality: AtypicalityModel
    spot: SPOT
    clustering: HDBSCANModel
    def analyze(self,frame):
        scores=self.atypicality.score(self.anomaly_preprocessor.transform(frame)); decisions=self.spot.predict(scores)
        clusters=self.clustering.fit_predict(self.cluster_preprocessor.transform(frame))
        result=frame.copy(); result["atypicality_score"]=scores; result["spot_anomaly"]=decisions; result["cluster_id"]=clusters
        return result

def train_models(frame, config):
    train,calibration,evaluation=chronological_split(frame,config["split"]); p=config["preprocessing"]
    af=tuple(p["atypicality_features"]); cf=tuple(p["hdbscan_features"])
    validate_features(af,"stft"); validate_features(cf,"stft")
    ap=FeaturePreprocessor(af,tuple(x for x in p.get("log1p",[]) if x in af),p.get("scaling","robust")); cp=FeaturePreprocessor(cf,tuple(x for x in p.get("log1p",[]) if x in cf),p.get("scaling","robust"))
    seed=config["reproducibility"]["random_seed"]; ac=dict(config["atypicality"]); backend=ac.pop("backend"); model_type=ac.pop("type"); ac.setdefault("random_state",seed)
    atyp=AtypicalityModel(backend=backend,model_type=model_type,**ac).fit(ap.fit_transform(train))
    spot=SPOT(**config["spot"]).fit(atyp.score(ap.transform(calibration)))
    hc=dict(config["hdbscan"]); hb=hc.pop("backend"); cluster=HDBSCANModel(backend=hb,**hc); cp.fit(train)
    return ModelPipeline(ap,cp,atyp,spot,cluster),(train,calibration,evaluation)
