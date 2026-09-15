import json
import numpy as np
import pandas as pd
import pytest
from spectral_anomaly import (Segment, SPOT, analyze_spectrum, chronological_split,
    extract_segment_features, load_config, mask_iou, postprocess_masks,
    resolve_device, save_artifact, load_artifact)

def spectral_config(rep="stft"):
    cfg=load_config("configs/spectral_analysis.json","spectral"); cfg["representation"]=rep; cfg["device"]="cpu"; return cfg

@pytest.mark.parametrize("representation",["stft","msst"])
def test_common_spectral_api(representation):
    result=analyze_spectrum(np.sin(np.arange(256)/7),spectral_config(representation))
    assert result.representation==representation
    assert result.values.shape==result.psd.shape==result.stft.shape

def test_three_configs_validate():
    for path,kind in [("configs/spectral_analysis.json","spectral"),("configs/sam.json","sam"),("configs/models.json","models")]:
        assert load_config(path,kind)

def test_auto_device_falls_back_and_explicit_gpu_is_honest(monkeypatch):
    monkeypatch.setattr("spectral_anomaly.devices.cuda_available",lambda:False)
    assert resolve_device("auto").resolved=="cpu"
    with pytest.raises(RuntimeError): resolve_device("cuda")
    with pytest.raises(RuntimeError,match="no genuine"): resolve_device("cuda",gpu_supported=False)

def test_iou_transitive_merge_traceability_and_recomputed_feature():
    a=np.zeros((3,6),bool); a[:,0:3]=1; b=np.zeros_like(a); b[:,1:5]=1; c=np.zeros_like(a); c[:,3:6]=1
    assert mask_iou(a,b)==pytest.approx(.4)
    merged=postprocess_masks([Segment(9,a),Segment(2,b),Segment(7,c)],iou_threshold=.4)
    assert len(merged)==1 and merged[0].mask.all()
    assert merged[0].source_segment_ids==(2,7,9) and merged[0].merge_count==3
    spectrum=analyze_spectrum(np.sin(np.arange(256)/7),spectral_config())
    # Match actual spectral shape to prove extraction consumes union, not old features.
    x=np.zeros(spectrum.values.shape,bool); x[:2,:3]=1; y=np.zeros_like(x); y[:2,2:5]=1
    union=postprocess_masks([Segment(1,x),Segment(2,y)],iou_threshold=.15)[0]
    feature=extract_segment_features(union,spectrum)
    assert feature["time_frequency_area"] > extract_segment_features(Segment(1,x),spectrum)["time_frequency_area"]

def test_containment_is_optional():
    large=np.ones((4,4),bool); small=np.zeros((4,4),bool); small[0,0]=1
    assert len(postprocess_masks([Segment(1,large),Segment(2,small)],iou_threshold=.8))==2
    assert len(postprocess_masks([Segment(1,large),Segment(2,small)],iou_threshold=.8,containment_enabled=True,containment_threshold=.9))==1

def test_chronological_split_no_leakage():
    frame=pd.DataFrame({"window_start":range(10)})
    a,b,c=chronological_split(frame,{"time_column":"window_start","train_fraction":.6,"calibration_fraction":.2,"evaluation_fraction":.2})
    assert set(a.index).isdisjoint(b.index|c.index) and a.window_start.max()<b.window_start.min()<c.window_start.min()

def test_spot_and_persistence(tmp_path):
    rng=np.random.default_rng(2); spot=SPOT(q=.01,initial_quantile=.8,min_excesses=5).fit(rng.exponential(size=200))
    assert spot.predict([0,100]).tolist()==[False,True]
    path=tmp_path/"spot.pkl"; save_artifact(spot,path); assert load_artifact(path).threshold_==spot.threshold_
