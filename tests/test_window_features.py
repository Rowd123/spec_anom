import copy
import json

import numpy as np
import pandas as pd
import pytest

from spectral_anomaly.artifacts import read_table, write_table, build_manifest
from spectral_anomaly.window_features import (
    prepare_windows, extract_window_features, split_window_features,
    window_descriptors, validate_window_features, validate_window_preprocessing,
)
from spectral_anomaly.preprocessing import chronological_split, selected_features
from spectral_anomaly.workflows import (
    train_isolation_forest, score_isolation_forest, fit_hdbscan, predict_hdbscan, apply_spot,
)
from spectral_anomaly.cli import main


def configs():
    def load(name):
        with open(f'configs/{name}.json') as stream:
            return json.load(stream)
    p, f, m = [load(n) for n in ('window_preprocessing','window_features','window_models')]
    p.update(sampling_frequency=10., sampling_period='100ms')
    p['windowing'] = {'size':100, 'overlap':50}
    p['normalization'] = {'V1.MAG': {'mu':100., 'sigma':2.}}
    f['spectral']['bands_hz'] = [[.1,.5],[.5,1.5],[1.5,3],[3,5]]
    f['history'].update(size=4,min_windows=2)
    m['atypicality'].update(backend='cpu',n_estimators=12,n_jobs=1)
    m['hdbscan'].update(backend='cpu',min_cluster_size=3,min_samples=2)
    return p,f,m


def data(n=4000):
    rng=np.random.default_rng(42)
    return pd.DataFrame({'value':100 + 2*rng.normal(size=n)},index=pd.date_range('2026-01-01',periods=n,freq='100ms',tz='UTC'))


def test_fixed_normalization_and_quality_reuse(tmp_path):
    from spectral_anomaly.energy import prepare_analysis_windows
    p,_,_=configs()
    x=data(500); x.iloc[20,0]=np.nan
    raw_meta, raw = prepare_analysis_windows(x,value_col='value',sampling_period='100ms',window_size=100,overlap=50)
    prepared=prepare_windows(x,tmp_path/'w.parquet',p,source_id='s',channel_id='V1.MAG')
    assert len(prepared)==len(raw)
    np.testing.assert_allclose(prepared.iloc[0].signal,(raw[0].signal-100)/2)
    assert prepared.iloc[0].interpolated_mask[20]
    p['normalization']['V1.MAG']['sigma']=None
    with pytest.raises(ValueError,match='finite mu'):
        validate_window_preprocessing(p)


def test_sinusoidal_band_power_and_acf():
    _,c,_=configs(); c['spectral'].update(window='boxcar',detrend=False)
    c['differences']=['variance']
    c=validate_window_features(c,10,100)
    x=3*np.sin(2*np.pi*np.arange(100)/10)
    result=window_descriptors(x,c,10)
    assert result['wf_band_1_power']==pytest.approx(4.5)
    assert result['wf_band_0_power']==pytest.approx(0,abs=1e-25)
    assert result['wf_variance']==pytest.approx(4.5)
    centered=x-x.mean()
    assert result['wf_acf_0']==pytest.approx(centered[:-1]@centered[1:]/(centered@centered))
    shifted=window_descriptors(x+200,c,10)
    assert shifted['wf_acf_0']==pytest.approx(result['wf_acf_0'])
    assert result['wf_diff_variance']==pytest.approx(np.diff(x).var())


def test_constant_policy_and_validation():
    _,c,_=configs()
    c=validate_window_features(c,10,100)
    r=window_descriptors(np.ones(100),c,10)
    assert r['wf_constant']==1 and r['wf_skewness']==0 and r['wf_acf_0']==0
    c['undefined_policy']='error'
    with pytest.raises(ValueError,match='constant'):
        window_descriptors(np.ones(100),c,10)
    _,c,_=configs(); c['spectral']['bands_hz']=[[.1,8]]
    with pytest.raises(ValueError,match='Nyquist'):
        validate_window_features(c,10,100)


def test_causality_and_gap_reset(tmp_path):
    p,c,_=configs()
    windows=prepare_windows(data(1000),tmp_path/'w.parquet',p,source_id='s',channel_id='V1.MAG')
    result=extract_window_features(tmp_path/'w.parquet',tmp_path/'f.parquet',c)
    original,manifest=read_table(tmp_path/'w.parquet')
    altered=original.copy()
    for i in altered.index[10:]:
        altered.at[i,'signal']=np.asarray(altered.at[i,'signal'])*1000
    write_table(altered,tmp_path/'altered.parquet',manifest)
    second=extract_window_features(tmp_path/'altered.parquet',tmp_path/'g.parquet',c)
    cols=[n for n in result if n.startswith('wf_')]
    np.testing.assert_allclose(result.loc[result.window_start<original.iloc[10].window_start,cols],second.loc[second.window_start<original.iloc[10].window_start,cols])
    # A missing scheduled window clears history, including for overlapping windows.
    write_table(original.drop(index=8),tmp_path/'gap.parquet',manifest)
    gap=extract_window_features(tmp_path/'gap.parquet',tmp_path/'h.parquet',c)
    assert not set(original.iloc[9:11].window_id)&set(gap.window_id)


def test_splits_selection_and_models(tmp_path):
    p,c,m=configs()
    prepare_windows(data(),tmp_path/'w.parquet',p,source_id='s',channel_id='V1.MAG')
    features=extract_window_features(tmp_path/'w.parquet',tmp_path/'f.parquet',c)
    train,cal,evaluation=split_window_features(tmp_path/'f.parquet',tmp_path/'splits',m['split'])
    assert train.window_end.max()<cal.window_start.min()
    assert cal.window_end.max()<evaluation.window_start.min()
    m['preprocessing']['atypicality_exclude']=['wf_acf_*']
    artifact=train_isolation_forest(tmp_path/'splits/train.parquet',tmp_path/'if.pkl',m)
    assert not any(n.startswith('wf_acf_') for n in artifact['feature_columns'])
    np.testing.assert_equal(artifact['pipeline'].anomaly_preprocessor.center_,0)
    for name in ('calibration','evaluation'):
        score_isolation_forest(tmp_path/f'splits/{name}.parquet',tmp_path/'if.pkl',tmp_path/f'{name}_scores.parquet',batch_size=3)
    # SPOT unchanged, with permissive calibration for this small synthetic fixture.
    decision=apply_spot(tmp_path/'calibration_scores.parquet',tmp_path/'evaluation_scores.parquet',tmp_path/'decisions.parquet',tmp_path/'spot.pkl',{'q':.1,'initial_quantile':.5,'min_excesses':2})
    assert len(decision)==len(evaluation)
    result=fit_hdbscan(tmp_path/'splits/train.parquet',tmp_path/'hdb.pkl',tmp_path/'clusters.parquet',m)
    assert len(result)==len(train)
    assigned=predict_hdbscan(tmp_path/'splits/evaluation.parquet',tmp_path/'hdb.pkl',tmp_path/'assigned.parquet')
    assert len(assigned)==len(evaluation)
    assert 'wf_acf_0' in features  # Exclusion never changed extraction.
    with pytest.raises(ValueError,match='unknown selected'):
        selected_features({'atypicality_features':['typo']},'atypicality',features.columns)


def test_cli_returns_success_and_saves_independent_artifacts(tmp_path):
    p,c,m=configs()
    raw=data(1000); raw.index.name='time'; raw.to_csv(tmp_path/'input.csv',sep=';')
    for name,config in [('p',p),('f',c),('m',m)]:
        (tmp_path/f'{name}.json').write_text(json.dumps(config))
    assert main(['prepare-windows',str(tmp_path/'input.csv'),str(tmp_path/'w.parquet'),'--config',str(tmp_path/'p.json'),'--source-id','s','--channel-id','V1.MAG','--sep',';']) is None
    assert main(['extract-windows',str(tmp_path/'w.parquet'),str(tmp_path/'f.parquet'),'--config',str(tmp_path/'f.json')]) is None
    assert main(['train-iforest',str(tmp_path/'f.parquet'),str(tmp_path/'if.pkl'),'--config',str(tmp_path/'m.json')]) is None
