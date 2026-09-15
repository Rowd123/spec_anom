"""Feature extraction from final masks (never averages pre-merge features)."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .spectral import SpectralResult

FEATURE_MEANING = {
 "integrated_energy":"physical_stft_psd", "mean_energy_density":"physical_stft_psd",
 "central_frequency":"physical_stft_psd", "frequency_dispersion":"physical_stft_psd",
 "local_energy_contrast":"physical_stft_psd", "time_frequency_area":"geometry",
 "duration":"geometry", "frequency_width":"geometry", "temporal_variation":"morphology",
 "frequency_variation":"morphology",
}

def extract_segment_features(segment, spectral: SpectralResult, *, window_id=0, window_start=None):
    mask=np.asarray(segment.mask,bool)
    if mask.shape != spectral.values.shape: raise ValueError("mask and spectral map shapes differ")
    rows,cols=np.nonzero(mask)
    base={"window_id":window_id,"window_start":window_start,"segment_id":segment.segment_id,
          "source_segment_ids":segment.source_segment_ids,"merge_count":segment.merge_count}
    if not len(rows): return {**base, **{x:0.0 for x in FEATURE_MEANING}}
    dt=float(np.median(np.diff(spectral.times))) if len(spectral.times)>1 else 1/spectral.sampling_frequency
    df=float(np.median(np.diff(spectral.frequencies))) if len(spectral.frequencies)>1 else spectral.sampling_frequency/2
    pixel_area=dt*df; area=len(rows)*pixel_area; weighted=spectral.psd[mask]; energy=float(weighted.sum()*pixel_area)
    weights=np.maximum(weighted,0); total=weights.sum(); freqs=spectral.frequencies[rows]
    center=float(np.average(freqs,weights=weights)) if total else float(freqs.mean())
    dispersion=float(np.sqrt(np.average((freqs-center)**2,weights=weights))) if total else float(freqs.std())
    outside=spectral.psd[~mask]; background=float(np.median(outside)) if outside.size else 0.0
    return {**base,"integrated_energy":energy,"time_frequency_area":area,
      "mean_energy_density":energy/area if area else 0.,"duration":(cols.max()-cols.min()+1)*dt,
      "frequency_width":(rows.max()-rows.min()+1)*df,"central_frequency":center,
      "frequency_dispersion":dispersion,"temporal_variation":float(np.std(np.bincount(cols,minlength=mask.shape[1]))),
      "frequency_variation":float(np.std(np.bincount(rows,minlength=mask.shape[0]))),
      "local_energy_contrast":float(weighted.mean()/max(background,1e-12))}

def segments_to_dataframe(segments, spectral, **metadata):
    return pd.DataFrame([extract_segment_features(s,spectral,**metadata) for s in segments])
