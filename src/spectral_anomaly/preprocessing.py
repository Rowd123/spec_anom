"""Chronological splitting and independently configurable feature preprocessing."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

def chronological_split(frame, config):
    ordered=frame.sort_values(config.get("time_column","window_start"),kind="stable")
    n=len(ordered); a=int(n*config["train_fraction"]); b=a+int(n*config["calibration_fraction"])
    return ordered.iloc[:a].copy(),ordered.iloc[a:b].copy(),ordered.iloc[b:].copy()

@dataclass
class FeaturePreprocessor:
    features: tuple[str,...]; log1p: tuple[str,...]=(); method: str="robust"
    center_: np.ndarray|None=None; scale_: np.ndarray|None=None
    def fit(self, frame):
        x=self._raw(frame); self.center_=np.median(x,axis=0) if self.method=="robust" else x.mean(0)
        self.scale_=1.4826*np.median(np.abs(x-self.center_),axis=0) if self.method=="robust" else x.std(0)
        self.scale_[self.scale_==0]=1; return self
    def _raw(self,frame):
        x=frame.loc[:,self.features].to_numpy(float).copy()
        for name in self.log1p: x[:,self.features.index(name)]=np.log1p(x[:,self.features.index(name)])
        return x
    def transform(self,frame):
        if self.center_ is None: raise RuntimeError("preprocessor is not fitted")
        return (self._raw(frame)-self.center_)/self.scale_
    def fit_transform(self,frame): return self.fit(frame).transform(frame)
