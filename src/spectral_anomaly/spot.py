"""Small CPU Peaks-Over-Threshold calibrator; decision remains separate from score."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.stats import genpareto

@dataclass
class SPOT:
    q: float=1e-3; initial_quantile: float=.98; min_excesses: int=5
    threshold_: float|None=None
    def fit(self,scores):
        x=np.asarray(scores,float); initial=float(np.quantile(x,self.initial_quantile)); excess=x[x>initial]-initial
        if len(excess)<self.min_excesses: raise ValueError("not enough excesses for SPOT calibration")
        shape,_,scale=genpareto.fit(excess,floc=0); tail=max(1-self.initial_quantile,1/len(x))
        probability=max(1-self.q/tail,0)
        self.threshold_=initial+float(genpareto.ppf(probability,shape,loc=0,scale=scale)); return self
    def predict(self,scores):
        if self.threshold_ is None: raise RuntimeError("SPOT is not calibrated")
        return np.asarray(scores)>self.threshold_
