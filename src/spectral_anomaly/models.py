"""Interchangeable CPU sklearn and genuine GPU cuML model backends."""
from __future__ import annotations
import pickle
import numpy as np
from .devices import resolve_backend

class AtypicalityModel:
    def __init__(self, *, backend="auto", model_type="isolation_forest", **params):
        self.backend=resolve_backend(backend); self.model_type=model_type; self.params=params; self.model=None
    def fit(self,x):
        if self.model_type != "isolation_forest": raise ValueError("unsupported atypicality model")
        if self.backend=="gpu":
            from cuml.ensemble import IsolationForest
        else:
            from sklearn.ensemble import IsolationForest
        self.model=IsolationForest(**self.params).fit(x); return self
    def score(self,x):
        if self.model is None: raise RuntimeError("model is not fitted")
        values=-self.model.score_samples(x)
        return np.asarray(values.get() if hasattr(values,"get") else values,float)

class HDBSCANModel:
    def __init__(self, *, backend="auto", **params): self.backend=resolve_backend(backend); self.params=params; self.model=None
    def fit_predict(self,x):
        if self.backend=="gpu": from cuml.cluster import HDBSCAN
        else:
            if self.params.get("prediction_data"):
                try: from hdbscan import HDBSCAN
                except ImportError as exc: raise ImportError("CPU HDBSCAN requires scikit-learn>=1.3 or hdbscan") from exc
            else:
                try: from sklearn.cluster import HDBSCAN
                except ImportError:
                    try: from hdbscan import HDBSCAN
                    except ImportError as exc: raise ImportError("CPU HDBSCAN requires scikit-learn>=1.3 or hdbscan") from exc
        self.model=HDBSCAN(**self.params); result=self.model.fit_predict(x)
        return np.asarray(result.get() if hasattr(result,"get") else result,int)

def save_artifact(value,path):
    with open(path,"wb") as stream: pickle.dump(value,stream)
def load_artifact(path):
    with open(path,"rb") as stream: return pickle.load(stream)
