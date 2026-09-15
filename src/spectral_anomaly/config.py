"""Loading and strict validation of the three independent configurations."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Mapping

REQUIRED = {
    "spectral": {"sampling_frequency", "windowing", "representation", "transform", "device"},
    "sam": {"model", "device", "image", "segmentation_mode", "automatic", "energy_contrast", "mask_postprocessing"},
    "models": {"preprocessing", "split", "atypicality", "spot", "hdbscan", "reproducibility"},
}

def validate_config(data: Mapping[str, Any], kind: str) -> dict[str, Any]:
    if kind not in REQUIRED:
        raise ValueError(f"unknown configuration kind: {kind}")
    missing = REQUIRED[kind] - data.keys()
    if missing:
        raise ValueError(f"missing {kind} configuration section(s): {sorted(missing)}")
    result = dict(data)
    if kind == "spectral":
        if result["representation"] not in {"stft", "msst"}: raise ValueError("representation must be stft or msst")
        if float(result["sampling_frequency"]) <= 0: raise ValueError("sampling_frequency must be positive")
    elif kind == "sam":
        if result["segmentation_mode"] not in {"automatic", "energy_contrast"}: raise ValueError("invalid segmentation_mode")
        post = result["mask_postprocessing"]
        if post["strategy"] not in {"merge", "deduplicate", "none"}: raise ValueError("invalid mask strategy")
        for key in ("iou_threshold", "containment_threshold"):
            if key in post and not 0 <= float(post[key]) <= 1: raise ValueError(f"{key} must be in [0, 1]")
    else:
        split = result["split"]
        if abs(sum(float(split[x]) for x in ("train_fraction", "calibration_fraction", "evaluation_fraction"))-1) > 1e-9:
            raise ValueError("split fractions must sum to one")
    return result

def load_config(path: str | Path, kind: str) -> dict[str, Any]:
    with Path(path).open(encoding="utf8") as stream:
        return validate_config(json.load(stream), kind)

def load_configs(spectral: str | Path, sam: str | Path, models: str | Path):
    return load_config(spectral, "spectral"), load_config(sam, "sam"), load_config(models, "models")
