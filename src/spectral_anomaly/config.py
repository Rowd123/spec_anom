"""Loading and strict validation of the three independent configurations."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

REQUIRED = {
    "spectral": {"sampling_frequency", "sampling_period", "windowing", "quality",
                 "signal_preprocessing", "representation", "transform", "msst",
                 "psd", "device", "visualization"},
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
        if set(result) != REQUIRED["spectral"]:
            raise ValueError(f"unknown spectral configuration section(s): {sorted(set(result) - REQUIRED['spectral'])}")
        if result["representation"] not in {"stft", "msst"}:
            raise ValueError("representation must be stft or msst")
        frequency = float(result["sampling_frequency"])
        if not frequency > 0:
            raise ValueError("sampling_frequency must be positive")
        period_seconds = float(pd.to_timedelta(result["sampling_period"]).total_seconds())
        if not np.isclose(period_seconds, 1 / frequency):
            raise ValueError("sampling_period must equal 1 / sampling_frequency")
        windowing = result["windowing"]
        if set(windowing) != {"size", "overlap"}:
            raise ValueError("windowing must contain only size and overlap")
        if not isinstance(windowing["size"], int) or not 0 <= windowing["overlap"] < windowing["size"]:
            raise ValueError("windowing requires an integer size and 0 <= overlap < size")
        quality = result["quality"]
        if set(quality) != {"quality_column", "valid_flags", "min_valid_fraction", "max_interpolation_gap"}:
            raise ValueError("invalid quality configuration keys")
        if not 0 <= quality["min_valid_fraction"] <= 1 or quality["max_interpolation_gap"] < 0:
            raise ValueError("invalid quality thresholds")
        preprocessing = result["signal_preprocessing"]
        if set(preprocessing) != {"remove_mean"} or not isinstance(preprocessing["remove_mean"], bool):
            raise ValueError("signal_preprocessing.remove_mean must be boolean")
        transform = result["transform"]
        expected_transform = {"window", "window_length", "n_fft", "hop_length", "center", "padtype", "dtype"}
        if set(transform) != expected_transform:
            raise ValueError("transform configuration has missing or unknown keys")
        if transform["window_length"] > windowing["size"] or transform["n_fft"] < transform["window_length"]:
            raise ValueError("require window_length <= window size and n_fft >= window_length")
        if result["psd"] != {"scaling": "density", "one_sided": True}:
            raise ValueError("only one-sided density PSD is currently supported")
        if set(result["msst"]) != {"iteration_count", "gamma"}:
            raise ValueError("msst must contain only iteration_count and gamma")
        if result["msst"]["iteration_count"] < 1:
            raise ValueError("msst.iteration_count must be positive")
        if set(result["visualization"]) != {"compare_representations", "output"}:
            raise ValueError("visualization must contain compare_representations and output")
        if not isinstance(result["visualization"]["compare_representations"], bool):
            raise ValueError("compare_representations must be boolean")
    elif kind == "sam":
        if result["segmentation_mode"] not in {"automatic", "energy_contrast"}: raise ValueError("invalid segmentation_mode")
        if set(result["model"]) != {"checkpoint", "config"}:
            raise ValueError("SAM model must contain checkpoint and config")
        if result["device"] not in {"auto", "cpu", "cuda"} and not (
                isinstance(result["device"], str) and result["device"].startswith("cuda:")
                and result["device"][5:].isdigit()):
            raise ValueError("SAM device must be auto, cpu, cuda, or cuda:N")
        image = result["image"]
        if set(image) != {"transform", "percentiles", "epsilon"}:
            raise ValueError("SAM image configuration has missing or unknown keys")
        if image["transform"] not in {"log", "linear"}:
            raise ValueError("SAM image transform must be log or linear")
        guided = result["energy_contrast"]
        guided_keys = {"exclusion_seconds", "neighborhood_seconds", "epsilon",
                       "min_valid_references", "contrast_threshold", "min_time_spacing",
                       "min_frequency_spacing", "max_points", "mask_iou_threshold",
                       "multimask_output"}
        if set(guided) != guided_keys or not isinstance(guided["multimask_output"], bool):
            raise ValueError("invalid energy_contrast configuration")
        if not 0 <= guided["mask_iou_threshold"] <= 1:
            raise ValueError("energy_contrast.mask_iou_threshold must be in [0, 1]")
        from .sam_segmentation import validate_automatic_mask_options
        validate_automatic_mask_options(result["automatic"])
        post = result["mask_postprocessing"]
        if post["strategy"] not in {"merge", "deduplicate", "none"}: raise ValueError("invalid mask strategy")
        if not isinstance(post.get("enabled"), bool) or not isinstance(post.get("containment_enabled"), bool):
            raise ValueError("mask postprocessing switches must be boolean")
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
