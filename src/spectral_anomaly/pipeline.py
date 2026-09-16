"""Composable DataFrame-to-segments and segments-to-models orchestration."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .energy import prepare_analysis_windows
from .features import segments_to_dataframe
from .masks import postprocess_masks
from .models import AtypicalityModel
from .preprocessing import FeaturePreprocessor, chronological_split
from .segmentation import SAMSegmentationSession
from .spectral import analyze_spectrum, validate_features


_FROM_CONFIG = object()


def dataframe_to_segments(
    frame,
    spectral_config,
    sam_config,
    *,
    value_col="value",
    quality_col=_FROM_CONFIG,
    valid_quality_flags=_FROM_CONFIG,
    window_overlap=None,
    automatic_segmenter=None,
    predictor=None,
    return_window_metadata=False,
):
    """Run quality-aware window preparation, STFT, SAM, and feature extraction."""
    windowing = spectral_config["windowing"]
    quality = spectral_config["quality"]
    if quality_col is _FROM_CONFIG:
        quality_col = quality.get("quality_column")
    if valid_quality_flags is _FROM_CONFIG:
        valid_quality_flags = quality.get("valid_flags")
    metadata, windows = prepare_analysis_windows(
        frame,
        value_col=value_col,
        sampling_period=spectral_config["sampling_period"],
        window_size=windowing["size"],
        overlap=windowing["overlap"] if window_overlap is None else window_overlap,
        min_valid_ratio=quality["min_valid_fraction"],
        max_interpolation_gap=quality["max_interpolation_gap"],
        quality_col=quality_col,
        valid_quality_flags=valid_quality_flags,
    )
    session = SAMSegmentationSession(
        sam_config, automatic_segmenter=automatic_segmenter, predictor=predictor
    )
    rows = []
    diagnostics = []
    for window_id, item in windows.items():
        spectrum = analyze_spectrum(item.signal, spectral_config)
        segmentation = session.segment(spectrum)
        final = postprocess_masks(
            segmentation.segments, **sam_config["mask_postprocessing"]
        )
        start = metadata.loc[window_id, "start_time"]
        rows.append(
            segments_to_dataframe(
                final, spectrum, window_id=window_id, window_start=start
            )
        )
        diagnostics.append(
            {
                "window_id": window_id,
                "spectral": spectrum,
                "sam_image": segmentation.image,
                "raw_segments": segmentation.raw_segments,
                "raw_sam_segments": segmentation.raw_sam_segments,
                "segments": final,
                "points": segmentation.points,
                "contrast": segmentation.contrast,
                "guided": segmentation.guided,
                "sam_device": session.device,
            }
        )
    segments = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if return_window_metadata:
        return segments, diagnostics, metadata
    return segments, diagnostics


def _assert_finite_features(frame, features, *, label):
    values = frame.loc[:, features].to_numpy(dtype=float)
    invalid = {
        "NaN": int(np.isnan(values).sum()),
        "+inf": int(np.isposinf(values).sum()),
        "-inf": int(np.isneginf(values).sum()),
    }
    if any(invalid.values()):
        details = ", ".join(f"{name}={count}" for name, count in invalid.items())
        raise ValueError(f"non-finite atypicality features in {label}: {details}")


@dataclass
class ModelPipeline:
    """Isolation Forest pipeline; SPOT and HDBSCAN are deliberately separate."""

    anomaly_preprocessor: FeaturePreprocessor
    atypicality: AtypicalityModel

    def score_atypicality(self, frame):
        """Return raw ``-score_samples`` values (higher means more atypical)."""
        _assert_finite_features(
            frame, self.anomaly_preprocessor.features, label="scoring data"
        )
        transformed = self.anomaly_preprocessor.transform(frame)
        return self.atypicality.score(transformed)

    def analyze(self, frame):
        """Attach only the continuous atypicality score to segment rows."""
        result = frame.copy()
        result["atypicality_score"] = self.score_atypicality(frame)
        return result


def train_atypicality(frame, config):
    """Fit preprocessing and Isolation Forest on train windows only."""
    train, calibration, evaluation = chronological_split(frame, config["split"])
    preprocessing = config["preprocessing"]
    features = tuple(preprocessing["atypicality_features"])
    validate_features(features, "stft")
    if train.empty:
        raise ValueError("the chronological train split contains no segments")
    for name, split in (
        ("train", train),
        ("calibration", calibration),
        ("evaluation", evaluation),
    ):
        _assert_finite_features(split, features, label=name)
    preprocessor = FeaturePreprocessor(
        features,
        tuple(name for name in preprocessing.get("log1p", []) if name in features),
        preprocessing.get("scaling", "robust"),
    )
    transformed_train = preprocessor.fit_transform(train)
    model_config = dict(config["atypicality"])
    backend = model_config.pop("backend")
    model_type = model_config.pop("type")
    model_config.setdefault("random_state", config["reproducibility"]["random_seed"])
    atypicality = AtypicalityModel(
        backend=backend, model_type=model_type, **model_config
    ).fit(transformed_train)
    return ModelPipeline(preprocessor, atypicality), (train, calibration, evaluation)


def score_atypicality_splits(pipeline, splits):
    """Score all splits with one fitted preprocessor/model and retain row metadata."""
    results = []
    for label, frame in zip(("train", "calibration", "evaluation"), splits):
        scored = pipeline.analyze(frame)
        scored["split"] = label
        results.append(scored)
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


def train_models(frame, config):
    """Backward-compatible name for Isolation-Forest-only training."""
    return train_atypicality(frame, config)
