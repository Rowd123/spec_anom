"""Composable DataFrame-to-segments and segments-to-models orchestration."""
from __future__ import annotations

from dataclasses import dataclass
import logging
from time import perf_counter

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
LOGGER = logging.getLogger(__name__)


def _elapsed(start):
    return perf_counter() - start


def _window_count(frame):
    """Count logical windows without materialising another copy of the rows."""
    if frame.empty:
        return 0
    return int(frame["window_id"].nunique()) if "window_id" in frame else len(frame)


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
    log_every=100,
):
    """Run quality-aware window preparation, STFT, SAM, and feature extraction."""
    if not isinstance(log_every, int) or isinstance(log_every, bool) or log_every < 1:
        raise ValueError("log_every must be a positive integer")
    windowing = spectral_config["windowing"]
    quality = spectral_config["quality"]
    if quality_col is _FROM_CONFIG:
        quality_col = quality.get("quality_column")
    if valid_quality_flags is _FROM_CONFIG:
        valid_quality_flags = quality.get("valid_flags")
    pipeline_started = perf_counter()
    LOGGER.info("[data-preparation] starting quality control and window preparation")
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
    preparation_duration = _elapsed(pipeline_started)
    rejection_counts = metadata.loc[~metadata["accepted"], "rejection_reason"].value_counts()
    LOGGER.info(
        "[data-preparation] finished in %.3fs: total=%d accepted=%d rejected=%d reasons=%s",
        preparation_duration, len(metadata), len(windows), len(metadata) - len(windows),
        rejection_counts.to_dict(),
    )
    sam_started = perf_counter()
    LOGGER.info(
        "[sam-load] loading checkpoint=%s configured_device=%s mode=%s",
        sam_config["model"]["checkpoint"], sam_config["device"],
        sam_config["segmentation_mode"],
    )
    session = SAMSegmentationSession(
        sam_config, automatic_segmenter=automatic_segmenter, predictor=predictor
    )
    sam_load_duration = _elapsed(sam_started)
    LOGGER.info("[sam-load] finished in %.3fs: device=%s", sam_load_duration, session.device)
    rows = []
    diagnostics = []
    processing_started = perf_counter()
    segments_kept = 0
    processed_without_segments = 0
    stft_total = 0.0
    sam_total = 0.0
    postprocessing_and_features_total = 0.0
    total = len(windows)
    LOGGER.info(
        "[windows] starting STFT, SAM segmentation, mask post-processing, and feature extraction: accepted=%d",
        total,
    )
    for window_id, item in windows.items():
        window_started = perf_counter()
        stft_started = perf_counter()
        spectrum = analyze_spectrum(item.signal, spectral_config)
        stft_duration = _elapsed(stft_started)
        stft_total += stft_duration
        sam_inference_started = perf_counter()
        segmentation = session.segment(spectrum)
        sam_duration = _elapsed(sam_inference_started)
        sam_total += sam_duration
        postprocessing_started = perf_counter()
        final = postprocess_masks(
            segmentation.segments, **sam_config["mask_postprocessing"]
        )
        start = metadata.loc[window_id, "start_time"]
        end = metadata.loc[window_id, "end_time"]
        segments_kept += len(final)
        processed_without_segments += not final
        rows.append(
            segments_to_dataframe(
                final, spectrum, window_id=window_id, window_start=start
            )
        )
        postprocessing_and_features_total += _elapsed(postprocessing_started)
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
                "stft_duration_seconds": stft_duration,
                "sam_duration_seconds": sam_duration,
            }
        )
        processed = len(diagnostics)
        LOGGER.debug(
            "[window] id=%s start=%s end=%s stft=%.3fs sam=%.3fs "
            "masks_raw=%d masks_before_postprocessing=%d masks_after_postprocessing=%d total=%.3fs",
            window_id, start, end, stft_duration, sam_duration,
            len(segmentation.raw_segments), len(segmentation.segments), len(final),
            _elapsed(window_started),
        )
        if processed % log_every == 0 or processed == total:
            elapsed = _elapsed(processing_started)
            rate = processed / elapsed if elapsed else 0.0
            remaining = (total - processed) / rate if rate else 0.0
            LOGGER.info(
                "[windows] processed=%d/%d accepted=%d rejected=%d "
                "without_segment=%d segments_kept=%d elapsed=%.1fs rate=%.2f windows/s eta=%.1fs",
                processed, total, processed, len(metadata) - len(windows),
                processed_without_segments, segments_kept, elapsed, rate, remaining,
            )
    segments = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    LOGGER.info(
        "[feature-extraction] finished: segments=%d accepted_windows=%d "
        "windows_without_segment=%d stage_durations=%s processing_duration=%.3fs "
        "total_duration=%.3fs",
        len(segments), total, processed_without_segments,
        {"preparation": round(preparation_duration, 3),
         "sam_load": round(sam_load_duration, 3), "stft": round(stft_total, 3),
         "sam": round(sam_total, 3),
         "postprocessing_and_features": round(postprocessing_and_features_total, 3)},
        _elapsed(processing_started), _elapsed(pipeline_started),
    )
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
        LOGGER.error(
            "[feature-validation] partition=%s non_finite=%s action=abort_no_exclusion",
            label, invalid,
        )
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
    LOGGER.info("[split] features=%s", list(features))
    for name, split in zip(("train", "calibration", "evaluation"), (train, calibration, evaluation)):
        LOGGER.info("[split] partition=%s windows=%d segments=%d", name, _window_count(split), len(split))
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
    preprocessing_started = perf_counter()
    LOGGER.info("[preprocessing] starting train_matrix=(%d, %d)", len(train), len(features))
    transformed_train = preprocessor.fit_transform(train)
    LOGGER.info("[preprocessing] finished in %.3fs output_shape=%s", _elapsed(preprocessing_started), transformed_train.shape)
    model_config = dict(config["atypicality"])
    backend = model_config.pop("backend")
    model_type = model_config.pop("type")
    model_config.setdefault("random_state", config["reproducibility"]["random_seed"])
    fit_started = perf_counter()
    LOGGER.info("[isolation-forest-fit] starting backend=%s parameters=%s", backend, model_config)
    atypicality = AtypicalityModel(
        backend=backend, model_type=model_type, **model_config
    ).fit(transformed_train)
    LOGGER.info("[isolation-forest-fit] finished in %.3fs resolved_backend=%s", _elapsed(fit_started), atypicality.backend)
    return ModelPipeline(preprocessor, atypicality), (train, calibration, evaluation)


def score_atypicality_splits(pipeline, splits):
    """Score all splits with one fitted preprocessor/model and retain row metadata."""
    results = []
    for label, frame in zip(("train", "calibration", "evaluation"), splits):
        started = perf_counter()
        LOGGER.info("[scoring] starting partition=%s matrix=(%d, %d)", label, len(frame), len(pipeline.anomaly_preprocessor.features))
        scored = pipeline.analyze(frame)
        scored["split"] = label
        results.append(scored)
        LOGGER.info("[scoring] finished partition=%s in %.3fs scores=%d", label, _elapsed(started), len(scored))
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


def train_models(frame, config):
    """Backward-compatible name for Isolation-Forest-only training."""
    return train_atypicality(frame, config)
