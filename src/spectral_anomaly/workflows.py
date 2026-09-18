"""Persistent stage workflows; none of these functions reads raw sensor data."""
from __future__ import annotations

import logging
from pathlib import Path
from time import perf_counter
import uuid

import numpy as np
import pandas as pd

from .artifacts import (SCORE_CONVENTION, build_manifest, fingerprint, read_table,
                        validate_segment_table, write_table)
from .models import HDBSCANModel, load_artifact, save_artifact
from .pipeline import ModelPipeline, train_atypicality
from .preprocessing import FeaturePreprocessor
from .spot import SPOT
from .features import FEATURE_MEANING
from .selection import validate_segment_selection
from .pipeline import dataframe_to_segments

LOGGER = logging.getLogger(__name__)


def _artifact_id(prefix: str, payload) -> str:
    return f"{prefix}-{fingerprint(payload)[:16]}"


def extract_segments(frame, output_path, spectral_config, sam_config, *, source_id,
                     channel_id, value_col="value", save_arrays=None, resume=False,
                     automatic_segmenter=None, predictor=None):
    """Run extraction once and persist raw, pre-model features and traceability."""
    spectral_config = {**spectral_config, "segment_selection":
                       validate_segment_selection(spectral_config.get("segment_selection", {}))}
    segments, diagnostics, windows = dataframe_to_segments(
        frame, spectral_config, sam_config, value_col=value_col,
        return_window_metadata=True, automatic_segmenter=automatic_segmenter,
        predictor=predictor, source_id=str(source_id),
    )
    segments = segments.drop(columns=["source_id"], errors="ignore")
    segments.insert(0, "source_id", str(source_id)); segments.insert(1, "channel_id", str(channel_id))
    if len(segments):
        bounds = []
        lookup = {item["window_id"]: item for item in diagnostics}
        for row in segments.itertuples(index=False):
            item = lookup[row.window_id]; spectral = item["spectral"]
            segment = next(x for x in item["segments"] if x.segment_id == row.segment_id)
            yy, xx = np.nonzero(segment.mask)
            start = pd.Timestamp(row.window_start)
            bounds.append((windows.loc[row.window_id, "end_time"], start + pd.to_timedelta(float(spectral.times[xx.min()]), unit="s"),
                           start + pd.to_timedelta(float(spectral.times[xx.max()]), unit="s"),
                           float(spectral.frequencies[yy.min()]), float(spectral.frequencies[yy.max()]),
                           getattr(segment, "predicted_iou", np.nan), getattr(segment, "stability_score", np.nan)))
        segments[["window_end", "time_start", "time_end", "frequency_min", "frequency_max", "predicted_iou", "stability_score"]] = pd.DataFrame(bounds, index=segments.index)
    else:
        for name in ("window_end", "time_start", "time_end", "frequency_min", "frequency_max", "predicted_iou", "stability_score"): segments[name] = pd.Series(dtype=float)
    features = tuple(name for name in FEATURE_MEANING if name in segments)
    manifest = build_manifest("segments", config={"spectral": spectral_config, "sam": sam_config}, features=features,
                              units={name: FEATURE_MEANING[name] for name in features},
                              provenance={"source_id": str(source_id), "channel_id": str(channel_id),
                                          "selection_audit_scope": "current_extraction_call",
                                          "segment_selection": {
                                              key: sum(item[key] for item in diagnostics)
                                              for key in ("number_of_segments_before_selection",
                                                          "number_of_segments_after_selection",
                                                          "number_of_segments_rejected_by_energy")},
                                          "rejected_segments": [record for item in diagnostics
                                                                for record in item["rejected_segments"]]})
    write_table(segments, output_path, manifest, resume=resume)
    if save_arrays:
        directory = Path(save_arrays); directory.mkdir(parents=True, exist_ok=True)
        for item in diagnostics:
            np.savez_compressed(directory / f"window-{item['window_id']}.npz", stft=item["spectral"].stft,
                               psd=item["spectral"].psd, frequencies=item["spectral"].frequencies,
                               times=item["spectral"].times, masks=np.asarray([x.mask for x in item["segments"]], bool))
    return segments


def train_isolation_forest(segments_path, model_path, config):
    frame, source = read_table(segments_path, expected_type="segments")
    features = tuple(config["preprocessing"]["atypicality_features"])
    validate_segment_table(frame, features)
    started = perf_counter()
    pipeline, splits = train_atypicality(frame, config)
    preprocessing_id = _artifact_id("prep", {"features": features, "center": pipeline.anomaly_preprocessor.center_.tolist(), "scale": pipeline.anomaly_preprocessor.scale_.tolist()})
    model_id = _artifact_id("iforest", {"config": config["atypicality"], "preprocessing_id": preprocessing_id, "nonce": uuid.uuid4().hex})
    artifact = {"kind": "isolation_forest", "schema_version": "1.0", "model_id": model_id,
                "preprocessing_id": preprocessing_id, "score_convention": SCORE_CONVENTION,
                "feature_columns": features, "source_config_hash": source.config_hash,
                "pipeline": pipeline, "config": config}
    save_artifact(artifact, model_path)
    LOGGER.info("trained Isolation Forest model_id=%s rows=%d backend=%s duration=%.3fs", model_id, len(splits[0]), pipeline.atypicality.backend, perf_counter()-started)
    return artifact


def score_isolation_forest(segments_path, model_path, output_path, *, batch_size=65536, resume=False):
    frame, source = read_table(segments_path, expected_type="segments")
    artifact = load_artifact(model_path)
    features = tuple(artifact["feature_columns"])
    validate_segment_table(frame, features)
    if source.config_hash != artifact["source_config_hash"]:
        LOGGER.warning("segment extraction config differs from training; feature contract is still compatible")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    scores = []
    started = perf_counter()
    for start in range(0, len(frame), batch_size):
        scores.append(artifact["pipeline"].score_atypicality(frame.iloc[start:start+batch_size]))
    result = frame[[c for c in frame if c.endswith("_id") or c in {"window_start", "window_end", "time_start", "time_end"}]].copy()
    result["model_id"] = artifact["model_id"]
    result["preprocessing_id"] = artifact["preprocessing_id"]
    result["score"] = np.concatenate(scores) if scores else np.empty(0)
    manifest = build_manifest("scores", config={"batch_size": batch_size}, provenance={"segments": str(segments_path), "model": str(model_path)}, model_id=artifact["model_id"], preprocessing_id=artifact["preprocessing_id"], score_convention=SCORE_CONVENTION)
    write_table(result, output_path, manifest, resume=resume)
    LOGGER.info("scored rows=%d batch_size=%d duration=%.3fs", len(result), batch_size, perf_counter()-started)
    return result


def fit_hdbscan(segments_path, model_path, output_path, config):
    frame, source = read_table(segments_path, expected_type="segments")
    features = tuple(config["preprocessing"]["hdbscan_features"])
    validate_segment_table(frame, features)
    prep = FeaturePreprocessor(features, tuple(x for x in config["preprocessing"].get("log1p", ()) if x in features), config["preprocessing"].get("scaling", "robust"))
    matrix = prep.fit_transform(frame)
    options = dict(config["hdbscan"]); backend = options.pop("backend")
    model = HDBSCANModel(backend=backend, **options)
    labels = model.fit_predict(matrix)
    strengths = getattr(model.model, "probabilities_", np.full(len(labels), np.nan))
    strengths = np.asarray(strengths.get() if hasattr(strengths, "get") else strengths, float)
    preprocessing_id = _artifact_id("prep", {"features": features, "center": prep.center_.tolist(), "scale": prep.scale_.tolist()})
    model_id = _artifact_id("hdbscan", {"config": options, "preprocessing_id": preprocessing_id, "nonce": uuid.uuid4().hex})
    artifact = {"kind": "hdbscan", "schema_version": "1.0", "model_id": model_id, "preprocessing_id": preprocessing_id, "feature_columns": features, "preprocessor": prep, "model": model, "config": config}
    save_artifact(artifact, model_path)
    result = frame[[c for c in frame if c.endswith("_id") or c in {"window_start", "time_start"}]].copy()
    result["model_id"] = model_id; result["cluster_label"] = labels; result["is_noise"] = labels == -1; result["membership_strength"] = strengths
    write_table(result, output_path, build_manifest("partitions", config=config["hdbscan"], provenance={"segments": str(segments_path)}, model_id=model_id, preprocessing_id=preprocessing_id))
    return result


def predict_hdbscan(segments_path, model_path, output_path):
    frame, _ = read_table(segments_path, expected_type="segments"); artifact = load_artifact(model_path)
    matrix = artifact["preprocessor"].transform(frame); model = artifact["model"].model
    try:
        from hdbscan import approximate_predict
    except ImportError as exc:
        raise RuntimeError("existing-cluster assignment requires the hdbscan package and prediction_data=True during fitting") from exc
    labels, strengths = approximate_predict(model, matrix)
    result = frame[[c for c in frame if c.endswith("_id") or c in {"window_start", "time_start"}]].copy()
    result["model_id"] = artifact["model_id"]; result["cluster_label"] = labels; result["is_noise"] = labels == -1; result["membership_strength"] = strengths
    write_table(result, output_path, build_manifest("partitions", config={"mode": "approximate_predict"}, provenance={"segments": str(segments_path)}, model_id=artifact["model_id"], preprocessing_id=artifact["preprocessing_id"]))
    return result


def apply_spot(calibration_path, scores_path, output_path, state_path, config, *, series_columns=("source_id", "channel_id"), resume=False):
    calibration, cm = read_table(calibration_path, expected_type="scores")
    scores, sm = read_table(scores_path, expected_type="scores")
    compatibility = (cm.model_id, cm.preprocessing_id, cm.score_convention)
    if compatibility != (sm.model_id, sm.preprocessing_id, sm.score_convention):
        raise ValueError("calibration and analysis scores use incompatible model/preprocessing/convention")
    missing = set(series_columns) - set(scores)
    if missing: raise ValueError(f"series columns missing from scores: {sorted(missing)}")
    order = [*series_columns, "window_start", "segment_id"]
    scores = scores.sort_values([x for x in order if x in scores], kind="stable")
    rows, states = [], {}
    for key, group in scores.groupby(list(series_columns), sort=False, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        selector = np.ones(len(calibration), bool)
        for column, value in zip(series_columns, key): selector &= calibration[column].eq(value).to_numpy()
        initial = calibration.loc[selector, "score"].to_numpy(float)
        spot = SPOT(**config).fit(initial)
        part = group.copy(); part["threshold"] = spot.threshold_; part["is_anomaly"] = spot.predict(part["score"])
        rows.append(part); states[str(key)] = spot
    result = pd.concat(rows, ignore_index=True) if rows else scores.assign(threshold=np.nan, is_anomaly=False)
    write_table(result, output_path, build_manifest("spot_decisions", config={**config, "series_columns": list(series_columns), "update_policy": "fixed"}, provenance={"calibration": str(calibration_path), "scores": str(scores_path)}, model_id=sm.model_id, preprocessing_id=sm.preprocessing_id, score_convention=sm.score_convention), resume=resume)
    save_artifact({"states": states, "last_keys": result[[c for c in order if c in result]].tail(1).to_dict("records"), "compatibility": compatibility}, state_path)
    return result
