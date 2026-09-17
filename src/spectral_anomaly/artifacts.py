"""Versioned, self-describing interchange files for independent pipeline stages."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

SCHEMA_VERSION = "1.0"
SEGMENT_KEYS = ("source_id", "channel_id", "window_id", "segment_id")
SCORE_CONVENTION = "higher_is_more_anomalous:-score_samples"


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def _versions() -> dict[str, str]:
    names = ("spectral-anomaly", "numpy", "pandas", "scipy", "scikit-learn")
    result = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return result


@dataclass(frozen=True)
class ArtifactManifest:
    artifact_type: str
    schema_version: str
    created_at: str
    config: dict[str, Any]
    config_hash: str
    feature_columns: tuple[str, ...]
    feature_units: dict[str, str]
    provenance: dict[str, Any]
    software_versions: dict[str, str]
    model_id: str | None = None
    preprocessing_id: str | None = None
    score_convention: str | None = None


def build_manifest(artifact_type: str, *, config: dict[str, Any], features: Iterable[str] = (),
                   units: dict[str, str] | None = None, provenance: dict[str, Any] | None = None,
                   model_id: str | None = None, preprocessing_id: str | None = None,
                   score_convention: str | None = None) -> ArtifactManifest:
    return ArtifactManifest(
        artifact_type, SCHEMA_VERSION, datetime.now(timezone.utc).isoformat(), config,
        fingerprint(config), tuple(features), units or {}, provenance or {}, _versions(),
        model_id, preprocessing_id, score_convention,
    )


def manifest_path(path: str | Path) -> Path:
    path = Path(path)
    return path.with_suffix(path.suffix + ".manifest.json")


def write_table(frame: pd.DataFrame, path: str | Path, manifest: ArtifactManifest, *,
                resume: bool = False, key_columns: Iterable[str] = SEGMENT_KEYS) -> None:
    """Atomically write Parquet and its manifest; resume is idempotent.

    Resume first verifies the schema/config and then merges by stable keys. This
    intentionally favours correctness over append speed; partitioned Parquet can
    be placed behind this API later without changing pipeline stages.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = tuple(key_columns)
    if resume and path.exists():
        previous, old_manifest = read_table(path, expected_type=manifest.artifact_type)
        if old_manifest.config_hash != manifest.config_hash or old_manifest.feature_columns != manifest.feature_columns:
            raise ValueError("resume artifact is incompatible with the effective configuration/features")
        available = [key for key in keys if key in frame and key in previous]
        if not available:
            raise ValueError("resume requires stable key columns")
        frame = pd.concat([previous, frame], ignore_index=True).drop_duplicates(available, keep="first")
    tmp = path.with_name(path.name + ".tmp")
    try:
        frame.to_parquet(tmp, index=False)
    except ImportError as exc:
        raise RuntimeError("Parquet I/O requires pyarrow; install spectral-anomaly[io]") from exc
    tmp.replace(path)
    meta_tmp = manifest_path(path).with_name(manifest_path(path).name + ".tmp")
    meta_tmp.write_text(json.dumps(asdict(manifest), indent=2, default=str) + "\n", encoding="utf8")
    meta_tmp.replace(manifest_path(path))


def read_table(path: str | Path, *, expected_type: str | None = None) -> tuple[pd.DataFrame, ArtifactManifest]:
    path = Path(path)
    raw = json.loads(manifest_path(path).read_text(encoding="utf8"))
    for name in ("feature_columns",):
        raw[name] = tuple(raw[name])
    manifest = ArtifactManifest(**raw)
    if manifest.schema_version != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {manifest.schema_version}")
    if expected_type and manifest.artifact_type != expected_type:
        raise ValueError(f"expected {expected_type} artifact, got {manifest.artifact_type}")
    return pd.read_parquet(path), manifest


def validate_segment_table(frame: pd.DataFrame, features: Iterable[str]) -> None:
    required = {*SEGMENT_KEYS, "window_start", "window_end", "time_start", "time_end",
                "frequency_min", "frequency_max", *features}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"segment table is missing columns: {sorted(missing)}")
    if frame.duplicated(list(SEGMENT_KEYS)).any():
        raise ValueError("segment identifiers are not unique")
    values = frame.loc[:, list(features)].to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("raw features contain NaN or infinity; policy is fail-fast")
