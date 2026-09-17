"""Event exclusions shared by raw-window and saved-feature workflows."""
from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Mapping

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


def exclusion_signature(config: Mapping[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def validate_exclusions(config: Mapping[str, Any]) -> dict[str, Any]:
    expected = {"events", "event_id_column", "unknown_id_policy"}
    if set(config) != expected:
        raise ValueError(f"exclusions must contain exactly {sorted(expected)}")
    if config["unknown_id_policy"] not in {"warning", "error"}:
        raise ValueError("exclusions.unknown_id_policy must be warning or error")
    if not isinstance(config["events"], list):
        raise ValueError("exclusions.events must be a list")
    result = dict(config); result["events"] = []
    for index, event in enumerate(config["events"]):
        if not isinstance(event, Mapping):
            raise ValueError(f"excluded event {index} must be an object")
        unknown = set(event) - {"event_id", "source_id", "start", "end"}
        if unknown:
            raise ValueError(f"unknown excluded-event keys: {sorted(unknown)}")
        has_id = event.get("event_id") is not None
        has_interval = event.get("start") is not None or event.get("end") is not None
        if not has_id and not has_interval:
            raise ValueError("each excluded event needs event_id or start/end")
        if has_interval and (event.get("start") is None or event.get("end") is None):
            raise ValueError("excluded time intervals require both start and end")
        if has_interval and pd.Timestamp(event["end"]) < pd.Timestamp(event["start"]):
            raise ValueError("excluded event end must not precede start")
        result["events"].append(dict(event))
    return result


def excluded_observations(frame: pd.DataFrame, config: Mapping[str, Any], *,
                          source_id: str | None = None) -> np.ndarray:
    """Return rows excluded by matching IDs or closed time intervals."""
    config = validate_exclusions(config)
    excluded = np.zeros(len(frame), dtype=bool)
    id_column = config["event_id_column"]
    known_ids = set(frame[id_column].dropna()) if id_column and id_column in frame else set()
    for event in config["events"]:
        configured_source = event.get("source_id")
        if configured_source is not None and str(configured_source) != str(source_id):
            continue
        matched = np.zeros(len(frame), dtype=bool)
        if event.get("event_id") is not None:
            event_id = event["event_id"]
            if id_column is None or id_column not in frame:
                message = f"excluded event id {event_id!r} cannot be resolved: event_id_column is unavailable"
                if config["unknown_id_policy"] == "error": raise ValueError(message)
                LOGGER.warning(message)
            elif event_id not in known_ids:
                message = f"excluded event id {event_id!r} is unknown"
                if config["unknown_id_policy"] == "error": raise ValueError(message)
                LOGGER.warning(message)
            else:
                matched |= frame[id_column].eq(event_id).to_numpy()
        if event.get("start") is not None:
            start, end = pd.Timestamp(event["start"]), pd.Timestamp(event["end"])
            try:
                matched |= np.asarray((frame.index >= start) & (frame.index <= end))
            except TypeError as exc:
                raise ValueError("excluded interval type is incompatible with the data index") from exc
        excluded |= matched
        LOGGER.info("excluded event=%s source=%s observations=%d", event, source_id, int(matched.sum()))
    return excluded


def filter_excluded_segments(frame: pd.DataFrame, config: Mapping[str, Any], *,
                             source_column: str = "source_id") -> pd.DataFrame:
    """Filter an already-saved feature table using its stable metadata."""
    config = validate_exclusions(config)
    keep = np.ones(len(frame), dtype=bool)
    for event in config["events"]:
        source_match = np.ones(len(frame), dtype=bool)
        if event.get("source_id") is not None:
            if source_column not in frame:
                raise ValueError(f"saved features lack source column {source_column!r}")
            source_match = frame[source_column].astype(str).eq(str(event["source_id"])).to_numpy()
        matched = np.zeros(len(frame), dtype=bool)
        if event.get("event_id") is not None:
            column = config["event_id_column"]
            if not column or column not in frame:
                message = f"saved features cannot resolve excluded event id {event['event_id']!r}"
                if config["unknown_id_policy"] == "error": raise ValueError(message)
                LOGGER.warning(message)
            else:
                def contains(value):
                    if isinstance(value, (list, tuple, set, np.ndarray)): return event["event_id"] in value
                    return value == event["event_id"]
                matched |= frame[column].map(contains).to_numpy()
        if event.get("start") is not None:
            required = {"window_start", "window_end"}
            if not required <= set(frame):
                raise ValueError("saved features need window_start/window_end for interval exclusions")
            start, end = pd.Timestamp(event["start"]), pd.Timestamp(event["end"])
            matched |= (pd.to_datetime(frame["window_start"]).le(end) &
                        pd.to_datetime(frame["window_end"]).ge(start)).to_numpy()
        keep &= ~(source_match & matched)
    return frame.loc[keep].copy()
