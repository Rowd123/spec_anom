"""Selection of final raw segment features, independent of mask generation."""
from collections.abc import Mapping
from numbers import Real

import numpy as np

ENERGY_FEATURES = frozenset({
    "representation_energy", "mean_representation_energy_density",
    "integrated_spectral_power", "mean_spectral_power_density",
    "integrated_energy", "mean_energy_density", "local_energy_contrast",
})
DEFAULT_SELECTION = {"enabled": False, "energy_feature": "representation_energy", "min_energy": None}


def validate_segment_selection(options):
    if not isinstance(options, Mapping) or set(options) - DEFAULT_SELECTION.keys():
        raise ValueError("segment_selection must contain only enabled, energy_feature, min_energy")
    config = {**DEFAULT_SELECTION, **options}
    if not isinstance(config["enabled"], bool):
        raise ValueError("segment_selection.enabled must be boolean")
    if not isinstance(config["energy_feature"], str) or config["energy_feature"] not in ENERGY_FEATURES:
        raise ValueError(f"segment_selection.energy_feature must be one of {sorted(ENERGY_FEATURES)}")
    threshold = config["min_energy"]
    if threshold is not None and (isinstance(threshold, bool) or not isinstance(threshold, Real)
                                  or not np.isfinite(threshold) or threshold < 0):
        raise ValueError("segment_selection.min_energy must be a finite non-negative number or null")
    if config["enabled"] and threshold is None:
        raise ValueError("segment_selection.min_energy is required when enabled=true")
    return config


def select_segments_for_analysis(frame, options):
    """Return retained rows and an audit; equality passes, non-finite values fail."""
    config = validate_segment_selection(options)
    keep = np.ones(len(frame), dtype=bool)
    reasons = np.full(len(frame), "", dtype=object)
    if config["enabled"]:
        feature = config["energy_feature"]
        if feature not in frame:
            raise ValueError(f"segment_selection.energy_feature column is missing: {feature}")
        values = frame[feature].to_numpy(dtype=float)
        finite = np.isfinite(values)
        keep = finite & (values >= config["min_energy"])
        reasons[~finite] = f"non_finite_{feature}"
        reasons[finite & ~keep] = f"below_min_{feature}"
    rejected = frame.loc[~keep, [name for name in ("window_id", "segment_id") if name in frame]].copy()
    rejected["reason"] = reasons[~keep]
    return frame.loc[keep].copy(), {
        "number_of_segments_before_selection": len(frame),
        "number_of_segments_after_selection": int(keep.sum()),
        "number_of_segments_rejected_by_energy": int((~keep).sum()),
        "rejected_segments": rejected.to_dict("records"),
    }
