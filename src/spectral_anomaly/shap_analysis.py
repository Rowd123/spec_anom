"""Post-hoc SHAP of the saved pipeline's exact higher-is-atypical score."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .artifacts import SCORE_CONVENTION, manifest_path, read_table
from .models import load_artifact
from .preprocessing import chronological_split

FAMILIES = ("statistics", "spectral_power", "acf", "acf_changes", "spectral_changes")
STATISTICS = {"minimum", "maximum", "mean", "variance", "median", "iqr",
              "skewness", "kurtosis", "constant"}


def feature_family(name):
    import re
    if name.removeprefix("wf_") in STATISTICS or name.startswith("wf_diff_"):
        return "statistics"
    if re.fullmatch(r"wf_band_\d+_power", name):
        return "spectral_power"
    if re.fullmatch(r"wf_acf_\d+_change", name):
        return "acf_changes"
    if re.fullmatch(r"wf_acf_\d+", name):
        return "acf"
    if re.fullmatch(r"wf_spectral_change_\d+(_frequency)?", name):
        return "spectral_changes"
    # Indicators such as history_ready must not silently become statistics.
    return "other"


def select_rows(frame, scores, *, top_n=None, window_id=None, date=None,
                source_id=None, channel_id=None):
    if top_n is not None and (window_id is not None or date is not None):
        raise ValueError("top_n cannot be combined with window_id/date")
    if top_n is not None and top_n < 1:
        raise ValueError("top_n must be positive")
    mask = np.ones(len(frame), dtype=bool)
    for column, value in (("source_id", source_id), ("channel_id", channel_id),
                          ("window_id", window_id)):
        if value is not None:
            if column not in frame:
                raise ValueError(f"missing selector column: {column}")
            mask &= frame[column].astype(str).eq(str(value)).to_numpy()
    if date is not None:
        mask &= pd.to_datetime(frame.window_start, utc=True).eq(pd.to_datetime(date, utc=True)).to_numpy()
    positions = np.flatnonzero(mask)
    if top_n is not None:
        positions = positions[np.argsort(-scores[positions], kind="stable")[:top_n]]
    if not len(positions):
        raise ValueError("selection contains no windows")
    if (window_id is not None or date is not None) and len(positions) != 1:
        raise ValueError("ambiguous window; combine window_id/date/source_id/channel_id")
    return positions


def verify_additivity(base, values, scores, *, atol=1e-8, rtol=1e-6):
    reconstructed = np.asarray(base) + np.asarray(values).sum(axis=1)
    residual = reconstructed - scores
    if not np.isfinite(values).all() or not np.isfinite(reconstructed).all():
        raise ValueError("SHAP produced non-finite values")
    if not np.allclose(reconstructed, scores, atol=atol, rtol=rtol):
        raise ValueError(f"SHAP additivity failed: max absolute error={np.max(np.abs(residual)):.3g}")
    return reconstructed, residual


def explain(artifact, train, observations, *, background_size=100, permutations=10,
            seed=42, batch_size=1024, top_n=None, window_id=None, date=None,
            source_id=None, channel_id=None):
    """Never fit: explain raw wf inputs through the saved transform and score."""
    import shap

    if artifact.get("kind") != "isolation_forest" or artifact.get("score_convention") != SCORE_CONVENTION:
        raise ValueError("expected an Isolation Forest artifact using -score_samples")
    pipeline = artifact["pipeline"]
    features = list(pipeline.anomaly_preprocessor.features)
    if tuple(features) != tuple(artifact["feature_columns"]):
        raise ValueError("artifact feature order differs from saved preprocessing")
    if not features or any(not f.startswith("wf_") for f in features):
        raise ValueError("SHAP window analysis requires wf_ features")
    if min(background_size, permutations, batch_size) < 1:
        raise ValueError("background_size, permutations and batch_size must be positive")
    for name, frame in (("train", train), ("observations", observations)):
        if frame.empty:
            raise ValueError(f"empty {name} table")
        if not {"window_id", "window_start"}.issubset(frame.columns):
            raise ValueError(f"{name} requires window_id and window_start")
        if not np.isfinite(frame.loc[:, features].to_numpy(float)).all():
            raise ValueError(f"non-finite raw features in {name}")
        pipeline._validate_compatibility(frame)
    # Evaluate the reference through the public pipeline, independently of SHAP.
    scores = np.concatenate([pipeline.score_atypicality(observations.iloc[i:i + batch_size])
                             for i in range(0, len(observations), batch_size)])
    positions = select_rows(observations, scores, top_n=top_n, window_id=window_id,
                            date=date, source_id=source_id, channel_id=channel_id)
    selected = observations.iloc[positions].reset_index(drop=True)
    background = train.sample(n=min(background_size, len(train)), random_state=seed).copy()
    # Validate log1p domains with the saved transform even on unselected train rows.
    pipeline.anomaly_preprocessor.transform(train)

    def score(raw):
        frame = pd.DataFrame(raw, columns=features)
        return pipeline.atypicality.score(pipeline.anomaly_preprocessor.transform(frame))

    class ExactIndependent(shap.maskers.Independent):
        """Skip evaluations only for exactly equal raw feature values.

        SHAP's default np.isclose can hide differences amplified by saved
        scaling or crossing an Isolation Forest split, breaking additivity.
        """
        def invariants(self, x):
            if x.shape != self.data.shape[1:]:
                raise ValueError("SHAP input shape differs from background feature shape")
            return np.equal(x, self.data)

    masker = ExactIndependent(background[features].to_numpy(float), max_samples=len(background))
    explainer = shap.PermutationExplainer(score, masker, feature_names=features,
                                        link=shap.links.identity, seed=seed)
    explanation = explainer(selected[features].to_numpy(float),
                            max_evals=permutations * (2 * len(features) + 1),
                            batch_size=batch_size, silent=True)
    expected_base = pipeline.score_atypicality(background).mean()
    if not np.allclose(explanation.base_values, expected_base, atol=1e-8, rtol=1e-6):
        raise ValueError("SHAP baseline differs from mean pipeline score on background")
    reconstructed, residual = verify_additivity(explanation.base_values, explanation.values, scores[positions])
    metadata = [c for c in selected if c.endswith("_id") or c in {"window_start", "window_end"}]
    result = selected[metadata].copy()
    result["score"] = scores[positions]
    result["base_value"] = explanation.base_values
    result["reconstructed_score"] = reconstructed
    result["additivity_error"] = residual
    for j, feature in enumerate(features):
        result[f"shap_{feature}"] = explanation.values[:, j]
    grouped = result[[*metadata, "score", "base_value"]].copy()
    families = list(FAMILIES)
    if any(feature_family(f) == "other" for f in features):
        families.append("other")
    for family in families:
        indices = [j for j, f in enumerate(features) if feature_family(f) == family]
        grouped[f"shap_{family}"] = explanation.values[:, indices].sum(axis=1)
    verify_additivity(grouped.base_value, grouped[[f"shap_{f}" for f in families]].to_numpy(), result.score)
    return explanation, result, grouped, background, selected


def save_results(output, explanation, result, grouped, background, selected, provenance):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output / "shap_values.parquet", index=False)
    grouped.to_parquet(output / "shap_families.parquet", index=False)
    background.to_parquet(output / "background.parquet", index=False)
    selected.to_parquet(output / "explained_windows.parquet", index=False)
    ranking = pd.DataFrame({"feature": explanation.feature_names,
                            "mean_abs_shap": np.abs(explanation.values).mean(axis=0)})
    ranking["family"] = ranking.feature.map(feature_family)
    ranking = ranking.sort_values("mean_abs_shap", ascending=False, kind="stable")
    ranking.to_csv(output / "global_importance.csv", index=False)
    family_columns = [c for c in grouped if c.startswith("shap_")]
    pd.DataFrame({"family": [c.removeprefix("shap_") for c in family_columns],
                  "mean_abs_group_shap": grouped[family_columns].abs().mean().to_numpy()}).sort_values(
                      "mean_abs_group_shap", ascending=False).to_csv(output / "family_importance.csv", index=False)
    for name, draw in (
        ("global_importance", lambda: shap.plots.bar(explanation, show=False)),
        ("beeswarm", lambda: shap.plots.beeswarm(explanation, show=False)),
        ("waterfall", lambda: shap.plots.waterfall(explanation[0], show=False)),
    ):
        plt.figure()
        draw()
        plt.gcf().savefig(output / f"{name}.svg", bbox_inches="tight")
        plt.close("all")
    provenance.update({"features": list(explanation.feature_names), "shap_version": shap.__version__,
                       "explained_rows": len(result), "background_rows": len(background),
                       "max_absolute_additivity_error": float(result.additivity_error.abs().max()),
                       "atol": 1e-8, "rtol": 1e-6,
                       "waterfall_window": result.iloc[0][["window_id", "window_start"]].to_dict(),
                       "family_mapping": {f: feature_family(f) for f in explanation.feature_names}})
    (output / "analysis.json").write_text(json.dumps(provenance, indent=2, default=str) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("observations", help="evaluation or new window features Parquet (not scores)")
    parser.add_argument("output")
    parser.add_argument("--train", required=True, help="training feature artifact with its manifest")
    parser.add_argument("--top-n", type=int)
    parser.add_argument("--window-id")
    parser.add_argument("--date", help="exact window_start; timezone-naive values interpreted as UTC")
    parser.add_argument("--source-id")
    parser.add_argument("--channel-id")
    parser.add_argument("--background-size", type=int, default=100)
    parser.add_argument("--permutations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args(argv)
    artifact = load_artifact(args.model)
    train, manifest = read_table(args.train, expected_type="window_features")
    partition = manifest.provenance.get("split")
    if partition not in {None, "train"}:
        raise ValueError("background must come from train, never calibration/evaluation")
    if partition is None:
        train, _, _ = chronological_split(train, artifact["config"]["split"])
    if "split" in train and not train["split"].eq("train").all():
        raise ValueError("background contains non-train rows")
    if manifest_path(args.observations).exists():
        observations, _ = read_table(args.observations, expected_type="window_features")
    else:
        observations = pd.read_parquet(args.observations)
    if "split" in observations:
        observations = observations.loc[observations["split"].eq("evaluation")].copy()
    outputs = explain(artifact, train, observations, **{k: getattr(args, k) for k in (
        "background_size", "permutations", "seed", "batch_size", "top_n", "window_id", "date", "source_id", "channel_id")})
    save_results(args.output, *outputs, provenance={**vars(args), "model_id": artifact["model_id"],
                 "preprocessing_id": artifact["preprocessing_id"], "score_convention": SCORE_CONVENTION,
                 "method": "PermutationExplainer(identity)", "masker_invariants": "exact_equality",
                 "background_partition": "train"})


if __name__ == "__main__":
    main()
