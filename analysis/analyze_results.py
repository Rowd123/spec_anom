#!/usr/bin/env python3
"""Read-only visual analysis of saved Isolation Forest and HDBSCAN artifacts.

This module intentionally lives outside the package and never trains or mutates a
model.  Every generated file is placed below ``--output``.
"""
from __future__ import annotations

import argparse
import html
import json
import logging
import pickle
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # Required for unpickling project classes.

LOGGER = logging.getLogger("result-analysis")
PREFERRED_KEYS = ("source_id", "channel_id", "window_id", "segment_id")
TIME_COLUMNS = ("time_start", "time_end", "window_start", "window_end")


class AnalysisError(RuntimeError):
    """A user-facing incompatibility in input artifacts."""


def require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise AnalysisError(f"{label} not found: {path}")
    return path


def read_manifest(table_path: Path) -> dict[str, Any] | None:
    path = table_path.with_suffix(table_path.suffix + ".manifest.json")
    if not path.exists():
        LOGGER.warning("No manifest found for %s (expected %s)", table_path, path)
        return None
    try:
        value = json.loads(path.read_text(encoding="utf8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AnalysisError(f"invalid manifest {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AnalysisError(f"manifest must contain a JSON object: {path}")
    return value


def read_table(path: Path, label: str) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    require_file(path, label)
    try:
        if path.suffix.lower() == ".csv":
            frame = pd.read_csv(path)
        else:
            frame = pd.read_parquet(path)
    except ImportError as exc:
        raise AnalysisError(
            f"cannot read {path}: install a Parquet engine with `pip install pyarrow`"
        ) from exc
    except Exception as exc:
        raise AnalysisError(f"cannot read {label} {path}: {exc}") from exc
    if frame.columns.duplicated().any():
        raise AnalysisError(f"{label} contains duplicate column names")
    return frame, read_manifest(path)


def read_pickle(path: Path, label: str) -> Any:
    require_file(path, label)
    try:
        with path.open("rb") as stream:
            return pickle.load(stream)
    except Exception as exc:
        raise AnalysisError(
            f"cannot load {label} {path}; use the same project/backend versions that created it: {exc}"
        ) from exc


def model_parts(artifact: Any, kind: str) -> tuple[Any, Any, tuple[str, ...], dict[str, Any]]:
    """Introspect the actual supported saved layouts, without hard-coded features."""
    metadata = artifact if isinstance(artifact, dict) else {}
    if kind == "iforest":
        pipeline = metadata.get("pipeline", artifact)
        preprocessor = getattr(pipeline, "anomaly_preprocessor", None)
        wrapper = getattr(pipeline, "atypicality", None)
        model = getattr(wrapper, "model", wrapper)
    else:
        preprocessor = metadata.get("preprocessor")
        wrapper = metadata.get("model", artifact)
        model = getattr(wrapper, "model", wrapper)
    features = metadata.get("feature_columns")
    if features is None and preprocessor is not None:
        features = getattr(preprocessor, "features", None)
    if preprocessor is None or model is None or not features:
        raise AnalysisError(
            f"unsupported {kind} artifact layout: expected saved preprocessor, fitted model, and feature names"
        )
    return preprocessor, model, tuple(features), metadata


def choose_score_column(frame: pd.DataFrame) -> str:
    for name in ("score", "atypicality_score"):
        if name in frame:
            return name
    raise AnalysisError("scores table has neither 'score' nor 'atypicality_score'")


def stable_keys(*frames: pd.DataFrame) -> list[str]:
    keys = [name for name in PREFERRED_KEYS if all(name in frame for frame in frames)]
    if "window_id" not in keys or "segment_id" not in keys:
        raise AnalysisError(
            "artifacts need common window_id and segment_id columns; row-order joins are forbidden"
        )
    return keys


def unique_on(frame: pd.DataFrame, keys: list[str], label: str) -> None:
    duplicate = frame.duplicated(keys, keep=False)
    if duplicate.any():
        preview = frame.loc[duplicate, keys].head(5).to_dict("records")
        raise AnalysisError(f"{label} identifiers are not unique; examples: {preview}")


def compatible_manifests(*manifests: dict[str, Any] | None, compare_model: bool = True) -> None:
    present = [item for item in manifests if item]
    versions = {item.get("schema_version") for item in present if item.get("schema_version")}
    if len(versions) > 1:
        raise AnalysisError(f"incompatible manifest schema versions: {sorted(versions)}")
    if compare_model:
        model_ids = {item.get("model_id") for item in present if item.get("model_id")}
        if len(model_ids) > 1:
            raise AnalysisError(f"artifacts reference different model ids: {sorted(model_ids)}")


def verify_model_id(manifest: dict[str, Any] | None, metadata: dict[str, Any], label: str) -> None:
    table_id = (manifest or {}).get("model_id")
    artifact_id = metadata.get("model_id")
    if table_id and artifact_id and table_id != artifact_id:
        raise AnalysisError(f"{label} model id mismatch: table={table_id}, pickle={artifact_id}")


def merge_artifacts(left: pd.DataFrame, right: pd.DataFrame, left_name: str,
                    right_name: str) -> tuple[pd.DataFrame, list[str]]:
    keys = stable_keys(left, right)
    unique_on(left, keys, left_name); unique_on(right, keys, right_name)
    overlap = [name for name in right.columns if name in left.columns and name not in keys]
    merged = left.merge(right.drop(columns=overlap), on=keys, how="inner", validate="one_to_one")
    if len(merged) != len(right):
        LOGGER.warning("%d/%d %s rows matched %s", len(merged), len(right), right_name, left_name)
    if merged.empty:
        raise AnalysisError(f"no rows match between {left_name} and {right_name} on {keys}")
    return merged, keys


def transformed(preprocessor: Any, frame: pd.DataFrame, features: tuple[str, ...]) -> np.ndarray:
    missing = set(features) - set(frame)
    if missing:
        raise AnalysisError(f"segment features required by saved model are missing: {sorted(missing)}")
    values = frame.loc[:, features].to_numpy(dtype=float, copy=False)
    if not np.isfinite(values).all():
        raise AnalysisError("model features contain NaN or infinity")
    try:
        result = preprocessor.transform(frame)
    except Exception:
        # Support standard sklearn transformers saved with models.
        result = preprocessor.transform(values)
    if hasattr(result, "get"):
        result = result.get()
    return np.asarray(result, dtype=float)


def inspect_inputs(output: Path, **inputs: tuple[Path | None, Any, Any]) -> None:
    report = {}
    for label, (path, value, manifest) in inputs.items():
        if path is None:
            continue
        if isinstance(value, pd.DataFrame):
            structure = {"rows": len(value), "columns": list(value.columns),
                         "dtypes": {name: str(dtype) for name, dtype in value.dtypes.items()}}
        else:
            structure = {"python_type": f"{type(value).__module__}.{type(value).__qualname__}",
                         "dict_keys": sorted(map(str, value.keys())) if isinstance(value, dict) else None}
        report[label] = {"path": str(path.resolve()), "structure": structure, "manifest": manifest}
    (output / "input_inspection.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n", encoding="utf8"
    )


def npz_path(windows: Path, window_id: Any) -> Path | None:
    candidates = [windows / f"window-{window_id}.npz"]
    try:
        numeric = int(window_id)
        candidates.extend((windows / f"window-{numeric}.npz", windows / f"window-{numeric:06d}.npz"))
    except (TypeError, ValueError):
        pass
    return next((path for path in candidates if path.is_file()), None)


def mask_index(row: pd.Series, window_rows: pd.DataFrame, count: int) -> int | None:
    """Resolve mask by explicit NPZ ids when possible, otherwise saved row order."""
    shared = [name for name in PREFERRED_KEYS if name in row.index and name in window_rows]
    selected = np.ones(len(window_rows), dtype=bool)
    for name in shared:
        selected &= window_rows[name].eq(row[name]).to_numpy()
    matches = np.flatnonzero(selected)
    if len(matches) == 1 and matches[0] < count:
        return int(matches[0])
    segment_id = row.get("segment_id")
    for candidate in (segment_id, int(segment_id) - 1 if pd.notna(segment_id) else None):
        if isinstance(candidate, (int, np.integer)) and 0 <= candidate < count:
            return int(candidate)
    return None


def spectral_figure(row: pd.Series, all_segments: pd.DataFrame, windows: Path | None,
                    title: str):
    import plotly.graph_objects as go
    if windows is None:
        return None, "windows directory not supplied"
    path = npz_path(windows, row["window_id"])
    if path is None:
        return None, f"window NPZ unavailable for window_id={row['window_id']}"
    try:
        with np.load(path, allow_pickle=False) as data:
            names = set(data.files)
            spectral_name = "psd" if "psd" in names else "stft" if "stft" in names else None
            frequency_name = "frequencies" if "frequencies" in names else "frequency" if "frequency" in names else None
            time_name = "times" if "times" in names else "time" if "time" in names else None
            if not spectral_name or not frequency_name or not time_name or "masks" not in names:
                return None, f"{path.name} lacks PSD/STFT, axes, or masks; keys={sorted(names)}"
            values = np.asarray(data[spectral_name]); values = np.abs(values) ** (2 if spectral_name == "stft" else 1)
            frequencies = np.asarray(data[frequency_name]); times = np.asarray(data[time_name]); masks = np.asarray(data["masks"], bool)
            window_rows = all_segments.loc[all_segments["window_id"].eq(row["window_id"])]
            index = mask_index(row, window_rows, len(masks))
            if index is None or masks[index].shape != values.shape:
                return None, f"cannot map segment_id={row['segment_id']} to a compatible mask in {path.name}"
            mask = masks[index]
    except Exception as exc:
        return None, f"cannot read {path}: {exc}"
    z = np.log1p(np.maximum(values, 0))
    figure = go.Figure(go.Heatmap(x=times, y=frequencies, z=z, colorscale="Viridis", colorbar_title="log1p power"))
    contour = np.where(mask, 1.0, np.nan)
    figure.add_trace(go.Contour(x=times, y=frequencies, z=contour, contours={"start": .5, "end": .5},
                                line={"color": "red", "width": 3}, showscale=False, hoverinfo="skip", name="segment mask"))
    figure.update_layout(title=title, xaxis_title="Time (s)", yaxis_title="Frequency (Hz)", height=520)
    return figure, None


def write_gallery(rows: pd.DataFrame, segments: pd.DataFrame, features: Iterable[str],
                  windows: Path | None, output: Path, title: str, score_column: str | None = None) -> None:
    import plotly.io as pio
    blocks = [f"<h1>{html.escape(title)}</h1>"]
    include_js: bool | str = "cdn"
    for _, row in rows.iterrows():
        identity = f"window={row.get('window_id')} segment={row.get('segment_id')}"
        heading = identity + (f" score={row[score_column]:.6g}" if score_column else "")
        figure, warning = spectral_figure(row, segments, windows, heading)
        columns = [name for name in (*TIME_COLUMNS, "frequency_min", "frequency_max", *features) if name in row.index]
        blocks.append(f"<section><h2>{html.escape(heading)}</h2>{row[columns].to_frame('value').to_html(border=0)}")
        if figure is not None:
            blocks.append(pio.to_html(figure, full_html=False, include_plotlyjs=include_js)); include_js = False
        else:
            blocks.append(f"<p class='warning'>{html.escape(warning or 'visual unavailable')}</p>")
        blocks.append("</section>")
    document = "<!doctype html><meta charset='utf-8'><title>Analysis gallery</title><style>body{font-family:sans-serif;max-width:1400px;margin:auto}section{border-top:1px solid #ccc;padding:1rem}.warning{color:#9b4d00}table{text-align:right}</style>" + "".join(blocks)
    (output / "gallery.html").write_text(document, encoding="utf8")


def write_table_html(frame: pd.DataFrame, output: Path, name: str, title: str) -> None:
    frame.to_csv(output / f"{name}.csv", index=False)
    document = f"<!doctype html><meta charset='utf-8'><title>{html.escape(title)}</title><h1>{html.escape(title)}</h1>" + frame.to_html(index=False, border=0)
    (output / f"{name}.html").write_text(document, encoding="utf8")


def load_iforest(args, output: Path):
    segments, segment_manifest = read_table(args.segments, "segments")
    scores, score_manifest = read_table(args.scores, "Isolation Forest scores")
    artifact = read_pickle(args.model, "Isolation Forest model")
    preprocessor, model, features, metadata = model_parts(artifact, "iforest")
    compatible_manifests(segment_manifest, score_manifest)
    merged, keys = merge_artifacts(segments, scores, "segments", "scores")
    score_column = choose_score_column(merged)
    convention = (score_manifest or {}).get("score_convention") or metadata.get("score_convention")
    if convention and "higher" not in str(convention):
        raise AnalysisError(f"unsupported score convention: {convention}")
    verify_model_id(score_manifest, metadata, "Isolation Forest")
    inspect_inputs(output, segments=(args.segments, segments, segment_manifest), scores=(args.scores, scores, score_manifest), iforest_model=(args.model, artifact, None))
    return segments, merged, score_column, preprocessor, model, features, keys


def run_iforest(args) -> None:
    output = prepare_output(args.output)
    segments, merged, score_column, preprocessor, model, features, _ = load_iforest(args, output)
    top = merged.nlargest(args.top_n, score_column).copy()
    write_table_html(top, output, "top_segments", "Isolation Forest — highest atypicality scores")
    write_gallery(top, segments, features, args.windows, output,
                  "Isolation Forest — highest atypicality scores", score_column)
    if args.shap:
        explain_shap(merged, top, preprocessor, model, features, score_column,
                     output, args.shap_samples, args.shap_background, args.shap_evals, args.seed)


def model_score(model: Any, matrix: np.ndarray) -> np.ndarray:
    values = -model.score_samples(matrix)
    return np.asarray(values.get() if hasattr(values, "get") else values, dtype=float)


def explain_shap(frame: pd.DataFrame, local: pd.DataFrame, preprocessor: Any, model: Any,
                 features: tuple[str, ...], score_column: str, output: Path,
                 sample_count: int, background_count: int, evals: int, seed: int) -> None:
    try:
        import shap
    except ImportError as exc:
        raise AnalysisError("SHAP is not installed; install it with `pip install shap`") from exc
    import plotly.express as px
    import plotly.graph_objects as go
    rng = np.random.default_rng(seed)
    size = min(sample_count, len(frame)); indices = rng.choice(len(frame), size=size, replace=False)
    explained = frame.iloc[np.sort(indices)]
    background_size = min(background_count, len(frame)); background_indices = rng.choice(len(frame), size=background_size, replace=False)
    background = frame.iloc[np.sort(background_indices)].loc[:, features].to_numpy(float)
    values = explained.loc[:, features].to_numpy(float)

    def exact_saved_score(raw: np.ndarray) -> np.ndarray:
        temporary = pd.DataFrame(np.asarray(raw), columns=features)
        return model_score(model, transformed(preprocessor, temporary, features))

    LOGGER.info("Computing model-agnostic SHAP for %d rows with %d background rows", size, background_size)
    np.random.seed(seed)
    explainer = shap.KernelExplainer(exact_saved_score, background)
    shap_values = np.asarray(explainer.shap_values(values, nsamples=evals), dtype=float)
    if shap_values.ndim == 3: shap_values = shap_values[..., 0]
    importance = np.abs(shap_values).mean(axis=0)
    global_table = pd.DataFrame({"feature": features, "mean_abs_shap": importance}).sort_values("mean_abs_shap", ascending=False)
    write_table_html(global_table, output, "shap_global_importance", "Global model-agnostic SHAP importance")
    px.bar(global_table, x="mean_abs_shap", y="feature", orientation="h", title="Mean absolute SHAP contribution").write_html(output / "shap_global_importance_plot.html")
    long = pd.DataFrame(shap_values, columns=features).assign(sample=np.arange(size)).melt(id_vars="sample", var_name="feature", value_name="shap_value")
    px.strip(long, x="shap_value", y="feature", color="feature", title="SHAP contribution distribution").write_html(output / "shap_beeswarm.html")
    local_values = local.loc[:, features].to_numpy(float)
    local_shap = np.asarray(explainer.shap_values(local_values, nsamples=evals), dtype=float)
    if local_shap.ndim == 3: local_shap = local_shap[..., 0]
    local_dir = output / "shap_local"; local_dir.mkdir()
    for position, (_, row) in enumerate(local.iterrows()):
        table = pd.DataFrame({"feature": features, "raw_value": local_values[position], "shap_contribution": local_shap[position]}).sort_values("shap_contribution")
        name = f"window-{row['window_id']}-segment-{row['segment_id']}"
        table.to_csv(local_dir / f"{name}.csv", index=False)
        figure = go.Figure(go.Waterfall(orientation="h", y=table.feature, x=table.shap_contribution,
                                       base=float(np.asarray(explainer.expected_value).reshape(-1)[0])))
        figure.update_layout(title=f"{name}: score={row[score_column]:.6g}; exact f(X)=-score_samples(preprocess(X))", xaxis_title="Atypicality score contribution")
        figure.write_html(local_dir / f"{name}.html")


def load_hdbscan(args, output: Path):
    segments, segment_manifest = read_table(args.segments, "segments")
    clusters, cluster_manifest = read_table(args.clusters, "HDBSCAN partitions")
    required = {"cluster_label", "membership_strength"}
    if missing := required - set(clusters): raise AnalysisError(f"clusters table lacks columns: {sorted(missing)}")
    artifact = read_pickle(args.model, "HDBSCAN model")
    preprocessor, model, features, metadata = model_parts(artifact, "hdbscan")
    compatible_manifests(segment_manifest, cluster_manifest)
    verify_model_id(cluster_manifest, metadata, "HDBSCAN")
    merged, keys = merge_artifacts(segments, clusters, "segments", "clusters")
    inspect_inputs(output, segments=(args.segments, segments, segment_manifest), clusters=(args.clusters, clusters, cluster_manifest), hdbscan_model=(args.model, artifact, None))
    return segments, merged, preprocessor, model, features, keys, cluster_manifest


def representatives(frame: pd.DataFrame, matrix: np.ndarray, sample_limit: int,
                    seed: int) -> tuple[pd.DataFrame, dict[int, str]]:
    from scipy.spatial.distance import cdist
    rng = np.random.default_rng(seed); chosen, methods = [], {}
    for label in sorted(int(x) for x in frame.cluster_label.unique() if x >= 0):
        positions = np.flatnonzero(frame.cluster_label.to_numpy() == label)
        strength_position = positions[np.nanargmax(frame.iloc[positions].membership_strength.to_numpy(float))]
        chosen.append((strength_position, "highest_membership"))
        if len(positions) > sample_limit:
            candidates = np.sort(rng.choice(positions, sample_limit, replace=False)); method = f"approximate_medoid_sample_{sample_limit}"
        else:
            candidates = positions; method = "exact_medoid"
        distances = cdist(matrix[candidates], matrix[candidates], metric="euclidean")
        medoid_position = candidates[int(np.argmin(distances.mean(axis=1)))]
        chosen.append((medoid_position, method)); methods[label] = method
    rows = []
    for position, role in chosen:
        row = frame.iloc[position].copy(); row["representative_role"] = role; row["_position"] = position; rows.append(row)
    return pd.DataFrame(rows), methods


def add_projection(frame: pd.DataFrame, matrix: np.ndarray, reps: pd.DataFrame,
                   output: Path, use_umap: bool, seed: int) -> pd.DataFrame:
    import plotly.express as px
    def pca(values: np.ndarray) -> np.ndarray:
        centered = values - values.mean(axis=0, keepdims=True)
        _, _, right = np.linalg.svd(centered, full_matrices=False)
        result = centered @ right[:2].T
        if result.shape[1] == 1:
            result = np.column_stack((result[:, 0], np.zeros(len(result))))
        return result
    method = "PCA"
    if use_umap:
        try:
            import umap
        except ImportError:
            LOGGER.warning("UMAP unavailable; using PCA. Install with `pip install umap-learn`")
            projection = pca(matrix)
        else:
            method = "UMAP"; projection = umap.UMAP(n_components=2, random_state=seed).fit_transform(matrix)
    else:
        projection = pca(matrix)
    projected = frame.copy(); projected["projection_x"] = projection[:, 0]; projected["projection_y"] = projection[:, 1]
    projected["cluster"] = projected.cluster_label.astype(str).where(projected.cluster_label.ne(-1), "-1 noise / unassigned")
    hover = [name for name in ("window_id", "segment_id", "membership_strength", "score", "atypicality_score") if name in projected]
    figure = px.scatter(projected, x="projection_x", y="projection_y", color="cluster", hover_data=hover, title=f"{method} visualization (not the HDBSCAN fitting space)")
    if len(reps):
        positions = reps["_position"].astype(int).to_numpy(); figure.add_scatter(x=projection[positions, 0], y=projection[positions, 1], mode="markers+text", text=reps.cluster_label.astype(str), marker={"symbol": "x", "size": 14, "color": "black"}, name="real representatives / medoids")
    figure.write_html(output / "cluster_projection.html")
    return projected


def run_hdbscan(args) -> None:
    output = prepare_output(args.output)
    segments, merged, preprocessor, model, features, keys, _ = load_hdbscan(args, output)
    score_column = None
    score_manifest = None
    if args.scores:
        scores, score_manifest = read_table(args.scores, "Isolation Forest scores")
        merged, _ = merge_artifacts(merged, scores, "segments/clusters", "scores")
        score_column = choose_score_column(merged)
    matrix = transformed(preprocessor, merged, features)
    reps, methods = representatives(merged, matrix, args.medoid_sample, args.seed)
    statistics = merged.groupby("cluster_label")[list(features)].describe()
    statistics.columns = [f"{feature}_{statistic}" for feature, statistic in statistics.columns]
    statistics = statistics.reset_index()
    write_table_html(statistics, output, "cluster_feature_statistics", "HDBSCAN cluster feature statistics")
    summary = merged.groupby("cluster_label", as_index=False).agg(segment_count=("segment_id", "size"), mean_membership_strength=("membership_strength", "mean"))
    summary["medoid_method"] = summary.cluster_label.map(methods).fillna("not applicable")
    write_table_html(summary, output, "cluster_summary", "HDBSCAN cluster summary")
    write_table_html(reps.drop(columns="_position"), output, "cluster_representatives", "Real cluster representatives")
    write_gallery(reps, segments, features, args.windows, output, "HDBSCAN representatives")
    add_projection(merged, matrix, reps, output, args.umap, args.seed)
    noise = merged.loc[merged.cluster_label.eq(-1)].copy()
    proportion = len(noise) / len(merged)
    (output / "noise_summary.json").write_text(json.dumps({"title": "HDBSCAN noise / unassigned points", "count": len(noise), "proportion": proportion}, indent=2) + "\n")
    if len(noise):
        noise_sample = noise.sample(min(args.noise_sample, len(noise)), random_state=args.seed)
        if score_column:
            noise_sample = pd.concat([noise.nlargest(min(args.noise_extremes, len(noise)), score_column), noise.nsmallest(min(args.noise_extremes, len(noise)), score_column), noise_sample]).drop_duplicates(keys)
        write_table_html(noise.describe(include="all").transpose().reset_index(names="feature"), output, "noise_feature_statistics", "HDBSCAN noise / unassigned points")
        create_child(output, "noise")
        write_gallery(noise_sample, segments, features, args.windows, output / "noise", "HDBSCAN noise / unassigned points", score_column)


def create_child(output: Path, name: str) -> bool:
    (output / name).mkdir(exist_ok=True); return True


def run_combined(args) -> None:
    output = prepare_output(args.output)
    clusters, cm = read_table(args.clusters, "HDBSCAN partitions")
    scores, sm = read_table(args.scores, "Isolation Forest scores")
    # These tables intentionally originate from two different models. Only their
    # schema and stable segment identities must agree.
    compatible_manifests(cm, sm, compare_model=False)
    merged, _ = merge_artifacts(clusters, scores, "clusters", "scores")
    score = choose_score_column(merged)
    import plotly.express as px
    hover = [name for name in ("window_id", "segment_id", "cluster_label") if name in merged]
    px.scatter(merged, x="membership_strength", y=score, color=merged.cluster_label.astype(str), hover_data=hover,
               title="Isolation Forest score vs HDBSCAN membership strength (neither is a probability)").write_html(output / "score_vs_membership.html")
    px.box(merged, x="cluster_label", y=score, points="outliers", title="Isolation Forest score distribution by HDBSCAN partition").write_html(output / "score_by_cluster.html")
    noise = merged.loc[merged.cluster_label.eq(-1)]
    write_table_html(noise.sort_values(score, ascending=False), output, "noise_scores", "HDBSCAN noise / unassigned points — Isolation Forest scores")
    quantile = merged[score].quantile(args.high_score_quantile)
    strength = merged.membership_strength.quantile(args.strong_membership_quantile)
    categories = pd.DataFrame({"category": ["high score + noise", "high score + strong cluster membership", "low score + noise", "low score + strong cluster membership"],
                               "count": [int((merged[score].ge(quantile) & merged.cluster_label.eq(-1)).sum()), int((merged[score].ge(quantile) & merged.cluster_label.ge(0) & merged.membership_strength.ge(strength)).sum()), int((merged[score].lt(quantile) & merged.cluster_label.eq(-1)).sum()), int((merged[score].lt(quantile) & merged.cluster_label.ge(0) & merged.membership_strength.ge(strength)).sum())]})
    write_table_html(categories, output, "combined_categories", "Complementarity categories")
    inspect_inputs(output, clusters=(args.clusters, clusters, cm), scores=(args.scores, scores, sm))


def prepare_output(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    if not path.is_dir(): raise AnalysisError(f"output is not a directory: {path}")
    return path


def add_iforest_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--segments", required=True, type=Path); parser.add_argument("--scores", required=True, type=Path); parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--windows", type=Path); parser.add_argument("--top-n", type=int, default=20); parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--shap", action="store_true", help="compute exact-score, model-agnostic Kernel SHAP")
    parser.add_argument("--shap-samples", type=int, default=1000); parser.add_argument("--shap-background", type=int, default=50); parser.add_argument("--shap-evals", type=int, default=256); parser.add_argument("--seed", type=int, default=42)


def add_hdbscan_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--segments", required=True, type=Path); parser.add_argument("--clusters", required=True, type=Path); parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--scores", type=Path, help="optional Isolation Forest scores for noise cross-analysis"); parser.add_argument("--windows", type=Path); parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--medoid-sample", type=int, default=2000); parser.add_argument("--noise-sample", type=int, default=20); parser.add_argument("--noise-extremes", type=int, default=10); parser.add_argument("--umap", action="store_true"); parser.add_argument("--seed", type=int, default=42)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    commands = root.add_subparsers(dest="command", required=True)
    iforest = commands.add_parser("iforest", help="top atypical segments, spectral gallery, optional SHAP"); add_iforest_arguments(iforest)
    hdbscan = commands.add_parser("hdbscan", help="clusters, real representatives, medoids, noise, projection"); add_hdbscan_arguments(hdbscan)
    combined = commands.add_parser("combined", help="cross-analysis of saved scores and partitions")
    combined.add_argument("--scores", required=True, type=Path); combined.add_argument("--clusters", required=True, type=Path); combined.add_argument("--output", required=True, type=Path)
    combined.add_argument("--high-score-quantile", type=float, default=.9); combined.add_argument("--strong-membership-quantile", type=float, default=.75)
    all_command = commands.add_parser("all", help="run Isolation Forest, HDBSCAN, and combined reports")
    add_iforest_arguments(all_command); all_command.add_argument("--clusters", required=True, type=Path); all_command.add_argument("--hdbscan-model", required=True, type=Path)
    all_command.add_argument("--medoid-sample", type=int, default=2000); all_command.add_argument("--noise-sample", type=int, default=20); all_command.add_argument("--noise-extremes", type=int, default=10); all_command.add_argument("--umap", action="store_true")
    all_command.add_argument("--high-score-quantile", type=float, default=.9); all_command.add_argument("--strong-membership-quantile", type=float, default=.75)
    return root


def validate_args(args) -> None:
    for name in ("top_n", "shap_samples", "shap_background", "shap_evals", "medoid_sample", "noise_sample", "noise_extremes"):
        if hasattr(args, name) and getattr(args, name) < 1: raise AnalysisError(f"--{name.replace('_', '-')} must be positive")
    for name in ("high_score_quantile", "strong_membership_quantile"):
        if hasattr(args, name) and not 0 < getattr(args, name) < 1: raise AnalysisError(f"--{name.replace('_', '-')} must be between 0 and 1")


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv); logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(message)s")
    try:
        validate_args(args)
        if args.command == "iforest": run_iforest(args)
        elif args.command == "hdbscan": run_hdbscan(args)
        elif args.command == "combined": run_combined(args)
        else:
            base = args.output
            iforest_args = argparse.Namespace(**vars(args)); iforest_args.output = base / "iforest"; run_iforest(iforest_args)
            hdbscan_args = argparse.Namespace(**vars(args)); hdbscan_args.output = base / "hdbscan"; hdbscan_args.model = args.hdbscan_model; run_hdbscan(hdbscan_args)
            combined_args = argparse.Namespace(**vars(args)); combined_args.output = base / "combined"; run_combined(combined_args)
    except AnalysisError as exc:
        LOGGER.error("%s", exc); return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
