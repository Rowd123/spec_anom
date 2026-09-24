"""Exact-score SHAP regression tests using a small saved Isolation Forest."""
import pickle

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("shap")
pytest.importorskip("pyarrow")

from spectral_anomaly.artifacts import SCORE_CONVENTION, build_manifest, write_table
from spectral_anomaly.models import AtypicalityModel, save_artifact
from spectral_anomaly.pipeline import ModelPipeline
from spectral_anomaly.preprocessing import FeaturePreprocessor
from spectral_anomaly.shap_analysis import explain, feature_family, main, select_rows, verify_additivity


@pytest.fixture
def data():
    rng = np.random.default_rng(8)
    features = ("wf_acf_0_change", "wf_mean", "wf_band_0_power", "wf_acf_0", "wf_spectral_change_0")
    frame = pd.DataFrame(rng.uniform(.1, 5, (32, 5)), columns=features)
    frame["window_id"] = np.arange(32)
    frame["window_start"] = pd.date_range("2026-01-01", periods=32, freq="s", tz="UTC")
    frame["window_feature_signature"] = "same-contract"
    frame["source_id"] = "PMU"
    frame["channel_id"] = "V1"
    train, evaluation = frame.iloc[:24].copy(), frame.iloc[24:].copy()
    pre = FeaturePreprocessor(features, ("wf_band_0_power",), "standard").fit(train)
    model = AtypicalityModel(backend="cpu", n_estimators=12, max_samples=16, random_state=4).fit(pre.transform(train))
    pipeline = ModelPipeline(pre, model, window_feature_signature="same-contract")
    artifact = dict(kind="isolation_forest", feature_columns=features, pipeline=pipeline,
                    score_convention=SCORE_CONVENTION, model_id="test-if", preprocessing_id="test-pre")
    return artifact, train, evaluation


def test_saved_exact_score_order_no_refit(data, tmp_path, monkeypatch):
    artifact, train, evaluation = data
    path = tmp_path / "window_iforest_all.pkl"
    save_artifact(artifact, path)
    before = path.read_bytes()
    artifact = pickle.loads(before)
    def forbid(*args, **kwargs):
        raise AssertionError("analysis attempted fitting")
    monkeypatch.setattr(FeaturePreprocessor, "fit", forbid)
    monkeypatch.setattr(AtypicalityModel, "fit", forbid)
    monkeypatch.setattr(type(artifact["pipeline"].atypicality.model), "fit", forbid)
    outputs = explain(artifact, train, evaluation[evaluation.columns[::-1]], top_n=3,
                      background_size=8, permutations=2)
    explanation, result, groups, background, selected = outputs
    pipeline = artifact["pipeline"]
    scores = -pipeline.atypicality.model.score_samples(pipeline.anomaly_preprocessor.transform(selected))
    np.testing.assert_allclose(result.score, scores)
    np.testing.assert_allclose(result.reconstructed_score, scores, atol=1e-8)
    np.testing.assert_allclose(explanation.base_values, pipeline.score_atypicality(background).mean())
    np.testing.assert_allclose(groups.filter(like="shap_").sum(axis=1) + groups.base_value, scores)
    assert result.score.is_monotonic_decreasing
    assert path.read_bytes() == before
    assert set(background.window_id) <= set(train.window_id)
    assert list(explanation.feature_names) == list(artifact["feature_columns"])
    again = explain(artifact, train, evaluation, top_n=3, background_size=8, permutations=2)
    np.testing.assert_allclose(again[0].values, explanation.values)


@pytest.mark.parametrize("kwargs", [{"window_id": "24"}, {"date": "2026-01-01T00:00:24Z"}])
def test_window_selection(data, kwargs):
    artifact, train, evaluation = data
    assert explain(artifact, train, evaluation, background_size=3, permutations=1, **kwargs)[1].window_id.tolist() == [24]


@pytest.mark.parametrize("name,family", [("wf_mean", "statistics"), ("wf_diff_variance", "statistics"),
    ("wf_band_3_power", "spectral_power"), ("wf_acf_2", "acf"), ("wf_acf_2_change", "acf_changes"),
    ("wf_spectral_change_0_frequency", "spectral_changes"), ("wf_history_ready", "other")])
def test_families(name, family):
    assert feature_family(name) == family


def test_bad_additivity_fails():
    with pytest.raises(ValueError, match="additivity"):
        verify_additivity([.5], [[.1, .1]], [.9])


@pytest.mark.parametrize("change,match", [("signature", "incompatible"), ("nan", "non-finite"),
                                        ("order", "feature order"), ("empty", "empty")])
def test_invalid_inputs(data, change, match):
    artifact, train, evaluation = data
    if change == "signature":
        evaluation["window_feature_signature"] = "wrong"
    elif change == "nan":
        evaluation.loc[evaluation.index[0], "wf_mean"] = np.nan
    elif change == "order":
        artifact["feature_columns"] = artifact["feature_columns"][::-1]
    else:
        train = train.iloc[:0]
    with pytest.raises(ValueError, match=match):
        explain(artifact, train, evaluation)


def test_ambiguous_and_missing_selection(data):
    _, _, evaluation = data
    duplicate = pd.concat([evaluation, evaluation])
    with pytest.raises(ValueError, match="ambiguous"):
        select_rows(duplicate, np.zeros(len(duplicate)), window_id="24")
    with pytest.raises(ValueError, match="no windows"):
        select_rows(evaluation, np.zeros(len(evaluation)), window_id="absent")
    with pytest.raises(ValueError, match="cannot be combined"):
        select_rows(evaluation, np.zeros(len(evaluation)), top_n=1, window_id="24")


def test_cli_outputs_and_train_only(data, tmp_path):
    artifact, train, evaluation = data
    model, background, observations = [tmp_path / n for n in ("model.pkl", "train.parquet", "new_window_features.parquet")]
    save_artifact(artifact, model)
    write_table(train, background, build_manifest("window_features", config={}, provenance={"split": "train"}))
    evaluation.to_parquet(observations)
    output = tmp_path / "out"
    main([str(model), str(observations), str(output), "--train", str(background),
          "--top-n", "2", "--background-size", "4", "--permutations", "1"])
    for name in ("shap_values.parquet", "shap_families.parquet", "background.parquet", "explained_windows.parquet",
                 "global_importance.csv", "family_importance.csv", "waterfall.svg", "beeswarm.svg", "global_importance.svg", "analysis.json"):
        assert (output / name).stat().st_size > 0
    write_table(train, background, build_manifest("window_features", config={}, provenance={"split": "evaluation"}))
    with pytest.raises(ValueError, match="background must come from train"):
        main([str(model), str(observations), str(output), "--train", str(background)])


def test_cli_reconstructs_original_train_split(data, tmp_path, monkeypatch):
    import spectral_anomaly.shap_analysis as module
    artifact, train, evaluation = data
    full = pd.concat([train, evaluation], ignore_index=True)
    full["window_end"] = full.window_start
    artifact["config"] = {"split": {"train_fraction": .5, "calibration_fraction": .25,
                                     "evaluation_fraction": .25, "purge_overlap": True}}
    model, original, observations = [tmp_path / n for n in ("model.pkl", "full.parquet", "evaluation.parquet")]
    save_artifact(artifact, model)
    write_table(full, original, build_manifest("window_features", config={}))
    evaluation.to_parquet(observations)
    captured = {}
    def capture(artifact, background, observations, **kwargs):
        captured["train_ids"] = background.window_id.tolist()
        return (None,) * 5
    monkeypatch.setattr(module, "explain", capture)
    monkeypatch.setattr(module, "save_results", lambda *a, **k: None)
    main([str(model), str(observations), str(tmp_path / "out"), "--train", str(original)])
    assert captured["train_ids"] == list(range(16))


def test_constant_feature_and_single_window(data):
    artifact, train, evaluation = data
    # Use one constant input dimension in the background and explained row.
    train["wf_mean"] = 2.0
    evaluation["wf_mean"] = 2.0
    explanation, result, *_ = explain(artifact, train, evaluation, window_id="24",
                                      background_size=4, permutations=1)
    assert explanation.values[0, list(explanation.feature_names).index("wf_mean")] == 0
    assert abs(result.additivity_error.iloc[0]) < 1e-8


@pytest.mark.parametrize("all_features", [False, True])
def test_near_equal_raw_features_keep_exact_additivity(data, all_features):
    # np.isclose on raw inputs hides small changes amplified by saved scaling.
    artifact, train, evaluation = data
    features = list(artifact['feature_columns'])
    close_features = features if all_features else ['wf_mean']
    for frame in (train, evaluation):
        frame[close_features] = 1.0 + 1e-7 * frame[close_features]
    pipeline = artifact['pipeline']
    pipeline.anomaly_preprocessor.fit(train)
    pipeline.atypicality.fit(pipeline.anomaly_preprocessor.transform(train))
    explanation, result, _, background, _ = explain(
        artifact, train, evaluation, background_size=8, permutations=2)
    assert np.isclose(evaluation[close_features].iloc[0].to_numpy(), background[close_features].to_numpy()).all()
    assert np.abs(explanation.values).max() > 1e-6
    np.testing.assert_allclose(result.reconstructed_score, result.score, atol=1e-8, rtol=1e-6)
