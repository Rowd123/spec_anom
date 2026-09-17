import numpy as np
import pandas as pd

from spectral_anomaly import (FEATURE_MEANING, SCORE_CONVENTION, apply_spot,
    build_manifest, load_artifact, load_config, read_table,
    score_isolation_forest, train_isolation_forest, write_table)


def _pickle_parquet(monkeypatch):
    """Exercise contracts in minimal CI where optional pyarrow is unavailable."""
    monkeypatch.setattr(pd.DataFrame, "to_parquet", lambda self, path, index=False: self.to_pickle(path))
    monkeypatch.setattr(pd, "read_parquet", lambda path: pd.read_pickle(path))


def _segments(config):
    features = config["preprocessing"]["atypicality_features"]
    rows = []
    for window in range(30):
        for segment in range(2):
            row = dict(source_id="source", channel_id="channel", window_id=window,
                       segment_id=segment, window_start=pd.Timestamp("2025-01-01") + pd.Timedelta(hours=window),
                       window_end=pd.Timestamp("2025-01-01") + pd.Timedelta(hours=window, minutes=30),
                       time_start=pd.Timestamp("2025-01-01") + pd.Timedelta(hours=window),
                       time_end=pd.Timestamp("2025-01-01") + pd.Timedelta(hours=window, minutes=1),
                       frequency_min=1.0, frequency_max=5.0)
            row.update({name: 1.0 + window + segment / 10 for name in features})
            rows.append(row)
    return pd.DataFrame(rows)


def test_models_use_only_saved_segments_and_batch_size_is_invariant(tmp_path, monkeypatch):
    _pickle_parquet(monkeypatch)
    config = load_config("configs/models.json", "models")
    config["atypicality"].update(backend="cpu", n_estimators=20, n_jobs=1)
    frame = _segments(config); features = tuple(config["preprocessing"]["atypicality_features"])
    segments = tmp_path / "segments.parquet"
    write_table(frame, segments, build_manifest("segments", config={"test": True}, features=features,
                units={name: FEATURE_MEANING[name] for name in features}))
    model = tmp_path / "model.pkl"; train_isolation_forest(segments, model, config)
    one = score_isolation_forest(segments, model, tmp_path / "one.parquet", batch_size=1)
    many = score_isolation_forest(segments, model, tmp_path / "many.parquet", batch_size=1000)
    np.testing.assert_allclose(one.score, many.score)
    assert load_artifact(model)["pipeline"].score_atypicality(frame[:3]).shape == (3,)


def test_spot_reads_scores_only_and_resume_has_no_duplicates(tmp_path, monkeypatch):
    _pickle_parquet(monkeypatch)
    rng = np.random.default_rng(4)
    base = dict(source_id="source", channel_id="channel", model_id="m", preprocessing_id="p")
    def scores(values, start):
        return pd.DataFrame([{**base, "window_id": start+i, "segment_id": 0,
            "window_start": pd.Timestamp("2025-01-01") + pd.Timedelta(hours=start+i), "score": value}
            for i, value in enumerate(values)])
    manifest = build_manifest("scores", config={}, model_id="m", preprocessing_id="p", score_convention=SCORE_CONVENTION)
    calibration = tmp_path / "cal.parquet"; analysis = tmp_path / "scores.parquet"
    write_table(scores(rng.exponential(size=200), 0), calibration, manifest)
    write_table(scores([0.1, 100.0], 300), analysis, manifest)
    output = tmp_path / "spot.parquet"; state = tmp_path / "spot.pkl"
    apply_spot(calibration, analysis, output, state, {"q": .01, "initial_quantile": .8, "min_excesses": 5})
    apply_spot(calibration, analysis, output, state, {"q": .01, "initial_quantile": .8, "min_excesses": 5}, resume=True)
    result, _ = read_table(output, expected_type="spot_decisions")
    assert len(result) == 2 and result.is_anomaly.tolist() == [False, True]
