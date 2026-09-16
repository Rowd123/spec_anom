import numpy as np
import pandas as pd
import pytest

from spectral_anomaly import (
    AtypicalityModel,
    load_config,
    score_atypicality_splits,
    train_atypicality,
)


def _config():
    config = load_config("configs/models.json", "models")
    config["atypicality"] = {
        **config["atypicality"],
        "backend": "cpu",
        "n_estimators": 30,
        "n_jobs": 1,
    }
    return config


def _segments():
    config = _config()
    features = set(config["preprocessing"]["atypicality_features"])
    rows = []
    for window_id in range(10):
        for segment_id in range(2):
            row = {
                "window_id": window_id,
                "window_start": pd.Timestamp("2025-01-01")
                + pd.Timedelta(hours=window_id),
                "segment_id": segment_id,
            }
            row.update({name: window_id + segment_id / 10 + 1 for name in features})
            rows.append(row)
    return pd.DataFrame(rows), config


def test_isolation_forest_training_and_scoring_do_not_call_spot(monkeypatch):
    frame, config = _segments()

    def forbidden(*args, **kwargs):
        raise AssertionError("SPOT must not be fitted or called")

    monkeypatch.setattr("spectral_anomaly.spot.SPOT.fit", forbidden)
    monkeypatch.setattr("spectral_anomaly.spot.SPOT.predict", forbidden)
    pipeline, splits = train_atypicality(frame, config)
    result = score_atypicality_splits(pipeline, splits)
    assert len(result) == len(frame)
    assert "spot_anomaly" not in result
    assert not hasattr(pipeline, "spot")


def test_higher_raw_score_means_more_atypical():
    rng = np.random.default_rng(3)
    normal = rng.normal(0, 0.1, size=(200, 2))
    model = AtypicalityModel(
        backend="cpu", n_estimators=100, random_state=4
    ).fit(normal)
    assert model.score([[12.0, 12.0]])[0] > np.median(model.score(normal))


def test_complete_windows_stay_together_and_only_train_fits_preprocessor():
    frame, config = _segments()
    pipeline, splits = train_atypicality(frame, config)
    train, calibration, evaluation = splits
    memberships = {}
    for name, split in zip(("train", "calibration", "evaluation"), splits):
        for window_id in split.window_id.unique():
            memberships.setdefault(window_id, set()).add(name)
    assert all(len(labels) == 1 for labels in memberships.values())

    features = pipeline.anomaly_preprocessor.features
    raw_train = pipeline.anomaly_preprocessor._raw(train)
    expected_center = np.median(raw_train, axis=0)
    np.testing.assert_allclose(pipeline.anomaly_preprocessor.center_, expected_center)
    assert not np.allclose(expected_center, np.median(
        pipeline.anomaly_preprocessor._raw(pd.concat(splits)), axis=0
    ))

    model_identity = id(pipeline.atypicality.model)
    pipeline.score_atypicality(calibration)
    pipeline.score_atypicality(evaluation)
    assert id(pipeline.atypicality.model) == model_identity


def test_scored_csv_preserves_segment_metadata(tmp_path):
    frame, config = _segments()
    pipeline, splits = train_atypicality(frame, config)
    result = score_atypicality_splits(pipeline, splits)
    output = tmp_path / "scores.csv"
    result.to_csv(output, index=False)
    loaded = pd.read_csv(output)
    required = {"window_id", "window_start", "segment_id", "split", "atypicality_score"}
    assert required <= set(loaded.columns)
    assert len(loaded) == len(frame)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_non_finite_features_fail_explicitly(bad_value):
    frame, config = _segments()
    frame.loc[0, config["preprocessing"]["atypicality_features"][0]] = bad_value
    with pytest.raises(ValueError, match="non-finite atypicality features in train"):
        train_atypicality(frame, config)
