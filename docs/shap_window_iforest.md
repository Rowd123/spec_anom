# Explain the saved window Isolation Forest

This is a separate, read-only analysis stage. It never fits the model or preprocessing and does not change `train-iforest` / `score-iforest`.

```bash
python -m pip install -e '.[shap]'
```

## Inputs and exact output explained

`train-iforest` saves a dictionary containing `pipeline`, ordered `feature_columns`, model/preprocessing IDs, effective training config, and score convention. `ModelPipeline.score_atypicality` validates signatures, applies its saved `FeaturePreprocessor` (feature selection/order, optional log1p, saved center and scale), then calls `AtypicalityModel.score`, which returns `-IsolationForest.score_samples`.

`score-iforest` writes identifiers, dates and scores **without the feature values**. Supply the original `window_features` Parquet, not the scores Parquet. `new_window_features.parquet` may have no manifest, but must contain `window_id`, `window_start`, the model's selected features and the compatibility signature columns required by that model. Extra features and input column order do not affect the explanation.

`--train` must point to the training feature artifact, with its `.parquet.manifest.json`. A `split-windows` artifact marked `train` is used as-is. If the manifest has no split label, provide the **complete original training input**, and the command reconstructs train using the effective split configuration saved in the model (including overlap purging). Never supply an unlabeled subset. Calibration/evaluation backgrounds are rejected. The old model artifact does not store training row IDs, so exact training membership cannot be independently proven; the caller must supply the original dataset/partition.

## Commands

Paths below are examples; use your actual artifacts. No retraining is required.

Explain the 50 highest IF scores in evaluation:

```bash
spectral-anomaly-shap artifacts/window_iforest_all.pkl \
  artifacts/window_splits/evaluation.parquet artifacts/shap/top50 \
  --train artifacts/window_splits/train.parquet \
  --top-n 50 --background-size 100 --permutations 20 --seed 42
```

Explain new windows:

```bash
spectral-anomaly-shap artifacts/window_iforest_all.pkl \
  artifacts/new_window_features.parquet artifacts/shap/new \
  --train artifacts/window_splits/train.parquet --top-n 50
```

Explain one window, with its waterfall:

```bash
spectral-anomaly-shap artifacts/window_iforest_all.pkl \
  artifacts/window_splits/evaluation.parquet artifacts/shap/window42 \
  --train artifacts/window_splits/train.parquet --window-id 42
```

Or replace `--window-id 42` with `--date '2026-09-01T12:00:00Z'` (exact `window_start`, not a date range). Naive dates are interpreted as UTC. IDs/dates must resolve to exactly one row. Combine them, or add `--source-id` / `--channel-id`, to disambiguate. `--top-n` cannot be combined with an ID/date. With no selector, all input rows are explained. If the observation table includes a `split` column, only `evaluation` rows are used. `--batch-size` controls scoring/masking batches; default 1024.

The equivalent module invocation is `python -m spectral_anomaly.shap_analysis ...`.

## Interpretation and numerical checks

[PermutationExplainer](https://shap.readthedocs.io/en/latest/generated/shap.PermutationExplainer.html) wraps the saved transform followed by the actual atypicality scoring function, with an identity link. Background rows are sampled without replacement from train with a fixed seed. The masker uses that exact sample (no hidden 100-row cap). SHAP inputs are the extracted `wf_*` values before the model's log/scaling transform, not necessarily physical raw signal units.

For **every** explained observation, analysis checks:

```text
base_value + sum(feature SHAP) ~= ModelPipeline.score_atypicality(observation)
```

The masker skips model evaluations only for **exactly equal** raw values. The default SHAP `np.isclose` shortcut can hide small feature changes that become significant after saved scaling or cross tree thresholds; this can break baseline/score reconstruction. Exact equality fixes that shortcut without changing the score or relaxing the check.

The tolerance is `atol=1e-8, rtol=1e-6`; failure raises an error before results are saved. The baseline is also checked against the mean pipeline score of the sampled background. A positive contribution increases atypicality relative to that background; this is not a probability or proof of a physical anomaly.

The function being explained is exact, but individual Shapley values are Monte Carlo approximations. Additivity does not prove their convergence. Increase `--permutations` and compare seeds to assess stability. Each observation has an evaluation budget of `permutations * (2 * feature_count + 1)` masks, each potentially evaluated over the background. Large backgrounds and many windows can be expensive.

This uses marginal masking: correlated features can create unrealistic combinations and share credit. Contributions describe model behavior, not causal effects.

## Outputs

- `shap_values.parquet`: IDs/dates, `score`, `base_value`, reconstructed score, residual and ordered `shap_wf_*` columns.
- `shap_families.parquet`: signed sums for statistics (including configured `wf_diff_*`), spectral power, ACF, ACF changes, spectral changes (including selected change frequencies). Unclassified features such as `wf_history_ready` remain explicitly in `other`.
- `global_importance.csv` and `.svg`: `mean(abs(SHAP))` over **the selected observations**. A top-N analysis describes that selected tail, not the whole evaluation population. Omit `--top-n` for the whole evaluation ranking.
- `family_importance.csv`: `mean(abs(sum of signed feature SHAP within family))`; opposite contributions can cancel. These are aggregated feature attributions, not a separately estimated group Shapley game.
- `beeswarm.svg`: selected-window contributions and feature values.
- `waterfall.svg`: the first selected window (highest score for top-N; exact chosen window for ID/date).
- `background.parquet`, `explained_windows.parquet`: actual inputs, preserving background identities and feature values for reproduction.
- `analysis.json`: model/preprocessing IDs, parameters, seed, feature order/families, SHAP version, numerical error and waterfall identity.

Use a separate output directory per analysis. Outputs in that directory are replaced on another successful run.

## Tests

```bash
python -m pip install -e '.[test,shap]'
python -m pytest tests/test_shap_analysis.py -q
```

Tests train only their own small synthetic fixture. They verify a reloaded artifact, no fitting during explanation, nontrivial saved log/scaling, reordered inputs, exact score and baseline reconstruction, deterministic sampling, family conservation, selectors, validation failures and CLI files/plots. They do not claim validation on your production pickle or data.
