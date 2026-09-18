# Representation energy and selection

Let M be the **final mask after postprocessing/union**, R = SpectralResult.values,
and N the number of selected pixels. Extraction computes:

- representation_energy = sum over M of |R[f,t]|²;
- mean_representation_energy_density = representation_energy / N.

Both are zero for an empty mask. Accumulation uses float64 (coefficients are
converted to complex128 before taking the squared modulus).

| representation | R used for these two features |
| --- | --- |
| stft | Raw unnormalized complex STFT coefficients |
| ssq_stft | ssqueezepy SST-STFT output, with the configured squeezing convention (`sum` or `lebesgue`) |
| msst | Reassigned complex coefficients from `msst_stft`; each source coefficient contributes STFT * frequency_step to its final reassigned bin |

For MSST, coefficients are summed at each target bin **before** taking the
squared modulus. Cancellation/interference is therefore part of this measure.
For SST, changing the squeezing convention changes the scale/meaning of R.
These descriptors measure representation intensity, not calibrated physical
energy. No extra dt*df, one-sided doubling, or PSD normalization is applied.
The mean is per occupied pixel, not per physical time-frequency unit.
Thresholds depend on representation, signal scale and transform settings
(window, FFT length, hop, squeezing, etc.); they are not transferable between
STFT, SST and MSST without calibration.

SAM receives an image built from |R|, with optional log transform, percentile
clipping, normalization and uint8 RGB conversion. Energy is computed from R,
**not** that image. Changing display scaling does not change energy for a fixed
mask, though it can change SAM's masks.

The existing `integrated_spectral_power`, `mean_spectral_power_density`,
`integrated_energy`, `mean_energy_density`, and `local_energy_contrast` continue
to use the original STFT PSD and their previous definitions, even in MSST mode.
Geometry continues to use the final mask on the representation's grid.

## Configuration

Add to `configs/spectral_analysis.json` (default is disabled):

```json
"segment_selection": {
  "enabled": true,
  "energy_feature": "representation_energy",
  "min_energy": 100.0
}
```

100 is an illustrative threshold, not a calibrated recommendation.
`min_energy` is expressed in the raw scale of `energy_feature`.
Equality is accepted. Supported features are the two new descriptors, the four
physical energy/power names above, and `local_energy_contrast`.
Enabled selection requires a finite nonnegative numeric threshold; unknown
features/keys and invalid types raise ValueError before SAM loads. NaN/infinite
feature values are rejected with an explicit reason when selection is enabled.
Disabled selection removes no rows. Omitted sections default to disabled for
older configuration files. No changes to the model feature lists are required;
the new features can optionally be added to them.

## Pipeline and audit

`dataframe_to_segments` calls `select_segments_for_analysis` from `selection.py`
immediately after `postprocess_masks` and `segments_to_dataframe` for each
window. Only retained rows reach `extract_segments` and `segments.parquet`.
Isolation Forest and HDBSCAN read that filtered raw table; preprocessing remains
in the model stages. The dedicated selection function can accept further
criteria later without embedding selection logic in the window loop.

Each diagnostic includes:

- number_of_segments_before_selection;
- number_of_segments_after_selection;
- number_of_segments_rejected_by_energy;
- rejected_segments: window/segment identifiers and reason, e.g.
  below_min_representation_energy or non_finite_representation_energy;
- segments_before_selection: all final postprocessed masks;
- segments: retained final masks.

The INFO log includes `segments_before_selection`, `segments_after_selection`,
and `segments_rejected_energy`. The Parquet manifest lists the new raw features,
the effective selection configuration, aggregate counters and rejected records
in provenance. These audit records describe the **current extraction call**,
including when using resume, not cumulative historical attempts. Resume with a
changed selection configuration is rejected by the existing config hash check.
Even an empty output retains the feature columns and manifest. Fitting models
still requires enough retained segments for the chosen estimator/split.

## Run

Install the project's SAM dependencies and set the checkpoint/model configuration
in `configs/sam.json`. Provide CSV columns `time` and `value`; configure the true
sampling frequency and window sizes. For MSST set `representation` to `msst` in
the same spectral configuration. Then run from the repository root:

```bash
spectral-anomaly --log-level INFO extract data/signal.csv output/segments.parquet \
  --spectral-config configs/spectral_analysis.json \
  --sam-config configs/sam.json \
  --source-id pmu-test --channel-id voltage \
  --index-col time --value-col value
```

Tests: `python -m pytest tests/test_segment_selection.py -q`.
Tests use deterministic SAM mask doubles (no checkpoint needed), real spectral
transforms, real Parquet, and real CPU Isolation Forest/HDBSCAN estimators.
