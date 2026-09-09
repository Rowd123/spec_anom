# Spectral anomaly — energy and MSST morphology

This package provides quality-aware signal windowing, the historical causal
energy detector, STFT/MSST transforms, and a first morphology prototype. Energy
scoring remains available but is no longer required to select windows for the
structural MSST path. Classification and clustering are intentionally out of
scope for the prototype.

## Pipeline architecture and energy dependency

The historical path is:

```text
raw data -> quality preparation -> window energy -> causal median/MAD score
         -> suspicious windows -> joined fixed periods -> MSST
```

`detect_energy_anomalies` currently owns both quality preparation and energy
scoring. Its `suspicious` column is consumed by `prepare_analysis_periods`, so
that older MSST path is explicitly gated by energy. `analyze_msst_periods` and
`msst_stft` themselves do not depend on energy: they only require complete
regularly sampled arrays.

The morphology prototype adds a parallel, non-breaking path:

```text
raw data
  -> prepare_analysis_windows (quality only)
  -> MSST for every accepted window
  -> local robust normalization
  -> structure tensor
  -> significant connected components
  -> descriptive window/component features
  -> later: detection decision and unsupervised classification
```

Window acceptance in this path depends only on NaNs, quality flags, valid-sample
requirements, and interpolation-gap limits. Neither energy nor the historical
`suspicious` flag decides whether MSST is run.

### Prototype feature rationale

The deliberately small first feature set is intended for visual validation, not
as a final anomaly score:

| Feature | Geometric meaning | Expected noise/structure behavior |
|---|---|---|
| `weighted_mean_coherence` | Tensor anisotropy averaged with local significance weights | Lower for diffuse isotropic texture; higher along locally organized shapes |
| `coherence_q90` | Upper tail of coherence among locally significant pixels | Shows whether at least part of the map is strongly organized |
| `coherent_pixel_fraction` | Map fraction retained by both significance and coherence criteria | Small isolated noise responses contribute little; persistent shapes occupy more support |
| `orientation_dispersion` | Axial circular dispersion, invariant under a 180° reversal | Low for a regular local direction, higher for random directions; orientation itself is not scored as good or bad |
| `component_count` | Number of connected significant coherent regions | Separates absent/fragmented support from one or more organized objects |
| `largest_component_area_fraction` | Relative support of the largest object | Rewards spatial persistence without using absolute MSST amplitude |
| `largest_component_significance_fraction` | Share of local significance carried by the largest object | Measures concentration relative to the locally estimated background |

Every component also reports pixel area, integrated local significance, duration
in seconds, bandwidth in hertz, bounding box, dimensionless normalized aspect
ratio, PCA orientation, and PCA linearity. Duration and bandwidth are kept in
their own physical units; PCA coordinates are normalized by the complete map
extent before axes are combined.

### Important prototype parameters

The parameters most likely to require calibration are the frequency/time size of
`normalization_neighborhood`, tensor smoothing scales `tensor_sigma`, local
`significance_threshold`, `coherence_threshold`, and
`minimum_component_area`. They interact with STFT `window_length`, `hop_length`,
frequency resolution, window duration, and expected structure thickness. The
normalization neighborhood must be wider than a typical structure but smaller
than background variation. Tensor smoothing must suppress pixel noise without
merging nearby shapes. Thresholds should eventually be calibrated on background
recordings rather than interpreted as a universal classifier.

The normalization is a running 2-D median/MAD of `abs(MSST)` and therefore does
not assume a single global noise level or white stationary noise. The structure
tensor is evaluated in dimensionless local scale coordinates, making its
coherence orientation-neutral. Connected components use 8-connectivity and do
not assume a single-valued ridge `f(t)`, so vertical and multi-frequency shapes
remain representable. Skeleton, curvature, loops, branch counts, HDBSCAN, and a
final decision rule are intentionally deferred until these maps and basic
features have been validated visually.

### Structural prototype usage

```python
from spectral_anomaly import analyze_structural_windows, plot_structural_window

metadata, analyses = analyze_structural_windows(
    frame,
    value_col="value",
    sampling_frequency=1.0,
    sampling_period="1s",
    window_size=256,
    overlap=128,
    msst_options={"window_length": 128, "n_fft": 256, "hop_length": 4},
)

# All quality-valid windows are present, regardless of their energy score.
window_id = next(iter(analyses))
item = analyses[window_id]
print(item.features)
print(item.components)
plot_structural_window(item).write_html("msst_structure_window.html")
```

The six panels show the time signal, STFT, MSST, locally normalized MSST,
structure coherence, and retained components overlaid on the MSST. Run the full
example with `python examples/structure_usage.py`.

## Controlled STFT versus MSST comparison

The morphology chain is representation-independent: `extract_spectral_structure`
selects either `abs(result.stft)` or `abs(result.msst)`, then applies the exact
same normalization, tensor, thresholds, connected components, and aggregation.
Use `representation="stft"` or `representation="msst"` with
`analyze_structural_windows`. The STFT-only choice calls `analyze_stft_periods`
and skips instantaneous-frequency estimation and synchrosqueezing entirely.

For a paired experiment on identical windows, use:

```python
from spectral_anomaly import (
    compare_structural_windows,
    plot_stft_msst_comparison,
)

metadata, comparisons = compare_structural_windows(
    frame,
    value_col="value",
    sampling_frequency=1.0,
    sampling_period="1s",
    window_size=256,
    overlap=128,
    window_ids=[0],
    msst_options={"window_length": 128, "n_fft": 256, "hop_length": 4},
    structure_options={"small_component_area": 16},
)
comparison = comparisons[0]
print(comparison.metrics)
plot_stft_msst_comparison(comparison).write_html(
    "stft_msst_comparison_window_0.html"
)
```

`comparison.stft` and `comparison.msst` each expose the raw and normalized
representation, coherence and orientation maps, masks, labels, components, and
features. `comparison.metrics` has `stft`, `msst`, and `msst_minus_stft` columns.
In addition to the original features it reports median and maximum component
area plus `small_component_fraction`, where “small” means an area no greater
than the configurable `small_component_area`.

The nine-panel diagnostic aligns signal, raw maps, normalized maps, coherence
maps, and component overlays on the same time/frequency axes. Normalized maps
share a 0–12 color range and coherence maps share 0–1. Raw STFT and MSST retain
separate color scaling because their coefficient units differ after
reassignment; forcing one raw scale would make the visual comparison misleading.

Run the real-data example with:

```bash
python examples/compare_stft_msst.py --window-id 0
```

The deterministic synthetic experiment currently shows that the default
prototype parameters retain several small MSST components even for noise, while
STFT often retains none and does retain larger regions for the brief impulse.
This is evidence that the fragmentation concern is measurable, not a conclusion
that STFT is superior: the same thresholds were initially selected around MSST,
and the weak-tone STFT is also rejected. The next experiment should compare
background/structure distributions across recordings and calibrate thresholds
per representation before choosing STFT, MSST, or both.

## Algorithmic choices

1. Duplicates are sorted and reduced with an explicit “keep first” policy.
2. The nominal period is the median positive adjacent difference. This is robust
   to occasional missing timestamps, but assumes over half of adjacent gaps are
   nominal. Pass `sampling_period` when that assumption is false. Samples must be
   exactly on the resulting grid: the detector deliberately never snaps a nearby
   measurement to a grid position.
3. Invalid quality flags and NaNs become unobserved positions. Exact `reindex`
   preserves separate observed and interpolated masks.
4. Only complete, internally bounded gaps of at most `max_interpolation_gap` are
   linearly interpolated. A long run is never partially filled, so windows at its
   edges are rejected as well. Interpolation is convenient but changes spectral
   content (typically attenuating high frequencies); keep this limit conservative.
5. The original signal values are not centred or otherwise changed for the energy
   calculation: they are only multiplied by the selected Fourier taper. Energy is
   computed from the one-sided real FFT with Parseval weights and normalised by
   the taper energy. By default, the complete contribution assigned to Fourier
   bin 0 is subtracted before comparison with history (`exclude_dc_bin=True`). A
   centred copy is still returned for optional plotting, but is not used by the
   FFT or the anomaly score. The result also exposes
   `energy_including_dc` and `dc_bin_energy` for auditing. Excluding bin 0 removes
   exactly that discrete bin—not a configurable set of low-frequency bins. Set
   `exclude_dc_bin=False` to include it. Because tapering spreads a constant or
   slow trend beyond bin 0, this is deliberately not equivalent to detrending;
   keep the taper fixed when comparing windows.
6. A causal median/MAD baseline uses only earlier accepted windows. The default
   excludes detected anomalies from history to limit contamination. During warmup
   (`min_history` points), scores remain unavailable. A relative machine-epsilon
   scale floor handles zero MAD: an energy equal to the baseline scores zero, while
   a different energy gets a finite, possibly very large score.

```python
from spectral_anomaly import (
    detect_energy_anomalies,
    plot_suspicious_windows,
    plot_window,
)

result, windows = detect_energy_anomalies(
    frame, value_col="value", quality_col="quality",
    valid_quality_flags={"good"}, sampling_period="1s",
    window_size=256, overlap=128, exclude_dc_bin=True,
)
anomalies = result[result["suspicious"]]
if not anomalies.empty:
    # Plot one selected anomaly.
    one_figure = plot_window(anomalies.index[0], result, windows)
    one_figure.show()

    # Or plot all anomaly windows in a single interactive figure.
    figure = plot_suspicious_windows(result, windows)
    figure.write_html("all_anomaly_windows.html")
```

## Complete runnable example

Install the project and run the example from the repository root:

```bash
python -m pip install -e .
python examples/basic_usage.py
```

The example creates a reproducible signal with a slowly drifting mean, missing
timestamps, NaNs, an invalid quality flag, a duplicate timestamp, and one
artificial energy anomaly. It prints the window metadata and writes all suspicious
windows as separate interactive subplots in `energy_anomaly_windows.html`. Use
`--show` to display the figure interactively or
`--output path/to/figure.html` to select another output.

The complete source is available in [`examples/basic_usage.py`](examples/basic_usage.py).

## Fixed-period SSQ-STFT/MSST analysis

The second stage groups anomalous windows whose grid intervals overlap or touch.
Starting at the first anomalous sample, each group is extended **forward** with
samples from subsequent accepted windows until `period_size` is reached. A group
is excluded, with an explicit reason in `period_metadata`, when it is already
longer than `period_size` or when there is not enough complete data after it.

```python
from spectral_anomaly import (
    analyze_msst_periods,
    plot_msst_periods,
    prepare_analysis_periods,
)

period_metadata, periods = prepare_analysis_periods(
    result,
    windows,
    period_size=1024,
)
analyses = analyze_msst_periods(
    periods,
    sampling_frequency=1.0,  # Hz; reciprocal of the 1-second sampling period
    iteration_count=3,
    window="hann",
    n_fft=256,
    window_length=128,
    hop_length=4,
)
if analyses:
    figure = plot_msst_periods(analyses)
    figure.write_html("msst_analysis_periods.html")
```

The interactive Plotly figure has one row per studied period: the fixed-length
time signal, its STFT, and its iterative multisynchrosqueezed representation.
Hovering reveals exact time, frequency, and magnitude values. Zooming and panning
remain linked to each individual subplot. The red signal points show
the extent covered by the original consecutive anomalous windows. This stage only
computes and plots representations; it does not yet confirm anomalies or extract
clustering features.

To preprocess raw data, obtain a single transformation over the **entire
monitoring period**, and save its interactive visualization in one call, use
`analyze_msst_monitoring_data`:

```python
from spectral_anomaly import analyze_msst_monitoring_data

full_analysis, figure = analyze_msst_monitoring_data(
    frame,
    value_col="value",
    quality_col="quality",
    valid_quality_flags={"good"},
    sampling_period="1s",
    sampling_frequency=1.0,
    max_interpolation_gap=3,
    iteration_count=3,
    window="hann",
    n_fft=256,
    window_length=128,
    hop_length=4,
    output_html="msst_full_monitoring_period.html",
)
full_period_msst = full_analysis.msst
```

Here, `sampling_period="1s"` describes the spacing of the input samples, while
`sampling_frequency=1.0` gives the same rate in hertz to the MSST. The returned
`full_analysis.msst` is the MSST matrix, `full_analysis.stft` is the original
STFT, and `figure` is the Plotly object. The HTML file is written automatically
to the path passed as `output_html`.

The function applies the detector's timestamp de-duplication, quality masking,
regular-grid construction, and bounded interpolation rules before passing the
complete sequence to MSST. It returns both the numerical `MSSTResult` and the
Plotly figure, and writes a self-contained HTML file to `output_html`. Since MSST
requires a finite signal, the function explicitly rejects a monitoring period
containing a missing run longer than `max_interpolation_gap`.

When preprocessing and energy detection have already been run, the lower-level
`analyze_msst_monitoring_period(result, windows, ...)` remains available for
assembling overlapping accepted windows without processing the raw frame again.

A complete executable example is provided in
[`examples/full_monitoring_msst_usage.py`](examples/full_monitoring_msst_usage.py).
After installing the package, run it from the repository root:

```bash
python examples/full_monitoring_msst_usage.py
```

To select the output file or also open the interactive figure:

```bash
python examples/full_monitoring_msst_usage.py \
    --output reports/full_monitoring_msst.html \
    --show
```

Run the complete example with:

```bash
python examples/msst_usage.py
```

### Using suspicious windows in the next pipeline stage

The result index is the key of the separate `windows` dictionary. Therefore no
NumPy arrays are stored inside DataFrame cells:

```python
for window_id in result.index[result["suspicious"]]:
    item = windows[window_id]
    regularly_sampled_signal = item.signal
    regularly_sampled_time = item.time
    genuinely_observed = item.observed_mask
    interpolated = item.interpolated_mask

    # Later: send regularly_sampled_signal to the MSST implementation.
```
