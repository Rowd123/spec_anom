# Spectral anomaly — STFT morphology with MSST comparison

This package provides quality-aware signal windowing, the historical causal
energy detector, STFT/MSST transforms, and a first morphology prototype. STFT is
the structural default; MSST remains available for controlled comparisons. Energy
scoring remains available but is no longer required to select windows for the
structural MSST path. Classification and clustering are intentionally out of
scope for the prototype.

## Optional SAM 2.1 spectrogram experiment

`sam_segmentation.py` is a removable experiment parallel to the existing
morphology path. It consumes the **existing complex STFT**, converts its
magnitude with `log1p`, clips at the 1st/99th percentiles, scales to uint8, and
repeats the grayscale channel as RGB. It never consumes the local-normalization,
significance, coherence, connected-component, or candidate-structure masks, and
it makes no anomaly decision.

SAM is intentionally not a mandatory package dependency. In a separate virtual
environment, follow Meta's official installation approach:

```bash
# Install a CPU/CUDA PyTorch build suitable for the machine first:
# https://pytorch.org/get-started/locally/
python -m pip install 'git+https://github.com/facebookresearch/sam2.git'
mkdir -p checkpoints
wget -P checkpoints \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt
```

The application never downloads weights. Supply the checkpoint explicitly and
choose `--device auto` (CUDA when `torch.cuda.is_available()`, otherwise CPU),
`--device cpu`, or `--device cuda`:

```bash
# Bounding-box prompt in image pixels: x=time column, y=frequency row
python examples/sam_structure_usage.py \
  --checkpoint checkpoints/sam2.1_hiera_small.pt \
  --device auto --box 20 10 50 35 --output sam_box.html

# Positive point in physical coordinates: TIME_SECONDS FREQUENCY_HZ
python examples/sam_structure_usage.py \
  --checkpoint checkpoints/sam2.1_hiera_small.pt \
  --device auto --point 120 0.15 --output sam_point.html
```

The default model config is `configs/sam2.1/sam2.1_hiera_s.yaml` (the config for
`sam2.1_hiera_small`). `--model-config` can override it. A box is
`[x_min, y_min, x_max, y_max]` in STFT image pixels. The example's point is
`(time, frequency)` and is converted using the actual `spectral_time` and
`frequencies` arrays. Python callers may use `segment_point([x, y])` or
`segment_points([[x1, y1], ...], [1, 0, ...])`, where 1 is a positive prompt
and 0 excludes a location.

With `multimask_output=True` (the wrapper default), SAM normally proposes
multiple masks. `SAMSegmentationResult.scores` exposes its predicted mask-quality
(predicted IoU) estimates; `best_mask` selects `argmax(scores)`. These scores are
model confidence estimates, not measured IoU against ground truth and not
physical relevance or anomaly scores. Use `compute_iou` and `compute_dice` only
when an independent reference mask is available. The four-panel Plotly output
shows the original STFT magnitude, exact grayscale input, selected mask, and
overlay, and prints all returned scores.

This prototype does not assume that generic natural-image pretraining transfers
to spectrograms. Results can be sensitive to prompt placement, STFT resolution,
image scaling, weak/diffuse boundaries, and the domain gap. CPU inference may be
slow. The automatic mask generator is deliberately deferred until box and point
prompts have been evaluated reliably.

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
  -> STFT for every accepted window (MSST optional)
  -> local robust normalization
  -> structure tensor
  -> significant connected components
  -> descriptive window/component features
  -> later: detection decision and unsupervised classification
```

Window acceptance in this path depends only on NaNs, quality flags, valid-sample
requirements, and interpolation-gap limits. Neither energy nor the historical
`suspicious` flag decides whether a spectral transform is run.

### Prototype feature rationale

The deliberately small window-level feature set is intended for visual
validation, not as a final anomaly score. In particular, **coherent structure is
not synonymous with anomaly**: nominal signals may contain persistent coherent
bands. Coherence and significance help segment and describe objects; only a
later model of historically observed component families can assess novelty.

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

The normalization is a running 2-D median/MAD of the selected spectral magnitude
and therefore does
not assume a single global noise level or white stationary noise. The structure
tensor is evaluated in dimensionless local scale coordinates, making its
coherence orientation-neutral. Connected components use 8-connectivity and do
not assume a single-valued ridge `f(t)`, so vertical and multi-frequency shapes
remain representable. Skeleton, curvature, loops, branch counts, HDBSCAN, and a
final decision rule are intentionally deferred until these maps and basic
features have been validated visually.

`minimum_component_area` defaults to 1: small components are retained as data,
not silently declared to be noise. Raising it is an explicit exploratory
segmentation choice whose effect should be reported.

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
    representation="stft",  # default; use "msst" only for comparison
    transform_options={"window_length": 128, "n_fft": 256, "hop_length": 4},
)

# All quality-valid windows are present, regardless of their energy score.
window_id = next(iter(analyses))
item = analyses[window_id]
print(item.features)
print(item.components)
plot_structural_window(item).write_html("stft_structure_window.html")
```

The six panels show the time signal, the selected raw representation, its local
normalization, structure coherence, retained components with their IDs, and a
component feature table. `representation="stft"` is the default; pass `"msst"`
explicitly for the experimental alternative. Run the full example with
`python examples/structure_usage.py`.

### Component population for a future normal-structure model

`analyze_structural_components` returns `(metadata, analyses, components)`, where
`components` has one row per `(window_id, component_id)`. It includes frequency
centroid/min/max, relative time centroid, duration, bandwidth, area, mean/max/
integrated local significance, mean/median/q90 coherence, normalized aspect
ratio, linearity, the interpretable raw orientation, and the clustering-safe
axial encoding `cos(2 theta)`, `sin(2 theta)`.

```python
from spectral_anomaly import (
    analyze_structural_components,
    component_features,
    fit_component_feature_scaler,
    transform_component_features,
)

metadata, analyses, components = analyze_structural_components(
    frame,
    value_col="value",
    sampling_frequency=1.0,
    sampling_period="1s",
    window_size=256,
)

# Identification columns and the raw angle are excluded from X automatically.
X = component_features(components)
scaler = fit_component_feature_scaler(components)
X_scaled = transform_component_features(components, scaler)
```

The scaler applies `log1p` by default to non-negative, typically right-skewed
size/intensity variables (area, duration, bandwidth, significance and aspect
ratio), then fits a per-column median and `1.4826 * MAD`. Frequency positions,
relative time, coherence, linearity, and axial orientation coordinates are not
log-transformed. Constant training columns receive scale 1 rather than creating
NaNs. The fitted scaler is reusable on later windows; no labels, clusters, or
anomaly decisions are produced.

Some candidate features are intentionally redundant: frequency min/max overlap
with centroid and bandwidth; area overlaps with duration, bandwidth and aspect
ratio; mean/max/integrated significance are related; and mean/median/q90
coherence summarize the same distribution. Before clustering, correlation and
stability analyses should select a smaller subset to avoid overweighting one
physical property merely because it has several correlated columns.

Segmentation still depends strongly on neighborhood size, tensor scale,
significance/coherence thresholds, STFT resolution and minimum area. Components
can split, merge, or disappear when these change, and overlapping physical
phenomena can become a single connected object. Before trying HDBSCAN on real
data, validate repeatability across nominal recordings, operating regimes,
signal-to-noise ratios and nearby parameter values; check feature stability and
missing frequency bands; fit scaling on training periods only; and quantify how
often each learned family occurs per window. A future cluster label `-1` must
mean only “not assigned to a dense learned family,” never automatic physical
anomaly.

### From connected fragments to candidate structures

A connected component is a segmentation object, not necessarily one physical
occurrence. `associate_component_fragments` builds an undirected graph whose
nodes are the original components. An edge is added only when all four physical
criteria pass: temporal gap, frequency-interval gap, frequency-centroid
difference, and axial orientation difference. Connected graph groups become
`CandidateStructure` objects; original component masks and IDs are never
discarded.

The four experimental controls are:

* `max_fragment_time_gap_seconds`;
* `max_fragment_frequency_gap_hz`;
* `max_fragment_frequency_centroid_difference_hz`;
* `max_fragment_orientation_difference_radians`.

Their defaults are deliberately conservative: all distance tolerances are zero,
apart from a 10-degree orientation tolerance. Useful values depend on STFT time/
frequency resolution and on the physical process, so real-data examples pass
explicit values. Association is transitive: if C1--C2 and C2--C3 pass, all three
form one candidate even when C1--C3 does not. This repairs short interruptions
but can also chain two distinct occurrences through intermediate fragments.
False merges remain possible for two nearby parallel occurrences with overlapping
frequency support, at crossings where local orientation is unstable, for broad
bands whose centroids happen to agree, or through a long transitive chain of
individually acceptable gaps. Conversely, frequency drift or a noisy PCA angle
can prevent a legitimate merge. The association is intentionally confined to
one window and does not join objects across window boundaries.

Candidate features are recomputed from the union of the retained pixels, never
by averaging component rows. Alongside component geometry/significance/
coherence, candidates expose `fragment_count`, `time_span_seconds`,
`active_duration_seconds`, `total_gap_duration_seconds`,
`maximum_gap_duration_seconds`, and `gap_fraction`. Thus a continuous band and a
fragmented band with the same outer span remain distinguishable.

`support_bandwidth_hz` means occupied frequency-bin count times frequency
resolution; it is therefore one `delta_f` for a one-bin object.
`frequency_span_hz` means `frequency_max - frequency_min` and is zero for that
same object. Legacy `bandwidth_hz` remains an alias of support bandwidth.
Likewise, candidate `time_span_seconds` includes the full bin support between
outer edges, whereas `active_duration_seconds` counts only time columns holding
candidate pixels.

```python
from spectral_anomaly import analyze_candidate_structures, plot_candidate_structures

metadata, results, components_df, candidates_df = analyze_candidate_structures(
    frame,
    value_col="value",
    sampling_frequency=1.0,
    sampling_period="1s",
    window_size=256,
    window_ids=[0],
    association_options={
        "max_fragment_time_gap_seconds": 8.0,
        "max_fragment_frequency_gap_hz": 0.01,
        "max_fragment_frequency_centroid_difference_hz": 0.02,
        "max_fragment_orientation_difference_radians": 0.26,
    },
)
print(components_df)
print(candidates_df)
plot_candidate_structures(results[0]).write_html("stft_candidates_window_0.html")
```

The candidate diagnostic keeps the original component IDs (`C1`, `C2`, ...)
beside candidate IDs (`S1`, `S2`, ...), plus both DataFrames for auditing. These
objects are neither normal nor anomalous at this stage. This local association
reconstructs occurrences within a window; future HDBSCAN-style work would group
similar candidates across many windows and is a separate problem.

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
    transform_options={"window_length": 128, "n_fft": 256, "hop_length": 4},
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
