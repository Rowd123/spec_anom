# Spectral anomaly — energy pre-filter

This package implements the inexpensive first stage of an anomaly pipeline. It
does **not** implement MSST or clustering.

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
