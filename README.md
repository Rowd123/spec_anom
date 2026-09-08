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
5. Each accepted window is locally mean-centred. Its energy is
   `sum((centered * taper)**2) / sum(taper**2)`. This is a window-energy-normalised
   local variance estimate: it removes slow level drift and makes white-noise
   energy comparable between tapers. It is not an unbiased estimate for every
   coloured/nonstationary signal, so the taper should remain fixed in production.
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
    window_size=256, overlap=128,
)
anomalies = result[result["suspicious"]]
if not anomalies.empty:
    # Plot one selected anomaly.
    plot_window(anomalies.index[0], result, windows)

    # Or plot all anomaly windows in a single figure.
    figure, axes = plot_suspicious_windows(result, windows)
    figure.savefig("all_anomaly_windows.png", dpi=150, bbox_inches="tight")
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
windows as separate subplots in `energy_anomaly_windows.png`. Use `--show` to
display the figure interactively or `--output path/to/figure.png` to select another
output.

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
    figure, axes = plot_msst_periods(analyses)
    figure.savefig("msst_analysis_periods.png", dpi=150, bbox_inches="tight")
```

The figure has one row per studied period: the fixed-length time signal, its STFT,
and its iterative multisynchrosqueezed representation. The red signal points show
the extent covered by the original consecutive anomalous windows. This stage only
computes and plots representations; it does not yet confirm anomalies or extract
clustering features.

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
