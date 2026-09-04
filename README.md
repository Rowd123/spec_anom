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
from spectral_anomaly import detect_energy_anomalies, plot_window

result, windows = detect_energy_anomalies(
    frame, value_col="value", quality_col="quality",
    valid_quality_flags={"good"}, sampling_period="1s",
    window_size=256, overlap=128,
)
anomalies = result[result["suspicious"]]
if not anomalies.empty:
    plot_window(anomalies.index[0], result, windows)
```
