# Operational photometry and finite-sample controls

`operational_photometry.matched_measure` is the original Model E forced photometry operator used for Eq. (2) of A1. Its two functions are extracted from the 17 October 2026 source-shot implementation of `operational_photometry.py`, SHA-256 `0a19d3c74b5005eaa0e2e9dc445021a5d27ece3a7f06bf3f22fbb9d3e3f375db`. Uniform and non-unit flats, signed negative amplitude, unusable support and subpixel placement are controls from its accompanying `operational_photometry_golden.py`.

For a supplied candidate trail, the inputs are the observed calibrated image minus estimated background (electrons), the base variance per pixel (electrons squared), the dimensionless flat, and a template normalized to unit flux over its complete support. The positional prediction is (x, y) in image pixels; the template's center is given in (row, column) order. Mask a defective pixel with infinite base variance. A partial footprint retains the original unit-flux normalization. A candidate crossing another source can acquire that source's light.

```python
import numpy as np
from operational_photometry import matched_measure

residual = np.array([[20., 60.]])
template = np.array([[0.25, 0.75]])
out = matched_measure(residual, np.full((1, 2), 4.), np.ones((1, 2)),
                      template, prediction_xy=(0., 0.), centre_y=0, centre_x=0)
print(out["flux_e"], out["photometric_snr"])  # 80 electrons, SNR 8.165
```

On the usable support, let D = Σ Q²/V₀ and a = (Q/V₀)/D. The signed fitted flux is F̂ = Σ a r. The operational denominator includes base variance 1/D and estimated source variance max(F̂,0) Σ a²Q/f. `null_z_base_only` uses just the base variance. The operator consumes a supplied candidate and fixed variance map. Detection completeness and false-alarm rates require running the search and measuring its outcomes.

`python goldens.py photometry` replays the five original 100,000-draw experiments in A1 supplement S8, using the production measurement operator to extract its weights. The comparison predicts ensemble variance at the injected F; the per-image photometric SNR plugs in its measured F̂. The first four observed/predicted flux variance ratios span 0.9952–1.0047. In the fifth, subtracting the median of eight independent reference frames raises the measured variance to 1.1605 times the known-background prediction. The associated standard deviation rises by 7.73%. This example diagnoses omitted reference noise and does not specify a universal correction.

The four 4 s native-camera controls use the archived diffuse-sky arrays (e⁻/pixel/s) and unit incident trail templates (15 arcsec/s). Each draws 1,024 independent images of a 207.43368 e⁻ source. C0 omits sky light; CW includes the historical W sky. Gaussian read noise and uniform quantization residuals follow Poisson electron production. Here the observed/predicted statistic is the flux **standard deviation**. `native_trails.npz` selects the native arrays from S8 `noise_inputs.npz`; `cameras.json` carries the original settings; `reference.csv` selects the corresponding rows of S7 `historical_mc_checks.csv`. S8 `image_trials.py` supplies the native-camera calculation and `reference_trials.py` the independent two-pixel draws.
