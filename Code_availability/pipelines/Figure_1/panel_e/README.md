# Fig. 1e — model–hardware error: ideal-multiplier assumption vs PIML twin

- Figure: bottom panel of `fig1c_theory.png` (top = Fig. 1d; same script).
- Code: `fig1c_theory.py`, section "(e)": bar chart 16.53 dB → 1.30 dB (12.7×),
  same unmodified hardware.
- Data: `twin_predictions_N1.npz` [PKG U3] — the 43×43 scalar characterization
  (keys p_if_meas_w / p_if_phys_w / p_if_full_w / p_lo_dbm / p_rf_dbm); both bar
  values are directly recomputable from it. Full chain: `..\b\upstream_scalar_PIML_calibration\`.
