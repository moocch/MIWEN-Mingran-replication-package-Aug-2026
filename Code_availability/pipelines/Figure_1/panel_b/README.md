# Fig. 1b — IF output power vs LO drive of the calibrated mixer model (simulation)

- Figure: top panel of `fig1b_motivation.png/.pdf/.svg` (bottom panel = Fig. 1c).
- Simulation code: `fig1b_motivation.py`, section "(b) transfer curve" — forward
  model of the calibrated physics block at P_RF = −35 dBm, LO swept −70…+14 dBm,
  vs the ideal multiplier; operating point P_LO = −3 dBm marked.
- Inputs: the seven calibrated physics parameters hard-coded in the script
  (G = −1.84 dB, P_sat = +1.73 dBm, β_LO = 1.045, P_comp = +4.47 dBm,
  β_RF = 2.13, P_n = −73.6 dBm, L_leak = −84.3 dB).

Calibration chain (copy in `upstream_scalar_PIML_calibration\`):
- `heatmap_2db.npz` [PKG U3] — raw measured 43×43 grid (−70…+14 dBm, 2 dB steps,
  3 repeats, Keysight N9020A readout)
- `Code_PIML_scalar_local_meeting\` [PKG U3] — scalar PIML twin trainer
  (`model\piml_mixer_digital_twin.py`, saves twin_predictions.npz) + plot scripts
- `twin_predictions.npz` [PKG U3] — trainer output (= twin_predictions_N1.npz)
- `acquisition_code_43x43_sweep\` [ARC/RIG] — instrument scripts for the sweep
  (usrp_mixer_heatmap.py + config.py; 1_heatmap_single_scalar.py is the rig revision)
- `twin_training_run_PIML_Digital_Twin_2\` [ARC] — training run record
  (summary.json: seed 0, 24 starts, RMSE 3.94 physics-only / 1.30 dB full twin;
  piml_mixer.pt = trained residual weights)
