# Fig. 2a — N=1 (scalar) measured vs twin IF-power surface, RMSE 1.30 dB

- Figure: ladder column 1 (top = measured, bottom = digital twin).
- Plot code: `fig2_v3_twin.py`, "b–f — the ladder"; reads `twin_predictions_N1.npz`
  (p_if_meas_w / p_if_full_w).
- Direct data: `twin_predictions_N1.npz` [FIG; = PKG U3\scalar\twin_predictions.npz, MD5-verified].

Upstream (`upstream_measured_data_and_twin_code\`):
1. Raw measurement `heatmap_2db.npz` [PKG U3] — USRP X310 dual-channel
   (RF@1.20 GHz, LO@0.90 GHz) → ZEM-4300, IF read by Keysight N9020A via VISA;
   both ports −70…+14 dBm, 2 dB steps, 43×43 = 1,849 points, 3 repeats each.
2. Acquisition code `acquisition_code_43x43_sweep\` [ARC/RIG]:
   `usrp_mixer_heatmap.py` + `config.py` (earliest version, predates the data);
   `1_heatmap_single_scalar.py` (rig revision).
3. Twin trainer `Code_PIML_scalar_local_meeting\model\piml_mixer_digital_twin.py`
   [PKG U3] — three-stage training (multi-start L-BFGS physics → residual with
   physics frozen → joint fine-tune); saves twin_predictions.npz.
4. Training-run record `twin_training_run_PIML_Digital_Twin_2\` [ARC]:
   `summary.json` (seed 0, 24 physics starts; RMSE 3.94 physics-only / 1.30 dB
   full twin; fitted 7 parameters) + `piml_mixer.pt` (trained residual weights).
