# Fig. 3c — Residual distributions of the three model tiers (scalar, N = 1)

**What is drawn:** distributions of the model−measurement residuals over the 1,849-point
(43 × 43) scalar sweep, for the ideal multiplier, the physics-only block, and the full PIML twin.
Residuals are computed live by `fig3_v12_payoff.py` (lines ~114–125) from the three tier
surfaces in `twin_predictions_N1.npz` (this folder = the copy loaded from `../data/`;
keys used: `p_lo_dbm`, `p_rf_dbm`, `p_if_meas_w`, `p_if_phys_w`, `p_if_full_w`).

## Full chain (in `upstream_measured_data_and_twin_code/`)

1. **Acquisition** — `acquisition_code_43x43_sweep/` (`1_heatmap_single_scalar.py`,
   `usrp_mixer_heatmap.py`, `config.py`): USRP X310 dual-channel TX → ZEM-4300 mixer →
   Keysight N9020A readout; both ports −70…+14 dBm in 2-dB steps, 3 repeats/point.
2. **Raw data** — `heatmap_2db.npz` (measured scalar surface).
3. **Twin training** — `Code_PIML_scalar_local_meeting/model/piml_mixer_digital_twin.py` +
   training run record `twin_training_run_PIML_Digital_Twin_2/` (`piml_mixer.pt`,
   `summary.json`; 1.30 dB RMSE full twin / 3.94 dB physics-only / 16.53 dB ideal).
4. **Export** — the training run exported the three tier surfaces to
   `twin_predictions_N1.npz` (the export invocation itself was not logged; the trainer code and
   run weights are archived, and recomputation from them was verified in the 2026-07 audit).

Identical copies of this chain also live in `../../fig2/a/upstream_measured_data_and_twin_code/`
and `../../fig1/b/upstream_scalar_PIML_calibration/` (byte-verified in the previous audit).
