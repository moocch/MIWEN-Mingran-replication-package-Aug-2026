# Fig. 3b — Twin-predicted compression of the weight waveform w(t)

**What is drawn:** one transmitted N = 65,536 LO data symbol (rebuilt exactly — same seed,
same comb geometry, same 10-dB PAPR clip — from the campaign metadata) and the same envelope
after the calibrated twin's compression law a(t) → a(t)·√η.

## Chain

1. **Direct data** — `gr_ip_scatter_N65536_20260810.npz` (this folder; MD5-identical to the
   2026-08-10 campaign original in `V2\inner_product_4096_655936\upload\`): supplies `seed`,
   `slot_seed_idx`, `slot_kinds`, `vec_N`, `fft_len`, `papr_clip_db = 10`, `p_lo_dbm_tx`.
2. **Rebuild + compression** — `fig3_v12_payoff.py` (this folder), lines ~127–164: rebuilds the
   'D' slot LO symbol from the seed and applies the compression ratio law with the two
   hard-coded constants `PCOMP_LO = -5.1990871823`, `BETA_C = 0.6101367971`.
3. **Provenance of those constants** — `upstream_twin_calibration_N4096/`: the N = 4096 PIML twin
   calibration on the all-USRP chain (acquisition script
   `gr_heatmap_vector_inner_product_v2_local.py`, raw sweep
   `gr_usrp_mixer_vector_heatmap_N4096.npz`, twin trainer `piml_4096_digital_twin.py`, twin class
   `mixer_digital_twin_Mingran.py`, training run `twin_training_run_4096DT_model/` whose
   `summary.json` physics_params `PcompLO`/`betaC` match the hard-coded values to full
   precision). The same 15-parameter set is embedded as `PIML_TWIN_PARAMS` in the correction
   pipeline (`../d/upstream_correction_20260826/1_opt_inner_product_with_DT.py`).
   This folder is a copy of the chain archived (and previously MD5-verified) in
   `../../fig2/e/upstream_measured_data_and_twin_code/`.
