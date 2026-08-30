# Fig. 2e — N=4096 measured vs twin IF-power surface, RMSE 0.79 dB

- Figure: ladder column 5 (all-USRP receive chain; amplitude uncalibrated, so
  this validates the response **shape**; absolute validation rests on N ≤ 8).
- Plot code: `fig2_v3_twin.py`, "b–f — the ladder"; reads `twin_predictions_N4096.npz`
  (meas_db_arb / full_db_arb).
- Direct data: `twin_predictions_N4096.npz` [FIG; copy of the 4096DT training-run output].

Upstream (`upstream_measured_data_and_twin_code\`):
1. Raw measurement `gr_usrp_mixer_vector_heatmap_N4096.npz` [PKG U4] — dual-USRP
   chain (X310 TX RF@1.20 GHz + LO@0.90 GHz → ZEM-4300 → 30 dB att → second USRP
   RX@0.30 GHz), original run 2026-06-05 02:17.
2. Acquisition code `gr_heatmap_vector_inner_product_v2_local.py` [PKG U4].
3. Twin trainer `piml_4096_digital_twin.py` [ARC Meeting 4096DT run; byte-identical
   copy in Meeting 20260712 CH] — inputs: the raw npz, `--init fit_dualpump`,
   N=1 twin summary; saves twin_predictions_N4096.npz. `README.md` + `summary.json`
   here are that run's own record.
4. Training products `twin_training_run_4096DT_model\` [ARC]:
   `piml_mixer_4096.pt`, `summary.json` (ideal 10.35 / physics 1.80 / full 0.79 dB
   + fitted parameters), `fit_dualpump_init.npy`.
5. `mixer_digital_twin_Mingran.py` [PKG U4] — standalone port of the trained twin
   (PHYS dict equals the trained parameters). `plot_mixer_heatmap.py` [PKG U4].
