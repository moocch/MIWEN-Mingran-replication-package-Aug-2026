# Fig. 2b — N=2 measured vs twin IF-power surface, RMSE 1.91 dB

- Figure: ladder column 2. Plot code: `fig2_v3_twin.py`, "b–f — the ladder".
- Direct data: `twin_predictions_N2.npz` [FIG] (vector-twin output).

Upstream (`upstream_measured_data_and_twin_code\`):
1. Raw measurement `usrp_mixer_vector_heatmap_N2.npz` [PKG U4] — N=2
   frequency-multiplexed inner-product sweep, 33×33 grid (−70…+10 dBm),
   USRP X310 TX → ZEM-4300 → N9020A center-tone readout (run 2026-05-22 06:45).
2. Acquisition code `2_heatmap_vector_inner_product_v3.py` [ARC Meeting] — the
   actual N=2/4/8 acquisition script (output stem matches the raw filenames).
   Note: the package's `gr_heatmap_vector_inner_product_v2_local.py` is the
   **N=4096** dual-USRP script, not this chain.
3. Original timestamped run archive `raw_acquisition_runs_3_N248.zip` [ARC].
4. Vector twin (generator of twin_predictions_N2.npz):
   `Code_Refined_Figs_N2N4N8_local_meeting\` [PKG U4] —
   `model\piml_vector_mixer_digital_twin.py` (scalar physics frozen; |⟨a,b⟩|²
   scaling, two PAPR knee shifts, bounded residual) +
   `piml_vector_data.py` + `make_all_figs.py`.
   The run that produced the published 1.91/1.67/1.55 dB used, per the run
   record, nn_max_db = 6 (code default is 4), nn_hidden = [48, 48],
   n_iter_nn = 8000, n_iter_joint = 4000 — i.e. |δ| ≤ 6 dB, as now stated
   in manuscript Methods.
5. Training-run record `vector_run_summary.json` [FIG; renamed run summary] —
   per-N metrics (N=2: ideal 18.71 / physics 3.37 / full 1.91 dB) and the
   full training config (source of the values quoted above).
