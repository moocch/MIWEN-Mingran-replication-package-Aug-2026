# Fig. 2f — N=4096 measured IF-power surface (baseline of the model-tier row)

- Figure: first of the g–j row. Plot code: `fig2_v3_twin.py`, "g–j" section;
  reads `measured_db` from `N4096_cascade_raw_data.npz`.
- Direct data: `N4096_cascade_raw_data.npz` [FIG; = PKG U4\N4096, MD5-verified].
  Verified to be a pure repackaging of `twin_predictions_N4096.npz`
  (arrays bit-identical; ideal surface = constant offset). The original one-off
  repackaging script was not archived; the backfilled
  `upstream_measured_data_and_twin_code\rebuild_make_cascade_raw_data.py`
  rebuilds it and verifies **17/17 keys bit-identical** (executed).

Upstream (`upstream_measured_data_and_twin_code\`): same N=4096 chain as `..\f`
(raw sweep, trainer `piml_4096_digital_twin.py`, `twin_training_run_4096DT_model\`,
`mixer_digital_twin_Mingran.py`).
