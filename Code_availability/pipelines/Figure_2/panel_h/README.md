# Fig. 2h — physics-only twin surface at N=4096, RMSE 1.80 dB

Same data file and chain as `..\g\README.md`; this panel plots
`physics_only_db`, RMSE = `error_metrics[1,0]` = 1.80 dB.

`physics_only_db` is the physics core of the **N=4096 twin** — the dual-pump
model recalibrated to the all-USRP chain (15 parameters: Hill-softened LO
turn-on, symmetric RF pump term, PAPR knee shifts, leakage, noise floor, and
the absolute-scale constant for the chain's uncalibrated amplitude) with the
bounded residual switched off. It is **not** the scalar 7-parameter block with
frozen parameters: the frozen-scalar tier ("naive N=1") scores 18.27 dB on
this sweep (`twin_training_run_4096DT_model\summary.json`,
`rmse_naive_n1_shape`), while the recalibrated physics core scores the plotted
1.80 dB (`rmse_physics_only`). See manuscript Methods, "Digital-twin
architecture and training", final paragraph.

Plot code: `fig2_v3_twin.py`, "g–j" section. Forward-model code:
`upstream_measured_data_and_twin_code\piml_4096_digital_twin.py` /
`mixer_digital_twin_Mingran.py`; training-run record
`upstream_measured_data_and_twin_code\twin_training_run_4096DT_model\summary.json`
(config: hidden 48, max_db 5.0, n_iter_phys 1200, n_iter_nn 6000,
n_iter_joint 4000).
