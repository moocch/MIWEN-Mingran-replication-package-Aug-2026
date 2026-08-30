# Reviewer 1, Comment 7 — activation-block supplementary note (simulation figures)

`make_c7_layer_scatter.py` generates `figures/c7_analog_layer_scatter.{pdf,png}`
— **the figure used** in the letter's "New Supplementary Note for Comment 7"
and as Supplementary Figure 3 of the revised SI (per-layer multiplication-
fidelity scatter of the simulated three-pass fully analog GTSRB chain: pass 1
dominated by deterministic envelope compression at the hottest drive, passes
2–3 showing the random scatter growing as the unpowered ladder decays,
relRMSE 0.54/0.07/0.26 with noise-off parts 0.54/0.02/0.00) — plus
`c7_layer_scatter_numbers.json`.

`make_c7_activation_figs.py` (auxiliary; its 4-panel figure was replaced by
the per-layer scatter at the author's request) generates
`figures/c7_activation_block_sim.{pdf,png}` and `c7_activation_numbers.json`;
it remains the validation record for the detector-transfer/knee/boost-ceiling
numbers quoted in the note text.

It is a pure-numpy re-implementation of the archived device physics behind
Fig. 5 of the revised manuscript:

- `V2_Manu/Code_availability/pipelines/fully_analog_simulation/analog_physics.py`
- `V2_Manu/Code_availability/pipelines/fully_analog_simulation/comb_analog_sim.py`
  (`ed_video`, `selfmix_features`, `selfmix_ideal`, `plan_Z`, `boost_db`)

**Validation before plotting:** the script asserts that its port reproduces the
archived `Data_availability/raw/fully_analog_simulation/results/link_budget.json` — the GTSRB pass-drive
ladder (−10.00 / −34.27 / −58.48 / −85.16 dBm) to < 0.05 dB and the three block
efficiencies (2.091 / 2.115 / 1.202 %) to < 0.02 points — and that the archived
"settings" block matches the constants hard-coded here (k_ED = 208.7 /V,
CL 7.47 dB, boost 18.5 dB, R_V 3.5 kΩ, step-down 1 dB). In the verified run
every printed digit matched the archive exactly.

Inputs (read-only, from the manuscript reproducibility archive):
- `Data_availability/raw/fully_analog_simulation/weights/ckpt_ZH_L3.npz` — fielded GTSRB L = 3 checkpoint
  (layer-1 weights used for the real-comb panels b/c/d)
- `Data_availability/raw/fully_analog_simulation/data/gtsrb_roi_32x32.npz` — GTSRB test photographs
- `Data_availability/raw/fully_analog_simulation/results/link_budget.json` — archived ladder (assertions)

The ablation numbers quoted in the note's S6.6 (88.29 % → 42.15 % when the
measured transfer is replaced by the idealised square law at execution) are the
archived `ZD_L3` vs `ZI_L3` entries of
`Data_availability/raw/fully_analog_simulation/results/results_summary.json` (same `Z_L3` weights, no noise
in either arm).

Run: `python make_c7_activation_figs.py`  (numpy + matplotlib only; ~40 s).
