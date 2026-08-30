# Fig. 4b — Comb-encoded accuracy against its paired digital references

Bars (± 1σ binomial, N = 1,200 frozen battery photographs):
**98.42 %** same-weights digital · **98.92 %** measured in-physics · **99.50 %** clean-trained digital.

## Chain — `upstream_comb_campaign/` = the complete curated comb campaign (V2\GTSRB_inference)

- `README_provenance.md` — the campaign's own file-by-file provenance map (read this first).
- `share_20260712/` — code, weights, logs and results:
  - **training**: `ladder_cnn_v2.py`, `r35_train_all.py`, `r35_ns25_train.py` (+ logs
    `r35_train_log.txt`, `r35_ns25_log.txt`); deployed noise-trained weights
    `r35_r3plus_ns25_s0_hw.npz`, clean weights `r35_r3plus_s0_hw.npz`.
  - **hardware run**: `miwen_frozen_reference.py` (frozen forward; `--hw --calib
    frozen_calibration.npz` is the energy-honest rerun path behind 98.92),
    `4_gtsrb_confusion_mlpN.py` (USRP/OFDM-comb core), `run_ladder_hw.py`,
    `make_frames.py`, `make_calibration.py` + `frozen_calibration.npz` (the frozen linear
    per-neuron decoder calibration, fitted once on 450 disjoint calibration images).
  - **measured results**: `battery_frozen_slim.npz` (the 98.9167 % predictions),
    `battery_slim.npz` (as-run battery incl. `digital_preds` = the 98.4167 % bit-identity
    gate), `battery_random1200_idx.npy` (frozen battery frame).
  - **recomputes/audit**: `results_grid.py`, `consolidated_numbers.py`, `battery_confusion.py`,
    `mcnemar_paired.py`, `audit_split.py` + `AUDIT.md` (split audit).
- `docs/` — campaign notes, including `notes/2026-08-06_digital_vs_miwen_table.md`
  (the mechanism-decomposition table quoted in Methods — text-only in the manuscript).

Gates enforcing all three bar values run in `../prep_comb_assets.py`
(re-executed 2026-08-28, all passed; see `../README.md`).
