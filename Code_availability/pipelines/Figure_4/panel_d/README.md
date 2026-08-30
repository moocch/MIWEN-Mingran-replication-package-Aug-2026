# Fig. 4d — Hardware-aware training: measure once, fit the twin, train through it

Three steps drawn, each backed here:

## ① Measure once + fit + validate

- `upstream_twin_fit/heatmap_unpadded_20260814.npz` — the **measured 41×41 CW power map**
  drawn in the panel: both ports −70…+10 dBm in 2-dB steps, 3 repeats/point, assembled all-USRP
  chain, acquired 2026-08-14 in ~58 s (meta_json embedded in the npz).
- `upstream_twin_fit/serial_twin_fit.py` — the fit: restricted to the product-dominated region
  (both ports ≥ −35 dBm, 520 cells), 20 % holdout, two one-pole knees × 20 bounded tanh ridges.
- `upstream_twin_fit/serial_twin_model.json` — the frozen twin surface (held-out 0.12 dB).
- `upstream_validation_capture/` — the (0,0)-dBm **validation scatter** displayed in the panel:
  `serial_stationA_20260823.npz` (archived drive-ladder payload; capture draw0/rep0) +
  `serial_stationA.py` (its acquisition code) + `serial_enob.py` / `serial_enob_00_20260824.json`
  (the ENOB convention; the json is a hard input of `../fig4_verify_package.py`).

## ② Train through f

- `upstream_training/train_serial_twin_short.py` — the fielded 15-epoch hardware-aware training
  (no noise injection, no bias correction; also pins the clean-under-twin predictions).
- `upstream_training/ladder_cnn_v2.py` (twin_forward, BN-folding export) and
  `upstream_training/miwen_serial_frozen_reference.py` (frozen serial algorithm).
- `upstream_training/train_serial_twin_short.log` — training log of the fielded checkpoint
  (ep15 selected; digital test full 98.56 %).
- Re-training reads the full `gtsrb_roi_32x32.npz` cache, which is rebuilt automatically from
  the public GTSRB mirror (`4_gtsrb_confusion_mlpN.py` / `../build_cache_from_master.py`).

## ③ Deploy

- `upstream_training/weights_twin_15ep_FIELDED/serial_twin_s0_hw.npz` — the deployed
  hardware-aware weights (sha 825ab4ee…), plus the pre-export checkpoint.
- `upstream_training/weights_clean_baseline/r35_r3plus_s0_hw.npz` — the clean-arm weights as
  archived by the serial campaign.
- No twin and no correction in the inference path (one frozen scalar gain per layer; see
  `../e/upstream_hardware_test_code/serial_nn_runner.py`).

Plot data for the panel (`cw_*` map keys, `scatter_captures`) live in
`../data/fig4_serial_results.npz` / `../data/fig4_panel_assets.npz`, built by
`../fig4_verify_package.py` + `../prep_fig4_assets.py` from the files above.
