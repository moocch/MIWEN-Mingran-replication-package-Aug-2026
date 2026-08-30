# Supplementary Note 6 — comb-encoded campaign: reproducibility folder

Verifies the numbers of **Supplementary Note 6** (platform calibrations,
comb construction, training recipe, and the paired statistics of the CNN
battery; SI `Supplementary_Information.tex`, section "Supplementary
Note 6").

Run `python verify_note6.py` (numpy only). Saved run: `verify_note6_output.txt`
(**25/25 assertions PASS**, including the optional Tier-2 cross-checks).

## Sources (read-only, not modified)

- **Provenance package**: `V2/GTSRB_inference/`
  (curated 2026-08-27 from GitHub `QPG-MIT/MIWEN_Mingran`, branch
  `handoff/rung3-session`; see the copied `README_provenance.md` for the
  commit table). The 55-MB `battery_slim.npz` and the ~2-MB weight files
  stay there — referenced, not copied.
- **Frozen comb chain**: `fig4/`
  (GTSRB test cache `data/gtsrb_roi_32x32_test.npz`, clean weights and
  frozen reference in `fig4/a/`) — used by Tier 2 of `verify_note6.py` to
  re-run the clean-trained digital forward.

## Claim → artifact map

| SI claim (Note 6) | archived artifact | verified by |
|---|---|---|
| S6.3 frozen-calibration rerun **98.92 ± 0.30 %** | `battery_frozen_slim.npz` (`preds`,`labels`: 1187/1200; stored `accuracy=0.9892`) | 1.1a–e (±0.30 = binomial s.e., recomputed) |
| S6.3 battery = 1,200 test photographs drawn once, fixed seed | `battery_random1200_idx.npy` == the npz `img_index` (seed 20260805; regenerable by `make_frames.py` at the source) | 1.2 |
| S6.3 frozen offline calibration: **one gain per output column**, fitted on **450 disjoint** images, **no host matrix product** | `frozen_calibration.npz` (`g_l1..g_l4` of length 32/64/128/43; `source='s4_random450 (disjoint calibration frame)'`; `note='…without any digital W@x'`) | 1.7a–c |
| S6.3 differs from first acquisition on **3 of 1,200** images | `battery_digital_preds_extract.npz` (`preds` = as-run 98.83 battery) vs `battery_frozen_slim.npz` | 1.5a–b |
| S6.3 (i) measured vs same-weights digital MAC (**98.42 %**): **10 vs 4** discordant, **+0.50**, exact McNemar **p ≈ 0.18** (n.s.) | `battery_frozen_slim.npz` + `battery_digital_preds_extract.npz` (`digital_preds`), recomputed with the archived `mcnemar_paired.py` itself | 1.3a–d (b=10, c=4, p=0.17957) |
| S6.3 (ii) measured vs clean-trained digital (**99.50 %**): **8 vs 1** discordant, **−0.58**, **p ≈ 0.04** | p and accuracy arithmetic from archived bytes (1.4a–c); full per-image recount by re-running `forward_digital` with the clean weights on the fig4 GTSRB test cache (2.2a–d: 1194/1200 = 99.50 %, clean-only-correct = 8, measured-only-correct = 1, p = 0.03906) | 1.4, 2.2 |
| S6.3 (iii) "a conventional signed-activation digital CNN reaches ≈99.7–99.8 % in the literature" | **FLAG — see below** | documented only |
| S6.2 ns25 noise vector **[0.177, 0.32, 0.102, 0.221] × 0.25**, injected before the BN affine | `r35_ns25_train.py` (`SIG = [0.177*.25, 0.32*.25, 0.102*.25, 0.221*.25]`); injection point: `ladder_cnn_v2.py` `train_model()` at the source | 1.6 |
| S6.2 training recipe (AdamW 1.5e-3, wd 1e-4, warm-up + cosine, label smoothing 0.05, augmentation, grad clipping, val-split checkpoint; ROI crop → 32×32 → [0,1] → 2–98 % stretch; folded BN at export) | code claims: `ladder_cnn_v2.py` + `miwen_frozen_reference.py` (`preprocess`, `PSTRETCH_LO/HI = 2/98`) at the source; logs archived here | documented |
| clean r3plus weights behind the 99.50 baseline (clean test 99.05 full) | `r35_train_log.txt` (final line: `r3plus: test 99.05; exported r35_r3plus_s0_hw.npz`) | 1.8 |
| S6.1 platform constants (TX law 18.0 + (g − 31.5) + 20·log10 a_RMS; 10 MS/s; 5-MHz LO offset; PSR gate 3.0, measured 28.6–31.4; drift < 3.2×10⁻³ rad/slot; 16,384-pt FFT / 610.35 Hz / 512 CP; comb 30 dB RX + 30 dB attenuator; serial 10 dB IF pads, no RX gain) | constants live in the frozen core `4_gtsrb_confusion_mlpN.py` (source: `P_MAX_DBM_DEFAULT=18.0` l.54, `GAIN_MAX_DB=31.5` l.56, `USRP_SAMPLE_RATE=10e6` l.45, `USRP_LO_OFFSET=5e6` l.46, `P_LO_DBM_DEFAULT=-3`, `P_RF_DBM_DEFAULT=-35`); measured PSR/drift values are session-log-only | documented, not asserted |
| S6.4 comb construction + Zadoff–Chu precoding equations | definitional; implementation in `4_gtsrb_confusion_mlpN.py` (comb synthesis/decode) at the source | documented |

## Files

Copied verbatim from `GTSRB_inference/` (`share_20260712/` unless noted):

- `README_provenance.md` (repo root) — curation record, commit table, headline-number map
- `mcnemar_paired.py` — exact paired McNemar (the code used for the p-values)
- `battery_frozen_slim.npz` — 2026-08-07 energy-honest rerun (98.92)
- `frozen_calibration.npz` — frozen per-column gains
- `battery_random1200_idx.npy` — the frozen 1,200-image frame
- `r35_train_log.txt` — **clean** training log (contains the full r3plus record → the 99.50 weights). *Note: despite the name this is not the ns25 log.*
- `r35_ns25_log.txt` — **ns25** training log (r3fast/r3wide sessions; per `README_provenance.md` the fielded r3plus-ns25 run itself has no committed log — its provenance is the identical recipe in `r35_ns25_train.py` plus the weight file's embedded `meta_json`)
- `r35_ns25_train.py` — ns25 wrapper (the noise-vector literal)

Derived here:

- `battery_digital_preds_extract.npz` (7.8 KB) — the four 1,200-element arrays (`preds`, `digital_preds`, `labels`, `img_index`) extracted from the 55-MB `battery_slim.npz`; regenerate + re-verify with `extract_digital_preds.py`; byte-verified against the source by check 2.1
- `extract_digital_preds.py`, `verify_note6.py`, `verify_note6_output.txt`, this README

Referenced, not copied: `battery_slim.npz` (55 MB), weights
`r35_r3plus_ns25_s0_hw.npz` (sha256[:16] `9be6e085f2d12e19`, verified during
curation) and `r35_r3plus_s0_hw.npz`, `4_gtsrb_confusion_mlpN.py`,
`ladder_cnn_v2.py`, `miwen_frozen_reference.py`, `docs/` notes.

## FLAG — uncited literature figure in S6.3(iii)

The SI sentence "a conventional signed-activation digital CNN reaches
≈99.7–99.8 % in the literature" carries **no citation in the
Supplementary Information**. It traces only to the internal repo note
`GTSRB_inference/docs/notes/2026-08-06_digital_vs_miwen_table.md`
("a conventional signed-activation digital CNN reaches ~99.7–99.8 in the
literature … A signed-ReLU baseline was not trained here"), which is
equally uncited. The range is consistent with well-known GTSRB results
(e.g. committee-of-CNNs ≈99.5 %, MCDNN 99.46–99.7 %, spatial-transformer
nets ≈99.8 %), but **no reference is archived anywhere in this chain**;
the SI should cite one (or label the figure as an external benchmark
range). Nothing in this folder can verify it.

## What is NOT verifiable from archived bytes

- The measured PSR values (28.6–31.4) and residual phase-drift slopes
  (< 3.2×10⁻³ rad/slot) — session logs of the Note-4 campaign only.
- The literature figure above (flagged).
- Re-running the hardware itself; the frozen algorithm replay
  (`miwen_frozen_reference.py --verify`) needs the GTSRB ROI cache and the
  55-MB archives at the source (verified there during curation, 2026-08-27).
