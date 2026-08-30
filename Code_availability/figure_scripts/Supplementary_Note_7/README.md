# Supplementary Note 7 — time-serial campaign: reproducibility folder

Verifies the numbers of **Supplementary Note 7** (session gates, twin
ridge sweep, amplitude-resolved ENOB; SI `Supplementary_Information.tex`,
section "Supplementary Note 7"), plus the main-text hardware headline the
note supports (clean 5.67 % vs hardware-aware 98.50 %, N=600).

Run `python verify_note7.py` (numpy only). Saved run:
`verify_note7_output.txt` (**40/40 assertions PASS**). The archived audit
script `verify_accuracy.py` also runs standalone from this folder
(`verify_accuracy_output.txt`).

## Source (read-only, not modified)

`V2/hardware_aware_training_v2/` (archived from
GitHub `QPG-MIT/MIWEN_Mingran`, branch `twin/high_power-rig`, commit
`2c938ef47862…`, 2026-08-26). The sub-folder structure (00/01/05/06/07/08)
is preserved here so `verify_accuracy.py` runs unmodified. Big session
logs (~4 MB), training weights, and `serial_run_audit.pdf/.tex` stay at
the source — referenced below.

## Claim → artifact map

| SI claim (Note 7) | archived artifact | verified by |
|---|---|---|
| S7.2 held-out RMSE vs ridge count: **6.90 / 4.03 / 3.50 / 3.09 / 2.44 / 0.46 / 0.29 / 0.19 / 0.15 / 0.06 dB** at K = 0/1/2/3/5/7/10/15/20/30 | `01_digital_twin_model/twin_ksweep_20260825.json` (sweep code: `twin_ksweep.py`) | 1a–b (all ten, to 0.005) |
| S7.2 repeatability floor **0.015 dB** | same json, `floor=0.01519` | 1c |
| S7.2 fielded surface **K = 20 (held-out 0.12 dB)** | `serial_twin_model.json` (`K=20`, `rmse_holdout=0.12446`, gates G1–G3 true) recomputed **exactly** as a function evaluation on `heatmap_unpadded_20260814.npz` with the frozen rng(9) 20 % split (520 cells, 102 held out) | 2a–2d |
| — the 0.15-vs-0.12 discrepancy | **FLAG, resolved — see below** | 2a/2b/2e + printed analysis |
| S7.3 drive-ladder deviations **0.21 / 0.41 / 0.50 / 0.58 / 0.66** at (−9,−9)/(−3,−3)/(0,0)/(+3,+3)/(+7,+7) dBm (mean of 4 captures; rel. RMS residual of each capture's best complex linear fit) | recomputed from `07_…/serial_stationA_20260823.npz` (`yhat_*`, `x_d*`, `w_d*`; g = ⟨xw,ŷ⟩/⟨xw,xw⟩, the `serial_stationA.py` convention) | 3b (all five, to 0.005) |
| S7.3 each ladder capture holds **256 pairs**; four (0,0) captures = two draws × two repeats | same npz (20 `yhat` arrays of length 256; `meta_json` nslot=256) | 3a |
| S7.3 overall ENOB **2.59 ± 0.06 → 4.32 ± 0.06 bits** (mean ± 1 s.d. over the four (0,0) captures) | recomputed per capture: full-scale-referred log₂[range(xw)/(RMSE·√12)], twin transfer inverted per product on a 4,096-point grid (`serial_enob.py` convention, twin = `serial_twin_model.json` product term) | 4a–b (2.588 ± 0.056 → 4.324 ± 0.061) |
| S7.3 eight amplitude bins, endpoints **6.5 → 9.7** (smallest bin) and **2.1 → 3.5 bits** (largest) | recomputed, four captures pooled (1,024 products, quantile bins, half-open) | 4c–d (6.455/2.119 → 9.707/3.472) |
| S7.1 pack-test gain reference **0.1155**, **eight consecutive sessions within 1.6 %** (0.1155–0.1173) | `00_report_and_audit/2026-08-24_serial_m10m24_prespec.md` (final section: "Eight consecutive sessions with consistent gates (packtest |g| 0.1155-0.1173 at (0,0))"); spread arithmetic recomputed = 1.56 % | 6c–d (text presence + arithmetic); per-session values: session logs + `serial_run_audit.tex` at source |
| S7.1 gate constants (±20 % tol, residual band [0.30, 0.75], chain band, sync floor 50 / typical ~1,700; no accuracy gates) | `04_hardware_test_code/serial_nn_runner.py` + session logs at source; verdict-band pre-registration archived here in `00_report_and_audit/` | documented, not asserted |
| S7.4 fielded 15-epoch checkpoint **98.56 %** under the twin forward (full digital test set) | `06_digital_comparators/serial_predictions.json` (`twin_under_twin_full=98.56`) | 5d |
| S7.4 60-epoch sibling **99.07 %**, never fielded; 150-epoch clean arm; ~2.9 h sprint | `03_training_results/train_serial_twin.log` + `weights_twin_60ep_sibling/` at source | documented |
| S7.5 one image = **7,211,904 slots ≈ 23 s** slot time | pure arithmetic: r3plus MAC count 28²·32·75 + 10²·64·800 + 128·1600 + 43·128 = 7,211,904; × 32 samples/slot (TSLOT=32, `serial_nn_runner.py` l.38) / 10 MS/s = 23.1 s | 6e–f |
| S7.5 ~100 s wall clock/image, ~8.5 h per 300-image half, four days; 256² bilinear twin table (interp. error ≤ 0.07 dB); chirp matched-filter timing-only decode; row-aligned chunking | session logs (~30,400 s per half) + `serial_run_audit.tex` (table max deviation 0.069 dB) at source | documented, not asserted |
| main text: clean **5.67 %** (34/600) vs hardware-aware **98.50 %** (591/600) at (0,0) dBm, identical frozen battery[0:600] | raw chunk predictions `05_hardware_result_0dBm_N600/{clean_arm,twin_arm_hw_aware}/serial_nn_*.npz` + labels `08_frozen_inputs_and_labels/battery_frozen_slim.npz`; also re-counted by the archived `verify_accuracy.py` | 5a–c, 5g |
| digital comparators pinned pre-hardware (twin-under-twin pooled 591/600 = 98.50; clean-ideal seg-2 300/300; clean-under-twin seg-2 3/300 — the predicted collapse) | `06_digital_comparators/serial_predictions.json` + `digital_pins_seg300600.npz` | 5e–f |
| serial chain: **10 dB IF padding, no receive gain** (stated in S6.1) | `serial_stationA_20260823.npz` `meta_json` (`chain='IF +10dB pads (2x5dB post-mixer)'`; runner `rx_gain=0`) | 6a |

## FLAG — resolved: sweep 0.15 dB vs fielded 0.12 dB at K = 20

S7.2 quotes 0.15 dB at K=20 in the sweep but "(held-out 0.12 dB)" for the
fielded surface. Investigation (checks 2a–2e of `verify_note7.py`):

- **Same data, same split.** Both `serial_twin_fit.py` (fielded) and
  `twin_ksweep.py` (sweep) build the identical product-dominated region
  (520 clean cells) and the identical deterministic 20 % holdout
  (`np.random.default_rng(9) < 0.2` → the same 102 cells). Not a split
  difference.
- **Not the feedthrough terms.** The fielded model carries two per-port
  feedthrough parameters the sweep omits, but evaluating the fielded
  surface without them changes the held-out RMSE by < 1e-4 dB (fitted
  leaks ≈ −210/−102 dB are negligible in-region) — check 2e.
- **It is two different fits of the same K=20 family.** The sweep re-fits
  from scratch at every K with a leaner staged budget (3-parameter physics
  stage; `max_nfev` 800/2500/1500) than the fielded production fit
  (4-start, 5-parameter stage; 2000/3000/2000). An 83-parameter tanh-ridge
  least-squares problem has many near-degenerate minima: the sweep's K=20
  refit landed at 0.1522 dB, the fielded fit at 0.1245 dB. Re-running both
  staged fits during archiving (different scipy/BLAS) landed at 0.115 and
  0.138 dB — ~0.03 dB of optimizer-trajectory scatter at fixed capacity,
  far above the 0.015-dB repeat floor and immaterial to the sweep's
  conclusion (capacity saturates at K ≈ 20–30).
- **The archived 0.12446 is exact.** Check 2b reproduces
  `rmse_holdout` bit-for-bit as a pure function evaluation of the stored
  parameters — environment-independent, unlike the refits.

Suggested SI touch-up (optional): note that the sweep's per-K refits and
the fielded fit are independent staged fits, so their K=20 values differ
by fit-to-fit scatter (~0.03 dB).

## Files (this folder)

- `verify_note7.py`, `verify_note7_output.txt` — the verification and its saved run
- `verify_accuracy.py`, `verify_accuracy_output.txt` — the source archive's own audit script (unmodified) + saved run
- `00_report_and_audit/2026-08-24_serial_m10m24_prespec.md` — pre-registration (pins committed before hardware; eight-session gate record)
- `01_digital_twin_model/` — `heatmap_unpadded_20260814.npz` (CW map, 3-repeat), `serial_twin_model.json` (fielded K=20 surface), `twin_ksweep_20260825.json`, and the two fit scripts `serial_twin_fit.py` / `twin_ksweep.py` (the flag evidence)
- `05_hardware_result_0dBm_N600/` — the 7 raw chunk-prediction npz files (both arms)
- `06_digital_comparators/` — `serial_predictions.json` (seg-1 pins), `digital_pins_seg300600.npz` (seg-2 per-image pins)
- `07_supporting_analysis_enob_and_driveladder/` — `serial_stationA_20260823.npz` (drive ladder, 20 captures), `serial_enob.py` (the ENOB/inversion convention), `serial_enob_00_20260824.json`
- `08_frozen_inputs_and_labels/` — `battery_frozen_slim.npz` (labels), `battery_random1200_idx.npy` (frame)

**Caveat on `serial_enob_00_20260824.json`:** that json is the *earlier
single-capture* diagnostic analysis (overall 2.53 → 4.38 bits, endpoints
6.62/2.06 → 9.79/3.24) of a capture (`serial_diag_cap.npy`) that is not in
the curated archive. The SI quotes the *four-capture Stage-A statistics*,
which `verify_note7.py` recomputes from `serial_stationA_20260823.npz`
(2.59 ± 0.06 → 4.32 ± 0.06; 6.5/2.1 → 9.7/3.5). Do not compare the SI
against the json directly.

Referenced, not copied (at the source archive): session logs
(`serial_*_2026082*.log`), `serial_run_audit.tex/.pdf`, training code and
weights (`02_training_code/`, `03_training_results/`), hardware runner
(`04_hardware_test_code/`), per-arm calibration npz
(`serial_cal_{clean,twin}_20260823.npz`), `s4_random450_idx.npy`.

## What is NOT verifiable from archived bytes

- Per-session gate readings (pack-test |g| per session, chain-gate levels,
  sync margins ~1,700) — session logs at source only; the eight-session
  1.6 % consistency is asserted here only as text-presence in the archived
  pre-registration record plus arithmetic.
- The 60-epoch sibling's 99.07 % and the ~2.9-h training time — training
  log at source.
- Wall-clock/rig-time figures (~100 s/image, ~8.5 h/half) — session logs.
- The twin 256² bilinear-table interpolation error (≤ 0.07 dB) — audit
  report at source (0.069 dB).
