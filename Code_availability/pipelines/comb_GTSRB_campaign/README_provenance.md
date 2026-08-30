# GTSRB inference — provenance package (curated 2026-08-27)

Source: GitHub `QPG-MIT/MIWEN_Mingran`, branch **`handoff/rung3-session`**
(work by jon-morag / Jonathan Morag, 2026-08-03 → 08-07).
Curated subset: ONLY the files directly behind the three headline numbers
below. Everything else on the branch (MNIST arms, MLP-era sessions, ablation
suites, Bussgang/twin fitting, later high-power-rig work) was deliberately
left out.

## The three numbers (all N=1,200, same frozen random frame, GTSRB test)

| # | number | what it is | where it lives |
|---|---|---|---|
| 1 | **98.92 ± 0.30** | MIWEN hardware, energy-honest rerun 2026-08-06→07: frozen offline calibration, host computes NO W@x. 1187/1200. vs digital (#2): b=4/c=10. vs as-run battery 98.83 (08-05): 3/1200 flips (statistically identical). | `share_20260712/battery_frozen_slim.npz` |
| 2 | **98.42** | The SAME ns25 weights executed as an exact linear digital MAC (`forward_digital`), exact-paired on the same 1,200 images. 1181/1200. | `battery_slim.npz['digital_preds']` |
| 3 | **99.50** | Clean-trained (no noise injection) digital model, same architecture, same 1,200 images — the FAIR system-level digital baseline. | recomputed by `results_grid.py` / `consolidated_numbers.py` from `r35_r3plus_s0_hw.npz` |

Interpretation used in the repo (issue #52, docs/issue52_fair_comparison.md):
measured 98.92 > same-weights digital 98.42 is a real, mechanism-attributed
chain property (the conv2 deterministic transform is beneficial; noise-injected
training exploits it), but the system-level fair comparison is
clean-digital 99.50 > measured — "hardware beats digital" as a system claim
was retracted on 2026-08-06. Do not quote #1 vs #3 without this context.

Operating point (frozen protocol): **LO = −3 dBm, RF = −35 dBm**, RX gain 30 dB,
packed frames, 1× (no averaging). See `docs/notes/2026-08-04_algorithm_freeze.md`.

Model: r3plus — 4 layers, complex weights: conv 32@5×5 → conv 64@5×5 →
dense 1600→128 → dense 128→43; magnitude readout, folded-BN affine,
2×2 max-pool, per-image max-norm, argmax of squared magnitudes. 43 classes.

## Key commits (branch `handoff/rung3-session`)

| commit | date | content |
|---|---|---|
| `17fdbd334e` | 08-03 | ns25 model exports (weights + trainer wrapper committed) |
| tag `algorithm-freeze-20260804` | 08-04 | algorithm freeze: r3plus-ns25, protocol, correction stack |
| `7fcc06a46d` | 08-05 | FULL BATTERY COMPLETE: measured 98.83 ± 0.31, digital 98.42 exact-paired |
| `5d93ca9beb` | 08-06 | Fair-comparison correction (issue #52): clean-digital 99.50; system claim retracted |
| `4303e466c3` | 08-07 | Energy-honest fix: frozen offline calibration replaces per-image W@x (adds `frozen_calibration.npz`) |
| `f4ac264ef2` | 08-07 | Energy-honest HARDWARE rerun: **98.92 ± 0.30** (adds `battery_frozen_slim.npz`) |

## File map

### share_20260712/ — code + data + results (repo-relative layout preserved; scripts expect to run in this directory)

Measurement / evaluation code
- `miwen_frozen_reference.py` — single-file audited reference of the frozen
  algorithm. `--verify` reproduces digital refs bit-exactly and replays all
  archived hardware runs; `--hw --calib frozen_calibration.npz` is the exact
  code path of the 98.92 rerun; `forward_digital` is the exact code path of
  98.42 and 99.50. Frozen weights sha256[:16] = `9be6e085f2d12e19` (verified).
- `4_gtsrb_confusion_mlpN.py` — core hardware/DSP module the reference loads
  (USRP drive, OFDM comb synthesis/decode, GTSRB cache builder). Defines
  `P_LO_DBM_DEFAULT = -3.0`, `P_RF_DBM_DEFAULT = -35.0`.
- `run_ladder_hw.py` — production hardware runner used for the as-run 08-05
  battery (98.83, chunks `bat_c1..10`).
- `make_calibration.py` — fits the per-column gain calibration (the kind of
  fit that produced `frozen_calibration.npz`, measured offline on a disjoint set).
- `results_grid.py` — computes/renders the 2×2 deployment matrix:
  99.50 / 98.42 / 98.83 / twin-estimate (figure in docs/notes/figs/).
- `consolidated_numbers.py` — single source of truth printing every N=1200
  number with its {weights, det-dev, noise} label. NOTE: row 2 (run-2
  persistence) needs `bat2_c*.npz`, which were never committed to git —
  comment that block out; rows for #1/#2/#3 run fine.
- `battery_confusion.py` — 43×43 confusion matrix of the measured battery.
- `mcnemar_paired.py` — exact paired McNemar between any two result npz
  (works on `battery_slim.npz` vs `battery_frozen_slim.npz`).
- `make_frames.py` — regenerates the frozen 1,200-image frame from seed
  20260805 and asserts byte-equality with `battery_random1200_idx.npy`.
- `audit_split.py` — train/test contamination check.

Training code + logs (trainer = `ladder_cnn_v2.py`: AdamW, cosine LR,
120 epochs, batch 96, augmentation, label smoothing 0.05, val-split
checkpoint selection)
- `ladder_cnn_v2.py` — `train_model()` injects additive magnitude noise
  `noise[i] * rms(S) * randn` before BN at every layer (the entire
  "hardware-aware" content of the recipe); `export_runner()` folds BN and
  exports the `_hw.npz` inference weights.
- `r35_ns25_train.py` — ns25 wrapper: noise = 25% of the session-measured
  per-layer rel-RMSE, i.e. `[0.177, 0.32, 0.102, 0.221] × 0.25`.
  CAVEAT: the committed wrapper loops over r3fast/r3wide; the r3plus-ns25
  run itself has no committed log — its provenance is the identical recipe
  (this wrapper + freeze note) and the `meta_json` inside the weight file
  (arch r3plus, seed 0, clean-test 98.04 full / 98.33 N300).
- `r35_train_all.py` + `r35_train_log.txt` — clean (no-noise) arms; the log
  contains the full r3plus clean training record → `r35_r3plus_s0_hw.npz`
  (clean test 99.05 full / 98.33 N300) = the weights behind 99.50.
- `r35_ns25_log.txt` — ns25 training log (r3fast/r3wide sessions; see caveat).

Weights (all with embedded meta_json / arch_json)
- `r35_r3plus_ns25_s0.npz` / `r35_r3plus_ns25_s0_hw.npz` — noise-injection
  trained (ns25). The `_hw` file is the FROZEN inference weights
  (sha256[:16] `9be6e085f2d12e19`) behind 98.92 / 98.83 / 98.42.
- `r35_r3plus_s0.npz` / `r35_r3plus_s0_hw.npz` — clean-trained, behind 99.50.

Calibration + evaluation frames
- `frozen_calibration.npz` — frozen per-column gains (offline, disjoint set);
  the correction stack of the 98.92 rerun.
- `battery_random1200_idx.npy` — THE frozen 1,200-image frame (official
  GTSRB test indices, pool 900–12629, seed 20260805).
- `bat_c1..10_idx.npy` — the 10 battery chunks (120 images each).

Results
- `battery_slim.npz` — 08-05 as-run battery: `preds` (measured, 98.8333%),
  `digital_preds` (98.4167%), `labels`, `img_index`, per-layer magnitudes
  (`c*_y_mag_l*`) and digital references (`c*_y_ref_l*`) for all 10 chunks.
- `battery_frozen_slim.npz` — 08-07 energy-honest rerun: `preds` (98.9167%),
  `labels`, `img_index`, `accuracy=0.9892`, `calibration='frozen (no host W@x)'`.
- `AUDIT.md` — auditor's guide for exactly this frozen configuration.

### docs/
- `notes/2026-08-04_algorithm_freeze.md` — freeze note incl. all addenda:
  the energy-honest fix and the 98.92 rerun result (Addendum 2026-08-06 (5)).
- `notes/2026-08-06_digital_vs_miwen_table.md` — the 99.50 / 98.83 / 98.42
  comparison table + deterministic-vs-stochastic decomposition
  (linear 98.42 → +det 98.75 → +noise 98.79 vs measured 98.83) + caveat that
  a conventional signed-activation digital CNN reaches ~99.7–99.8.
- `notes/figs/battery_results_grid_20260805.png` — the 2×2 deployment matrix
  figure (99.50 / 98.42 / 98.83 / twin estimate).
- `notes/figs/battery_confusion_20260805.png` — measured battery confusion matrix.
- `issue52_fair_comparison.md` — full GitHub issue #52 thread (the
  fair-comparison audit and the retraction of the system-level claim).

## What is NOT here (and why)

- `gtsrb_roi_32x32.npz` (GTSRB 32×32 ROI cache, train+test) — gitignored in
  the repo (too large). `4_gtsrb_confusion_mlpN.py` rebuilds it automatically
  on first use (downloads the GTSRB mirror, builds the cache); `make_frames.py`
  then verifies the frame indices. Needed to re-run `results_grid.py` /
  `consolidated_numbers.py` / `forward_digital`; NOT needed to read the
  committed results (the npz archives carry preds + labels).
- `bat_c*.npz` / `bat2_c*.npz` full chunk archives (~100 MB, run-2
  persistence) — never committed to git; only the slim archives exist.
- hwn/r3fast/r3wide sibling arms, MLP-era code, twin/Bussgang fitting,
  later high-power-rig (AB16/AB17) sessions — different experiments,
  intentionally excluded.

## Verification performed while curating (2026-08-27)

Recomputed directly from the copied npz files:
`battery_slim`: measured 98.8333% (1186/1200), digital 98.4167% (1181/1200);
`battery_frozen_slim`: 98.9167% (1187/1200), 3 flips vs as-run, identical
1,200-image frame; frozen-weights sha256[:16] matches `9be6e085f2d12e19`.
