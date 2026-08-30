# fig4_v5 — Figure 4: serial in-physics inference + hardware-aware training through the digital twin

**Figure file used by `main_PANS.tex`:** `fig4_v5_preview.pdf` (180 × 120 mm)
**Build:** `python fig4_verify_package.py` → `python prep_fig4_assets.py` → `python fig4_v5.py`
**Source campaign:** 2026-08-23…26 serial-encoding 2×2 experiment at (LO, RF) = (0, 0) dBm,
archived in `../V2/hardware_aware_training_v2/` (repo `QPG-MIT/MIWEN_Mingran`,
branch `twin/high_power-rig`, commit `2c938ef4`, curated v2 package).
The comb-context numbers quoted in the main text come from
`../V2/GTSRB_inference/` (2026-08-03…07 comb battery campaign).

Every number in the figure was **recomputed from the raw prediction files**
(`fig4_verify_package.py`) and cross-checked against the campaign's own
`verify_accuracy.py` and every session log's `FINAL:` line — zero mismatches.

---

## Panels

| panel | content | data | key numbers |
|---|---|---|---|
| a | Serial inference protocol: GTSRB photos, r3plus network, slot encoding, mixer MAC, digital activation, recirculation | photos: `data/gtsrb_roi_32x32_test.npz` indexed by `battery_random1200_idx[0:600]` | 263,904 complex weights; D = 75/800/1600/128; 7,211,904 slots/image; 3.2 µs/slot |
| b | Training abstractions: ideal multiplier vs twin f(x,w) inside the sum | schematic | 15 epochs; twin held-out 0.12 dB |
| c | 2×2 accuracy grid {clean, twin} × {digital, mixer 0 dBm} | `data/fig4_serial_results.npz` | 99.5 / **5.7 ± 0.9** ; 98.5 / **98.5 ± 0.5** ; twin forecast 0.8 % |
| d | Confusion, clean-trained on hardware | `confusion_clean` (43×43, N=600) | 34/600; class 12 predicted 590× |
| e | Confusion, twin-trained on hardware | `confusion_twin` | 591/600 = digital comparator 591/600 |
| f | Per-product transfer at (0,0) + twin inversion | `data/fig4_panel_assets.npz` (from archived stationA payload) | ENOB 2.59 ± 0.06 → 4.32 ± 0.06 bits (4 captures) |
| g | Layer-output relRMSE vs each arm's training model | cal logs / `serial_cal_*_20260823.npz` | clean [0.53, 0.61, 0.32, 2.94]; twin [0.142, 0.087, 0.021, 0.162] |

## Number provenance (all in `data/fig4_numbers.json` with per-entry provenance strings)

- **5.67 % (34/600), class-12 collapse 590/600** — merged from the 5 clean-arm
  npz chunk files in `05_hardware_result_0dBm_N600/clean_arm/` against
  `08_frozen_inputs_and_labels/battery_frozen_slim.npz`; segment checks
  6.33 % / 5.00 % match the session-log FINAL lines.
- **98.50 % (591/600)** — twin-arm npz files; segments 97.67 % / 99.33 %
  match `serial_twin_20260824.log` / `serial_twin_00_fresh_20260825.log`.
- **Digital comparators (pinned pre-hardware):** twin-under-twin 98.33 %
  (seg 1, `06_digital_comparators/serial_predictions.json`) + 98.67 % (seg 2,
  `digital_pins_seg300600.npz`) → pooled **98.50 %** = hardware exactly;
  clean-under-twin 0.67 % + 1.00 % → **0.83 %** (the forecast of the collapse);
  clean-ideal 99.50 % on the battery frame (audit pin; seg-2 comparator 100 %).
- **Binomial σ (N=600):** ±0.9 (5.7 %), ±0.5 (98.5 %), ±0.3 (99.5 %).
- **ENOB 2.59 ± 0.06 → 4.32 ± 0.06 bits** — recomputed by
  `prep_fig4_assets.py` from the archived Stage-A payload
  (`07_supporting_analysis…/serial_stationA_20260823.npz`, (0,0) dBm point,
  2 draws × 2 reps) with the exact convention of `serial_enob.py`
  (FS-referred log2[range/(rms·√12)], duty −3.05 dB, per-pair inversion on a
  4096-point grid). Draw-0 captures reproduce the archived diag-capture
  values (2.53 / 4.38) to 0.01 bit.
- **Drive ladder 0.21 → 0.66 (−9…+7 dBm)** — recomputed from the same npz
  (per-capture linear-fit relative residual, mean over 4 captures:
  0.208 / 0.407 / 0.499 / 0.576 / 0.657).
- **Twin (Methods):** `serial_twin_model.json` sha `b07bb82b…` — held-out
  0.1245 dB; knees +6.38 / +9.05 dBm; K-sweep `twin_ksweep_20260825.json`;
  slot-curve validation worst bin 3.7 %; probe-hash bit-exact
  (`5cb94af8cc5b257a`).
- **Per-layer relRMSE** — calibration blocks of the session logs
  (teacher-forced, each arm against its own training forward; clean vs ideal,
  twin vs twin).
- **Comb context (main text only):** 98.92 ± 0.30 % vs clean-digital 99.50 %
  (McNemar b=9, c=1) from `../V2/GTSRB_inference/` (`battery_frozen_slim.npz`,
  `README_provenance.md`); MNIST 96.2 ± 0.4 % from the July campaign
  (unchanged, `../fig4_v3/data/mnist_mlp3_25db.npz`).

## Data files

- `data/fig4_serial_results.npz` — consolidated verified package (54 keys):
  confusions, accuracies, calibrations, twin surface + CW map, drive ladder.
- `data/fig4_numbers.json` — every scalar with a one-line provenance.
- `data/fig4_panel_assets.npz` — per-product scatter/ENOB + battery indices.
- `data/gtsrb_roi_32x32_test.npz` — rebuilt official test cache
  (`build_cache_from_master.py`, mirrors `fig4_v4/build_gtsrb_test_cache.py`;
  gate: labels match `battery_frozen_slim.npz` 1200/1200). The extracted
  mirror folder `data/German-Traffic-Signs-Dataset-GTSRB-master/` (~300 MB)
  is only needed to rebuild this cache and can be deleted.
- `photos_battery/`, `contact_sheet.png` — evaluated-photo browsing aids.

## Photo picks in panel a (positions k in battery[0:600], all twin-arm correct)

main: k=34 (dark STOP, class 14); montage: k=1 (dark 70), k=5 (crisp 30),
k=79 (motion blur), k=91 (glare), k=4 (blue arrow), k=87 (roadworks).

## Deliberately NOT in the figure (SI candidates; see the `% NOTE(SI 候选)`
comment in `main_PANS.tex`)

second operating point (−10, −24) dBm clean 81.83 %; full-span twin-v1
falsification; comb same-weights elevation (98.92 vs 98.42, mechanism only);
15-vs-150-epoch asymmetry / 60-epoch sibling; per-bin ENOB decomposition;
per-session gate values. The July 10-layer-MLP ladder figure (fig4_v3) is
superseded; its archive and reproduction chain remain in `../fig4_v3/` and
`../response/Dirk_extracted/`.
