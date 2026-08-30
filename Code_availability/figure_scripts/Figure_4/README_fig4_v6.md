# fig4_v6 — Figure 4 (final): CNN on the passive mixer + hardware-aware training

**Figure file used by `main_PANS.tex`:** `fig4_v6_preview.pdf` (180 × 142 mm)
**Build:** `python prep_comb_assets.py` → `python fig4_v6.py`
(also requires `../fig4_v5/data/` — `fig4_serial_results.npz`,
`fig4_panel_assets.npz`, `gtsrb_roi_32x32_test.npz` — built by
`../fig4_v5/fig4_verify_package.py` + `../fig4_v5/prep_fig4_assets.py`)
**Layout:** s41586 (Nature 601, 549) Fig.-4 grammar — two vertical
columns, one claim each, dashed separator; PANS (2603.23974v1) visual
language (Physical/Digital tinted bands, pink method accent, red
measured results). FINAL, approved 2026-08-27 after 11 revision rounds.
Final palette: blue = measured/twin-trained, salmon = ideal-multiplier /
collapse, violet = RF data stream (as fig1a), teal = outputs/argmax,
gold tint = Physical band, gray = digital references, ink = mixer glyph
and twin chips, viridis = measured CW map (as fig2). Column headers:
"CNN inference on a passive mixer" / "hardware-aware training".
Terminology: time-serial encoding, LO, RF = (0, 0) dBm; hardware
columns/matrices explicitly labeled (measured); table and confusion
values carry ±1σ binomial.

## Left column — comb encoding (V2/GTSRB_inference, 2026-08-03…07)

| panel | content | provenance |
|---|---|---|
| a | In-physics CNN protocol: real STOP photo (battery k=34) + real 5×5×3 patch (max-variance window), real conv-1 kernel tiles (\|w\| of `r35_r3plus_s0_hw.npz` c0), real layer-1 feature maps (top-variance channels 20/29/11 of that photo), mixer, digital activation, recirculate ℓ<4, argmax box outside the loop | cache + weights, computed in `prep_comb_assets.py` |
| b | Bars 98.42 ± 0.36 / **98.92 ± 0.30** / 99.50 ± 0.20 (±1σ binomial, N=1,200) | all three recomputed from raw archives with gates (below) |
| c | Confusion matrices: measured in-physics vs clean-trained digital | `conf_measured`, `conf_clean_digital` in `data/comb_assets.npz` |

**Gates (all enforced in `prep_comb_assets.py`, zero tolerance):**
- measured 98.9167 % = `battery_frozen_slim.npz` preds vs labels (frozen-calibration, energy-honest rerun; labels cross-checked against the rebuilt test cache 1200/1200);
- clean-trained digital 99.50 % = `miwen_frozen_reference.forward_digital` with `r35_r3plus_s0_hw.npz`, recomputed locally on the battery;
- same-weights digital 98.4167 % = same forward with `r35_r3plus_ns25_s0_hw.npz`, **bit-identical** to the archived `battery_slim.npz['digital_preds']`;
- paired counts: vs same-weights 10:4 discordant (+0.50, exact p≈0.18); vs clean digital 1:8 (−0.58, exact McNemar p≈0.04). (Recomputed 2026-08-28 from battery_frozen_slim.npz + forward_digital; the earlier 1:9/p≈0.02 pairing belongs to the 08-05 as-run battery 98.83, not the frozen rerun.)
- Mechanism decomposition quoted in Methods (98.42→98.75→98.79 vs as-run 98.83): `V2/GTSRB_inference/docs/notes/2026-08-06_digital_vs_miwen_table.md`.

## Right column — serial 0 dBm (V2/hardware_aware_training_v2, 2026-08-23…26)

| panel | content | provenance |
|---|---|---|
| d ① | **measured CW map** (twin fit data): `01_digital_twin_model/heatmap_unpadded_20260814.npz` — 41×41, both ports −70…+10 dBm in 2-dB steps, 3 repeats/point, all-USRP chain (dual-ch X310 TX both CW tones, second X310 RX at IF 300 MHz, rx_gain 0), acquired 2026-08-14 in ~58 s; twin fit region ≥ −35 dBm (520 cells) | npz meta_json; `serial_twin_fit.py` line 17/27 |
| d ① | twin f chip (two one-pole knees × 20 tanh ridges, held-out 0.12 dB) | `serial_twin_model.json` sha `b07bb82b…` |
| d ① | validation scatter: serial slots at (0,0) dBm | archived stationA payload `serial_stationA_20260823.npz` (capture draw0/rep0). NOTE: the fit script's own G3 check used `serial_diag_cap.npy` (same day, same geometry, not archived); the displayed capture is the archived equivalent at the same operating point |
| d ② | training loop: W → f ← x → Σ → \|·\|,act → layers 1–4 → loss; backprop through f | structure of `ladder_cnn_v2.twin_forward` / `train_serial_twin_short.py` (15 epochs, no noise injection, no bias correction) |
| d ③ | deploy W_HA → same mixer, same power; no twin at inference | `serial_nn_runner.py` (`serial_twin_s0_hw.npz`, sha `825ab4ee…`) |
| e | table: clean 99.50 / **5.67 %**; twin-trained 98.50 / **98.50 %** (N=600) | `verify_accuracy.py` + `06_digital_comparators` pins (digital committed before hardware) |
| f | Confusion matrices on hardware: clean-trained (590/600 one class) vs hardware-aware (591/600 = pinned comparator 591/600) | merged chunk preds, `../fig4_v5/data/fig4_serial_results.npz` |

Text-only numbers (Methods): drive ladder 0.21→0.66 (−9…+7 dBm),
per-product ENOB 2.59 ± 0.06 → 4.32 ± 0.06 bits (4 archived captures),
per-layer relRMSE clean [0.53, 0.61, 0.32, 2.94] vs twin
[0.142, 0.087, 0.021, 0.162], twin forecast of the collapse 0.8 %.

## Fit-data verification (user query, 2026-08-27)

The fielded twin **was fit on the CW map**, not on serial-slot data:
`serial_twin_fit.py` loads `heatmap_unpadded_20260814.npz` (line 17),
restricts to ≥ −35 dBm (line 27), holds out 20 %. The (0,0) serial-slot
measurement enters only as the **G3 validation** (lines 103–150,
"fitted on none of that data", worst bin 3.7 %). Figure d-① draws
exactly this order: map → fit → f → validate.
