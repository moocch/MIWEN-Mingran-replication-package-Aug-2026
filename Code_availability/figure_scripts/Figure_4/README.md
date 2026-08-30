# Fig. 4 — A convolutional network on a passive mixer, and hardware-aware training that unlocks its strongly nonlinear regime

**Figure file used by `main_PANS.tex`:** `fig4_v6_preview.pdf`
(the tex includes `fig4_v6/fig4_v6_preview.pdf` — a byte-identical copy sits in the sibling
stub folder `../fig4_v6/` so the archive compiles self-contained).

**Figure restructure (author, 2026-08-28/29, `fig4_v6.py` edited in place):** the former
panel e (2 × 2 outcome table) was removed — its four numbers are quoted in the Results
text — and panel d was simplified (step-3 box deleted; step subtitles and the
CW-map/twin/time-serial sub-labels moved into the caption). Panels are now **a–e**
(old f → new e). No data file and no gate changed; `prep_comb_assets.py`,
`fig4_verify_package.py`, `prep_fig4_assets.py` and every npz/json in `data/` are
byte-identical to the 2026-08-28-verified state.

**Verified by execution:**
- 2026-08-29 (this build): `fig4_v6.py` re-rendered from the archived bytes in this folder —
  regenerated PNG **pixel-identical to the published one**.
- 2026-08-28 (unchanged inputs): `prep_comb_assets.py` re-ran end-to-end with **all
  zero-tolerance gates passing** — measured battery 98.9167 % recomputed from
  `battery_frozen_slim.npz` predictions vs labels; clean-trained digital 99.5000 % recomputed
  live by `miwen_frozen_reference.forward_digital`; same-weights digital 98.4167 % recomputed
  and **bit-identical** to the archived `battery_slim.npz['digital_preds']`; the regenerated
  `comb_assets.npz` is array-for-array equal to the archived copy. These results remain valid
  verbatim: every input of that pipeline is hash-unchanged.

## Layout of this folder

The original build layout is three folders (`fig4_v6/`, `fig4_v5/`, `V2/`); here everything is
gathered under one `fig4/` root per the archive convention. The scripts keep their original
relative paths, so **to re-run them, arrange a mirror**:

```
<anywhere>\fig4_v6\{fig4_v6.py, prep_comb_assets.py, data\comb_assets.npz}
<anywhere>\fig4_v5\data\{fig4_serial_results.npz, fig4_panel_assets.npz, gtsrb_roi_32x32_test.npz}
<anywhere>\V2\GTSRB_inference\   <- copy of b\upstream_comb_campaign
cd <anywhere>\fig4_v6 ; python fig4_v6.py            # figure only (needs the two fig4_v5 npz)
cd <anywhere>\fig4_v6 ; python prep_comb_assets.py   # full left-column gate recompute
```
(`fig4_verify_package.py` additionally hard-codes the manuscript root path at its line 11 —
edit that one line to point at the mirror; it rebuilds `fig4_serial_results.npz`,
`fig4_panel_assets.npz` and `fig4_numbers.json` from the raw serial-campaign archives in `d/`
and `e/` and asserts every reported number.)

## Files at this root

| file | role |
|---|---|
| `fig4_v6.py` | final figure script (panels a–e); loads `data/comb_assets.npz` + `../fig4_v5/data/{fig4_serial_results,fig4_panel_assets}.npz` |
| `prep_comb_assets.py` | left-column asset builder with the zero-tolerance gates above |
| `fig4_verify_package.py` | serial-campaign verification/package builder (asserts every table/confusion number from the raw chunk predictions) |
| `prep_fig4_assets.py` | panel-d validation-scatter + battery-index asset builder (serial_enob.py convention) |
| `build_cache_from_master.py` | rebuilds `data/gtsrb_roi_32x32_test.npz` from the public GTSRB mirror (label gate: battery labels 1200/1200) |
| `README_fig4_v6.md`, `README_fig4_v5_pipeline.md` | the build folders' own READMEs (written for the pre-restructure panel lettering a–f; panel-to-provenance content still applies with e→text, f→e) |
| `data/` | direct inputs: `comb_assets.npz`, `fig4_serial_results.npz`, `fig4_panel_assets.npz`, `fig4_numbers.json` (every scalar with a provenance string), `gtsrb_roi_32x32_test.npz` (rebuilt official 12,630-image test cache, label-gated) |

## Panels

| panel | content | chain lives in |
|---|---|---|
| a | in-physics CNN protocol (real battery photo k=34, real conv-1 kernels/feature maps of the clean checkpoint) | `a/` + `data/` |
| b | comb-encoded accuracy bars 98.42 / **98.92** / 99.50 (± 1σ binomial, N = 1,200) | `b/upstream_comb_campaign/` (the complete curated V2\GTSRB_inference campaign package) |
| c | confusion matrices, measured comb cascade vs clean-trained digital | `c/` (+ b's campaign) |
| d | hardware-aware training: measured 41×41 CW map (2026-08-14) → twin fit (held-out 0.12 dB) → (0,0)-dBm validation scatter → train through f → deploy with no twin | `d/` (twin fit, validation capture, training code + fielded weights) |
| e | confusion matrices on hardware, clean (590/600 one class, 5.7 ± 0.9 %) vs hardware-aware (591/600 = pinned comparator, 98.5 ± 0.5 %); the same chain backs the text-quoted 2 × 2 numbers | `e/` (raw chunk predictions, hardware runs, pinned comparators, pre-registration audit, frozen inputs, test code) |
