# Fig. 5 — A fully analog client, simulated from published device numbers

**Figure file used by `main_PANS.tex`:** `fig5_analog_v3_preview.pdf`
(the tex includes `fig5_analog_v3/fig5_analog_v3_preview.pdf` — a byte-identical copy sits in
the sibling stub folder `../fig5_analog_v3/`).

**PURE SIMULATION** — no panel contains a hardware measurement. Every device number is a
datasheet value, a published measurement, or a constant calibrated on this work's own platform
(the mixer twin of Fig. 2 and the noise floor P_n = −73.6 dBm; that calibration chain is
archived under `../fig1/b/upstream_scalar_PIML_calibration/` and `../fig2/b/`).

**2026-08-29 script revision (author, edited in place at 00:31):** background color blocks and
in-panel annotations (pass-drive ladder, fully-analog banner, booster/self-mixer/equation
lines) removed from the artwork; the self-mixer parameter ($k_{ED} = 208.7$ V$^{-1}$) now
lives in the caption with its source citation, the rest in the Results text. Panels a–d,
inputs, and the zero-tolerance gate mechanism are unchanged. (`README_fig5_analog_v3.md`,
the author's build README, predates this revision and still describes the annotated artwork;
its provenance and gate documentation remain valid.)

**Verified 2026-08-29 (and identically 2026-08-28):** `fig5_analog_v3.py` re-executed from the
archived `data/` in this folder — regenerated PNG **pixel-identical to the published one
(0 / 9,237,423 differing pixels)**, all gates passed (every displayed drive, SNR, constant and
accuracy is asserted against the loaded JSONs, never re-typed). The two `data/` JSONs are
byte-identical to the simulation archive's originals in `simulation/results/`.

## The chain (one line)

`simulation/comb_analog_sim.py` + `analog_physics.py` (calibrated device models) →
`reproduce.py budget` (deterministic, rng(0)) → `results/link_budget.json` (panels a, b, c) and
`reproduce.py train` (checkpoints) + `reproduce.py infer` (5 noise draws, full test sets) →
`results/results_summary.json` (panel d) → verbatim copies in `data/` →
`fig5_analog_v3.py` (gates) → `fig5_analog_v3_preview.pdf`.

Modes: **ZH** = calibrated device twin + link noise + element jitter ("through the chain");
**ZT** = the identical twin forward with noise stripped ("executed digitally", same weights —
the ZT bars reuse the ZH checkpoints); ZC (ideal multiplier, no twin/noise) is used only by the
Supplementary-Note-9 2×2, which this figure does not display.

## Files

| item | role |
|---|---|
| `fig5_analog_v3.py` | figure script (SCALE = 1.20 typography; gates on every displayed number) |
| `prep_fig5_assets.py` | builds `data/input_glyphs.npz` (panel-a thumbnails) from `simulation/data/mnist.npz` (pass it as argv[1]) + the Fig.-4a STOP photo (`../Data_availability/Figure_4/comb_assets.npz` / copy in `a/`) |
| `README_fig5_analog_v3.md` | the build folder's own README (v3 = v2 layout with type ×1.20; scale-knob verification) |
| `README_fig5_analog_v2_provenance.md` | the v2 README that the v3 README defers to for all caption/provenance/gate documentation |
| `data/` | exactly the three files the figure script loads: `link_budget.json`, `results_summary.json`, `input_glyphs.npz` |
| `simulation/` | the fully-analog simulation archive (from `V2\fully analog\files (1).zip::miwen_fully_analog_archive.zip`): `MANIFEST.md`, `comb_analog_sim.py`, `analog_physics.py`, `reproduce.py`, `prepare_gtsrb.py`, `data/{mnist.npz, gtsrb_roi_32x32.npz}`, `weights/` — the two fielded L3 checkpoints behind Fig. 5d **plus (2026-08-29) the six checkpoints behind the main-text depth-ladder and control numbers** (`ckpt_MZH_L2/ZH_L2/MZH_L4` for the L = 2/L = 4 sentences, `ckpt_MHa_L3/MO_L3/MD_L3` for the 92.38/93.18/96.05 controls) — and `results/{link_budget.json, results_summary.json, results_summary.csv}`. Per the author's scoping (2026-08-29), SI-only material (the simulated 2×2 and ablation arms of Supplementary Note 8) stays in the V2 zip for the author's separate SI effort |

## How to reproduce

```
cd fig5
python fig5_analog_v3.py                       # the figure, from the archived JSONs

cd simulation                                  # the simulation itself (CPU-only)
python reproduce.py budget                     # -> results/link_budget.json  (seconds)
python reproduce.py infer                      # -> results/results_summary.json  (~2 min)
python reproduce.py train                      # full retrain of the checkpoints (~30 min, 8 cores)
```
Requirements: numpy, matplotlib, Pillow, and JAX (CPU build) for the simulation; all seeds fixed
(`rng(0)` for the budget, `PRNGKey(1000+s)` for the 5 inference draws). See `simulation/MANIFEST.md`.

## Note on `data/results_summary.json`

`data/results_summary.json` is kept **whole** — it is the verbatim, gate-checked input the
figure script loads — and therefore also contains tags for runs beyond the four the figure
displays (MZT_L3d / MZH_L3 / ZT_L3d / ZH_L3).
