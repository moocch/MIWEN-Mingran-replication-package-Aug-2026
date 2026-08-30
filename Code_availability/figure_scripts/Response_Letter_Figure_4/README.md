# Letter Fig. 4 — the fully analog client (verbatim manuscript Fig. 5)

The response letter's Fig. 4 is `figures/fig5_analog_v3_preview.pdf`, a **verbatim
reproduction of manuscript Fig. 5** (the fully analog client). It is **pure
simulation** — no panel contains a hardware measurement; every device number is a
datasheet value, a published measurement, or a constant calibrated on this work's
own platform. All published copies are byte-identical
(MD5 `cbe3979452474b180a9716a6faa7e4b6`): the letter's `figures/` copy, the
canonical generation folder (`fig5/`), and
`published_fig5_analog_v3_preview.pdf` archived here.

## The chain

```
..\fully_analog_simulation\          (the fully-analog simulation archive:
    comb_analog_sim.py + analog_physics.py    calibrated device models,
    data\{mnist.npz, gtsrb_roi_32x32.npz}     datasets,
    weights\ckpt_*.npz                        trained checkpoints,
    reproduce.py, MANIFEST.md)                deterministic driver
        |
        |  reproduce.py budget  -> results\link_budget.json      (panels a, b, c)
        |  reproduce.py train   -> weights\ckpt_*.npz
        |  reproduce.py infer   -> results\results_summary.json  (panel d)
        v
data\{link_budget.json, results_summary.json}   verbatim copies (MD5-verified below)
data\input_glyphs.npz                            panel-a thumbnails (prep_fig5_assets.py)
        |
        v
fig5_analog_v3.py            gate-checked figure script: every displayed drive, SNR,
        |                    constant and accuracy is asserted (zero tolerance)
        |                    against the loaded JSONs, never re-typed
        v
fig5_analog_v3_preview.pdf   = letter Fig. 4 = manuscript Fig. 5
```

Simulation modes: **ZH** = calibrated device twin + link noise + element jitter
("through the chain"); **ZT** = the identical twin forward with noise stripped
("executed digitally", same weights — the ZT bars reuse the ZH checkpoints).

## Folder contents

| item | role |
|---|---|
| `fig5_analog_v3.py` | the figure script (v2 layout, type scaled x1.20; paths resolve relative to the script's own location) |
| `prep_fig5_assets.py` | builds `data/input_glyphs.npz` (panel-a thumbnails) from the simulation archive's `mnist.npz` + the Fig.-4a STOP photo |
| `data/` | exactly the three files the figure script loads |
| `a/ b/ c/ d/` | per-panel provenance (each has its own README naming the code and data behind that panel) |
| `README_upstream.md` | the canonical generation folder's README, copied verbatim (its `simulation/` folder is this archive's `..\fully_analog_simulation`) |
| `published_fig5_analog_v3_preview.pdf` | the published figure, byte-identical to the letter's copy |
| `reproduced/` | the 2026-08-29 re-run: the copied script + `data/`, and the outputs it wrote (`fig5_analog_v3_preview.pdf/.png/.svg/_small.png`) |

The full simulation archive is **not** duplicated here — it lives once at
`..\fully_analog_simulation\` (copied from the frozen manuscript archive's
`fig5\simulation`).

## Provenance MD5s

| file | MD5 |
|---|---|
| `published_fig5_analog_v3_preview.pdf` (= letter `figures/fig5_analog_v3_preview.pdf` = canonical copy) | `cbe3979452474b180a9716a6faa7e4b6` |
| `fig5_analog_v3.py` | `cf6284ce7badb66819afadabd392e427` |
| `prep_fig5_assets.py` | `85689349974871d947b1dd33c2540c14` |
| `data/link_budget.json` (byte-identical to `..\fully_analog_simulation\results\link_budget.json`) | `87dfe4a8cef3bdd196b1c591ea94a6ab` |
| `data/results_summary.json` (byte-identical to `..\fully_analog_simulation\results\results_summary.json`) | `67d78dae114a17f71c76fd2c007e50a8` |
| `data/input_glyphs.npz` | `2b33f14fcb771e11984dcfb45e1a1b38` |
| `reproduced/fig5_analog_v3_preview.pdf` (2026-08-29 re-run) | `b96a95fd170fdffe722269763d550401` |
| `reproduced/fig5_analog_v3_preview.png` (2026-08-29 re-run; **byte-identical to the canonical published PNG**) | `7a0ce9263e2cd3daacd5fc3d9b58392d` |

## How to re-run

The figure (seconds; needs numpy, matplotlib, Pillow):

```
cd reproduced            # or any folder holding fig5_analog_v3.py next to data\
python fig5_analog_v3.py # writes fig5_analog_v3_preview.pdf/.png/.svg/_small.png
                         # next to the script; aborts with "GATE: ..." if any
                         # displayed number deviates from the archived JSONs
```

The underlying simulation (CPU-only; needs numpy + JAX CPU build; all seeds fixed —
`rng(0)` for the budget, `PRNGKey(1000+s)` for the 5 inference draws):

```
cd ..\fully_analog_simulation
python reproduce.py budget   # -> results\link_budget.json      (seconds)
python reproduce.py infer    # -> results\results_summary.json  (~2 min)
python reproduce.py train    # full retrain of the checkpoints  (~30 min, 8 cores)
```

See `..\fully_analog_simulation\MANIFEST.md` for the environment details.

## Verification (2026-08-29, this archive's build)

* `published_fig5_analog_v3_preview.pdf` MD5 = `cbe3979452474b180a9716a6faa7e4b6`,
  equal to the letter's `figures/fig5_analog_v3_preview.pdf` and to the canonical
  `Data_availability/Figure_5/fig5_analog_v3_preview.pdf`. Verified.
* `data/link_budget.json` and `data/results_summary.json` byte-identical to the
  simulation archive's `results/` originals. Verified.
* Re-run: `fig5_analog_v3.py` executed inside `reproduced\`
  (Python 3.13.14, numpy 2.5.1, matplotlib 3.11.0, Pillow 12.3.0, Windows 10).
  **All zero-tolerance gates passed** (settings constants, the CL split
  6.65 + 0.82 dB, all 8 drive/SNR rows, block efficiency, and the four displayed
  accuracies); the script finished with
  `wrote fig5_analog_v3_preview.* — v2 layout, type x1.2 (main 9.00 pt), 180 x 92 mm`.
* Reproduced vs published PDF: same size (52,970 bytes); rendered at **200 dpi with
  PyMuPDF 1.28.0: 0 / 1,028,050 differing pixels**. The two files differ in exactly
  5 bytes, all inside the embedded `/CreationDate` timestamp (PDF metadata), which
  is why the whole-file MD5s differ; every content stream is byte-identical.
* Reproduced 600 dpi PNG vs the canonical published PNG: **byte-identical**
  (MD5 `7a0ce9263e2cd3daacd5fc3d9b58392d`), i.e. 0 / 9,237,423 differing pixels —
  the same count the frozen manuscript archive's audit recorded on 2026-08-28/29.
