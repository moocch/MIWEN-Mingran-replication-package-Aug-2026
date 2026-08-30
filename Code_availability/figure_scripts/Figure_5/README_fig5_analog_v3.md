# fig5_analog_v3 — Figure 5, v2's layout with the type ladder scaled up

**Figure file for `main_PANS.tex`:** `fig5_analog_v3_preview.pdf` (180 × 92 mm)
**Build:** `python fig5_analog_v3.py`
**One knob:** `SCALE` at the top of the script (or `FIG5_SCALE=1.25 python
fig5_analog_v3.py` to try a value without editing).

## What this is

v3 is `fig5_analog_v2` with **exactly one change**: every type size is
routed through `S()`, which multiplies it by `SCALE`. Layout, panel
geometry, canvas (180 × 92 mm), strings, numbers, gates, palette and
stroke widths are all v2's, untouched.

```python
SCALE = 1.20          # the only typography knob
FS, FS_S, FS_TAG = 7.5, 6.3, 9.0        # v2's ladder, verbatim
def S(pt): return pt * SCALE
```

`SCALE = 1.0` reproduces `fig5_analog_v2_preview.pdf` **exactly** — checked
span by span against the v2 PDF (128 text spans, identical strings and
sizes). So the diff really is "same figure, bigger type".

## Why 1.20

| | v2 | v3 @ 1.20 | manuscript body |
|---|---|---|---|
| main label (`FS`) | 7.5 pt | **9.0 pt** | 10 pt |
| small label (`FS_S`) | 6.3 pt | **7.56 pt** | — |
| panel tag (`FS_TAG`) | 9.0 pt | **10.8 pt** | — |
| smallest label on the figure | **5.2 pt** | **6.24 pt** | — |

1.20 is the smallest bump that lifts the *smallest* thing on the artwork
(the 5.2 pt IF / LO / RF port tags) above the **6 pt floor** most journals
set for figure text, while the main label stays at 9.0 pt — below the 10 pt
body size that was rejected as too large. Measured in the built PDF, page 6:
figure text now runs 6.21–8.96 pt with the tags at 10.76 pt, against 9.96 pt
body.

**The knob is verified collision-free over `SCALE` = 1.00–1.30**
(main 7.5–9.75 pt), so it is safe to dial either way with one number.

## Nudges that made the larger type fit

Positions only — no string, number, gate, panel or canvas change. Each one
is a place where v2 had a *fixed* millimetre offset that stopped working
once the glyphs grew:

* **a** — the three-line stack in the activation glyph (`analog` /
  `activation` / `block (b)`) was on a fixed 2.6 mm pitch; it is now
  `2.6 × SCALE`, so the lines keep their proportional leading.
* **a** — the `IF` and `LO` port tags were 0.1 mm from touching; moved to
  (31.6, 19.4) and (35.0, 22.1).
* **b** — `to next` / `IF port` was a two-line stack on a fixed 2.0 mm
  pitch; now `2.4 + 2.8 × SCALE`.
* **b** — the bottom-left cluster was the tightest spot on the figure:
  `1 : n` 10.9 → 12.0, the self-mixer note (38.0, 10.6) → (33.0, 9.2) —
  re-centred as well as lowered, because at 1.20 it ran past the panel
  edge into panel c's y-axis label — and the squarer identity 5.0 → 4.4.
* **c** — `MNIST` 53.0 → 51.5 (it was crossing the axis top) and `GTSRB`
  34.6 → 33.0 (it was sitting on its own curve).

## Verification

A script renders the module, walks every **drawn** `Text` artist and
reports out-of-canvas placement and pairwise overlap in mm. Current
state at every scale from 1.00 to 1.30: **clean** — nothing outside the
canvas, no overlaps.

## History of this request (author, 2026-08-28)

1. Uniform 10 pt inside the 180 × 92 mm canvas → reverted the same day
   ("太丑了"). 10 pt does not fit a layout budgeted for 5.2–7.5 pt.
2. Uniform 10 pt with the composition re-cut to three bands at
   180 × 151 mm → rejected: too big, and it changed the layout. Kept for
   reference as `rejected_2026-08-28_uniform10pt_threeband.py` (it does
   work and it does put every label at exactly body size — run it if that
   trade ever becomes attractive again).
3. **This file** — v2's layout back, type scaled ×1.20.

## Still worth knowing

**Figs. 1–4 are on the old ladder.** Measured on page in the built PDF:
Fig. 2 and Fig. 5's neighbours run at ~7.3–7.5 pt main, fig1 at 5.8 pt.
Fig. 5 is now the largest-set figure in the manuscript at 9.0 pt. The same
`S()` / `SCALE` transformation applies cleanly to `fig2_best.py`,
`fig3_piml_ladder*.py` and `fig4_v6.py` (they use the identical
`FS` / `FS_S` / `FS_TAG` idiom) if you want the set consistent.

## Panels — unchanged content

| panel | content | provenance |
|---|---|---|
| a | Unrolled cascade: DAC → [ring mixer (LO = broadcast weight comb $w^{(\ell)}$ at 0.30/0.35/0.40 GHz) → LPF → passive activation block] ×2 → mixer → LPF → ADC → class. Band: "fully analog — no ADC / DAC / amplifier between passes". Pass drives −10 → −34 → −58 dBm | topology: `plan_Z()` + `forward()` in `comb_analog_sim.py`; drives gate-checked from `data/link_budget.json` |
| b | The activation block: 1:n impedance-transform matching network (+18.5 dB measured boost), centre-tapped secondary driving a zero-bias anti-phase pair (odd terms cancel → squarer), baseband LPF returning $r_k=\sum_m y_{m+k}y_m^*$; −1 dB step-down into the next IF port | topology: sub-nW self-mixer wake-up front ends (Mangal & Kinget); constants: `BOOST_DB`, `K_ED`, `IL_STEP` in `comb_analog_sim.py` |
| c | Per-tone SNR vs pass, MNIST + GTSRB level plans; SNR = 0 line; pass 4 dashed/open = out of budget → the client supports L = 3 | `data/link_budget.json` verbatim |
| d | Same checkpoint executed twice (digital = mode ZT, chain = mode ZH): MNIST L3 94.30/93.90, GTSRB L3 88.83/87.53 | `data/results_summary.json`, tags MZT_L3d/MZH_L3/ZT_L3d/ZH_L3 |

Dropped panels, dropped part numbers, the detector-supply wording and all
caption provenance are exactly as documented in `fig5_analog_v2/README.md`
— none of that was touched.

## Gates (enforced in `fig5_analog_v3.py`, zero tolerance)

Carried over from v2 unchanged: settings (cl_pass 7.47 dB, boost 18.5 dB,
k_ED 208.7 /V, R_v 3500 Ω, step-down 1.0 dB, first pass −10 dBm; displayed
strings FORMATTED from the loaded settings, not re-typed); the CL split tie
6.65 + 0.82 == cl_pass_dB; P_n = −73.6 dBm pinned as a named constant;
the MNIST and GTSRB drive/SNR ladders; block efficiency 2.255; and the four
displayed accuracies 94.30 / 93.90 ± 0.1 / 88.83 / 87.53 ± 0.14.

## Reverting

`fig5_analog_v2/` is untouched. To go all the way back: point the
`\includegraphics` in `main_PANS.tex` at
`fig5_analog_v2/fig5_analog_v2_preview.pdf` and swap the two
`!fig5_analog_v3…` lines in `.gitignore` back to `!fig5_analog_v2…`.
Or just set `SCALE = 1.0` here, which gives the same artwork.
