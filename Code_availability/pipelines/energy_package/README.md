# Fig. 5 — energy per real MAC vs. inner-product size `N`

Self-contained package for the energy-scaling figure: raw measurement data,
the code that turns it into the figure, vector output ready for the
manuscript, a plot-data `.npz` for re-drawing without the raw data, full
documentation of every formula, and the group-meeting slides.

Nothing here needs the instrument, the network or a proprietary library —
`numpy` + `matplotlib` rebuild everything.

---

## Layout

```
fig5_energy_package/
├── README.md                       ← you are here
├── METHODS.md                      ← ★ every symbol, formula and number defined
│
├── code/
│   ├── fig5_energy_scaling.py      raw .npz  →  figure + plot-data .npz + .csv
│   └── replot_from_npz.py          plot-data .npz  →  vector figure (standalone)
│
├── data/
│   ├── raw/                        the measurements, exactly as acquired
│   │   ├── gr_fig3c_ip_scatter_20260810_011915_N4096/
│   │   │     gr_fig3c_ip_scatter.npz          all measured quantities
│   │   │     gr_fig3c_ip_scatter.png          decoded-vs-true scatter
│   │   │     gr_fig3c_ip_scatter_enob.png     ENOB ± 1 s.d., both panels
│   │   ├── gr_fig3c_ip_scatter_20260810_002043_N65536/   (same three files)
│   │   ├── 1_inner_product_scatter_v4.py      acquisition script (N=65536)
│   │   └── 1_inner_product_scatter_v4_N4096.py  acquisition script (N=4096)
│   ├── fig5_plot_data.npz          ★ everything plotted, for re-drawing
│   └── fig5_plot_data_curves.csv   the four curves as plain text
│
├── figures/
│   ├── fig5_energy_scaling.pdf     ★ vector — use this in the manuscript
│   ├── fig5_energy_scaling.svg     vector — editable in Illustrator/Inkscape
│   └── fig5_energy_scaling.png     600 dpi raster — for slides
│
└── slides/
    ├── main.tex                    10-slide walkthrough (beamer, XeLaTeX)
    ├── fig5_energy_scaling.pdf     figure the deck embeds
    └── slides_preview.pdf          compiled preview
```

---

## Quick start

```bash
# 1. rebuild the figure from the raw measurements (verifies every number)
python code/fig5_energy_scaling.py

# 2. restyle for the manuscript, from the plot data alone
python code/replot_from_npz.py --font "Arial Nova" --width 3.4 --out figures/single_col
python code/replot_from_npz.py --no-bottom          # top panel only
```

`fig5_energy_scaling.py` reads the operating points **directly out of the
raw `.npz`** — transmit power, waveform length, RMSE and ENOB are never
hard-coded — and asserts each published number against them. If the data and
the figure ever drift apart, the script stops instead of drawing a stale
value.

For LaTeX: `\includegraphics[width=\columnwidth]{fig5_energy_scaling.pdf}`.

---

## The figure in one paragraph

Client-side energy per real MAC, `e_ip = e1 + e2 + e3`, against
inner-product size `N`, at fixed accuracy (`RMSE < 0.0625`, i.e. ≈ 2.4-bit
ENOB). `e1` (orange) is the radio bill and is flat in `N`, because airtime
and MAC count grow together — every MAC gets the same 117 ns. `e2` (pink)
and `e3` (purple) are a fixed 14 pJ read-out fee per answer, so per MAC they
fall as `1/N`. The blue sum therefore bends at `N ≈ 6.5×10³` and flattens
onto the radio floor. The two triangles are measured: **1.47 fJ/MAC at
`N=4096` (48× below an H100 GPU)** and **0.59 fJ/MAC at `N=65536` (119×
below)**. The bottom panel shows the room-temperature floor, labelled at a round
2-bit ENOB, where the Landauer and thermodynamic bounds happen to be exactly
equal (11.5 zJ) — still ~5×10⁴ below where we are.

**Read `METHODS.md` for the derivation of every one of those numbers**,
including where the “6 ADC conversions” and “8 digital MACs” come from, why
everything is divided by `4N`, and how the original WISE “5-bit” criterion
maps onto “2.4-bit ENOB” under the current definition.

---

## Two conventions worth flagging

* **Precision.** `ENOB = log2(std[y]/RMSE)`. The criterion `RMSE < 0.0625`
  is “5-bit” under the WISE full-scale convention and **2.4-bit** under this
  one; the offset is `log2(2 ÷ std[y]) = log2 6 = 2.585 bit`, because
  `std[y] = 1/3` analytically for `U(0,1)`-amplitude random vectors
  (METHODS §5.3, §6). Don't mix the two across figures. The bottom-panel
  limits are *labelled* at a round 2 bit (the measured value rounded down,
  which is the conservative direction); at exactly 2 bit the Landauer and
  thermodynamic bounds coincide, so they share one line (METHODS §6.3).

* **No energy error bars.** The three repeats were taken at *fixed*
  transmit power, so `e_ip` has no direct repeat scatter. Repeat statistics
  are reported as RMSE and ENOB (± 1 s.d., `n = 3`) instead (METHODS §7).

---

## Provenance

Both runs were acquired on 2026-08-10 over a wired link with raw decode (no
twin/post correction), using the closed-loop power tuner: the criterion is
fixed first, the power is lowered until it is just barely met, then `n = 3`
repeats are captured at that fixed power. Energy accounting follows
**Gao et al., *Sci. Adv.* 12, eadz0817 (2026)**, Eqs. 1–3.
