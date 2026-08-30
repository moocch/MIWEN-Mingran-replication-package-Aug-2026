# fig5_analog_v2 — Figure 5 (new): the fully analog client, simulated from published device numbers

**Figure file for `main_PANS.tex`:** `fig5_analog_v2_preview.pdf` (180 × 92 mm)

**Typography:** fig4_v6 house sizes (7.5 / 6.3 pt hierarchy). A
uniform-10-pt (= body size) variant was tried on 2026-08-28 at the
author's request and reverted the same day ("太丑了"); the compact
single-row layout (a full-width; b, c, d in one row) came back with it.
**Panel d shows only the two L = 3 pairs** — the L = 4 link-failure
pair was removed from the figure (numbers remain in
`data/results_summary.json` for the text: digital 85.02 vs chain
15.39 ± 0.23). Panel a keeps the 2026-08-28 geometry fixes: the two
input thumbnails stacked on one x-centre, the gold band hugging the
chain symmetrically, and the whole composition centred.
**Build:** `python prep_fig5_assets.py <path-to-archive-mnist.npz>` (once) →
`python fig5_analog_v2.py`
**Layout:** WISE-hybrid stack (schematic band on top, data panels below);
fig4_v6 visual language. Palette: gold band = Physical/analog domain,
warm gold = the fully analog chain, gray = digital execution, salmon =
ideal-assumption collapse / out-of-budget, blue = twin/hardware-aware,
violet = data stream, teal = outputs. Built from scratch 2026-08-27; does
NOT reuse anything from fig5/ or fig5_analog/.

Source of truth: `V2/fully analog/files (1).zip` →
`miwen_fully_analog_archive.zip` → `code/` (comb_analog_sim.py,
analog_physics.py, reproduce.py) and `code/results/*`; the manuscript
section draft `analog_section.tex` in the same archive; the slide deck
`V2/fully analog/analog.pptx` (2026-08-24). This is a **pure simulation
study** — nothing in this figure is a hardware measurement; the figure
says so on panel a and carries the provenance table as panel f.

## Panels

| panel | content | provenance |
|---|---|---|
| a | Unrolled cascade: DAC → [ring mixer (LO = broadcast weight comb $w^{(\ell)}$ at 0.30/0.35/0.40 GHz) → LPF → passive activation block] ×2 → mixer → LPF → ADC → class. Gold band: "fully analog — no ADC / DAC / amplifier between passes". Pass drives −10 → −34 → −58 dBm | topology: `plan_Z()` + `forward()` in `comb_analog_sim.py`; drives gate-checked from `data/link_budget.json` |
| b | The activation block: 1:n impedance-transform matching network (+18.5 dB measured boost), centre-tapped secondary driving a zero-bias anti-phase Schottky pair (odd terms cancel → squarer), baseband LPF returning the difference-frequency comb $r_k=\sum_m y_{m+k}y_m^*$; −1 dB step-down into the next IF port | topology: sub-nW self-mixer wake-up front ends (Mangal & Kinget); constants: `BOOST_DB`, `K_ED`, `IL_STEP` in `comb_analog_sim.py` |
| c | Per-tone SNR vs pass, MNIST + GTSRB level plans; SNR = 0 line; pass 4 dashed/open = out of budget → the client supports L = 3. Tick drive labels quote the MNIST plan (GTSRB within 0.7 dB) | `data/link_budget.json` verbatim (gates below) |
| d | Same checkpoint executed twice (digital = mode ZT, chain = mode ZH): MNIST L3 94.30/93.90, GTSRB L3 88.83/87.53, MNIST L4 85.02/**15.39** (link failure, salmon) | `data/results_summary.json`, tags MZT_L3d/MZH_L3/ZT_L3d/ZH_L3/MZT_L4d/MZH_L4 |
**Dropped panels (author decisions, 2026-08-27).** (i) The provenance table
(former panel f): this is a paper — citations belong in the caption and the
reference list, not on the artwork. (ii) The hardware-aware 2×2 (former
panel e, `data/twin_2x2.json`): dropped from the figure; candidate for
text/SI. (iii) Part-number labels ZEM-4300+/LFCN-490+ removed from panel a.
All provenance lives in `caption_draft.tex`: datasheets ZEM-4300+/LFCN-490+
(caption a), measured boost + detector + self-mixer topology (caption b),
link-noise law / mixer twin = this work Figs. 2–3 and the discrete-Schottky
16-dB boundary (caption tail).

**Detector supply question (verified against the SSCL 2018 paper,
2026-08-27).** Wang et al., IEEE SSC-L 1(5) 134–137: the k_ED = 208.7/V
device is the *passive* pseudo-balun ED — a 2N-stage (N = 5) rectifier of
higher-Vt diode-connected nMOS, "no dc currents", hence no 1/f noise; the
receiver's 0.4-V / 6.1-nW budget powers the baseband amplifier (3.44 nW),
comparator (0.66), correlator (0.80), oscillator (0.52), SPI (0.28) — none
of which the block uses — **plus an "ED bias" of 0.36 nW** (a 4-bit
diode-connected reference ladder that forward-biases the bulk < 200 mV to
trim Vt). So the rectifying core is passive, but the published k_ED relies
on a sub-nW bulk-bias trim: the block is sub-nW, not strictly 0 W. On-figure
wording therefore avoids "no supplied component" / "zero bias"; the caption
states the precise claim (core draws no dc current; 0.36 nW bulk-threshold
biasing in the published implementation).

## Gates (enforced in `fig5_analog_v2.py`, zero tolerance)

Contract: every number in the data panels (c/d/e) and every settings constant
shown in b/f is loaded from `data/*.json` and asserted; the residual
provenance text carries archive code constants pinned by explicit assertions.

- settings: cl_pass 7.47 dB, boost 18.5 dB, k_ED 208.7 /V, R_v 3500 Ω,
  step-down 1.0 dB, first pass −10 dBm; displayed strings for boost / k_ED /
  step-down are FORMATTED from the loaded settings, not re-typed;
- CL split tie gate: 6.65 (ZEM-4300+) + 0.82 (LFCN-490+) == cl_pass_dB;
  panel-f strings formatted from those named constants;
- P_n = −73.6 dBm: archive `comb_analog_sim.py` constant, no JSON anchor —
  pinned as a named constant next to the gates (main text Fig. 2 quotes it);
- LO plan 0.30/0.35/0.40 GHz: archive frequency plan (only 0.40 GHz has a
  code anchor, F0_HZ); nothing in the results JSONs to gate it against;
- MNIST plan drives −10 / −33.94 / −58.02 / −84.46 dBm, SNRs 48.97 / 33.97 /
  11.83 / −14.61 dB; GTSRB −10 / −34.27 / −58.48 / −85.16 dBm, 43.04 / 32.57 /
  8.36 / −18.31 dB; block efficiency 2.255 % (gated though not displayed);
- accuracies: MZT_L3d 94.30, MZH_L3 93.90±0.10, ZT_L3d 88.83, ZH_L3 87.53±0.14,
  MZT_L4d 85.02, MZH_L4 15.39±0.23;
- twin_2x2: all ten cells verbatim (see table above).

## Verification (2026-08-27)

Three adversarial passes (numbers / design-grammar / reproducibility) ran
against the archive ground truth. Outcome: every displayed number matches the
archive exactly. Fixes applied from the findings: mixer port labels corrected
to the archive convention (weights → LO, activation comb → DC-coupled IF,
product band out the RF port through the LPF — panel b's "to next IF port"
now consistent); panel-f self-mixer row re-attributed (topology precedent =
Mangal & Kinget ISSCC 2019; the zero-bias detector numbers belong to Wang
SSC-L 2018), video step-down row added ("engineering value"); panel-c GTSRB
drive disclaimer put on the panel; panel-c series recolored inside the gold
family (no off-palette brown); forecast line recolored twin-blue and reworded
("the twin's software forecast", no pinned-in-time claim — both cells come
from one `twin_2x2.py` run); d/e cross-panel precision unified (93.9/87.5);
legend box replaced by in-bar direct labels; the n:1 step-down network is now
drawn, so the −1 dB names a drawn component.

## Notes / decisions

- **LO drive is not printed on the figure.** The archive Methods state the
  mixers run at their **rated LO level** (datasheet CL, no soft-switching
  penalty); the old draft TikZ figure said "−3 dBm", which contradicts that.
  The figure shows only the LO frequencies; the Methods text should say
  "rated LO drive".
- Networks: MNIST 784→100→64→10 (L3), 784→100→64→64→10 (L4); GTSRB
  3072→128→128→43. Full test sets (10,000 / 12,630), analog = mean ± s.d.
  over 5 independent noise realizations, digital execution deterministic.
- Panel d bars are the ZH-trained weights (twin + link in the loss). Panel e's
  twin arm is twin-only in the loss (serial-campaign protocol); its footnote
  ties the two: adding the link noise to the loss recovers d.
- `data/input_glyphs.npz` built by `prep_fig5_assets.py` from the archive's
  own `code/data/mnist.npz` (test index 8975, label 3) and the Fig.-4a STOP
  photograph (`../fig4_v6/data/comb_assets.npz`, battery k=34).
- **Inserted into `main_PANS.tex` on 2026-08-28** (declared after the
  Introduction so its caption cites number after the intro's; [t]
  figure* lands on TOP OF PAGE 6). Six new bib keys added to
  refs_PANS.bib (lfcn490, mercier2017wurx, wang2018pseudobalun,
  mangal2017selfmix — venue CICC 2017 pp. 1–4 verified from the SSCL
  2018 reference list —, mangal2019wurx, sms7630); `ZEM-4300+` reused
  from refs.bib. Block renamed "analog activation block" in panels a/b
  and the caption; "baseband LPF" → "LPF" (2026-08-28).
  `caption_draft.tex` is now historical — the live caption is in
  main_PANS.tex.

- 2026-08-28 (later): panel-b label renamed to ''transformer-based passive voltage booster, +18.5 dB'' (author instruction; caption aligned). The block physics (booster, n:1 step-down, available-power accounting) is documented as Supplementary Note 6 in Supp_M/Supplementary_Information.tex (Eqs. S15-S17).

- 2026-08-28 palette (final): band = fig4d light-blue tint #E9F0F9 with blue headers; activation blocks + panel-b frame = amber/cream (#B87A0F on #FDF1E3, restored); c/d data = fig4b pairing (salmon #E58C7D = digital / MNIST, blue #2458A6 = fully analog chain / GTSRB). Gold and green band variants were tried and vetoed by the author the same day.
