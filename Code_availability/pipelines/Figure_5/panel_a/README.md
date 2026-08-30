# Fig. 5a — The unrolled cascade

**What is drawn:** DAC → [ring mixer (LO = broadcast weight comb w^(ℓ) at 0.30/0.35/0.40 GHz) →
LPF → passive activation block] ×2 → mixer → LPF → ADC → class; band "fully analog — no ADC /
DAC / amplifier between passes"; pass drives −10 → −34 → −58 dBm; two real input thumbnails
(MNIST digit '3', test index 8975, and the Fig.-4a GTSRB STOP photo).

- **Topology evidence**: `comb_analog_sim.py` (this folder) — `plan_Z()` (level planner) and
  `forward()` define exactly the drawn chain; LO band plan 0.30/0.35/0.40 GHz.
- **Pass drives**: `link_budget.json` (this folder) — the figure script gate-checks the displayed
  −10/−34/−58 dBm against the MNIST rows (−10.0/−33.94/−58.02) with zero tolerance.
- **Thumbnails**: `input_glyphs.npz`, built by `prep_fig5_assets.py` (this folder) from
  `../simulation/data/mnist.npz` (deterministic pick) and `comb_assets.npz` (this folder — the
  same STOP photo as Fig. 4a; its own provenance is in `../../fig4/a/`).
