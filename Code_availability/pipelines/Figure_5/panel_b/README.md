# Fig. 5b — The analog activation block

**What is drawn:** 1:n impedance-transform passive voltage booster (+18.5 dB measured boost),
centre-tapped secondary driving a zero-bias anti-phase Schottky pair (odd terms cancel →
squarer, k_ED = 208.7 /V), baseband LPF returning r_k = Σ_m y_{m+k} y_m*, and an n:1 step-down
(−1 dB) into the next mixer's IF port.

- **Constants**: `BOOST_DB = 18.5`, `K_ED = 208.7`, `IL_STEP = 1.0`, `R_V = 3500`,
  `CL_PASS = 7.47` (= CL_MIXER 6.65 + IL_FILT 0.82, tie asserted) in `comb_analog_sim.py`
  (this folder), each anchored in comments to its source: Wang et al., IEEE SSC-L 2018
  (pseudo-balun detector k_ED), the Mercier-group wake-up receivers (+18.5 dB boost at 400 MHz),
  ZEM-4300+ / LFCN-490+ datasheets (mixer/filter losses), and this work's own P_n = −73.6 dBm.
- **Device physics**: `analog_physics.py` (this folder) — exact Shockley solve of the zero-bias
  anti-phase pair (the squarer), Johnson–Nyquist noise.
- **Displayed strings** (+18.5 dB, −1 dB, k_ED) are FORMATTED from the loaded
  `link_budget.json` settings block (this folder), not re-typed, and gate-checked.
- The block's power efficiency (2.255 % → "2.2 %" in the text) is gate-checked from the same
  JSON but not displayed; the full available-power accounting is Supplementary Note 6.
