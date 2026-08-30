# Fig. 5c — Link budget of the chain (per-tone SNR vs pass)

**What is drawn:** per-tone SNR at each pass for the MNIST and GTSRB level plans; the SNR = 0
line; pass 4 open/dashed = out of budget (would arrive below the measured −73.6-dBm floor) →
the fully analog client supports L = 3.

- **Generator**: `reproduce.py budget` (this folder) — deterministic (`rng(0)` inside
  `plan_Z()`); needs no datasets or checkpoints; writes `results/link_budget.json`.
- **Models it evaluates**: `comb_analog_sim.py` + `analog_physics.py` (copies in `../b/`,
  canonical in `../simulation/`); P_N_DBM = −73.6 pinned from this work's twin calibration.
- **Plot data**: `link_budget.json` (this folder) — loaded verbatim by the figure script; all
  8 drives and 8 SNRs are gate-asserted (MNIST 48.97/33.97/11.83 dB → text "49/34/12";
  GTSRB 43.04/32.57/8.36 dB; pass-4 −14.61/−18.31 dB below floor).
- Byte-identity of this JSON with the simulation archive's `results/link_budget.json` was
  verified during assembly.
