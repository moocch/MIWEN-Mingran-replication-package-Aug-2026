# Fig. 3f — Inner-product RMSE vs SNR and the measured receiver noise floor

**What is drawn:** before/after RMSE (mean, capped ± 1 s.d. bars over the 3 captures) at the two
measured operating points, against the measured guard-bin noise floor in its closed form
1/√(27·SNR). N = 65,536 only (the N = 4096 companions are quoted in the manuscript text only).

- x-positions: per-panel mean of `snr3_db_reps` (guard-bin SNR measured in situ per capture)
  from `gr_ip_scatter_N65536_20260810.npz` (this folder).
- y-values: `rmse_before/after` mean and error from `ip_optimized_N65536_20260826.npz`
  (this folder).
- floor curve: computed live by the figure script from the measured guard-bin convention —
  no free parameter (drawn in `fig3_v12_payoff.py`, panel-f block, lines ~440–469).

**Full upstream chain: see `../d/README.md`** (same acquisition and correction runs).
