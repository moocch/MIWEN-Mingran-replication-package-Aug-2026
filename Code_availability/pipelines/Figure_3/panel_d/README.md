# Fig. 3d — Measured N = 65,536 inner products before/after correction, 15 dB SNR

RMSE 0.064 → 0.036 (1.8×); ENOB annotations are mean ± 1 s.d. over the 3 independent captures.
Drawn by `fig3_v12_payoff.py` → `ip_panel(AX_D, 0, ...)`.

## Full chain

1. **Acquisition code** — `upstream_acquisition_and_raw_data/1_inner_product_scatter_v4.py`:
   GNU Radio + UHD dual-USRP X310 script (RF 1.20 GHz / LO 0.90 GHz / IF 0.30 GHz), 262,144-point
   FFT frames, 200 random complex vector pairs + 26 pilots, closed-loop data-comb power tuner
   against the guard-bin SNR estimate, weight comb at −3 dBm, 3 independent captures per
   operating point. Campaign run 2026-08-10 00:20:43.
2. **Raw measured data (+ run-record figures)** —
   `upstream_acquisition_and_raw_data/gr_fig3c_ip_scatter_20260810_002043_N65536/`:
   `gr_ip_scatter.npz` (= `gr_ip_scatter_N65536_20260810.npz` in this folder and in `../data/`,
   MD5-identical) plus the as-acquired scatter/ENOB PNGs.
3. **Out-of-sample correction (2026-08-26)** —
   `upstream_correction_20260826/1_opt_inner_product_with_DT.py`
   (from `V2\inner_product_4096_655936\output\files.zip`): rebuilds the transmitted waveforms
   from the stored seeds (ideal-mixing self-check < 1e-3), builds the four distortion features
   (twin compression template, parameter-free hard limiter, square law, residual complex gain),
   fits one complex coefficient per feature on training folds only, applies to held-out folds
   (5-fold CV, defaults `--folds 5 --cv-seed 0`); PIML twin frozen at the N = 4096 calibration
   (`PIML_TWIN_PARAMS` embedded; provenance in `../b/upstream_twin_calibration_N4096/`).
   `inner_product_optimized_N65536.png` is the correction run's own record figure.
4. **Corrected results** — `ip_optimized_N65536_20260826.npz` (this folder; MD5-identical to
   `files.zip::inner_product_optimized_N65536.npz`): `c_true`, `c_hat_before_reps`,
   `c_hat_after_reps`, per-capture RMSE/ENOB before/after. The same run's **N = 4096
   companion**, `ip_optimized_N4096_20260826.npz` (this folder and `../data/`; MD5-identical
   to `files.zip::inner_product_optimized_N4096.npz`, run-record png in
   `upstream_correction_20260826/`), backs the text's four-operating-point statements.
5. **Figure** — `fig3_v12_payoff.py` (this folder; canonical at fig3 root; since 2026-08-29
   it also gate-checks the caption's "mean ± 1 s.d., n = 3" semantics against the npz fields).
6. **Repeat decomposition** — `verify_s43_repeat_decomposition.py` (this folder) reproduces,
   with asserts, every value of the main text's noise-floor paragraph and SI S4.3 from the
   files above (σ̂ vs closed-form floor at 0.98×/1.01×; reproducible residue |d̂| and its
   power fractions, corrected and uncorrected, at both N; the ≤ 0.4-dB tuning-capture gate).
