# Fig. 3e — Measured N = 65,536 inner products before/after correction, 25 dB SNR

RMSE 0.056 → 0.017 (3.3×). Drawn by `fig3_v12_payoff.py` → `ip_panel(AX_E, 1, ...)` —
the 25-dB column (panel index 1) of exactly the same files as panel d.

Direct data in this folder: `ip_optimized_N65536_20260826.npz` (corrected results, both SNR
panels) and `1_opt_inner_product_with_DT.py` (the correction pipeline that produced it).

**Full upstream chain (acquisition code, raw campaign data, run records): see `../d/README.md`** —
panels d and e share one acquisition run and one correction run; the chain is archived once, in `d/`.
