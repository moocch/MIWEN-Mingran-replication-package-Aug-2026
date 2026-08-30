# Data_availability/Figure_3 — byte-identical working copies

The four files loaded by `../fig3_v12_payoff.py` (all MD5-verified against their campaign
originals; provenance in the panel READMEs):

- `gr_ip_scatter_N65536_20260810.npz` — raw 2026-08-10 campaign, N = 65,536 (panels b, d, e, f, g)
- `gr_ip_scatter_N4096_20260810.npz` — raw 2026-08-10 campaign, N = 4096 (panel g only)
- `ip_optimized_N65536_20260826.npz` — 2026-08-26 out-of-sample correction results (panels d, e, f)
- `twin_predictions_N1.npz` — scalar three-tier surfaces (panel c)

**Shim for Fig. 2:** `twin_predictions_N2.npz`, `twin_predictions_N4.npz`,
`twin_predictions_N8.npz`, `twin_predictions_N4096.npz` are *not* Fig. 3 inputs — together with
`twin_predictions_N1.npz` they are read by `../../Code_availability/figure_scripts/Figure_2/fig2_v3_twin.py` via its hard-coded
relative path `..\fig3\data\`, exactly as in the manuscript working tree. Their provenance
(vector sweeps N = 2–8 on the signal-analyzer readout, N = 4096 on the all-USRP chain, and the
vector-twin training records) is documented panel-by-panel in `../../fig2/`.
