# Fig. 3g — Client-side energy per real MAC

**What is drawn:** the Eq.-(energy) model curve e_ip = e1 + e2 + e3 (radio floor e1 pinned at the
N = 65,536 operating point, 1/N read-out fees), the H100 ≈ 70 fJ/MAC arithmetic-energy line, the
two measured triangles **1.47 fJ (N = 4096, 48×)** and **0.59 fJ (N = 65,536, 119×)**, the GTSRB
CNN inference overlay in green (below), and the Landauer = thermodynamic 11.5-zJ strip at the
criterion-equivalent 2-bit ENOB.

**CNN inference overlay (green):** the Fig.-4 comb network priced by the same Eq.-(energy)
accounting at its as-fielded device-plane operating point (P_LO, P_RF) = (−3, −65) dBm — open
circles for the four layers at their inner-product lengths (conv1 N = 75: 46.98 fJ, dense2
N = 128: 27.65 fJ, conv2 N = 800: 4.69 fJ, dense1 N = 1600: 2.50 fJ; each pays the same flat
0.31-fJ radio term plus the fixed 14-pJ read-out fee amortized over its own N), and a star for
the full CNN: 452 nJ/image = **15.7 fJ per real MAC at the answer-averaged N ≈ 228, 4.5×
below H100**. Source: the rung3-session energy audit
(`QPG-MIT/MIWEN_Mingran` @ `handoff/rung3-session`, `analysis/energy_budget_nn.py` →
`energy_budget_nn_results.json`); the figure script recomputes every
overlay value from the JSON's primitives (airtime 2.8366272 s/image, 28,847,616 real MACs,
31,659 answers) and **asserts** them against the exported point values.
In this replication package the audit code is archived in `../../energy_budget_nn/`, its
artifacts (JSON + report) in `Data_availability/raw/energy_budget_nn/`, and the working copy of
the JSON read by the figure script in `Data_availability/Figure_3/`.

**The figure script recomputes every number live from the two raw campaign files in `../data/`**
(`gr_ip_scatter_N4096_20260810.npz`, `gr_ip_scatter_N65536_20260810.npz`: transmitted power
`p_rf_dbm_tx` minus the meta `rf_atten_db` → P_x = −62.81 / −63.41 dBm; frame geometry → T_ip;
`rmse_reps` all < 0.0625 = the five-bit criterion) and **asserts** them against the package
values (`energy_point()`, `fig3_v12_payoff.py` lines ~166–216: e1 = 0.534 fJ, e_ip = 0.587 fJ,
119.2×; 0.613 / 1.467 fJ, 47.7×; bend at N = 6,553; 11.48 zJ).

## `fig5_energy_package/` — the standalone energy-accounting package (V2\energy)

| item | role |
|---|---|
| `METHODS.md` | full derivation: benchmark convention (Gao et al.), e1/e2/e3 definitions, 6 ADC conversions + 8 digital MACs read-out count, 4N real-MAC convention, airtime amortization, Landauer/thermodynamic limits, WISE 5-bit ↔ 2.4-bit distortion-inclusive ENOB mapping |
| `code/fig5_energy_scaling.py` | raw npz → energy numbers, with the CHECKS assertion table replicated by the manuscript figure script |
| `code/replot_from_npz.py` | replot from the exported curves |
| `data/fig5_plot_data.npz`, `data/fig5_plot_data_curves.csv` | exported curves/points (redundant record) |
| `data/raw/` | the two acquisition scripts (`1_inner_product_scatter_v4.py`, `..._N4096.py`) and both raw run folders (npz + as-acquired PNGs) — MD5-identical to the `V2\inner_product_4096_655936\upload\` originals and to `../data/` |
| `figures/` | the package's own rendering of the energy figure (reference output) |
