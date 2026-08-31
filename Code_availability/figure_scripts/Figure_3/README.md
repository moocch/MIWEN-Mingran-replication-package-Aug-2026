# Fig. 3 — Out-of-sample correction of measured N = 65,536 inner products, and the client-side energy of the primitive

**Figure file used by `main_PANS.tex`:** `fig3_v12_preview.pdf` (included as `Data_availability/Figure_3/fig3_v12_preview.pdf`)
**One script draws all seven panels:** `fig3_v12_payoff.py` (reads `data/` relative to itself — run `python fig3_v12_payoff.py` in this folder)

**2026-08-28 script revision (author, edited in place):** in-figure annotations (panel b's
peaks-compressed note, the d/e RMSE and ENOB numbers) moved into the main-text caption, and
panel g's legend is spelled out in the caption. Panels, inputs, and every zero-tolerance
assert are unchanged. The caption now quotes the per-capture values
RMSE $0.0643 \pm 0.0011 \to 0.0360 \pm 0.0011$ (15 dB) and
$0.0561 \pm 0.0006 \to 0.0172 \pm 0.0002$ (25 dB), ENOB $2.38 \pm 0.02 \to 3.22 \pm 0.05$
and $2.58 \pm 0.01 \to 4.28 \pm 0.02$ — these are exactly the values the script recomputes
live from `data/ip_optimized_N65536_20260826.npz` (printed at every run; confirmed in the
2026-08-28 and 2026-08-29 verification runs).

**Verified 2026-08-29 (and identically 2026-08-28):** re-executed from the archived data in
this folder; the regenerated `fig3_v12_preview.png` is **pixel-identical to the published one
(0 / 10,520,103 differing pixels)**, and every zero-tolerance assert inside the script passed.

**2026-08-31 script revision (green CNN inference overlay):** panel g now also prices the
comb-encoded GTSRB CNN of Fig. 4 by the same Eq.-(energy) accounting at its as-fielded
operating point (P_LO, P_RF) = (−3, −65) dBm at the mixer: open green circles, the four layers
at their inner-product lengths (N = 75 / 128 / 800 / 1600 → 46.98 / 27.65 / 4.69 / 2.50 fJ per
real MAC); green star, the full-CNN aggregate at the answer-averaged N ≈ 228 (452 nJ/image =
**15.7 fJ per real MAC, 4.5× below H100**). The overlay is recomputed live from
`energy_budget_nn_results.json` (rung3-session energy audit; provenance and audit code in
`../../pipelines/energy_budget_nn/`, artifacts in `Data_availability/raw/energy_budget_nn/`)
and **asserted** against the exported point values. **Verified 2026-08-31:** re-executed in
this package from the archived data — the regenerated `fig3_v12_preview.png` is
**pixel-identical to the published one (0 / 10,520,103 differing pixels)**, and every assert
(including the five new CNN-overlay asserts) passed.

## Data files actually loaded by the script (all four MD5-verified against their campaign originals)

| file (`data/`) | role | origin |
|---|---|---|
| `gr_ip_scatter_N65536_20260810.npz` | raw measured N = 65,536 campaign (2026-08-10 00:20:43): 200 products x 2 SNRs x 3 repeats, seeds + frame metadata + guard-bin SNRs | `V2\inner_product_4096_655936\upload\gr_fig3c_ip_scatter_20260810_002043_N65536\gr_fig3c_ip_scatter.npz` (MD5 identical) |
| `gr_ip_scatter_N4096_20260810.npz` | raw measured N = 4096 campaign (2026-08-10 01:19:15) — **panel g only** (energy point 1.47 fJ) | `V2\inner_product_4096_655936\upload\...N4096\gr_fig3c_ip_scatter.npz` (MD5 identical) |
| `ip_optimized_N65536_20260826.npz` | 2026-08-26 out-of-sample correction results (5-fold CV; before/after products, RMSE/ENOB per capture) | `V2\inner_product_4096_655936\output\files.zip::inner_product_optimized_N65536.npz` (MD5 identical) |
| `twin_predictions_N1.npz` | scalar 43x43 three-tier surfaces (measured / physics-only / full twin) | scalar PIML twin training run (chain archived in `c/upstream_measured_data_and_twin_code`) |
| `energy_budget_nn_results.json` | GTSRB-CNN client-energy audit — **panel g green overlay** (per-layer points + full-CNN star; airtime, MAC and answer counts frozen as primitives) | `QPG-MIT/MIWEN_Mingran` @ `handoff/rung3-session`, `analysis/energy_budget_nn.py` (byte-identical copy; audit code archived in `../../pipelines/energy_budget_nn/`) |

`data/` additionally holds `twin_predictions_N{2,4,8,4096}.npz` — **not** fig3 inputs; they are the
shim read by `../Code_availability/figure_scripts/Figure_2/fig2_v3_twin.py` (see `data/README_repro_shim.md`) — and, since the
2026-08-29 revision, `ip_optimized_N4096_20260826.npz` (MD5 3c8536fc… =
`files.zip::inner_product_optimized_N4096.npz`): the corrected **N = 4096 companion** results
behind the text's four-operating-point statements (ρ = 0.79–0.91, the noise-floor tracking "at
both vector lengths", and the S4.3 repeat-decomposition values in parentheses). The full
repeat decomposition quoted in the main text and SI S4.3 is reproduced, with zero-tolerance
asserts, by `d/verify_s43_repeat_decomposition.py` (run it in `d/`; verified 2026-08-29 —
every value exact, incl. the 0.98×/1.01× floor ratios and the 0.4-dB tuning-capture gate).

## Panels

| panel | content | kind | chain |
|---|---|---|---|
| a | correction scheme | schematic | drawing code only (script lines ~287–332) — see `a/` |
| b | twin-predicted compression of the weight waveform w(t) | simulation from measured calibration | LO symbol rebuilt from campaign seeds; compression law = N=4096 PIML twin fit constants (PCOMP_LO = −5.199..., BETA_C = 0.610...) — full calibration chain in `b/upstream_twin_calibration_N4096/` |
| c | residual distributions of the three model tiers | measured + twin | 43×43 scalar sweep → scalar PIML twin → `twin_predictions_N1.npz`; full chain in `c/` |
| d | N = 65,536 inner products before/after correction, 15 dB (0.064 → 0.036, 1.8×) | measured | acquisition → raw npz → 2026-08-26 correction pipeline → `ip_optimized_N65536_20260826.npz`; full chain in `d/` |
| e | same at 25 dB (0.056 → 0.017, 3.3×) | measured | identical chain — see `d/` (panel index 1) |
| f | RMSE vs SNR + measured guard-bin noise floor 1/√(27·SNR) | measured | same two files as d/e; floor computed live from guard-bin fields |
| g | client-side energy per real MAC (1.47 fJ @ N=4096, 0.59 fJ @ N=65,536; green CNN overlay: four layer circles + full-CNN star, 15.7 fJ/MAC; H100 line; Landauer strip) | measured + accounting model | raw 25-dB operating points of both campaign npz → Eq. (energy) accounting of `g/fig5_energy_package/` (Gao et al. benchmark methodology); CNN overlay from `energy_budget_nn_results.json` (rung3-session audit); the figure script recomputes and **asserts** every number from the raw files and the JSON primitives |

## How to reproduce

```
cd fig3
python fig3_v12_payoff.py       # writes fig3_v12_preview.pdf/.png/.svg
```

To re-run the 2026-08-26 out-of-sample correction itself (regenerates `ip_optimized_N65536_20260826.npz`):
`d/upstream_correction_20260826/1_opt_inner_product_with_DT.py` on the raw campaign npz
(defaults: `--folds 5 --cv-seed 0`; the N = 4096 PIML twin parameters are embedded as
`PIML_TWIN_PARAMS`, provenance in `b/upstream_twin_calibration_N4096/`).
