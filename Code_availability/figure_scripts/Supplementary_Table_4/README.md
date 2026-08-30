# Supplementary Table 4 — Measured accuracy–energy comparison of the two adaptation routes

Reproducibility package for Supplementary Table 4 (also reproduced in the
response letter, Reviewer 3 Comment 2) and its S3.5 companion numbers.
Assembled 2026-08-29. Every table cell is verified against the archived
pipeline outputs in this folder by `verify_table4.py` (31/31 checks pass;
see `verify_table4_output.txt`).

## Contents

| file | what it is |
|---|---|
| `verify_table4.py` | cell-by-cell verification script (numpy only) |
| `verify_table4_output.txt` | its saved output, run 2026-08-29, 31/31 PASS |
| `fig5_energy_package/` | complete measured client-energy accounting package of the inner-product primitive (code + METHODS + `data/fig5_plot_data.npz` + raw scatter runs + figure) |
| `ip_optimized_N65536_20260826.npz` | out-of-sample twin-correction results at N=65,536 (rows 15 dB / 25 dB: `rmse_before/after_mean`, `enob_before/after_mean`, the 4 fitted complex coefficients, per-repeat data) |
| `provenance_comb_accuracies.md` | provenance of the comb-route classification numbers 98.92 / 99.50 (copy of the GTSRB inference curation README) |
| `README_hardware_aware_training_v2.md` | provenance of the serial-route classification numbers 98.50 / 5.67 (copy of the serial hardware-aware-training campaign README) |
| `verify_accuracy.py` | campaign script that recomputes 98.50 / 5.67 from the raw prediction files (raw npz stay in the campaign folder, see below) |
| `route_comparison_numbers_original_recovered.py.txt` | recovered original computation script that first derived the table's energy numbers (kept verbatim; its hard-coded paths point at the older V2_Manu layout) |

## Table cell → artifact map

### Top block — inner-product primitive, N=65,536, 25 dB

| table cell | value | artifact | field / derivation |
|---|---|---|---|
| raw decode RMSE | 0.056 | `fig5_energy_package/data/fig5_plot_data.npz` | `points_rmse_mean[N=65536]` = 0.056095 (identical to `rmse_before_mean[25 dB]` of the ip_optimized npz) |
| raw ENOB | 2.58 | same | `points_enob_mean[N=65536]` = 2.57524 (= `enob_before_mean[25 dB]`) |
| raw energy per real MAC | 0.588 fJ | same | `points_e_ip[N=65536]` = 0.587794 fJ |
| raw advantage vs 70-fJ GPU | 119× | same | `points_speedup_vs_h100[N=65536]` = 119.089 (= `h100_line`/`points_e_ip`) |
| + twin correction RMSE | 0.017 | `ip_optimized_N65536_20260826.npz` | `rmse_after_mean[25 dB]` = 0.017169 |
| + twin correction ENOB | 4.28 | same | `enob_after_mean[25 dB]` = 4.28328 |
| + correction energy | 0.649 fJ | derived | 0.587794 fJ + 16 pJ/(4·65536) = 0.648830 fJ (see convention below) |
| + correction advantage | 108× | derived | 70 fJ / 0.648830 fJ = 107.89 |

S3.5 companion numbers (same two artifacts):
N=4096 energy 1.467 → 2.444 fJ and advantage 47.7× → 28.6×
(`points_e_ip[N=4096]` = 1.467273 fJ, fee 16 pJ/(4·4096) = 0.9766 fJ);
15-dB RMSE 0.064 → 0.036 at N=65,536
(`rmse_before_mean[15 dB]` = 0.064349, `rmse_after_mean[15 dB]` = 0.035962).

### Bottom block — 43-class GTSRB classification

| table cell | value | artifact |
|---|---|---|
| noise-injection training, measured | 98.92 % | `provenance_comb_accuracies.md` row #1: 1187/1200, energy-honest rerun 2026-08-07, frozen offline calibration, `battery_frozen_slim.npz` |
| digital baseline (clean-trained) | 99.50 % | same doc, row #3: clean-trained digital model, same architecture, same frozen 1,200-image frame (`r35_r3plus_s0_hw.npz`) |
| twin-based training, measured | 98.50 % | `README_hardware_aware_training_v2.md` main table: 591/600, (0,0) dBm, N=600 = battery[0:600]; recomputable by `verify_accuracy.py` |
| clean weights on hardware | 5.67 % | same: 34/600 (network collapses — 590/600 predict class 12); the pre-registered clean-under-twin digital pin 0.83 % predicted the collapse |
| added inference energy (both routes) | 0 | by construction, doc-backed: noise injection and the training-loop twin change training only; the comb 98.92 run is the energy-honest rerun ("host computes NO W@x") and the serial README states the inference path contains no twin component |

Context caveat carried over from the provenance doc: 98.92 (measured) vs
99.50 (clean digital) is the system-level fair comparison — quote the pair
together, as the table does (the earlier "hardware beats digital" system
claim was retracted on 2026-08-06; see `provenance_comb_accuracies.md`).

## The 16-pJ deployment-cost convention

The twin correction is four complex coefficients (basis: twin, limiter,
square, gain — `basis_names`/`coeffs` in `ip_optimized_N65536_20260826.npz`)
applied per answer. Convention used by the table:

- The feature **templates depend only on the transmitted waveforms and are
  precomputed off-client**; the four **coefficients are fitted offline**.
  Neither enters the client's per-answer energy.
- **Deploying** the correction on the client costs 4 complex MACs = **16 real
  MACs = 16 pJ per answer** at the benchmark constant e_dig = 1 pJ per real
  MAC (from `meta_json.e_dig_J` of `fig5_plot_data.npz`).
- An N-point complex inner product is 4N real MACs, so the amortized fee is
  **+16 pJ/(4N) per real MAC**: +0.061 fJ at N=65,536 (0.588 → 0.649 fJ,
  119× → 108×) and +0.977 fJ at N=4096 (1.467 → 2.444 fJ, 47.7× → 28.6×).

This is the only arithmetic in the table not read directly from an archived
artifact; `verify_table4.py` recomputes it from the npz constants.

## Provenance (sources read-only; copies verified byte-identical by MD5 on 2026-08-29)

| copy here | MD5 | source |
|---|---|---|
| `fig5_energy_package/` (whole folder, verbatim) | all 17 files identical; key file `data/fig5_plot_data.npz` = `19e2739083f1ee28be3c008d9e231b00` | `fig3\g\fig5_energy_package` |
| `ip_optimized_N65536_20260826.npz` | `750abe21b9976cbf6341d84527109487` | `fig3\data\ip_optimized_N65536_20260826.npz` |
| `provenance_comb_accuracies.md` | `bd89ad4951cc2d6e91bec9743fffedda` | `V2\GTSRB_inference\README_provenance.md` |
| `verify_accuracy.py` | `5f9e957942d04c9128e258a4d77372cb` | `V2\hardware_aware_training_v2\verify_accuracy.py` |
| `README_hardware_aware_training_v2.md` | `026f56fe3d791c0f3835631dc88301b0` | `V2\hardware_aware_training_v2\README.md` |
| `route_comparison_numbers_original_recovered.py.txt` | `a7a98948787d5f6681c244bb8881a3e7` | recovered original computation script (pre-existing in this folder) |


Per-file MD5s of the full `fig5_energy_package/` copy: METHODS.md
`0090255c8b766a31af359bada3150d2b`, README.md `f33b26931cf5536fe93dd0aadeb01050`,
code/fig5_energy_scaling.py `a605d791f7d13c03f2d39365c97072d1`,
code/replot_from_npz.py `6cf7ee5e28e55a0b6d175f05bc1266f9`,
data/fig5_plot_data.npz `19e2739083f1ee28be3c008d9e231b00`,
data/fig5_plot_data_curves.csv `e85828ee1b06f746325fcc6036750b7e`,
data/raw/1_inner_product_scatter_v4.py `d7ff7b1f1b61fbaae64ee7c57729954d`,
data/raw/1_inner_product_scatter_v4_N4096.py `6864a4811d3d1659618c1d99d4cec8fd`,
data/raw/...N65536/gr_fig3c_ip_scatter.npz `3ff3c54f4e81234c2b8062b62f950821`,
data/raw/...N4096/gr_fig3c_ip_scatter.npz `f9b38769c6ed4390f859afe86c1f2906`
(pngs/pdf/svg omitted here; all 17 files matched their sources at copy time).

### Deliberately NOT copied (large raw data — referenced instead)

- The ~55-MB comb battery npz (`battery_frozen_slim.npz` and the surrounding
  measured battery archives) behind 98.92 / 99.50 is **not** duplicated here.
  The **full comb chain is archived at**
  `fig4\`
  (and the curated code+data package at
  `V2\GTSRB_inference\share_20260712\`).
- The serial-route raw prediction npz (folders `05_hardware_result_0dBm_N600`,
  `06_digital_comparators`, `08_frozen_inputs_and_labels`) behind 98.50 / 5.67
  stay in the campaign folder
  `V2\hardware_aware_training_v2\`.
  Running `python verify_accuracy.py` there (done 2026-08-29) prints
  clean 34/600 = 5.67 % and twin 591/600 = 98.50 %, plus the frame-identity
  and single-weights/single-power assertions.

## How to re-run

1. `python verify_table4.py` — verifies every table cell from the local
   copies; needs only numpy. Expected: `31/31 checks passed`, exit code 0
   (compare with `verify_table4_output.txt`).
2. Energy accounting from scratch:
   `python fig5_energy_package/code/fig5_energy_scaling.py` regenerates
   `fig5_plot_data.npz` from the raw scatter runs in
   `fig5_energy_package/data/raw/`; see `fig5_energy_package/METHODS.md`
   for the Eqs. 1–3 methodology and constants.
3. Classification numbers from raw predictions: run `verify_accuracy.py`
   inside the campaign folder named above (its inputs do not resolve from
   this archive folder); for the comb chain use the fig4 archive named above.
