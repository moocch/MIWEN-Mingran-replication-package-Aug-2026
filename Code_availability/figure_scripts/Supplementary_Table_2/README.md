# Supplementary Table 2 — PIML-twin parameters recalibrated at N = 4096

**What is backed:** Supplementary Table 2 of `Supplementary_Information.tex`
(label `tab:twin4096`, "Final calibrated physics parameters of the PIML digital
twin used, frozen, for the N = 65,536 experiments") and the identical red table
in the response letter (`Main.tex`), plus the quoted text
metrics of Supplementary Note 1, S1.4: 0.79 dB full-twin RMSE on the measured
N = 4096 surface, 96% within ±2 dB, versus 10.35 dB for the ideal multiplier
and 1.80 dB for the physics alone. The same record supplies the two frozen
constants reused at N = 65,536 in S1.5 (P_LO,comp = −5.20 dBm, β_C = 0.610).

## The chain (all in `upstream_measured_data_and_twin_code/`)

1. **Acquisition** — `gr_heatmap_vector_inner_product_v2_local.py` (GNU
   Radio / USRP): sweeps the two-port drive grid with N = 4096 comb-encoded
   vectors through the mixer and records the resolved IF response.
2. **Raw measured heatmap** — `gr_usrp_mixer_vector_heatmap_N4096.npz`
   (the measured N = 4096 surface; the `aliasing` block of `summary.json`
   records its tone bookkeeping: fs = 10 MHz, Δf = 1 MHz, 10 folded tones).
3. **PIML recalibration** — `piml_4096_digital_twin.py`, warm-started from
   `twin_training_run_4096DT_model/fit_dualpump_init.npy` (dual-pump physics
   initialization): symmetric two-port physics (Eq. S4; 15 parameters) fitted
   in stages (1,200 physics iters, 6,000 residual-NN iters, 4,000 joint
   iters; residual 6→48→48→1, |δ| ≤ 5 dB — `config.hidden = 48`,
   `config.max_db = 5.0`).
4. **Training-run record** — `twin_training_run_4096DT_model/summary.json`
   (calibrated `physics_params` + `metrics_db` + `config` + `history_tail`)
   and `piml_mixer_4096.pt` (trained residual weights). The top-level
   `summary.json` is a byte-identical copy (same MD5) kept next to the code.
5. **Downstream twin module** — `mixer_digital_twin_Mingran.py` is the twin
   implementation that consumes this record in the manuscript pipeline
   (Fig. 2e chain); `plot_mixer_heatmap.py` is the plotting helper.
6. **Table / text** — every displayed value is the `summary.json` value rounded
   to the printed number of decimals.

## Verification

Run `python verify_table2.py` (standard library only). It asserts the identity
of the two `summary.json` copies, all 15 table rows, the 4 text metrics, and
the 2 frozen S1.5 constants, rounded exactly as displayed; output archived in
`verify_table2_output.txt` (all 22 checks PASS, 2026-08-29).

## Value map (displayed → `summary.json` key, full precision)

| # | Displayed | Source key | Full precision |
|---|---|---|---|
| 1 | G_max,dB = +2.73 dB | `physics_params.G` | 2.7293514785998596 |
| 2 | P_LO,sat = −2.80 dBm | `physics_params.PsatL` | −2.8049513064337788 |
| 3 | β_LO = 1.85 | `physics_params.betaL` | 1.8507810921098646 |
| 4 | P_RF,sat = −8.02 dBm | `physics_params.PsatR` | −8.022814238429426 |
| 5 | β_RF = 2.92 | `physics_params.betaR` | 2.9163471944551613 |
| 6 | w = 1.00 | `physics_params.w_hill` | 0.9998751648148396 |
| 7 | P_RF,comp = −3.94 dBm | `physics_params.PcompRF` | −3.944236460733695 |
| 8 | P_LO,comp = −5.20 dBm | `physics_params.PcompLO` | −5.199087182330942 |
| 9 | β_C = 0.61 | `physics_params.betaC` | 0.6101367971342286 |
| 10 | κ = −1.49 dB | `physics_params.kappa` | −1.4910963469974874 |
| 11 | c_papr,LO = 0.54 | `physics_params.c_papr_lo` | 0.5383216348803388 |
| 12 | c_papr,RF = 0.07 | `physics_params.c_papr_rf` | 0.0681181224296381 |
| 13 | leak_dB = −97.7 dB | `physics_params.leak` | −97.70295030590809 |
| 14 | C_cal = +13.2 dB | `physics_params.C_cal` | 13.231423935543141 |
| 15 | P_floor = −97.6 dB | `physics_params.floor` | −97.58202830211204 |
| — | 0.79 dB RMSE (full) | `metrics_db.rmse_physics_plus_nn` | 0.7858207318089814 |
| — | 1.80 dB (physics only) | `metrics_db.rmse_physics_only` | 1.8022691737821477 |
| — | 10.35 dB (ideal multiplier) | `metrics_db.rmse_ideal_shape` | 10.345887808863704 |
| — | 96% within ±2 dB | `metrics_db.frac_abs_err_le_2db` | 0.9595959595959596 |

## Provenance

Copied unmodified (2026-08-29) from the read-only source
`fig2\e\upstream_measured_data_and_twin_code\`
(the Fig. 2e chain of the manuscript reproducibility archive).

MD5 checksums of the load-bearing files:

| File | MD5 |
|---|---|
| `summary.json` (top level) | `6a79f292287b21dac4b803c6f279852c` |
| `twin_training_run_4096DT_model/summary.json` | `6a79f292287b21dac4b803c6f279852c` (byte-identical) |
| `twin_training_run_4096DT_model/piml_mixer_4096.pt` | `505d216a66584aba957c9949efc6f620` |
| `twin_training_run_4096DT_model/fit_dualpump_init.npy` | `ad81ef91dff27ec237d9a8c7529f3ee0` |
| `gr_usrp_mixer_vector_heatmap_N4096.npz` | `54a1cdb02d0d016a7c5a45ede398d4e0` |
