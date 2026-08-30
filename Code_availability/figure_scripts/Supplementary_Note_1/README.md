# Supplementary Note 1 — physical model, residual network, and calibrated parameters of the PIML twin

**What is backed:** every numeric claim of Supplementary Note 1 of
`Supplementary_Information.tex` (source read:
`Supp_M\Supplementary_Information.tex`, lines 131–386;
the response letter `Main.tex` carries the same content in red).

Artifacts referenced below:

* **[T1]** `..\..\supp_tables\table1_scalar_twin_parameters\` — scalar (N = 1)
  calibration chain (43×43 acquisition → `heatmap_2db.npz` → 3-stage PIML →
  `summary.json` + `twin_predictions.npz`); verified by `verify_table1.py` there.
* **[T2]** `..\..\supp_tables\table2_twin4096_parameters\` — N = 4096
  recalibration chain (acquisition → `gr_usrp_mixer_vector_heatmap_N4096.npz`
  → `piml_4096_digital_twin.py` → `summary.json`); verified by `verify_table2.py` there.
* **[V]** `vector_run_summary.json` (this folder; copied unmodified from
  `fig2\j\`,
  MD5 `18d2bfc2106cd5b631cc723f7648355c`) — the N = 2–8 vector-twin training
  record; verified by `verify_note1.py` here (output in `verify_note1_output.txt`,
  all 14 checks PASS, 2026-08-29).
* **[T4/N4]** `..\..\supp_tables\table4_adaptation_routes\` and
  `..\note4_coherence_recovery\` — the N = 65,536 campaign and its
  out-of-sample correction numbers (owned by those components).
* **[ALG]** pure algebra / arithmetic — no data artifact required.

## Claim-by-claim map

### Preamble and diode-ring transfer law

| Claim | Backing |
|---|---|
| N = 65,536 experiments use these frozen parameters | design point (2¹⁶); campaign backed by **[T4/N4]** |
| Taylor ratio 1/6 for the odd-order cross terms, O(u⁵) remainder | **[ALG]** — Taylor expansion of the four-junction Shockley combination; parameter-free |

### S1.1 Physics block (Eqs. S1a–S1c)

| Claim | Backing |
|---|---|
| 1,849-point measurement, 43 × 43 grid | **[T1]** `heatmap_2db.npz` / `twin_predictions.npz` arrays are 43×43 (asserted in `verify_table1.py`) |
| physics block alone: 3.94 dB RMSE | **[T1]** `summary.json` → `metrics_db.rmse_physics_only` = 3.9442… |
| Supplementary Table 1: −1.84 dB / +1.73 dBm / 1.045 / +4.47 dBm / 2.13 / −73.62 dBm / −84.25 dB / Weibull | **[T1]** `summary.json` → `physics_params` (all 8 asserted) |

### S1.2 Bounded residual network and calibration

| Claim | Backing |
|---|---|
| δ_max = 8 dB | **[T1]** `summary.json` → `config.nn_max_db` = 8.0 |
| 2→32→32→1 tanh MLP | **[T1]** `config.nn_hidden` = [32, 32]; architecture in `Code_PIML_scalar_local_meeting\model\piml_mixer_digital_twin.py` |
| regularizers 10⁻⁴ / 10⁻³ / 10⁻² | **[T1]** `config.lam_nn` = 1e-4, `config.lam_delta` = 1e-3, `config.lam_smooth` = 1e-2 |
| 24 L-BFGS starts; 6,000 residual iters; 4,000 joint iters | **[T1]** `config.n_physics_starts` = 24, `config.n_iter_nn` = 6000, `config.n_iter_joint` = 4000 |
| 77-dB dynamic range | **[T1]** measured surface span in `twin_predictions.npz` = 77.43 dB (printed by `verify_table1.py`) |
| 1.30 dB RMSE | **[T1]** `metrics_db.rmse_physics_plus_nn` = 1.3037… (asserted) |
| R² = 0.9969 | **[T1]** recomputed from `twin_predictions.npz` = 0.99687 (asserted; not stored in `summary.json`) |
| 90.4% within ±2 dB | **[T1]** recomputed from `twin_predictions.npz` = 90.427% (asserted) |

### S1.3 Transfer to vectors, N = 2–8

| Claim | Backing |
|---|---|
| seven parameters frozen | **[V]** `n1_physics_params` block equals the Table-1 values (asserted in `verify_note1.py`) |
| c_LO = 0.885 | **[V]** `vector_papr_params.c_papr_lo` = 0.8848… (asserted) |
| c_RF = −0.040 | **[V]** `vector_papr_params.c_papr_rf` = −0.04017… (asserted) |
| bounded residual \|δ_NN\| ≤ 6 dB | **[V]** `config.nn_max_db` = 6.0 (asserted) |
| six inputs (2 powers, log₁₀\|⟨a,b⟩\|², log₂N, 2 PAPRs); 48-wide MLP | **[V]** `config.nn_hidden` = [48, 48]; input list implemented in the vector-twin code of the `fig2\j` chain (`fig2_v3_twin.py` and its upstream trainer) |
| RMSE 1.91 / 1.67 / 1.55 dB at N = 2 / 4 / 8 | **[V]** `metrics_db.{2,4,8}.full.RMSE_dB` = 1.9114 / 1.6723 / 1.5485 (asserted) |

### S1.4 Recalibration at N = 4096 (Eq. S4)

| Claim | Backing |
|---|---|
| N = 4096 recalibration on the receive chain of the N = 65,536 campaign | **[T2]** entire chain |
| fit selects w ≈ 1 | **[T2]** `physics_params.w_hill` = 0.99988 (Table-2 row 6, asserted as 1.00) |
| Supplementary Table 2: all 15 values (+2.73 / −2.80 / 1.85 / −8.02 / 2.92 / 1.00 / −3.94 / −5.20 / 0.61 / −1.49 / 0.54 / 0.07 / −97.7 / +13.2 / −97.6) | **[T2]** `summary.json` → `physics_params` (all 15 asserted in `verify_table2.py`) |
| bounded residual \|δ_NN\| ≤ 5 dB, 6→48→48→1 | **[T2]** `config.max_db` = 5.0, `config.hidden` = 48 |
| 0.79 dB RMSE on the measured N = 4096 surface | **[T2]** `metrics_db.rmse_physics_plus_nn` = 0.7858… (asserted) |
| 96% within ±2 dB | **[T2]** `metrics_db.frac_abs_err_le_2db` = 0.9596 (asserted) |
| 10.35 dB ideal multiplier | **[T2]** `metrics_db.rmse_ideal_shape` = 10.3459 (asserted) |
| 1.80 dB physics alone | **[T2]** `metrics_db.rmse_physics_only` = 1.8023 (asserted) |

### S1.5 Frozen use at N = 65,536

| Claim | Backing |
|---|---|
| sixteenfold longer vector | **[ALG]** 65,536 / 4,096 = 16 |
| η(t) compression template formula | **[ALG]** — same law as Eq. S1b with the two frozen constants below |
| exactly two frozen constants: P_LO,comp = −5.20 dBm, β_C = 0.610 | **[T2]** `physics_params.PcompLO` = −5.1991, `physics_params.betaC` = 0.61014 (asserted at both displayed precisions in `verify_table2.py`) |
| one complex coefficient per distortion feature, 5-fold CV, 200 measured products | **[T4/N4]** |
| RMSE 0.064 → 0.036 at 15 dB SNR; 0.056 → 0.017 at 25 dB SNR | **[T4/N4]** (also main-text Fig. 3d,e chain, `fig3`) |

### S1.6 Noise formulation and validation

| Claim | Backing |
|---|---|
| exactly one stochastic term; no w·n_x, x·n_w, n_w·n_x terms | structural claim about the model equations — verifiable by inspection of Eqs. S1c/S4 and the twin code in **[T1]**/**[T2]** |
| repeatability better than 0.5 dB | **[T1]** raw repeats: `heatmap_2db.npz` → `if_amp_uv_all` (43×43×3, three full sweeps). Derived, not stored: the median sweep-to-sweep difference over the full grid is 0.48 dB and the well-driven region (above the noise-floor corner) agrees to ≈0.06 dB median / <0.8 dB RMS; the full-grid *RMS* including the noise-floor-dominated corner is ≈2.3 dB. The "better than 0.5 dB" wording therefore holds for the device response above the floor (median statistic), not as a full-grid RMS — see honesty note below. |
| 16.5-dB error of the ideal-multiplier assumption (letter: 16.53 dB, 12.7×) | recorded in the scalar training run and carried in the Fig. 3c chain (`fig3\c\README.md`; value hard-coded from the run record in `fig3\c\fig3_v12_payoff.py`, `RM_IDEAL[0] = 16.53`). **Not recomputable from the archived `summary.json`** (it stores no ideal-tier RMSE) — see honesty note below. 12.7× = 16.53 / 1.30 **[ALG]**. |
| "thirty times smaller" | **[ALG]** 16.5 / 0.5 ≈ 33 |
| 1.30 dB RMSE at N = 1; 0.79 dB at N = 4096 | **[T1]** / **[T2]** (asserted there) |
| error prediction √σ²_bin/(\|Ĝ\|√N); residual matches within 1.1× (15 dB) and 1.7× (25 dB) | formula **[ALG]**; the 1.1× / 1.7× match is main-text Fig. 3f, backed by the manuscript reproducibility chain `fig3` (not owned by this folder) |

## Honesty notes (values not exactly reproducible from the archives here)

1. **"better than 0.5 dB" repeatability (S1.6).** The three repeated sweeps are
   archived (**[T1]** `heatmap_2db.npz:if_amp_uv_all`), but no artifact stores a
   single "0.5 dB" number. Recomputation here gives: median absolute
   sweep-to-sweep difference 0.48 dB over the full grid (consistent with the
   claim as a median), ≈0.06 dB in the well-driven region, but ≈2.3 dB RMS if
   the noise-floor-dominated low-power corner is included. The claim is
   supported for the deterministic device response, but the statistic behind
   the exact "0.5 dB" wording is not recorded.
2. **"16.5-dB error of the ideal-multiplier assumption" (S1.6; 16.53 dB in the
   letter).** The archived scalar `summary.json` stores no ideal-multiplier
   RMSE and `twin_predictions.npz` exports no ideal-tier surface; the value
   16.53 dB survives only as the training-run record quoted in
   `fig3\c\README.md` and hard-coded in `fig3\c\fig3_v12_payoff.py`. A naive
   log-domain least-squares ideal-multiplier fit on the archived measured
   surface gives 16.94 dB (same 16–17 dB scale, different anchoring
   convention). The exact 16.53 convention is not reconstructible from the
   files archived here.

Everything else in Note 1 is asserted PASS by `verify_table1.py` (14 checks),
`verify_table2.py` (22 checks) and `verify_note1.py` (14 checks) — 50/50 PASS
on 2026-08-29.

## Verification in this folder

Run `python verify_note1.py` (standard library only): asserts c_LO = 0.885,
c_RF = −0.040, the 6-dB residual cap, RMSE 1.91/1.67/1.55 dB at N = 2/4/8, and
that the frozen `n1_physics_params` equal the Table-1 values. Output archived
in `verify_note1_output.txt`.
