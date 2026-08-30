# Supplementary Note 4 — carrier coherence, inner-product recovery, noise floor, correction controls

Verifies the numbers of **Supplementary Note 4** of
`Supp_M/Supplementary_Information.tex` (lines
668–816: S4.1–S4.4) from archived bytes.

```
python verify_note4.py        # numpy only, ~30 s (the N=65536 transmit frame
                              # is rebuilt from its stored seed); 59 asserts;
                              # output archived in verify_note4_output.txt
```

## Inputs

* **Raw campaign npzs — referenced, not duplicated**:
  `../note3_client_energy/raw/gr_fig3c_ip_scatter_20260810_011915_N4096/gr_fig3c_ip_scatter.npz`
  (MD5 `f9b38769…`) and
  `../note3_client_energy/raw/gr_fig3c_ip_scatter_20260810_002043_N65536/gr_fig3c_ip_scatter.npz`
  (MD5 `3ff3c54f…`), plus the archived acquisition script
  `../note3_client_energy/raw/1_inner_product_scatter_v4.py` (its source text
  is itself an artifact here: the sync-acceptance threshold and the guard-bin
  choice are asserted from it).
* **Corrected per-product repeats — copied here** (needed for S4.3/S4.4):

  | File | MD5 | Copied from |
  |---|---|---|
  | `ip_optimized_N65536_20260826.npz` | `750abe21b9976cbf6341d84527109487` | `Data_availability/Figure_3/` |
  | `ip_optimized_N4096_20260826.npz` | `3c8536fca58e89b420976dfbc80e05eb` | same |

  Both are outputs of the 2026-08-26 out-of-sample correction run
  (`Code_availability/pipelines/ip_scatter_sweep/correction_20260826/1_opt_inner_product_with_DT.py`;
  full provenance chain in `Code_availability/pipelines/Figure_3/panel_d/README.md`,
  MD5-identical to `V2/inner_product_4096_655936/output/files.zip` members).
  They hold `c_true`, `c_hat_before_reps`, `c_hat_after_reps`,
  `c_hat_twin_reps` (2 panels × 3 captures × 200 products). The N=4096
  companion was **added beyond the original one-file copy list** because
  S4.3's quoted N=4096 *corrected* statistics (|d̂| 0.0163/0.0158,
  fractions 22.0%/71.5%) are computable only from its `c_hat_after_reps`;
  the N=4096 *uncorrected* fractions are computed from the raw npz's own
  `c_hat_reps`, as are all raw-side quantities. `verify_note4.py` asserts
  that the two archives agree (`c_hat_before_reps == c_hat_reps`,
  `c_true == c_true_norm`, both panels, atol 1e-12).

## Claim → artifact / assert map

| SI claim | artifact + derivation | outcome |
|---|---|---|
| S4.1 Eqs. S12–S13 (sync estimators, leakage law) | definitions; nothing to verify | — |
| 16,384 / 262,144-pt symbols; 610.35 / 38.15 Hz spacing; 512 / 8,192 CP | raw npz `fft_len, cp_len, fs_hz` | PASS |
| every 3rd FFT bin; products at one output bin, +3.66 kHz | `k0` = 6/96 (≡0 mod 3), `k0·fs/L` = 3662.1 Hz; comb grid `3(n−N/2)+1` regenerated and proven by the true-IP match below | PASS |
| combs occupy ≈7.5 MHz = 75% of Nyquist | regenerated bin span × fs/L = 7.50 MHz; kmax/(L/2) = 0.7505/0.7507 | PASS |
| preamble PSR 28.6–31.4, threshold 3 | `peak_metric` = 28.64/29.54/31.43/31.29; threshold from archived code (`peak_metric < 3.0` reject branch) | PASS |
| 26 pilots one-per-eight with 200 data slots; drift slopes < 3.2e-3 rad/slot | `n_pilot/n_data/meta.pilot_every`; `drift_rad_per_slot` (max 3.17e-3) | PASS |
| 10-dB PAPR digital clip | `meta_json.papr_clip_db = 10` | PASS |
| clip *"touching 0.006% of samples"* | byte-exact frame regeneration from stored seed (proof of fidelity: all 226 stored true inner products reproduced to ≤1.3e-13) | **NOT REPRODUCED — see below** |
| receiver ADC clipping fraction zero | `clip_frac = [0, 0]`, both runs (threshold 0.98 FS in archived code) | PASS |
| guard-bin SNRs 15.14–15.44 and 24.37–25.20 dB (union); 25.03–25.20 at N=65,536 alone | `snr3_db_reps` (range endpoints match to rounding) | PASS |
| noise from 4 guard bins at k0±4, k0±5 | archived acquisition code, `noise_bins` line | PASS |
| σ²_bin agrees within 0.6 dB across the ~10-dB power difference | `sigma2_bin` (0.351 / 0.590 dB apart), `p_rf_dbm_tx` (9.88 / 10.49 dB apart) | PASS |
| S4.3 σ̂ = 0.0324/0.0108 vs floors 0.0331/0.0107 → 0.98×/1.01× | bias-corrected estimators on `c_hat_after_reps − c_true`; floors `1/sqrt(27·SNR)` at mean `snr3_db_reps` | PASS |
| \|d̂\| = 0.0156/0.0134 carrying 18.9%/60.6% (N=65,536); 0.0163/0.0158, 22.0%/71.5% (N=4096) | same decomposition | PASS |
| uncorrected deterministic fractions 74.6%/96.3% and 79.9%/97.3% | same decomposition on the **raw npz** `c_hat_reps` | PASS |
| S4.4 template correlation ρ = 0.79–0.91 across all four operating points | reconstructed from `c_hat_twin_reps` (see caveat below): 0.787 / 0.914 / 0.807 / 0.904 | PASS |
| S4.4 compression ratio 0.24 at −3 dBm | pure algebra on the frozen twin constants: η(P) = (1+(P/P_comp)^βC)^(−1/βC), P_comp,LO = −5.1990871823 dBm, βC = 0.6101367971 → 0.2445 | PASS |

Twin-constant provenance: `PIML_TWIN_PARAMS` embedded in the archived
correction script
`Code_availability/pipelines/ip_scatter_sweep/correction_20260826/1_opt_inner_product_with_DT.py`
(lines 26–28; calibration chain in `fig3/b/upstream_twin_calibration_N4096/`).

## Items NOT verified from archived bytes (documented instead)

1. **"digitally clipped to a 10-dB PAPR (touching 0.006% of samples)"** —
   the 10-dB clip itself is verified (`meta_json`), but the **0.006% sample
   fraction is not reproduced**. The transmitted combs were regenerated
   exactly as built by the acquisition script (same seeded RNG, comb grid,
   preamble, CP, unit-RMS normalization; fidelity proven by reproducing all
   226 stored true inner products to ≤1.3e-13). The clip at
   `thr = 10^(10/20)` touches **0.019–0.021% of frame samples** (LO/RF,
   both runs), while the **clipped-energy fraction is 0.0045–0.0053%**
   (≈0.005%, the script's own EVM figure, −43 dBc). The quoted "0.006% of
   samples" matches neither exactly; the nearest derivable quantity is the
   clipped-energy fraction. Recorded here as a likely SI erratum ("of
   samples" should presumably read "of the waveform energy", or the sample
   figure should read ≈0.02%); `verify_note4.py` prints this as a `[NOTE]`
   and deliberately does **not** assert 0.006%.
2. **ρ = 0.79–0.91 — estimator caveat** — the SI states the correlation but
   not its estimator. The correction pipeline's raw replayed template is not
   stored per se; what is stored is the twin-only-corrected decode
   `c_hat_twin_reps`. Since the correlation magnitude is invariant to the
   one fitted complex coefficient, `verify_note4.py` uses the
   scale-invariant proxy: per capture,
   ρ = |⟨t, r⟩| / (‖t‖‖r‖) with t = `c_hat_before − c_hat_twin` (the twin
   template component actually removed) and r = `c_hat_before − c_true`,
   averaged over the three captures of each operating point. The four values
   0.787, 0.914, 0.807, 0.904 land exactly on the quoted 0.79–0.91. (Small
   residual dependence on the per-fold CV coefficients is inherited from
   what was archived; the raw-template statement itself originates in the
   correction run of `fig3/d/upstream_correction_20260826/`.)
3. **S4.4 controls #2 and #3** (coefficient agreement across SNR and N;
   sign pattern of the envelope-variant correlations placing the LO on the
   compression side) — qualitative claims with no quoted numbers; the fitted
   coefficients are archived for inspection in the `coeffs` field of the two
   `ip_optimized_*.npz` (basis order `twin, limiter, square, gain`), and the
   limiter/square bases live in the raw npzs (`mixer_dev_lim`,
   `mixer_dev_sq`). Not asserted.

## Relation to sibling folders

* Raw bytes are shared with `../note3_client_energy/` (MD5-identical to
  `Data_availability/Figure_3/gr_ip_scatter_N*_20260810.npz` used by the fig3 chains).
* The S4.3 decomposition is also reproduced upstream by
  `Code_availability/pipelines/Figure_3/panel_d/verify_s43_repeat_decomposition.py`;
  this folder re-implements it independently against the archive's own
  copies.
