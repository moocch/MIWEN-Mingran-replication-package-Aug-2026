# Supplementary Note 3 — client-side energy accounting (main-text Fig. 3g)

Verifies every number of **Supplementary Note 3** of
`Supp_M/Supplementary_Information.tex` (lines
449–665: S3.1–S3.5, Eqs. S6–S11, Supplementary Table 4 `tab:routes` energy
column) from the archived raw campaign bytes.

```
python verify_note3.py        # numpy only; 57 asserts; output archived in
                              # verify_note3_output.txt (all PASS)
```

## Contents and provenance (all copies byte-identical to source, MD5-checked)

| File | MD5 | Copied from |
|---|---|---|
| `raw/gr_fig3c_ip_scatter_20260810_011915_N4096/gr_fig3c_ip_scatter.npz` | `f9b38769c6ed4390f859afe86c1f2906` | `Code_availability/pipelines/Figure_3/panel_g/fig5_energy_package/data/raw/<run>/` |
| `raw/gr_fig3c_ip_scatter_20260810_002043_N65536/gr_fig3c_ip_scatter.npz` | `3ff3c54f4e81234c2b8062b62f950821` | same package |
| run-record PNGs (`...ip_scatter.png`, `..._enob.png`, both runs) | `86fce3af…`, `09c581f9…`, `b463765b…`, `fd7c3cd4…` | same package |
| `raw/1_inner_product_scatter_v4.py` (N=65536 acquisition) | `d7ff7b1f1b61fbaae64ee7c57729954d` | same package |
| `raw/1_inner_product_scatter_v4_N4096.py` (N=4096 acquisition) | `6864a4811d3d1659618c1d99d4cec8fd` | same package |

The same two npz files also live (MD5-identical) as
`Data_availability/Figure_3/gr_ip_scatter_N{4096,65536}_20260810.npz`
and back the fig3 panels; `../note4_coherence_recovery/verify_note4.py` reads
*these* copies, so the two notes share one set of raw bytes. The acquisition
scripts are the GNU Radio / dual-USRP X310 campaign code (RF 1.20 GHz /
LO 0.90 GHz / IF 0.30 GHz); the two scripts differ only in the default
geometry constants (verified by diff).

## Claim → assert map (every quoted Note-3 number)

All asserts read the raw npz; the only external constants are the benchmark
conventions of the methodology (Gao et al. 2026, documented in the package's
`METHODS.md`): `eta_radio = 10%`, `e_ADC = e_dig = 1 pJ`, H100 = 70 fJ/MAC,
criterion `RMSE < 0.0625`.

| SI claim | derivation from archived bytes | assert |
|---|---|---|
| `P_x = −62.81 dBm (0.524 nW)` / `−63.41 dBm (0.457 nW)` | `p_rf_dbm_tx[25dB] − rf_atten_db` (30-dB pad in `meta_json`) | `P_x = …` (4 asserts) |
| LO (weight) comb at −3 dBm, both runs | `p_lo_dbm_tx` | 2 asserts |
| `T_ip = 1.918 / 30.68 ms` (Eq. S8) | 227 = (frame−gaps)/(fft+cp) bursts × (fft+cp)/fs / 200 | 4 asserts |
| gap-charging variant +0.43% (1.926/30.815 ms), changes nothing | `frame_len/fs/200` | 2 asserts |
| 117 ns airtime per real MAC, common to both sizes | `T_ip/(4N)` | 2 asserts |
| `e1/e2/e3 = 0.613/0.366/0.488` and `0.534/0.023/0.031 fJ` (Eq. S7) | model with measured `P_x`, `T_ip` | 6 asserts |
| totals `1.467 / 0.588 fJ`; `47.7× / 119×` vs 70 fJ | sum; ratio | 4 asserts |
| RMSE `0.0609±0.0003` / `0.0561±0.0006`; every repeat < 0.0625 | `rmse_mean/sd/reps[25dB]` | 6 asserts |
| 15-dB points miss the criterion (0.0672, 0.0643) | `rmse_mean[15dB]` | 2 asserts |
| `std_y = 0.3395 / 0.3343` | `std_y` | 2 asserts |
| criterion-equivalent ENOB 2.42–2.44 bits (Eq. S6 ↔ ENOB) | `log2(std_y/0.0625)` = 2.4415 / 2.4192 | 1 assert |
| tuner: 0.4-dB tolerance, 2–3 iterations, power then frozen, n=3 | `meta_json.tune_tol_db`, `tune_history_json`, `n_repeats` | 8 asserts (see below) |
| `N_bend ≈ 6.5e3` (Eq. S9 = radio floor) | `14 pJ/(4·e1)` = 6550 | 1 assert |
| Eq. S10 curve is not a fit; N=4096 sits a genuine ≈6% above it | `e1(65536) + 3.5 pJ/N` vs measured 1.467 fJ → +5.6% | 1 assert |
| Landauer = thermodynamic = 11.48 zJ at b=2 (Eq. S11) | `b²·kT·ln2 = 2b·kT·ln2` at b=2, T=300 K | 1 assert |
| measured point ≈5×10⁴ above the room-temperature floor | 0.588 fJ / 11.48 zJ = 5.12e4 | 1 assert |
| correction deployed: 0.588→0.649 fJ (119×→108×), 1.467→2.444 fJ (47.7×→28.6×) (S3.5) | `e_ip + 16 pJ/(4N)` | 4 asserts |

(S3.5's RMSE-improvement numbers 0.056→0.017 / 0.064→0.036 and the
Table-4 accuracy column belong to the correction chain — asserted from
`ip_optimized_*.npz` by `../note4_coherence_recovery/verify_note4.py` and by
the fig3/d chain — and to the fig4 training campaigns respectively; the
energy cells are asserted here.)

## The two session-log-style claims — now verified from bytes

The task of this folder was told to treat two S3.2 statements as
documentation-only. Both turned out to be **verifiable from archived bytes**,
because the raw npz stores the closed-loop tuner's own trace:

1. **"converging within 2–3 iterations to a 0.4-dB tolerance"** —
   `meta_json` carries `"tune_tol_db": 0.4` and `"max_tune_iters": 4`;
   `tune_history_json` shows the per-iteration trace: 2 and 2 iterations for
   the N=4096 panels, 2 and 3 for the N=65536 panels, with final
   `|err_db|` = 0.159, 0.363, 0.141, 0.044 dB — all within 0.4 dB.
   Asserted. The same procedure is documented in the package README
   ("Provenance": *"using the closed-loop power tuner: the criterion is
   fixed first, the power is lowered until it is just barely met, then
   `n = 3` repeats are captured at that fixed power"*).
2. **Power frozen for the repeats, `n = 3`** — `n_repeats = [3, 3]`;
   asserted.

## The legacy N=4096 run (documented, not archived here)

S3.2: *"An earlier N = 4096 acquisition, in which a high-PAPR waveform was
emitted ≈10 dB below its commanded power and underdrove the LO port (raw
RMSE 0.0795), was replaced by the present run after a peak-aware
transmit-power fix; the legacy file is retained in the released data."*

* **Where it is stated in the archived package** —
  `fig5_energy_package/METHODS.md` §7: *"**On the `N = 4096` run.** It
  replaces a legacy pre-power-fix acquisition (raw RMSE 0.0795, LO
  underdriven) that never met the criterion. With the peak-aware TX-power
  fix the LO sits at its −3 dBm target, `P_x` moved from −58.15 to
  −62.81 dBm, and the point now falls on the model curve."*
* **The ≈10-dB underdrive mechanism, in archived bytes** — the docstring of
  `power_to_gain_amp` in the archived acquisition scripts (this folder,
  `raw/1_inner_product_scatter_v4.py`, lines 128–135) documents the fixed
  bug: the old code satisfied the digital peak constraint by cutting `amp`
  without compensating gain, so high-PAPR waveforms were emitted *"10+ dB
  below target (LO target −3 dBm actually emitted −14.9 dBm, pushing the
  ZEM-4300 into its underdriven expansion regime, a deterministic
  RMSE ≈ 0.115 floor)"* (translated); the fix folds the peak constraint
  into the (gain, amp) solve.
* **The legacy npz itself is NOT in the energy package** (its `data/raw/`
  holds only the two replacement runs). The retained released copy was
  located, read-only, at
  `MIWEN_Mingran_我上传的文件\MIWEN_Revision_Mingran\U6_Inner_Product_Optimization_Digital_Twin\gr_fig3c_ip_scatter.npz`
  (MD5 `7caddfdff59219f7ed76588c15cd77f5`), and its stored values confirm
  the quote: 25-dB raw `rmse = 0.07948` ("0.0795") and
  `p_rf_dbm_tx = −28.149 dBm` → `P_x = −58.15 dBm` after the 30-dB pad,
  4.66 dB above the replacement run's −62.81 dBm at the same SNR target.
  It was left uncopied (outside this folder's sanctioned copy list); this
  README is the archive's record of where it lives.

## Not claimed, and knowingly so

The SI itself flags (S3.3–S3.4) that `e2`/`e3` describe an optimized
implementation (not the laboratory SDR's wall-plug power), that the H100
line excludes memory traffic, and that no energy error bars exist because
the repeats share one frozen `P_x` — conventions, not numbers; nothing to
assert beyond what is above.
