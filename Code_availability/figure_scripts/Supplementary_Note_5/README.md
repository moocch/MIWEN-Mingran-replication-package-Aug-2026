# Supplementary Note 5 — the analog activation block (claim → artifact map)

Verifies every number of Supplementary Note 5 (booster, detector, squaring
derivation Eqs. S15–S18, level plan S5.6, per-layer fidelity S5.8) of
`Supp_M/Supplementary_Information.tex`.

Shared simulation archive (read-only, referenced, not copied):
`../../fully_analog_simulation` — its `results/results_summary.json`
and `results/link_budget.json` are **byte-identical (sha256)** to
`code/results/*` inside `miwen_fully_analog_archive.zip` of
`V2/fully analog/files (1).zip`.

## Files here

| file | what it does |
|---|---|
| `verify_note5.py` | asserts every derivable number of the note (59 checks) |
| `verify_note5_output.txt` | its output — **59/59 PASS** |
| `recompute_no_booster.py` | derives the no-booster −52/−95 dBm levels from the archived physics (gated numpy port of the archive's `plan_Z` recursion) |
| `recompute_no_booster_output.txt`, `recompute_no_booster.json` | its output |

## Claim → artifact / derivation map

| # | Note 5 claim | backing | status |
|---|---|---|---|
| S5.1 | per-pass loss −7.47 dB = 6.65 + 0.82 | ZEM-4300+ datasheet conversion loss (6.65 dB) + LFCN-490+ datasheet insertion loss (0.82 dB); equals `link_budget.json settings.cl_pass_dB = 7.47` and `CL_MIXER`/`IL_FILT` in the archive's `comb_analog_sim.py` (lines 384–385) | PASS |
| S5.2 | Eq. S15 bound: Q=40, C=1.5 pF, 0.40 GHz → R_in ≤ 10.6 kΩ | R_Q = Q/(ωC) = 10 610 Ω; same formula as `comb_analog_sim.boost_db()` | PASS |
| S5.2 | 19.3 dB boost ceiling after 1 dB IL | 10·log10(10 610/100) − 1 = 19.257 dB; also stored in the old letter's `c7_activation_numbers.json` (`boost_ceiling_dB`) | PASS |
| S5.2 | adopted A_v = 18.5 dB ≤ bound | measured constant [Wang et al. ESSCIRC 2017]; `settings.boost_dB = 18.5` | PASS |
| S5.3/S5.4 | k_ED = 208.7 V⁻¹; knee v* = 1/(2k_ED) = 2.4 mV | measured constant [Wang et al. SSC-L 2018]; `settings.k_ED_per_V`; 1/(2·208.7) = 2.396 mV; Eq. S16 → k_ED·v² small-signal limit verified numerically | PASS |
| S5.5 | R_v = 50·10^1.85 ≈ 3.5 kΩ | formula gives 3 540 Ω; archive uses the conservative `R_v_ohm = 3500` (`comb_analog_sim.py` line 347: "50 x 10^(BOOST/10)") | PASS |
| S5.5 | Eq. S18: η ≈ 0.056 (−12.5 dB) | 4·50·3500/3550² = 0.0555 → −12.55 dB | PASS |
| S5.5 | n² ≈ R_v/R_0 ≈ 70; further 1 dB step-down | 3500/50 = 70 exactly; `settings.step_down_loss_dB = 1.0` | PASS |
| S5.6 | block efficiency 2.3 %/2.1 % (MNIST/GTSRB) | `link_budget.json` `MNIST.block_efficiency = [2.255, 2.181, 1.267]`, `GTSRB.block_efficiency = [2.091, 2.115, 1.202]`; first-pass values round to 2.3/2.1; arithmetic identity eff = P_next/(P_prev·10^(−7.47/10)) verified | PASS |
| S5.6 | ladder −10 → −34 → −58 dBm; 4th pass near −84 dBm, below −73.6 dBm floor | `link_budget.json` rows: MNIST −10.0/−33.94/−58.02/−84.46 dBm, GTSRB −10.0/−34.27/−58.48/−85.16 dBm; "near −84" is the MNIST row; the −73.6 dBm floor is `P_N_DBM = -73.6` in `comb_analog_sim.py` (line 59) and is the constant of the archived tone-SNR law, verified against every `tone_snr_dB` entry to <0.01 dB | PASS |
| S5.7 | SMS7630 k_ED = 1/(4nV_T) ≈ 9.3 V⁻¹ (nV_T ≈ 27 mV, n = 1.05) | 1/(4·0.027) = 9.26 V⁻¹; n·kT/q at 300 K = 27.1 mV | PASS |
| S5.7 | 27 dB video deficit | 20·log10(208.7/9.3) = 27.02 dB | PASS |
| S5.7 | 5.1 kΩ junction resistance caps Eq. S15 near 16 dB | 10·log10(5100/100) − 1 = 16.1 dB | PASS |
| S5.8 | per-pass drives/SNRs −10.0/43.0, −34.3/32.6, −58.5/8.4 | `link_budget.json` GTSRB rows (−34.27/32.57, −58.48/8.36) | PASS |
| S5.8 | "within 1.30 points of digital execution" | `results_summary.json`: `ZT_L3d` 88.83 − `ZH_L3` 87.53 = 1.30 | PASS |
| S5.8 | ideal square law drops GTSRB 88.3 % → 42.2 % | **found** — see investigation (a) | PASS |
| S5.3 | no booster: passes near −52/−95 dBm, ≤ 2 passes | **recomputed** — see investigation (b) | PASS with caveat |

## Investigation (a): 88.3 % → 42.2 % — BACKED

The pair exists in the released archive, in
`Data_availability/raw/fully_analog_simulation/results/results_summary.json`
(byte-identical to the original zip's copy):

- **88.3 %** = tag `ZD_L3`, `accuracy_pct = 88.29`, weights `Z_L3`,
  "same weights, digital execution (no link)". Corroborated by
  `hardware_aware_ablation.json` (in the original zip): `Z_L3.digital_pct = 88.29`.
- **42.2 %** = tag `ZI_L3`, `accuracy_pct = 42.15`, weights `Z_L3`,
  "same weights, ideal square-law activation".

Both tags execute the same chain-trained GTSRB weights (`Z_L3`); the only
change between them is the measured detector transfer (Eq. S16) versus the
idealized square law — exactly the note's sentence. Rounding note: 42.15 →
"42.2" is round-half-up (both values within 0.05 of the tex). **No GAP.**

(The tex's "chain-trained network" here is the `Z_L3` weights of the archive's
`results_summary.json`; the neighbouring "within 1.30 points" sentence uses the
hardware-aware `ZH_L3` weights — two different weight tags, both archived.)

## Investigation (b): no-booster ≈ −52 / −95 dBm — NOT ARCHIVED; RECOMPUTED

Searched: every JSON in `Data_availability/raw/fully_analog_simulation/results/`, every JSON
inside the original `miwen_fully_analog_archive.zip`, `analog_section.tex` in
that zip, and the old letter folder
`Response_Letter_Reproductivity/Reviewer1_Comment7/b_activation_block_note/c7_activation_numbers.json`
(which has knee/boost-ceiling/ladder/efficiencies but **no** no-booster
levels). The −52/−95 dBm pair appears in **no archived artifact** — it is a
derived statement. `recompute_no_booster.py` therefore derives it from the
archived physics constants, after a gate that reproduces the archived ladder
to < 0.005 dB (both datasets, all four passes, and all block efficiencies).

Result (GTSRB / MNIST):

- **Variant L** — the note's own bookkeeping ("the booster is worth ≈18.5 dB
  of delivered video power per recirculation"): archived ladder minus 18.5 dB
  at pass 2 and 2×18.5 dB at pass 3 → **−52.4/−52.8 and −95.0/−95.5 dBm**.
  Reproduces "near −52 / near −95" to < 0.8 dB; pass 3 is 21.4/21.9 dB below
  the −73.6 dBm floor ("some 20 dB below" ✓); at most two passes ✓.
- **Variant A** — the archived `plan_Z` recursion with `g_boost = 0` and all
  else untouched: pass 2 = **−52.4/−52.7 dBm** (reproduces −52 ✓) but pass 3 =
  **−107.0/−107.9 dBm**, not −95: without the boost the pass-2 detector node
  falls from the rectifier regime into the square-law regime, where the 18.5-dB
  deficit doubles. Still ≤ 2 passes, a fortiori.
- **Variant B** — `g_boost = 0` with the video source impedance also returned
  to 50 Ω: essentially restores the with-boost ladder (−34/−58 dBm), because in
  the saturated first-pass regime the boost and the boosted source impedance
  cancel; this reading contradicts the note and is reported only for
  completeness (it is not the archive's bookkeeping, which fixes
  `R_v = 50·10^(BOOST/10)`).

**Plain statement:** −52 dBm is reproduced by both the note's bookkeeping and
the literal no-boost recursion. −95 dBm is reproduced only by the linear
"18.5 dB per recirculation" bookkeeping stated in the same sentence of the
note; the full archived recursion puts the third pass near −107 dBm — even
further below the floor. The note's conclusions ("some 20 dB below the
−73.6-dBm floor", "at most two passes", "the booster is what buys the third
pass") hold under both readings; only the specific figure "−95 dBm" is
bookkeeping-derived rather than recursion-derived. Not a GAP in support of the
conclusion, but a provenance caveat worth recording.

## Citations for published (non-derivable) constants

- **6.65 dB conversion loss** — Mini-Circuits, ZEM-4300+ Level 7 frequency
  mixer, 300–4300 MHz, datasheet, https://www.minicircuits.com/pdfs/ZEM-4300+.pdf
  (tex bib key `zem4300ds`).
- **0.82 dB insertion loss** — Mini-Circuits, LFCN-490+ LTCC low-pass filter,
  DC–490 MHz, datasheet, https://www.minicircuits.com/pdfs/LFCN-490+.pdf
  (`lfcn490ds`).
- **18.5 dB passive boost at 400 MHz, R_in > 30 kΩ detector** — P.-H. P. Wang,
  H. Jiang, L. Gao, P. Sen, Y.-H. Kim, G. M. Rebeiz, D. A. Hall, P. P. Mercier,
  "A 400 MHz 4.5 nW −63.8 dBm sensitivity wake-up receiver employing an active
  pseudo-balun envelope detector," Proc. IEEE ESSCIRC, 35–38 (2017)
  (`mercier2017esscirc`; the archive's code comment cites it as
  "Wang et al., ESSCIRC 2017, pp. 35–38" — same paper).
- **k_ED = 208.7 V⁻¹, R_in > 750 kΩ, 30.6 dB boost at 109 MHz, 0.36 nW bias
  ladder, 6.1 nW budget** — P.-H. P. Wang, H. Jiang, L. Gao, P. Sen, Y.-H. Kim,
  G. M. Rebeiz, P. P. Mercier, D. A. Hall, "A 6.1-nW wake-up receiver achieving
  −80.5-dBm sensitivity via a passive pseudo-balun envelope detector," IEEE
  Solid-State Circuits Lett. 1, 134–137 (2018) (`wang2018sscl`).
- **N-stage stacking (+10·log10 N) and its capacitance cost** — V. Mangal,
  P. R. Kinget, "An ultra-low-power wake-up receiver with voltage-multiplying
  self-mixer and interferer-enhanced sensitivity," Proc. IEEE CICC, 1–4 (2017)
  (`mangal2017cicc`); also V. Mangal, P. R. Kinget, "A 0.42 nW 434 MHz
  −79.1 dBm wake-up receiver with a time-domain integrator," ISSCC Dig. Tech.
  Papers, 438–440 (2019) (`mangal2019isscc`).
- **SMS7630 zero-bias Schottky (n ≈ 1.05, ~5.1 kΩ junction resistance)** —
  Skyworks Solutions, SMS7630 series datasheet,
  https://www.skyworksinc.com/Products/Diodes/SMS7630-Series (`sms7630ds`).
- **−73.6 dBm per-bin floor** — calibrated constant of the manuscript's own
  PIML twin (`P_N_DBM = -73.6` in the archive's `comb_analog_sim.py`, "as
  published"); verified here to be the constant of the archived tone-SNR law.

## Verification results

- `verify_note5.py`: **59/59 checks passed** (see `verify_note5_output.txt`).
- `recompute_no_booster.py`: gate passed (archived ladder reproduced to
  < 0.005 dB); verdicts as in investigation (b)
  (see `recompute_no_booster_output.txt` / `.json`).
- No unbacked number remains: (a) is fully archived; (b) is derivable, with
  the provenance caveat stated above.
