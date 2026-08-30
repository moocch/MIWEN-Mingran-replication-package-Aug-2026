# Supplementary Note 8 — text numbers (fully analog client) and flag resolution

Verifies the Note 8 text numbers that are not Table 5/6 cells, and resolves
the two tensions flagged by the earlier census. SI source: Supplementary
Note 8 (S8.1–S8.3) of `Supplementary_Information.tex`.

## Files

| file | what |
|---|---|
| `verify_note8.py` | 50 assertions: S8.1 recovery, S8.3 depth controls + link budget, flag (a)/(b) provenance | 
| `verify_note8_output.txt` | saved run output — **50 passed, 0 failed** |
| `data/twin_2x2.json`, `data/hardware_aware_ablation.json`, `data/paper_numbers.json` | small JSONs missing from `Data_availability/raw/fully_analog_simulation/results/` (see provenance) |

Run: `python verify_note8.py` (exit 0 = everything verified).

## Text number → JSON tag mapping (all verified)

| SI claim | source | exact tag / key |
|---|---|---|
| recovery 92.62 → 93.90 (MNIST) | `twin_2x2.json` → `results_summary.json` | `mnist.twin_hardware.acc` = 92.62 → tag `MZH_L3` = 93.90 ± 0.10 |
| recovery 75.13 → 87.53 (GTSRB) | same | `gtsrb.twin_hardware.acc` = 75.13 → tag `ZH_L3` = 87.53 ± 0.14 |
| L=2 gap 0.22 (94.04 vs 93.82 ± 0.03) | `results_summary.json` | `MZT_L2d` (mode ZT) − `MZH_L2` (mode ZH) |
| L=2 gap 0.87 (87.68 vs 86.81 ± 0.05) | same | `ZT_L2d` − `ZH_L2` |
| L=3 gaps 0.40 / 1.30 ("main text") | same | `MZT_L3d` 94.30 − `MZH_L3` 93.90; `ZT_L3d` 88.83 − `ZH_L3` 87.53 |
| L=4 collapse 85.02 vs 15.39 ± 0.23 | same | `MZT_L4d` − `MZH_L4` |
| efficiencies 2.255/2.181/1.267 % and 2.091/2.115/1.202 % | `link_budget.json` | `MNIST.block_efficiency`, `GTSRB.block_efficiency` |
| GTSRB ladder −10.0/−34.27/−58.48 dBm | `link_budget.json` | `GTSRB.rows[0..2].drive_dBm` |
| SNRs 43.0/32.6/8.4 dB | same | `GTSRB.rows[0..2].tone_snr_dB` = 43.04/32.57/8.36, quoted to 1 decimal |
| 4th pass −85.2 dBm (−18.3 dB) | same | `GTSRB.rows[3]` = −85.16 dBm / −18.31 dB, quoted to 1 decimal |
| "within 0.7 dB of the MNIST plan" | same | max |GTSRB−MNIST| drive gap over passes 1–3 = 0.46 dB ≤ 0.7 |

All noise-bearing accuracies: mean ± 1 s.d. over 5 realizations
(`repeats: 5` in every `results_summary.json` entry).

## FLAG (a) resolution — why 93.90 / 87.53 match no table row

**The Note 8 tables and the main-text Fig. 5d numbers come from two distinct
experiment generations in the same release.**

**Generation 1 — the calibrated-twin (`ZC`/`ZT`/`ZH`) family** (the deployed
configuration; `MANIFEST.md`'s mode table describes only this generation):

- mode `ZH` = mixer-compression twin on the composite comb + measured detector
  transfer (k_ED = 208.7 V⁻¹) + per-pass link noise + element jitter — the
  chain as deployed;
- mode `ZT` = the same device twin, **no noise** — "digital execution" and the
  software forecast;
- mode `ZC` = ideal multiplier + ideal square law — the clean arm.

Fig. 5d ("Fig. 4" in `paper_numbers.json`, the standalone section's
numbering) = checkpoints `MZH_L3` / `ZH_L3` (trained under `ZH`):
digital (`ZT`) 94.30 / 88.83, chain (`ZH`) 93.90 ± 0.10 / 87.53 ± 0.14
(`results_summary.json` tags `MZT_L3d`, `ZT_L3d`, `MZH_L3`, `ZH_L3`; the
`…d` tags share the ZH checkpoint via their `weights` field). Table 5
(`twin_2x2.json`) is the same generation: `MZC_L3`/`ZC_L3` and
`MZT_L3`/`ZT_L3` checkpoints, chain cells evaluated under `ZH`. So S8.1's
recovery sentence (92.62 → 93.90, 75.13 → 87.53) is a **coherent
within-generation comparison**: both endpoints run on the identical `ZH`
chain with 5 noise draws; the checkpoints differ only in whether the
link-noise model was in the training loss (`ZT`-trained vs `ZH`-trained).

**Generation 2 — the earlier mode-`Z` family** (`MANIFEST.md` itself calls
`hardware_aware_ablation.json` the "**earlier noise-only ablation**"):

- mode `Z` = measured detector activation + link noise + jitter but **no
  mixer-compression twin** (`comb_analog_sim.py forward()` skips
  `comb_compress_mode` for mode `Z`);
- digital baseline mode `ZD` = same detector activation, exact arithmetic,
  no noise; `ZI` = ideal square law.

Table 6 lives entirely in this generation: checkpoints `MZ_L3`/`Z_L3`
(trained under `Z`), `MZD_L3_nohw`/`ZD_L3_nohw` (trained under `ZD`,
i.e. noise-free), `MZI_L3_ideal`/`ZI_L3_ideal` (trained under `ZI`); chain
column = mode `Z`, 5 draws; digital column = `ZD`/`ZI`, 1 draw
(`reproduce.py ablation`). Hence its "twin + link noise" row reads
96.07 / 95.62 ± 0.09 (MNIST) and 88.29 / 86.26 ± 0.13 (GTSRB) — different
checkpoints, different forward model, and a different digital protocol
(`ZD`, not `ZT`) from the Fig. 5d numbers. Same test sets and same
5-realization protocol in both generations, so the realization count is
NOT the explanation.

## FLAG (b) resolution — 0.40/1.30 vs 0.45/2.03

Both pairs are genuine "digital − chain at L = 3" gaps, but on the two
different generations:

- 0.40 / 1.30 = `MZT_L3d` − `MZH_L3` and `ZT_L3d` − `ZH_L3`
  (generation 1, the deployed `ZH` networks — what S8.3 and the main text
  quote);
- 0.45 / 2.03 = `MZD_L3` − `MZ_L3` and `ZD_L3` − `Z_L3`
  (generation 2, Table 6).

`verify_note8.py` asserts all four gaps and that the two checkpoint sets are
disjoint.

## Verdict — is the SI internally consistent?

**Numerically, every quoted value is real and traces to exactly one JSON
field; no number is wrong. But S8.2/Table 6 genuinely conflates the two
experiments in its labels:**

1. Table 6's row label "twin + link noise" (and S8.2's sentence "with both
   the twin and the link noise in the loss the analog penalty is 0.45/2.03")
   attaches the Note's own definition of *twin* (S8 preamble: mixer
   compression + measured detector) to the generation-2 `MZ_L3`/`Z_L3`
   checkpoints, which were trained **without** the mixer-compression twin.
   Under the actual twin+link-noise configuration (generation 1) the penalty
   is 0.40/1.30 — the main-text numbers. A reader comparing Table 6 row 1
   against Fig. 5d or against S8.1's recovery endpoints will find numbers
   that cannot be reconciled without the archive's tag structure.
2. Table 6's row label "link noise, no twin" inverts the actual ablation:
   `MZD_L3_nohw`/`ZD_L3_nohw` were trained **noise-free** (mode `ZD` adds no
   link noise) while *keeping* the measured detector activation — the
   archive's own label is "no-hw, noise-free".
3. Everything else — Table 5, S8.1's recovery sentence, all of S8.3, and the
   main-text Fig. 5d values — is one consistent generation-1 story.

Suggested minimal fix for the SI: relabel Table 6's rows (e.g.
"noise-aware (earlier chain model)" / "noise-free" / "ideal square law"),
or recompute the ablation on the `ZH` generation, and add one sentence
noting Table 6 predates the calibrated mixer twin.

## Provenance note (missing files in the shared folder)

`Data_availability/raw/fully_analog_simulation/results/` contains only
`results_summary.{json,csv}` and `link_budget.json`; the `twin_2x2.json`,
`hardware_aware_ablation.json` and `paper_numbers.{json,csv}` promised by its
own `MANIFEST.md` (and `twin_2x2.py` / `make_paper_numbers.py`) are absent.
The `data/` copies here were extracted from the canonical release
`miwen_fully_analog_archive.zip` (inside
`Manuscript/Data/V2/fully analog/files (1).zip`, entries under
`code/results/`) and are byte-identical (SHA-256) to
`Manuscript/Overleaf/fig5_analog{,_v2,_v3}/data/`:
`twin_2x2.json` `bee2c1a6…` (608 B), `hardware_aware_ablation.json`
`ad6330e3…` (1012 B), `paper_numbers.json` `b0558f7f…` (8552 B). The zip's
`results_summary.json` and `weights/train_summary.json` are byte-identical
to the shared archive's copies, confirming a single release. The script
prefers the shared `results/` folder and falls back to `data/`, printing the
source used.
