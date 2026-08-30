# Supplementary Table 6 — training ablation on the simulated chain

Verifies every cell of Supplementary Table 6 (`\label{tab:analogablation}`,
Supplementary Note 8, S8.2 of `Supplementary_Information.tex`) against the
released fully-analog simulation archive.

## Files

| file | what |
|---|---|
| `verify_table6.py` | asserts all 6 rows (digital / chain / s.d. / drop) + cross-checks + protocol |
| `verify_table6_output.txt` | saved run output — **49 passed, 0 failed** |
| `data/hardware_aware_ablation.json` | the source JSON (see provenance below) |

Run: `python verify_table6.py` (exit 0 = all cells verified).

## Row → JSON mapping

Source: `hardware_aware_ablation.json` (produced by `reproduce.py ablation`).
Each SI row is one top-level key with fields `digital_pct`, `analog_pct`,
`analog_std`, `drop_pts`:

| SI row (Task / Training loss) | JSON key | JSON `label` | digital / chain / drop |
|---|---|---|---|
| MNIST, twin + link noise | `MZ_L3` | `MNIST  hardware-aware` | 96.07 / 95.62 ± 0.09 / 0.45 |
| MNIST, link noise, no twin | `MZD_L3_nohw` | `MNIST  no-hw, noise-free` | 96.03 / 94.20 ± 0.15 / 1.83 |
| MNIST, ideal square law | `MZI_L3_ideal` | `MNIST  no-hw, ideal square` | 95.77 / 31.97 ± 0.23 / 63.80 |
| GTSRB, twin + link noise | `Z_L3` | `GTSRB  hardware-aware` | 88.29 / 86.26 ± 0.13 / 2.03 |
| GTSRB, link noise, no twin | `ZD_L3_nohw` | `GTSRB  no-hw, noise-free` | 89.19 / 65.70 ± 0.17 / 23.49 |
| GTSRB, ideal square law | `ZI_L3_ideal` | `GTSRB  no-hw, ideal square` | 85.28 / 8.16 ± 0.35 / 77.12 |

Rounding: the SI table prints the two GTSRB collapse drops to one decimal
(23.49 → 23.5, 77.12 → 77.1; S8.2's prose quotes 23.49 exactly). All other
cells match to two decimals.

## Evaluation protocol (exact, from `reproduce.py ablation`)

- Chain column: the named checkpoint deployed under **mode `Z`** — measured
  self-mixer/detector activation + per-pass link noise + element jitter, but
  **without** the calibrated mixer-compression twin in the forward model —
  mean ± 1 s.d. over **repeats = 5** noise realizations of the full test set.
- Digital column: one deterministic draw under mode `ZD` (exact arithmetic,
  same detector activation, no noise) for rows 1–2 and 4–5, or mode `ZI`
  (ideal square law) for rows 3 and 6:
  `dig, ana = run(dmode, 1), run("Z", repeats)`.
- Cross-checked: `results_summary.json` tags `MZ_L3` (95.62 ± 0.09, repeats 5),
  `MZD_L3` (96.07, weights `MZ_L3`, mode `ZD`), `Z_L3` (86.26 ± 0.13),
  `ZD_L3` (88.29, weights `Z_L3`, mode `ZD`); and
  `weights/train_summary.json` accs `MZD_L3_nohw` 0.9603, `MZI_L3_ideal`
  0.9577, `ZD_L3_nohw` 0.89192, `ZI_L3_ideal` 0.85281 (the digital baselines).

## Caveat — SI row labels vs. what the code actually ablates

All numbers verify exactly, but the SI's "Training loss" labels do not match
the archive's own labels or the training code (`reproduce.py train_nohw`,
`comb_analog_sim.py forward()`):

- **Row 1 "twin + link noise"**: checkpoint `MZ_L3`/`Z_L3` was trained under
  mode `Z` — link noise + measured detector transfer in the loss, but **no
  mixer-compression device twin** (mode `Z` skips `comb_compress_mode`). It is
  a different checkpoint family from the deployed twin+link-noise networks of
  main-text Fig. 5d (`MZH_L3`/`ZH_L3`, mode `ZH`), whose digital/chain pairs
  are 94.30/93.90 (gap 0.40) and 88.83/87.53 (gap 1.30) — not 96.07/95.62
  (0.45) and 88.29/86.26 (2.03) as in this table.
- **Row 2 "link noise, no twin"**: checkpoint `MZD_L3_nohw`/`ZD_L3_nohw` was
  trained under mode `ZD`, which is **noise-free** (forward() adds no link
  noise for `ZD`) while keeping the measured detector activation. The archive
  labels it "no-hw, noise-free". So the row ablates *noise removed from the
  loss*, not *twin removed while keeping noise*.
- **Row 3 "ideal square law"**: accurate (`MZI_L3_ideal`/`ZI_L3_ideal`, mode
  `ZI`, ideal square law, no noise).

`MANIFEST.md` of the release itself calls this file the "**earlier
noise-only ablation**". See
`../../Code_availability/figure_scripts/Supplementary_Note_8/README.md` for the full resolution of
the Table 6 vs. main-text/Fig. 5d discrepancy.

## Provenance note (missing file in the shared folder)

`Data_availability/raw/fully_analog_simulation/results/` in this reproducibility archive
does **not** contain `hardware_aware_ablation.json`, although the archive's
`MANIFEST.md` lists it. The copy in `data/` was extracted from the canonical
released archive `miwen_fully_analog_archive.zip` (inside
`Manuscript/Data/V2/fully analog/files (1).zip`, entry
`code/results/hardware_aware_ablation.json`) and is byte-identical (SHA-256
`ad6330e32dcdc50f…`, 1012 bytes) to the copies in
`Manuscript/Overleaf/fig5_analog{,_v2,_v3}/data/`.
`verify_table6.py` prefers the shared `results/` folder when the file exists
there and falls back to `data/`, printing which source it used.
