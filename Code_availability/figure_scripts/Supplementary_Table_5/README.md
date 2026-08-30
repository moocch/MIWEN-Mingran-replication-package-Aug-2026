# Supplementary Table 5 — hardware-aware 2×2 on the simulated fully analog chain

Verifies every cell of Supplementary Table 5 (`\label{tab:analog2x2}`,
Supplementary Note 8, S8.1 of `Supplementary_Information.tex`) against the
released fully-analog simulation archive.

## Files

| file | what |
|---|---|
| `verify_table5.py` | asserts all 16 table cells + 4 digital cross-checks + 4 protocol checks |
| `verify_table5_output.txt` | saved run output — **24 passed, 0 failed** |
| `data/twin_2x2.json` | the source JSON (see provenance below) |
| `data/twin_2x2.py` | the script that produced it (evaluation-protocol evidence) |

Run: `python verify_table5.py` (exit 0 = all cells verified).

## Cell → JSON mapping

Source: `twin_2x2.json` (produced by `twin_2x2.py`). Every cell is
`twin_2x2.json[<dataset>][<cell>]` with fields `acc` / `std` (percent,
rounded to 2 decimals):

| Table 5 cell | dataset key | cell key | value |
|---|---|---|---|
| MNIST clean, Digital | `mnist` | `clean_digital` | 95.77 ± 0.00 |
| MNIST clean, Under twin (forecast) | `mnist` | `clean_under_twin` | 18.25 |
| MNIST clean, Chain | `mnist` | `clean_hardware` | 15.77 ± 0.30 |
| MNIST twin-trained, Digital | `mnist` | `twin_digital` | 94.60 |
| MNIST twin-trained, Chain | `mnist` | `twin_hardware` | 92.62 ± 0.08 |
| GTSRB clean, Digital | `gtsrb` | `clean_digital` | 85.28 ± 0.00 |
| GTSRB clean, Under twin (forecast) | `gtsrb` | `clean_under_twin` | 8.65 |
| GTSRB clean, Chain | `gtsrb` | `clean_hardware` | 6.06 ± 0.08 |
| GTSRB twin-trained, Digital | `gtsrb` | `twin_digital` | 87.08 |
| GTSRB twin-trained, Chain | `gtsrb` | `twin_hardware` | 75.13 ± 0.18 |

## Which checkpoints and forward models (exact tags)

Per `twin_2x2.py` and the provenance rows `"Table I (MNIST)"` /
`"Table I (GTSRB)"` in `paper_numbers.json`:

- **clean arm** = checkpoint `MZC_L3` (MNIST) / `ZC_L3` (GTSRB), trained under
  mode `ZC` (ideal multiplier + ideal square law, no device model, no noise).
  - Digital column = its own model, mode `ZC`, deterministic (1 draw).
  - "Under twin (forecast)" = the same weights executed under mode `ZT`
    (calibrated device twin: mixer compression on the composite comb + measured
    detector transfer k_ED = 208.7 V⁻¹, **no** noise) — the software forecast.
  - Chain column = mode `ZH` (device twin **plus** per-pass link noise and
    element jitter), mean ± 1 s.d. over **5** noise realizations
    (`twin_2x2.py: main(..., repeats=5)`, `acc("ZH", ds, w, repeats=repeats)`).
- **twin-trained arm** = checkpoint `MZT_L3` / `ZT_L3`, trained under mode `ZT`
  (device twin in the loss, no noise injection, no bias correction).
  - Digital column = mode `ZT` (its own model, deterministic).
  - Chain column = mode `ZH`, 5 noise realizations.

Digital cells were additionally cross-checked against the training-time
accuracies in `fully_analog_simulation/weights/train_summary.json`
(tags `MZC_L3` = 0.9577, `MZT_L3` = 0.9460, `ZC_L3` = 0.85281,
`ZT_L3` = 0.87078) — all match after ×100 and 2-decimal rounding.

## Provenance note (missing file in the shared folder)

`Data_availability/raw/fully_analog_simulation/results/` in this reproducibility archive
does **not** contain `twin_2x2.json`, although the archive's own
`MANIFEST.md` lists it (and `twin_2x2.py` is likewise absent). The copies in
`data/` here were extracted from the canonical released archive
`miwen_fully_analog_archive.zip` (inside
`Manuscript/Data/V2/fully analog/files (1).zip`, entries
`code/results/twin_2x2.json` and `code/twin_2x2.py`) and are byte-identical
(SHA-256) to the copies in
`Manuscript/Overleaf/fig5_analog{,_v2,_v3}/data/`:

- `twin_2x2.json` — sha256 `bee2c1a6867b7375…` (608 bytes)
- `twin_2x2.py` — sha256 `f98ab5c1c86bd202…` (2879 bytes)

`verify_table5.py` prefers the shared `results/` folder when the file exists
there and falls back to `data/` otherwise, printing which source it used.
