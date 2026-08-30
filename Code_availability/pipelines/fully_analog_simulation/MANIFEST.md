# Fully analog inter-layer recirculation — simulation archive

Everything that backs the manuscript section `analog_section.tex`.
No GPU needed; inference ~2 min, full re-train ~30 min on 8 CPU cores.

## Entry points

| command | what it does | writes |
|---|---|---|
| `python3 reproduce.py budget`  | level plan + per-pass SNR of the chain | `results/link_budget.json` |
| `python3 reproduce.py infer`   | loads the stored weights, runs inference (5 noise draws) | `results/results_summary.{json,csv}` |
| `python3 reproduce.py train`   | re-trains every network | `weights/ckpt_*.npz` |
| `python3 reproduce.py train_nohw` | re-trains the two no-hardware-aware ablations | `weights/ckpt_*_nohw.npz`, `*_ideal.npz` |
| `python3 twin_2x2.py`          | {clean, twin} x {digital, hardware} table + pinned forecast | `results/twin_2x2.json` |
| `python3 make_paper_numbers.py`| every number quoted in the section, with its provenance | `results/paper_numbers.{json,csv}` |
| `python3 make_reference_figures.py` | PNG versions of Figs. 3-4 (the manuscript draws them in TikZ) | `results/slide_*.png` |
| `python3 prepare_gtsrb.py ...` | rebuilds `data/gtsrb_roi_32x32.npz` from the official archives | `data/` |

## Forward models (the `mode` field everywhere)

| mode | multiplier / activation | device twin | link noise | used for |
|---|---|---|---|---|
| `ZH` | comb mixing, self-mixer | yes | yes | the analog chain — deployment |
| `ZT` | comb mixing, self-mixer | yes | no  | digital execution of the same weights; twin arm of the 2x2 |
| `ZC` | ideal multiplier, ideal square law | no | no | clean arm of the 2x2 |
| `D`  | Eq. (recirc): magnitude + max-norm | — | yes | digital recirculation protocol (main text) |
| `F`  | float reference | no | no | initialisation / upper bound |
| `Ha` | passive levels, no activation | yes | yes | function-class control |
| `O`  | linear cascade, drive restored | yes | yes | function-class control |

The device twin = the mixer's calibrated large-signal compression evaluated on
the composite comb at the planned drive of each pass, plus the measured
detector transfer (k_ED = 208.7 /V).  Twin arms use **no noise injection and
no bias correction**, following the serial hardware-aware campaign.

## Files

```
analog_physics.py        calibrated device models (mixer twin, detector table)
comb_analog_sim.py       comb model, level planner, forward models, training
reproduce.py             budget / train / train_nohw / infer / ablation
twin_2x2.py              the hardware-aware 2x2 (clean vs twin)
make_paper_numbers.py    provenance table for every quoted number
make_reference_figures.py  PNG twins of the manuscript figures
prepare_gtsrb.py         rebuild the GTSRB ROI cache

data/    mnist.npz                 60k/10k, 784 floats in [0,1]
         gtsrb_roi_32x32.npz       ROI-cropped 32x32 RGB, 43 classes
weights/ ckpt_<tag>.npz            one per trained network (see below)
         train_summary.json        training-time accuracies
results/ link_budget.json          drives, SNRs, block efficiency
         results_summary.{json,csv} paired digital/analog inference
         twin_2x2.json             hardware-aware ablation
         hardware_aware_ablation.json  earlier noise-only ablation
         paper_numbers.{json,csv}  every number in the section + provenance
```

## Weight tags

`M` prefix = MNIST, no prefix = GTSRB; `_L3` = three passes.

| tag | trained under | role |
|---|---|---|
| `MZH_L3`, `ZH_L3` (also `_L2`, `_L4`) | `ZH` | headline networks; Fig. 4 |
| `MZT_L3`, `ZT_L3` | `ZT` | twin arm of Table I (no noise in the loss) |
| `MZC_L3`, `ZC_L3` | `ZC` | clean arm of Table I |
| `MZD_L3_nohw`, `ZD_L3_nohw` | `ZD` | ablation: noise-free training |
| `MZI_L3_ideal`, `ZI_L3_ideal` | `ZI` | ablation: idealized square law |
| `MD_L*`, `D_L*` | `D` | digital recirculation protocol |
| `MF_L*`, `F_L*` | `F` | floating-point references / init |
| `MHa_L3`, `H_L3`, `MO_L3`, `O_L3` | `Ha` / `H` / `O` | function-class controls |
