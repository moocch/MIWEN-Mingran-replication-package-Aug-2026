# Digital vs MIWEN — the N=1200 test

Accuracy on the **N=1200 test**: 1,200 random official GTSRB test images
(frozen frame `share_20260712/battery_random1200_idx.npy`, regenerable via
`make_frames.py`), identical image set for every row.

## Definitions

The multiply–accumulate's departure from the ideal linear `ΣW·x` has two
physically distinct parts (the magnitude activation is a separate digital
step, not a MAC deviation):

- **Deterministic deviation** — reproducible, input-dependent (Bussgang-frame
  distortion / the transform T). Same input → same deviation.
- **Stochastic noise** — irreproducible (thermal/measurement).

Measured per layer (rel-RMSE), from the two runs (mean of the two = the
deterministic part; (run1−run2)/√2 = the stochastic part):

| layer | total deviation | deterministic | stochastic |
|---|---|---|---|
| conv1 | 0.207 | 0.201 | 0.070 |
| conv2 | 0.258 | 0.253 | 0.091 |
| dense1 | 0.076 | 0.079 | 0.060 |
| dense2 | 0.182 | 0.162 | 0.129 |

At the conv layers ~89% of the deviation power is deterministic.

## The key results

| result | weights | MAC | N=1200 accuracy | produced by |
|---|---|---|---|---|
| **Digital** (no-MIWEN) | own, clean-trained | exact linear | **99.50** | `results_grid.py` |
| **MIWEN measured** | hardware-aware | real: deterministic + stochastic | **98.83 ± 0.31** | `run_ladder_hw.py` → `bat_c*.npz` |
| deterministic only | hardware-aware | deterministic, no noise | 98.75 | `det_reconstruct.py` |
| reconstruction | hardware-aware | deterministic + stochastic (modelled) | 98.79 ± 0.06 | `det_reconstruct.py` |

- **System comparison**: Digital (own optimized weights) exceeds measured
  MIWEN by **0.67**. Given the task, the digital system is better.
- **Understanding check**: starting from the exact linear MAC on the same
  (hardware-aware) weights, adding the measured deterministic deviation and
  then the true stochastic noise reconstructs the measurement:

  linear MAC 98.42 → + deterministic → 98.75 → + stochastic noise →
  **98.79**, versus measured **98.83** (gap 0.04). We can build the MIWEN
  result up from linear + deterministic + noise. The deterministic deviation
  is the large, beneficial part; the stochastic noise is small.

  (The linear-MAC 98.42 here is only the zero-deviation anchor of this
  fixed-weights decomposition — hardware-aware weights on an ideal linear
  MAC — not a deployable system.)

## Other runs (not systems; completeness only)

- ns25 weights + Gaussian noise at the **full** deviation magnitude
  (0.20–0.23): 97.68. A pessimistic what-if that models the ~89%
  deterministic part as if it were noise — NOT a valid noise level; do not
  quote as "the noise effect." (`results_grid.py`)
- clean weights + full-magnitude noise: 96.17 — robustness illustration
  (clean weights are not noise-hardened). (`results_grid.py`)

## Audit index (result → script → data)

| number | script | reads | notes |
|---|---|---|---|
| 98.83 measured | `run_ladder_hw.py` (or audited `miwen_frozen_reference.py --hw`) | weights `r35_r3plus_ns25_s0_hw.npz`, frame `battery_random1200_idx.npy` | archives `bat_c1..10.npz`; slim `battery_slim.npz` |
| 98.42 linear anchor | `miwen_frozen_reference.forward_digital` | same weights + frame | = `battery_slim.npz['digital_preds']` |
| 98.75 / 98.79 | `det_reconstruct.py` | `bat_c*.npz` (run1) + `bat2_c*.npz` (run2) | model-free deterministic part |
| deviation table | `decompose_det_noise.py` | `bat_c*.npz`, `bat2_c*.npz` | run1 vs run2 |
| 99.50 digital | `results_grid.py` | clean weights `r35_r3plus_s0_hw.npz` | `forward_digital` |
| all-in-one | `consolidated_numbers.py` | the above | prints every labelled number |
| statistics | `bootstrap_ci.py`, `s2_adjudication_analysis.py` | run archives | paired CIs, McNemar |
| frames/integrity | `make_frames.py`, `audit_split.py` | cache + committed idx | regenerate + contamination |

## Caveat

The Digital 99.50 still applies the magnitude activation (a MIWEN-imposed
step, computed digitally); a conventional signed-activation digital CNN
reaches ~99.7–99.8 in the literature, so the digital advantage is at least
0.67 and likely larger. A signed-ReLU baseline was not trained here.
