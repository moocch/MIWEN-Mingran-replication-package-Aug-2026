# AUDIT.md — auditor's guide to the frozen MIWEN GTSRB algorithm

Everything referenced is on branch `handoff/rung3-session`; the frozen state
is tagged `algorithm-freeze-20260804`.

## 1. Quick start (three machine checks before reading anything)

```bash
cd share_20260712
python3 miwen_frozen_reference.py --verify   # sha + digital refs bit-exact +
                                             # every hardware run replays
python3 make_frames.py                       # eval frames regenerate from seeds
sha256sum r35_r3plus_ns25_s0_hw.npz          # 9be6e085f2d12e19...
```

## 2. File map (what to read, in order)

The `share_20260712/` directory also contains ~50 scripts and ~60 data files
from the full campaign history. The AUDIT-RELEVANT set is exactly the files
listed here plus the two notes in §2a; everything else is historical
(exploration arms, earlier MLP work, superseded trainers) and is not part of
the frozen result.

| file | role |
|---|---|
| `docs/notes/2026-08-04_algorithm_freeze.md` | freeze declaration: definition, evidence, addenda incl. the 2026-08-05 claim rescope |
| `docs/notes/2026-08-06_digital_vs_miwen_table.md` | **the headline results** (N=1200): Digital 99.50 vs MIWEN 98.83, the deterministic/stochastic decomposition, the reconstruction, and a result→script→data audit index |
| `share_20260712/miwen_frozen_reference.py` | **the algorithm** — single file, numpy-only; also `--dry-run`/`--hw` execution modes (bit-identity-gated vs the production runner) |
| `share_20260712/r35_r3plus_ns25_s0_hw.npz` | **the model**: `arch_json` (architecture) + all parameters (complex conv kernels `c{i}r/i`, folded-BN affine `c{i}s/b`, dense `d{j}r/i`) |
| `share_20260712/ladder_cnn_v2.py` | training-time definition: `ARCHS["r3plus"]`, train recipe, noise hook, `export_runner` (BN folding) |
| `share_20260712/4_gtsrb_confusion_mlpN.py` | instrument layer (OFDM frame synthesis, UHD/GNU Radio session, capture, decode). Auditable separately; the reference file consumes its magnitudes |
| `docs/plans/2026-08-0*.md` | frozen pre-specs (battery, persistence repeat) with pre-declared endpoints |

### 2a. Result → script → data (the audit-relevant analysis set)

| number | script | data it reads |
|---|---|---|
| MIWEN measured 98.83 | `run_ladder_hw.py` / `miwen_frozen_reference.py --hw` | weights + `battery_random1200_idx.npy` → `bat_c1..10.npz` (slim `battery_slim.npz`) |
| Digital 99.50 | `results_grid.py` | clean weights `r35_r3plus_s0_hw.npz` |
| det-only 98.75 & reconstruction 98.79 | `det_reconstruct.py` | `bat_c*.npz` (run1) + `bat2_c*.npz` (run2) |
| deviation decomposition | `decompose_det_noise.py` | run1 + run2 archives |
| every labelled number at once | `consolidated_numbers.py` | all of the above |
| frames / contamination | `make_frames.py`, `audit_split.py` | cache + committed idx |
| paired statistics | `bootstrap_ci.py`, `s2_adjudication_analysis.py`, `s2_layer_attribution.py`, `persistence_analysis.py` | run archives |

## 3. The algorithm in one paragraph

Input image (official GTSRB test, 32×32 RGB) → per-image 2–98 percentile
contrast stretch → four layers, each: analog matrix–vector multiply (vector
on the RF comb, weights on the LO comb; conv layers stream im2col patches
against a held kernel bank) → magnitude readout |y| → **Rice debias**
√(max(m²−σ²,0)) with σ² measured from empty bins → **equalization**: divide
each output column by g_j = Σ(m·r)/Σ(r²), a per-column least-squares gain
against the digital reference (label-free; 43–128 scalars/layer; THE entire
correction stack) → **folded-BN affine** s·a+t (see §4) → clamp at 0 →
2×2 max-pool (conv layers) → per-image max-norm → next layer. Decision:
argmax of squared final magnitudes. Digital reference = identical code with
the analog MVM replaced by exact |X·Wᵀ| (equalization then ≡ identity).

## 4. BN folding (why "BatchNorm" appears nowhere at inference)

Training uses BatchNorm per conv channel: BN(a) = γ(a−μ)/√(σ²+ε)+β. At
inference μ,σ²,γ,β are constants, so BN collapses to an affine with
**scale s = γ/√(σ²+ε)** and **shift t = β−μ·s**, precomputed by
`ladder_cnn_v2.export_runner` and stored as `c{i}s`/`c{i}b`. Runtime cost:
one multiply-add per channel, in the already-digital inter-pass step.

## 5. Training provenance (one scalar of "hardware-awareness")

Standard recipe (AdamW, cosine LR, 120 epochs, geometric+brightness
augmentation, label smoothing 0.05, val-split checkpoint selection) plus
additive Gaussian magnitude noise at **25% of the measured per-layer
rel-RMSE** — the single hardware-aware ingredient. Exact invocation in the
freeze note.

## 6. Claims (rescoped 2026-08-05) and where each is proven

1. **Measured accuracy 98.83 ± 0.31** (N=1,200 frozen random frame, seed
   20260805; binomial σ). Artifacts: `battery_slim.npz` + chunk logs.
2. **Aggregate measured-over-digital elevation** — paired, same images:
   Tier A (frozen algorithm, disjoint frames) b=5/c=18, exact p=5.3×10⁻³;
   Tier B (+ sibling recipe, disjoint) b=6/c=28, **p=9.8×10⁻⁵**, bootstrap
   CI [+0.43,+1.33]; Tier C (everything incl. the negative r3wide arm)
   b=9/c=31, p=3.4×10⁻⁴. Seven consecutive arms positive, five days, no
   reversal. Mechanism attribution: the elevation enters at conv2
   (`s2_layer_attribution.py`); alternatives excluded
   (`s2_adjudication_analysis.py`).
3. **Fair system comparison (2026-08-05, issue #52)**: the pairings above
   use the SAME weights on both sides — they isolate the chain transform,
   not system superiority. Each-side-trained-for-itself on the battery
   frame: clean-digital 99.50 vs measured 98.83 (b=9/c=1) — **the digital
   system wins by 0.67**; "hardware ≥ digital" at system level is
   explicitly NOT claimed. The honest system statement: the price of
   analog at matched architecture is ~0.67 pt on this frame.
4. **What we do NOT claim**: image-for-image determinism of the elevation.
   The pre-registered persistence repeat returned 7/9 (SUGGESTIVE band,
   p=0.09): global flip rate 0.17%/session, but flips concentrate in the
   boundary images the transform acts on. The elevation is a reliable
   aggregate whose per-image realization varies session-to-session.

## 7. Didactic references (math relevant to the audited architecture)

- `docs/notes/conv_on_miwen_20260731.pdf` — how a convolutional layer is
  realized on this hardware: im2col patch-streaming, kernel bank held on the
  LO comb, per-symbol MVM slices. Directly describes both conv layers of
  the frozen network.
- `docs/notes/2026-08-03_conv_layer_implementation.pdf` — companion
  implementation note: comb plan, frame build, decode, and the digital
  inter-pass steps, i.e., the pipeline `miwen_frozen_reference.py` encodes.
- `docs/notes/rung3_algorithm_20260730.pdf` — the full rung-3 correction
  mathematics. CAVEAT for this audit: the frozen algorithm uses only its
  §equalization (the per-column least-squares gain, `fit_gj`) and the Rice
  debias; the physics terms defined there (per-tone response h, LO-clip
  Bussgang shadow ΔW, the β refit machinery) belong to the earlier MLP
  arms and are NOT part of the frozen chain — audit them only for
  historical claims, not for the certified configuration.

## 8. Data integrity

Official GTSRB split (39,209/12,630; track-disjoint by construction).
`audit_split.py`: 8/12,630 exact 32×32 duplicates test-in-train, none in
any evaluation frame. Frames drawn by committed seeds from the never-
measured pool; all eval images disjoint across arms except the deliberate
same-frame repeat.

## 9. Known caveats (disclosed, none label-leaking)

- Checkpoint selection used a random train/val split (GTSRB tracks make it
  leaky) → selection ≈ late-epoch choice; test evaluated once per model.
- Training-noise scalars calibrated from hardware runs of test images
  0–299 (label-free; all certified frames disjoint from 0–299).
- Equalization fits on the evaluation run's own data by protocol
  (label-free per-column scalars; half-run stability ≤1% median).
- Ops: chunk-1 startup failures at session boundaries occurred twice
  (UHD-release transient; foreign-claim fail-closed) — both logged on
  issue #51; retried cleanly.
- Run 3 (same frame via `miwen_frozen_reference.py --hw`) is staged to put
  the certification on this audited file and measure session variance.
