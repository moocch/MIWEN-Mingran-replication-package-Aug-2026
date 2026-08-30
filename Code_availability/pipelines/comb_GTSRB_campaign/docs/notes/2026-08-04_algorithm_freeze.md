# ALGORITHM FREEZE — 2026-08-04 (exploration closed)

Frozen algorithm: **r3plus-ns25** — chosen for measured accuracy, the
certified hardware>digital property, and minimal auditable surface.
Weights: `share_20260712/r35_r3plus_ns25_s0_hw.npz` (sha256 prefix recorded
in the freeze commit). Nothing about the algorithm, weights, correction, or
protocol changes after this note; the full-battery session certifies THIS
configuration; later changes would require a new freeze note.

## Definition (complete)

- **Architecture** (4 layers, complex weights): conv 32@5×5×3 valid stride-1
  → magnitude → per-channel affine (folded BN) → clamp(≥0) → 2×2 max-pool →
  per-image max-norm → conv 64@5×5×32 (D=800 comb) → same inter-pass →
  dense 1600→128 → same (no affine on dense; clamp+max-norm) → dense
  128→43 → argmax of squared magnitudes.
- **Training** (`ladder_cnn_v2.py`): AdamW, cosine LR, 120 epochs, batch 96,
  standard geometric+brightness augmentation, label smoothing 0.05,
  val-split checkpoint selection, and additive magnitude noise at **25% of
  the measured per-layer rel-RMSE** (one scalar × [0.177,0.32,0.102,0.221])
  injected before BN — the entire "hardware-aware" content of the recipe.
- **Hardware protocol**: packed frames (`--images-per-frame 999`), run-4
  operating point (LO −3 dBm / RF −35 dBm), 1× (no averaging), quantizer
  off, occupancy claim before TX.
- **Correction stack — equalization only**: per-column scalar least-squares
  vs the digital reference (`fit_gj`, 5 lines) + Rice debias
  √(max(m²−σ²,0)). No shadow terms, no per-tone response, no β refits.

## Evidence (all pre-registered where marked)

| date | frame | measured | digital (same imgs) | b/c | note |
|---|---|---|---|---|---|
| 08-03 | 600–899 (frozen) | **99.00** | 97.67 | 0/4 | ΔT=+1.33 ≥ +0.7 ✓ pre-reg |
| 08-04 | random-450, seed 20260804 (frozen) | **98.44** | 97.56 | 1/5 | ΔT=+0.88 ≥ +0.7 ✓ pre-reg |

Family context (same chain, sibling configs): frames 0–299/300–599 (hwn
recipe) 98.33/98.33 with frozen-band confirmation; architecture and plan
variants mapped (r3fast, r3wide) including one negative arm attributed to
unpooled conv1 (audit 2026-08-04). **Pooled paired measured-vs-digital
across ALL six arms incl. the negative: b=5, c=22, exact p = 7.6×10⁻⁴
(N=1,500).** Absolute-accuracy claims carry frame-granularity error bars
per the 2026-08-04 stats resource plan; the ±0.3 claim awaits the battery.

## Why this one (selection criteria, declared)

(1) Only candidate with multi-frame pre-registered hardware confirmations;
(2) smallest auditable surface: ~350-line core, one-scalar noise recipe,
5-line correction; (3) conv2 on the D=800 plan where the beneficial
deterministic transform is strongest and replicated 5/5.

## Deliverables that follow the freeze

Reference single-file implementation + AUDIT.md + machine-checked
equivalence harness (reproduces archived digital refs bit-exactly and all
measured-npz decision chains); pipelined runner (host-compute overlap only,
RF timeline unchanged) gated on dry-run bit-identity + 15-min rig
equivalence pilot; then the ~4.5 h full battery (1,200 random paired
images) on THIS frozen configuration.

## Addendum 2026-08-05: paired bootstrap CIs (bootstrap_ci.py)

Battery arm alone: diff +0.42, 95% CI [−0.17, +1.00], P(diff≤0)=0.099 —
standalone-n.s. as pre-declared for the easy frame. **Tier-B disjoint-image
pool (n=2,550): diff = +0.86, 95% CI [+0.43, +1.33], P(diff≤0) < 5×10⁻⁵
(0/20,000 resamples)** — the bootstrap and the exact McNemar (p=9.8×10⁻⁵)
agree. Trial-count labels in two earlier posts corrected (3,000/2,550, not
2,700/1,500); b/c counts unchanged.

## Addendum 2026-08-05 (2): claim rescoped after the persistence repeat

Run 2 of the battery frame (persistence pre-spec 49b5597): c-set 7/9 —
pre-registered SUGGESTIVE band; the ≥8/9 CONFIRM was missed. Only 2 flips
in 1,200 (0.17%), both c-set members. RESCOPED CLAIM for all downstream
documents: the image-for-image deterministic form is NOT supported; the
certified form is a reliable AGGREGATE elevation of measured over digital
(7 consecutive positive arms across 5 days, no reversal; Tier-B exact
p≈1e-4; bootstrap CI [+0.43,+1.33]) carried by decision-boundary images
whose individual outcomes vary session-to-session. Run 3 (audited-file
vehicle, same frame) will measure the session variance of the elevation.

## Addendum 2026-08-05 (3): fair system-vs-system comparison (claim retraction)

Jonathan's methodological catch: all prior measured-vs-digital pairings use
the SAME (ns25) weights — a system-identity comparison isolating the chain
transform, NOT a fair system comparison. Fair comparison (each side trained
for its own deployment), battery frame N=1,200 same images: clean-trained
r3plus digital = 99.50; frozen measured = 98.83±0.31; paired b=9/c=1
(p~0.02) — the digital-optimal system wins by 0.67. RETRACTED: any
system-level "hardware ≥ digital" framing. RETAINED: (a) 98.83±0.31
absolute; (b) the same-weights elevation as a mechanism finding (conv2
transform; tiered p~1e-4); (c) "price of analog" ≈ 0.67 pt at matched
architecture on this frame. Issue #52 carries the full exchange
(reference-free digital confirmed; auditor's y_ref_l4 check explained as
the measured-trajectory hybrid).

## Addendum 2026-08-06 (4): energy-honest inference — frozen calibration

Mingran's code review (issue summary 2026-08-06) correctly showed the
inference path computed the full digital W@x per image per layer
(`build_frame_ml` line 290) to form the per-image equalization reference —
so those runs had the host perform the matrix operation the mixer is meant
to replace, invalidating them as evidence for the energy claim (C1). The
accuracy (C2) is unaffected (the equalization is decision-irrelevant:
removing it, or replacing it with a frozen constant, flips 0/1200
predictions). FIX: the equalization is a per-column calibration measured
ONCE, offline, on a disjoint set and frozen (`frozen_calibration.npz`,
derived from the random-450 frame, 0 image overlap with the battery test).
The deployment path (`miwen_frozen_reference.py --hw`,
`build_frame_ml(compute_digital=False)`) applies the frozen table and
computes NO host W@x. Offline validation on the battery archives: 98.83,
0/1200 flips vs the as-run result. A from-scratch hardware rerun with the
frozen code is the definitive demonstration (trajectory is genuinely
produced under frozen calibration). The `forward_digital` DIGITAL BASELINE
still computes W@x by design — it is the digital system, not the MIWEN path.
Refrozen; the frozen weights are unchanged, the correction stack changes
from per-image equalization to frozen calibration.

## Addendum 2026-08-06 (5): energy-honest HARDWARE rerun — result

From-scratch hardware battery via the frozen-calibration code
(`miwen_frozen_reference.py --hw`, `frozen_calibration.npz`, N=1200), run
overnight 2026-08-06→07: **98.92% ± 0.30, host computed NO W@x** (verified
in every chunk log). Versus the as-run per-image-equalized battery (98.83):
3/1200 flips — i.e. statistically identical, the difference is measurement
noise between two independent hardware runs. Versus the digital baseline on
the same images (98.42): b=4/c=10. Conclusion: the energy-claim PREMISE is
now demonstrated on hardware — the mixer performs the matrix operation and
the host computes no W@x — with accuracy fully preserved. (This restores the
premise; the energy ADVANTAGE still rests on the fJ/MAC analysis, unchanged.)
Artifact: `battery_frozen_slim.npz`.
