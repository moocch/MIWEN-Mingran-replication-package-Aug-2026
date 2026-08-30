# Serial 2x2 at (LO -10, RF -24) dBm — pre-spec (2026-08-24, frozen at commit)

## Decision (Jonathan, 2026-08-24)
Operating point (LO -10, RF -24): LO at the -10 floor, RF 14 dB below
(working-group RF<<LO geometry), ~250x client-RF energy vs (0,0).
Clean arm runs first (overnight); twin arm retrains offline through the
FULL-SPAN twin surface and runs after. Same frozen battery frame
images 0-299; chain unchanged (bare input ports, 10 dB IF pads, RX gain
0); no pads added anywhere.

## Pinned predictions (committed BEFORE hardware)
- clean @ (-10,-24), N=300 frame: **83.00%** (full-span surface +
  protocol gain cal — the pipeline that reproduces measured (0,0)
  hardware: clean 6.33 = 6.33 measured, sprint-twin 98.67 vs 97.67).
  Noise expectation ~1-3 pts below pin (thermal <=~3%/product at this
  sum; no debias). Flanking pins: (-12,-24) 88.00, (-10,-30) 84.00.
- twin (retrained at point, 15-epoch sprint per the disclosed (0,0)
  precedent): pin = the trainer's selected-checkpoint test-N300-frame
  under its own shifted twin, recorded in train log BEFORE the twin
  hardware session. Expectation ~98-99 (the (0,0) sprint reached 98.33
  under a harsher curve).
- Context pins (already committed): full drive-sweep tables in campaign
  record; role inversion at low drive; LO-port-binding law.

## Verdict bands (pre-registered)
- clean: CONFIRM no-breakdown if >= 70; COLLAPSE-CLASS if <= 20
  (constant-classifier range); 20-70 = analyze before claiming.
- twin: CONFIRM if within 2.5 pts of its recorded pin.
- Both-arms headline requires both CONFIRMs.

## Code deltas (everything else byte-identical to the frozen files)
serial_nn_runner_m10m24.py vs frozen serial_nn_runner.py — 6 changes:
1. POWER = (-24.0, -10.0)   [(RF, LO) tuple order]
2. PACK_G_REF scaled by 10^(-34/20) -> 0.002312 (product gain at the
   new point); PACK_G_TOL unchanged (0.20)
3. PACK_RESID_BAND lower edge 0.30 -> 0.05 (the linear point's
   per-product residual is expected ~0.2; upper edge = corruption
   detector, unchanged)
4. cal path -> serial_cal_{arm}_m10m24_20260824.npz (prevents silent
   reload of the (0,0) frozen calibration present on the rig)
5. results -> serial_nn_{arm}_{i0}_{i1}_m10m24_20260824.npz
6. twin-arm cal reference -> serial_twin_model_fullspan.json evaluated
   with drive shift (pl-10, pr-24), ridges clamped to [-50,+10]
   (bit-identical to the validated pin machinery); twin weights ->
   serial_twin_m10m24_s0_hw.npz (tonight's retrain)
train_serial_twin_m10m24.py vs train_serial_twin_short.py: same
surface/shift/clamp substitution inside _pdb_np (table build), output
names namespaced. No recipe changes (15 epochs, no noise, mag readout,
val_frac 0.1, seed 0).

## Full-span twin surface (serial_twin_model_fullspan.json)
Joint fit: USRP map product region (520 cells) + stationA per-product
ladder (2531 pts to -38 dBm); knees LO +4.22 / RF +9.07; held-out map
0.273 dB, station 0.76-1.40 dB/drive; ridges flat (+7.3-7.9 dB) from
-40 to -60 (extrapolation pedestal eliminated by data); clamp -50.
Network-validated at (0,0) both arms. Fit script: twin_fullspan_fit.py.

## Session mechanics
- Rig claim 84fb3cc856122ebd (held); RIG_SESSION_TOKEN exported in the
  launch shell (chunk-1 lesson). Launch: serial_nn_runner_m10m24.py
  clean 0 300 (chunk-saved every 10, stop-anywhere). ~20-25 min cal +
  ~102 s/img => ~9 h.
- Twin training in parallel on the local 2070 (~3 h); twin hardware
  session fires only after (a) clean completes, (b) training's pinned
  N300 number is recorded here, (c) sanity check of the val curve.
- Safety: commanded (-24,-10) is >=14 dB below every prior operating
  point; sync v2 relative scaling => port peaks far under the +14 dBm
  rule. Gates fail closed.

## Addendum 1 (23:40, pre-data): chain-gate halt and resolution
First launch halted fail-closed at the CHAIN GATE: capture peak 0.074 FS
vs the (0,0)-era band (0.1, 0.7) — expected physics at -34 dB product
power (sync lock healthy, pm=3891; hardware released cleanly; no data
taken). Resolution (delta 7): rx_gain 0 -> 12.0 dB, restoring capture
level to ~0.29 FS (inside the ORIGINAL gate band, which is retained
unchanged); RX gain is absorbed exactly by the per-layer scalar
calibration (measured step fidelity <=0.06 dB). Knock-on: PACK_G_REF
recomputed to 0.1155x10^((-34+12)/20) = 0.00919. No other changes.

## Addendum 2 (23:55, pre-data): packtest |g| re-anchored, twin-derived
Attempt 2: chain gate PASSED as predicted (peak 0.284 vs ~0.29; resid
0.207 = clean linear decode); halted fail-closed on packtest |g|
0.0327 vs naively-scaled ref 0.00917. Root cause: the (0,0) reference
was measured in the COMPRESSED regime; bilinear -34 dB scaling
understates the linear-regime gain. Offline reproduction of the
packtest (same rng seed 11) through the full-span twin predicts the
measured value to -0.10 dB (0.03308). PACK_G_REF -> 0.03308 (tol
unchanged). Doubles as hardware validation of the full-span surface:
chain-gain ratio across 34 dB of operating-point change predicted to
0.1 dB. No data taken in attempts 1-2.

## Addendum 3 (02:20, pre-twin-hardware): twin arm pin RECORDED
Retrained twin arm (15-ep sprint through the shifted full-span twin;
train_serial_twin_m10m24.log; val-under-twin 99.92; weights
serial_twin_m10m24_s0_hw.npz sha256[:16] dd553cf4ee7ab053, shipped to
rig): battery-frame N=300 under its own twin = **99.00%** (first-50
subset 98.00). Per the verdict bands: twin CONFIRM if hardware within
2.5 pts (>= 96.5). Clean arm mid-evaluation at this writing; its
measured cal residuals 0.217/0.291/0.137/0.320 vs noiseless pins
0.188/0.178/0.115/0.643 (L4 half the pinned deviation).

## RESULT 1 (2026-08-25 early morning): CLEAN ARM CONFIRM
Measured clean @ (LO-10,RF-24), N=300 battery frame: **81.67%** vs pin
83.00 (Δ = 4 images; noiseless-pin gap inside the 1-3 pt noise
expectation; CONFIRM band was >=70). Independently recounted from the
pulled npz (81.67%, meta power [-24,-10] confirmed); 38 distinct
classes predicted (a functioning classifier — vs 1-class collapse at
(0,0)). Cal residuals 0.217/0.291/0.137/0.320. The no-breakdown
demonstration at the working-group geometry (LO at the -10 floor,
RF<<LO) stands, forecast-first. Artifacts:
serial_nn_clean_0_300_m10m24_20260824.npz, serial_cal_clean_m10m24_
20260824.npz, serial_clean_m10m24_20260824c.log. Twin session (pin
99.00, CONFIRM >=96.5) launched next on the same chain/cal protocol.

## Addendum 4 (pre-twin-FINAL): interpretive bracket recorded
Twin cal L1 measured 0.511 vs own reference. Adjudicated: runner
reference == training forward (1e-4); the surface's own L1 deviation-
from-ideal on the twin weights is 0.448; quadrature with hw-vs-ideal
~0.22 gives 0.498 ~= measured — consistent with the twin's L1
structure being ORTHOGONAL to the hardware's actual deviation (had it
contained it, expected ~0.40). Bracket forecast (computed BEFORE the
twin FINAL): twin arm under an IDEAL channel with protocol cal =
10.67% N300; self-consistent pin = 99.00. Discrimination via cal
L2-L4: surface-right hypothesis ~[0.28+, small, 0.17+] vs ideal-like
hypothesis quadratures ~[0.38, 0.16, 0.36]. Decision rule: session
continues (protocol has no cal gate); if the 30-image running accuracy
shows collapse-class behavior, notify Jonathan (abort is his call).
Either FINAL is a clean verdict: ~99 validates the full-span surface
as channel model on trained weights; near-bracket-low falsifies its
deviation structure at deep-linear drive — alongside the banked clean
CONFIRM either way.

## RESULT 2 (2026-08-25 morning): TWIN ARM FALSIFIES THE SURFACE — ABORTED
Twin arm @ (LO-10,RF-24), aborted at Jonathan's direction after N=20:
running accuracy 5.00% (10 distinct classes predicted, top class 25 at
9/20 — feature-scrambling degradation, not the single-class constant of
the (0,0) clean collapse). Landed at the LOW end of the pre-committed
bracket (ideal-channel 10.67 vs self-consistent pin 99.00): the
full-span v1 surface's early-layer deviation structure is NOT in the
real channel at deep-linear drive — the compensation it taught was
phantom (cal fingerprint L1-L3 matched the ideal-like hypothesis;
L4 partially aligned). Verdict per bands: twin NOT CONFIRMED; surface
falsified as channel model below its data floor (-38 dBm). The clean
CONFIRM (RESULT 1) is unaffected. Recovery path (directed by Jonathan):
low-drive station-B ladder -> K=30 refit on map+stationA+stationB
(low RMSE at high power retained — one surface for all points) ->
retrain -> re-run twin arm. Partial artifacts committed.

## Fresh-segment replication (2026-08-25 eve, Jonathan-directed): digital pins COMMITTED PRE-HARDWARE
Plan: N=300 fresh images (battery frame 300-599) per arm, byte-frozen
runners, frozen calibrations reloaded. Digital comparators on images
300-599 (frozen conventions; per-image preds in
digital_pins_seg300600.npz):
- clean ideal digital: 100.00% (an easier segment than 0-299's 99.50)
- (0,0) twin-under-twin: 98.67%   [run 1 comparator]
- (0,0) clean-under-twin: 1.00%   [run 3 comparator]
- (-10,-24) clean forecast (validated pipeline): 85.67%  [run 2]
Run 1 ((0,0) twin 300-600) launched with all gates passed (chain
0.332/0.335; packtest 0.1173/0.1155, 7th consistent session; frozen
cal reloaded). Runs 2-3 auto-follow.

## RESULT 3 (2026-08-26): (0,0) twin arm, fresh images 300-599 = 99.33
FINAL 99.33 vs segment pin 98.67 (measured beats digital by 2 images —
the certified-era elevation direction, replicated on a disjoint
segment). Recount-verified from the pulled npz (99.33; 40 classes;
99.00% per-image agreement with the digital comparator). Pooled (0,0)
twin arm: 591/600 = 98.50 (N=600, two disjoint segments). Gates:
chain 0.332/0.335, packtest 0.1173 (7th consistent session), frozen
cal reloaded. Artifacts: serial_nn_twin_300_600_20260823.npz + session
log.

## RESULT 4 (2026-08-26): (-10,-24) clean, fresh images 300-599 = 82.00
FINAL 82.00 vs segment pin 85.67 (noiseless; measured sits at the
lower edge of the 1-3 pt noise haircut). THE REPLICATION: 81.67
(images 0-299) vs 82.00 (300-599) — segment agreement within one
image. Pooled (-10,-24) clean: 491/600 = 81.83 (N=600). Recount
verified (82.00; 37 classes; 80.0% per-image agreement with the
forecast preds). Gates: chain 0.278/~0.29, packtest 0.0325/0.03308
(0.4%), frozen cal reloaded.

## RESULT 5 (2026-08-26): (0,0) clean, fresh images 300-599 = 5.00 — CAMPAIGN EXTENSION COMPLETE
Collapse replicated on the disjoint segment (5.00 vs first-segment
6.33; segment pin 1.00; constant-classifier fingerprint identical:
297/300 predictions = class 12, 3 classes total). FINAL POOLED TABLE
(N=600 per arm, two disjoint 300-image segments, all pins committed
pre-hardware):
- (0,0) clean:  6.33 / 5.00  -> pooled 5.67   (digital 99.50/100.00)
- (0,0) twin:  97.67 / 99.33 -> pooled 98.50  (under-twin 98.33/98.67)
- (-10,-24) clean: 81.67 / 82.00 -> pooled 81.83 (forecast 83.00/85.67)
Eight consecutive sessions with consistent gates (packtest |g|
0.1155-0.1173 at (0,0); 0.0325-0.0327 twin-derived at (-10,-24)).
