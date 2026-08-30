# Issue #52: Blocking Q for 08/06 DARPA review: was the frozen 97.67 digital reference computed reference-free? (99.00% itself verifies clean)

- **Author**: dirkenglund  **Opened**: 2026-08-05T23:46:50Z  **State**: closed

## Original post

@jon-morag — one blocking question before the DARPA TOPCHIP review tomorrow (2026-08-06). Short version: **your 99.00% verifies cleanly**; I only need to know how the *digital reference* it is compared against was computed.

## What I verified independently

Pulled `share_20260712/s3_arm1_ns25.npz` from `origin/handoff/rung3-session` and recomputed from the stored per-image predictions:

| check | result |
|---|---|
| accuracy recomputed from `preds`/`labels`/`keep_mask` | **99.00%**, 3 errors in 300 — matches the stored `accuracy` exactly |
| `img_index` range | **600–899** — the fresh frame you claimed, no image reuse |
| `meta_json` | `dry_run: False`, real `tx_args`/`rx_args` USRP addresses → genuine hardware run |
| `sim_mixer: ideal` | benign — it is only referenced inside `if args.dry_run:` branches in `run_ladder_hw.py`, so it is an unused argparse default, not a simulation |

So the hardware result stands on its own. No concern there.

## The question

The claim I could **not** reproduce is the comparison — `ΔT = +1.33` against a frozen digital reference of 97.67.

Using the artifact's own in-file reference `y_ref_l4`:

```
measured (argmax |y_mag_l4|) : 99.00%
digital  (argmax |y_ref_l4|) : 99.00%
Δ = 0.00 pts, McNemar discordants b = 0, c = 0   (identical predictions, same 3 errors)
```

That is consistent with the two paths not being independent. In `run_ladder_hw.py` (lines 138, 249, 281) each layer does:

```python
am_eq = am_raw / fit_gj(am_raw, ar)[None, :]
```

and `fit_gj` (`rung3_correction.py:54`) is a per-column least-squares gain of **measured against the digital reference**, fitted on the evaluation batch:

```python
num = np.sum(am * ar, axis=0)
den = np.maximum(np.sum(ar * ar, axis=0), 1e-30)
g = num / den
```

There is also the per-column complex alignment `a = Σ conj(Y)·Yref / Σ|Y|²` applied to `Y` before summing in the split path.

So the measured activations are normalized toward the per-layer digital reference at inference time. To be clear about what this does and does not imply: `ar` carries no label information, so this cannot trivially inflate accuracy. But it does mean measured and reference are coupled, and a measured-vs-digital *accuracy* comparison is only meaningful if the digital comparator is computed independently of that equalization.

I saw your note that eq circularity was ruled out via gains being half-run-stable to ≤1% median. That establishes the gains are **stable**; I do not think it establishes that the comparison is **independent**, which is the specific thing the +1.33 claim needs.

## What would resolve it (one line is enough)

**Was the frozen 97.67 digital reference computed reference-free** — i.e. a pure end-to-end numpy chain with no `fit_gj` equalization and no `Yref`-derived alignment anywhere in its path?

- **If yes** → the +1.33 stands, the coupling is confined to the measured path, and I will put the claim back on the slide as you reported it.
- **If no** (e.g. it came through the same runner/equalization) → the comparison needs a reference-free evaluation before it can be claimed.

## What we have done in the meantime

For tomorrow's deck we have **not** dropped your result — we present the verified 99.00% (3/300 on fresh frame 600–899) as the headline, and carry measured-vs-digital as a disclosed open item with the mechanism named. That is a defensible position either way; your answer just tells us whether to restore the stronger claim.

Deck slide (breakout copy): "Results: what each rung bought".

cc @dirkenglund — reviewing 2026-08-06.

---


## Comments

### jon-morag — 2026-08-06T00:08:00Z

The 99% figure is slightly stale. I froze the algorithm and ran a battery
on N=1200 random images in the past two days, achieving 98.8+/-0.3%
accuracy.

Separately, I ran inference on the exact same ML algorithm off-line
digitally (no-MIWEN) on the same 1200 images, and got 98.4% on the digital.

However, I believe I found an important logical gap. I used the exact same
weights for both inferences. I should actually train the digital model to
get its own weights. The digital will probably perform better after I do
this.

On Wed, 5 Aug 2026 at 19:47 Dirk R Englund ***@***.***> wrote:

> Assigned #52 <https://github.com/QPG-MIT/MIWEN_Mingran/issues/52> to
> @jon-morag <https://github.com/jon-morag>.
>
> —
> Reply to this email directly, view it on GitHub
> <https://github.com/QPG-MIT/MIWEN_Mingran/issues/52?email_source=notifications&email_token=BAOYGAAHRP5TLWWE3YTWBYT5IPBQBA5CNFSNUABQM5UWIORPF5TWS5BNNB2WEL2JONZXKZKFOZSW45CON52GSZTJMNQXI2LPNYXTEOJQGM4TKNJYGMYTTJTSMVQXG33OUZQXG43JM5XKKZLWMVXHJLDGN5XXIZLSL5RWY2LDNM#event-29039558319>,
> or unsubscribe
> <https://github.com/notifications/unsubscribe-auth/BAOYGAGVLWLLXYNZO4ALKRT5IPBQBAVCNFSNUABGKJSXA33TNF2G64TZHMYTCOBRGIZTCMBWGI5US43TOVSTWNJQG43DCOBWGQ4DPILWAI>
> .
> You are receiving this because you were assigned.Message ID:
> ***@***.***>
>


---

### jon-morag — 2026-08-06T00:18:22Z

Answering the blocking question first, then a material update that should shape the slide.

## 1. Yes — the digital reference is reference-free

The frozen 97.67 (and every digital reference in the record) is computed by a pure end-to-end numpy chain: exact `|X·Wᵀ|` per layer, digital recirculation, **no `fit_gj`, no `Yref`, no measured data anywhere in its path**. It is now enshrined as `forward_digital` in `share_20260712/miwen_frozen_reference.py` (single-file audit implementation) — grep it: no equalization call exists. `python3 miwen_frozen_reference.py --verify` recomputes 97.67 on frame 600–899 (plus all other frozen references bit-exact) from weights + images alone. Re-verified live just now.

## 2. Your `y_ref_l4` computation — correct observation, different quantity

`y_ref_l4` in the artifact is a **hybrid**: the digital L4 product applied to the *measured* chain's recirculated L1–L3 activations (each layer's `fr.y_digital` is built from the measured previous layer's output). It therefore shares the measured trajectory and matches the measured predictions by construction — your b=c=0 is expected and is not evidence of coupling in the *comparison*, which uses the independent chain of §1. The per-layer hybrids are actually what our layer-attribution analysis exploits deliberately (`s2_layer_attribution.py`). Also: the complex alignment in the split path you flagged was never used in any certified run (tone-split was bounded out on 08-03; certified arms are eq-only, serial path).

## 3. Material update for the slide (Jonathan's methodological catch, quantified today)

Jonathan identified that ΔT compares measured vs digital **at the same (hardware-noise-trained) weights** — a system-identity comparison, not a system-vs-system one. The fair system comparison (each side trained for its own deployment) is now computed on the battery frame (N=1,200, same images):

| system | training | execution | accuracy |
|---|---|---|---|
| digital-optimal | clean, own weights | digital | **99.50** |
| hardware-optimal (frozen) | ns25 | **measured** | **98.83 ± 0.31** |
| same-weights digital | ns25 | digital | 98.42 |

Paired, digital-optimal vs hardware-optimal: b=9, c=1 (p≈0.02) — **the digital system wins the fair comparison by 0.67 pts**. Recommended slide framing, in order of defensibility: (a) headline 98.83 ± 0.31 measured at 1×, unsigned, N=1,200 pre-registered — 6.8 pts above campaign start; (b) the *price of analog* at matched architecture is **0.67 pts** on this frame; (c) the same-weights elevation (+0.42 here; tiered exact p≈10⁻⁴ across 7 arms) is a real, mechanism-attributed *chain* property (conv2 transform) — present it as physics, not as "beats digital." Do **not** claim system-level hardware ≥ digital; that framing is retracted as of today and the repo record (freeze note + AUDIT.md) is being amended accordingly.

— Jonathan (posted via the session agent, authorized)


---

### jon-morag — 2026-08-06T00:24:03Z

To make it clear now for those that don't want to read AI-slop.

There is no standing claim that MIWEN beats digital.
For our best ML layer stack, when digital is trained on its own weights, its accuracy rises to 99.5%, relative to current MIWEN 98.8%.

---

### jon-morag — 2026-08-06T00:56:10Z

Updated confusion graph for N=1200 for MIWEN GTSRB

<img width="1900" height="1720" alt="Image" src="https://github.com/user-attachments/assets/eb544e76-c4ea-4dd8-8e45-e3a400288c71" />
