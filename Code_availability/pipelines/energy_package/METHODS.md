# METHODS — the energy-scaling figure, defined line by line

Complete definition of every quantity, curve, marker and constant in
`figures/fig5_energy_scaling.pdf`, together with the derivation of each
number and its provenance in the raw data.

Energy accounting follows the in-physics RF-computing benchmark methodology
of **Gao et al., *Sci. Adv.* 12, eadz0817 (2026)**, Eqs. 1–3.

**Contents**

1. [What is computed, and the unit of work](#1-what-is-computed-and-the-unit-of-work)
2. [What the client actually pays for](#2-what-the-client-actually-pays-for)
3. [The energy model: e1, e2, e3](#3-the-energy-model-e1-e2-e3)
4. [The curves in the top panel](#4-the-curves-in-the-top-panel)
5. [Accuracy: RMSE, ENOB, and why std\[y\] = 1/3](#5-accuracy-rmse-enob-and-why-stdy--13)
6. [Precision conversion: WISE 5-bit → 2.4-bit ENOB, and the limit lines](#6-precision-conversion-wise-5-bit--24-bit-enob-and-the-limit-lines)
7. [The two measured points](#7-the-two-measured-points)
8. [File and array reference](#8-file-and-array-reference)
9. [Reproducing and restyling the figure](#9-reproducing-and-restyling-the-figure)

---

## 1. What is computed, and the unit of work

The operation under test is a **complex inner product** of length `N`:

```
y = Σ_{i=1..N}  a_i* · b_i          a, b ∈ ℂ^N
```

(The conjugation is on `a`; this is verified against the raw data — see
§8.1, `slot_c_true`.)

**Why every energy is divided by `4N`.** The industry unit of work is the
real **MAC** (multiply–accumulate). One *complex* multiply expands into four
*real* multiplies:

```
(a_r + i·a_i)(b_r + i·b_i) = (a_r b_r − a_i b_i) + i·(a_r b_i + a_i b_r)
                               └──┬──┘ └──┬──┘      └──┬──┘ └──┬──┘
                                 #1      #2           #3      #4
```

so one inner product of length `N` performs **`4N` real MACs**
(`N = 65536` → 262 144 MACs). Dividing the total joules spent on one inner
product by `4N` gives **energy per real MAC (J/MAC)** — the same unit
digital accelerators are rated in, which is what makes the comparison with
an NVIDIA H100 (≈ 70 fJ/MAC) meaningful.

| symbol | meaning | value |
|---|---|---|
| `N` | inner-product length (complex elements) | 4096, 65536 |
| `4N` | real MACs per inner product | 16 384, 262 144 |
| `e_ip` | client energy per real MAC | J/MAC |
| `H100` | GPU reference line | 70 fJ/MAC |

---

## 2. What the client actually pays for

```
    transmit             RF mixer            one output tone        digitise 3
  waveforms a(t), b(t) ─► (analog ×,   ─►   carries Σ a*b     ─►   samples + tiny
  (N tones each)          passive)                                 digital rescale
       │                     │                    │                     │
     pay e1                free                 free              pay e2 + e3
```

Each vector is mapped onto `N` frequency tones. When the two waveforms meet
in a passive mixer, wave mixing performs all `4N` real multiplications at
once and the summation appears automatically, concentrated in a single
output tone. **The multiplication and the summation cost the client
nothing** — only generating the waveforms, digitising a few samples, and a
small digital clean-up are billed. Those are exactly `e1`, `e2`, `e3`.

---

## 3. The energy model: e1, e2, e3

```
                P_x · T                6 · e_adc            8 · e_dig
e_ip(N)  =  ───────────────────  +  ───────────────  +  ───────────────
              4N · η_radio               4N                   4N
             └──────┬──────┘         └─────┬─────┘        └─────┬─────┘
            e1: waveform generation   e2: I/Q sampling   e3: digital decoding
              (ORANGE dashed)           (PINK dash-dot)    (PURPLE dotted)
```

### 3.1 Benchmark constants (fixed by the methodology, not fitted)

| symbol | meaning | value | where it comes from |
|---|---|---|---|
| `η_radio` | transmit-chain efficiency | `0.1` (10 %) | benchmark assumption: 1 J radiated costs 10 J at the wall |
| `e_adc` | energy per ADC conversion | `1 pJ` | benchmark assumption |
| `e_dig` | energy per digital real MAC | `1 pJ` | benchmark assumption |
| `k_B·T₀` | thermal energy at 300 K | `4.14e−21 J` | `k_B = 1.380649e−23 J/K` |
| `k_B·T₀·ln2` | Landauer quantum | `2.87 zJ` | — |

### 3.2 `e1` — waveform generation (**orange dashed line**)

```
e1 = P_x · T_ip / (4N · η_radio)
```

* **`P_x` — RF power delivered to the computing mixer's RF port.**
  Read from the raw file as
  `P_x[dBm] = p_rf_dbm_tx − rf_atten_db`, with `rf_atten_db = 30 dB`.
  The 30-dB pad is instrumentation isolation; the client-side requirement is
  the power *at the mixer*, which is what the pad's output delivers.
  Measured: **−62.81 dBm (0.524 nW)** at `N=4096`,
  **−63.41 dBm (0.457 nW)** at `N=65536` — sub-nanowatt, roughly half a
  billion times weaker than a phone transmitter.

* **`T_ip` — airtime amortised per inner product.**
  The transmitted frame contains `n_slots` bursts of `(fft_len + cp_len)`
  samples at sample rate `fs`, of which `q_data = 200` carry data inner
  products (the rest are pilots, which are genuinely transmitted and so are
  charged):

  ```
  n_slots = (frame_len − gap0 − gap1) / (fft_len + cp_len) = 227
  T_ip    = n_slots · (fft_len + cp_len) / fs / q_data
  ```

  `N=4096`: 227 · 16 896 / 10 MHz / 200 = **1.918 ms**
  `N=65536`: 227 · 270 336 / 10 MHz / 200 = **30.683 ms**

  *Convention note.* A stricter variant also charges the two inter-frame
  guard gaps (`T_ip = frame_len/fs/q_data`), which raises `T_ip` — and hence
  `e_ip` — by **+0.4 %**: 1.926 ms and 30.815 ms. This is well inside any
  reasonable error budget and does not change any quoted figure; the
  script computes both (`T_ip_with_gaps`) and uses the slot-based value.

* **`η_radio = 10 %`** converts radiated energy to wall-plug energy.

### 3.3 Why `e1` does not depend on `N` (the flat orange line)

`T_ip ∝ N` (longer vectors need proportionally longer waveforms), while the
denominator carries `4N`. The two scale together and `N` cancels:

```
e1 = P_x · (T_ip/N) / (4 η)   with   T_ip/N = 468 ns  (both datasets)
   = P_x · 117 ns / η
```

Each real MAC always receives **the same 117 ns of airtime**, so `e1`
depends only on the RF power the accuracy criterion demands. The two
measured powers differ by just 0.6 dB, which is why the orange floor drawn
from `N=65536` also describes `N=4096` to within 15 %.

### 3.4 `e2` — I/Q sampling (**pink dash-dot line**)

```
e2 = 6 · e_adc / (4N)
```

**Where the “6” comes from.** Per inner product the benchmark digitises
**3 complex samples**: the output tone carrying the answer, plus reference
(pilot) samples from which the link's complex gain and phase drift are
estimated. Since 1 complex sample = 2 real conversions (I and Q):

```
3 complex × 2 = 6 real ADC conversions  →  6 × 1 pJ = 6 pJ per inner product
```

Crucially this is a **fixed** fee — independent of `N`. Only 3 samples are
ever digitised, whether the vectors have 4096 or 65 536 elements.

### 3.5 `e3` — digital decoding (**purple dotted line**)

```
e3 = 8 · e_dig / (4N)
```

**Where the “8” comes from.** The raw tone must be un-distorted before it is
an answer: divide by the measured complex link gain `G_hat`, correct the
per-slot phase drift, and rescale — about **2 complex multiplies**. Since
1 complex multiply = 4 real MACs (§1):

```
2 complex × 4 = 8 real digital MACs  →  8 × 1 pJ = 8 pJ per inner product
```

Also a fixed fee, independent of `N`.

### 3.6 The fixed read-out fee

```
e2 + e3 = (6 + 8) pJ / (4N) = 14 pJ / (4N) = 3.5 pJ / N
```

A constant 14 pJ per answer, split over `4N` MACs. On a log–log plot a
`1/N` law is a straight line of slope −1 — the two falling lines.

---

## 4. The curves in the top panel

| element | colour / style | definition |
|---|---|---|
| `e_ip(N)` | **blue solid** | `e1 + e2 + e3` = `0.534 fJ + 3.5 pJ/N` |
| `e1` | orange dashed | flat floor, `0.534 fJ/MAC` |
| `e2` | pink dash-dot | `6 pJ/(4N)`, slope −1 |
| `e3` | purple dotted | `8 pJ/(4N)`, slope −1 |
| H100 | black solid | `70 fJ/MAC` |
| experiment | orange ▼ | the two measured operating points (§7) |

**The blue curve is not a fit.** It is the sum of the three bills, with the
radio floor `e1` pinned to its value measured at `N=65536` — the point where
the radio term dominates and is therefore least contaminated by read-out
overhead. Everything else is arithmetic.

```
N        3.5 pJ / N     + e1        who dominates
100      35 fJ          35.5 fJ     read-out fee
4096     0.85 fJ        1.39 fJ     comparable
6550     0.53 fJ        1.07 fJ     ← the bend
65536    0.05 fJ        0.59 fJ     radio floor
```

**Where the bend sits.** Read-out fee equals the radio floor when

```
N_bend = (6·e_adc + 8·e_dig) / (4·e1) = 14 pJ / (4 × 0.534 fJ) ≈ 6.55e3
```

Below `N_bend` the curve rides the slope −1 line (fixed overhead dominates);
above it the curve flattens onto the orange floor. **The scheme pays off for
large inner products.** The curve crosses the H100 line already at `N ≳ 50`.

*Note on the `N=4096` triangle vs. the blue curve.* The curve uses the
`N=65536` floor; the `N=4096` point needed 0.6 dB more power, so it sits
marginally (≈ 6 %) above the curve. This is a genuine measured difference,
not a plotting artefact.

---

## 5. Accuracy: RMSE, ENOB, and why std[y] = 1/3

Energy is meaningless without fixing accuracy: any system gets cheaper if
allowed to be wrong. The criterion is fixed **first**, and the power is then
lowered until the criterion is just barely met.

### 5.1 Normalisation and RMSE

Outputs are normalised by `√N`:  `y = c / √N`, `c = Σ a_i* b_i`
(field `c_true_norm`). RMSE is the root-mean-square error between decoded
and true normalised values over the `Q = 200` data slots:

```
RMSE = sqrt( mean( |ŷ_q − y_q|² ) )
```

**Criterion: `RMSE < 0.0625` = 1/16.** Met by every repeat at both sizes.

### 5.2 ENOB

```
ENOB = log2( std[y] / RMSE )
```

i.e. the error is compared against **how much the answers actually vary**.
This is the ADC-industry convention (signal spread ÷ noise, in bits).

### 5.3 Why `std[y]` = 1/3 — analytically

The test vectors are random with amplitude `r ~ U(0,1)` and phase
`φ ~ U(0,2π)`, independent (see `gen_vec_pair` in the acquisition script).
Everything follows from that one choice:

1. **Power of one element.** `E[r²] = ∫₀¹ r² dr = 1/3`.
2. **One product term** `t_i = a_i* b_i`: the phase difference of two
   independent uniform phases is again uniform, so `E[t_i] = 0`, and
   `E|t_i|² = (1/3)·(1/3) = 1/9`.
3. **The sum is a random walk.** With random phases the `N` terms do not add
   head-to-toe; `E|Σ t_i|² = N/9`, so `|c|` grows like `√N` — which is
   exactly why `y` is normalised by `√N` and not by `N`. Hence

   ```
   std[y] = sqrt( E|y|² ) = sqrt(1/9) = 1/3 ≈ 0.3333
   ```

By the CLT `y` is approximately a circular complex Gaussian, so the answers
occupy only the inner part of `[−1, 1]`: over the 200 stored true values,
`mean|y| ≈ 0.30`.

**Measured:** `0.3395` (`N=4096`), `0.3343` (`N=65536`) — field `std_y`.
Both are the finite-sample estimate from `Q = 200` values, whose ±1 s.d.
scatter is ≈ 0.012; the deviations from 1/3 (+1.9 % and +0.3 %) are ordinary
sampling noise. `std[y]` depends only on the seed, not on the hardware, so
all three repeats of a run share the same denominator.

---

## 6. Precision conversion: WISE 5-bit → 2.4-bit ENOB, and the limit lines

Same hardware, same physical error — only the yardstick changes.

### 6.1 The two conventions

| convention | reference for the error | value at `RMSE = 0.0625` |
|---|---|---|
| **WISE (original paper)** | full scale, width 2 (`[−1,1]`) | `log2(2/0.0625) = log2 32 =` **5 bit** |
| **ENOB (here, ADC standard)** | actual spread `std[y] ≈ 1/3` | `log2(0.34/0.0625) ≈ log2 5.4 ≈` **2.4 bit** |

### 6.2 The conversion is one line

```
b_WISE − ENOB = log2( range / std[y] ) = log2( 2 ÷ 1/3 ) = log2 6 = 2.585 bit
```

so the predicted equivalent precision is `5 − 2.585 =` **2.415 bit**.
Measured: `log2(0.3395/0.0625) = 2.44` (`N=4096`) and
`log2(0.3343/0.0625) = 2.42` (`N=65536`) — bracketing the analytic value.
The reference limits in the figure are labelled at the round, conservative
value **`b* = 2 bit`** (§6.3).

Of the 2.585-bit offset, **1 bit** comes from the width-2 full scale and
**log2 3 ≈ 1.58 bit** from the `U(0,1)` amplitude distribution. Constant-
amplitude (e.g. pure-phase) elements would give `std[y] = 1` and a gap of
only 1 bit.

> ⚠ Do not mix conventions across figures. Anything quoted as “5-bit” under
> the WISE convention is the *same measurement* as “2.4-bit ENOB” here.

### 6.3 The floor in the bottom panel

The limits are labelled at the round value **`b* = 2 bit`** — the
criterion-equivalent precision (2.42–2.44 bit) rounded *down*. Rounding down
is the conservative choice: a lower `b` gives a lower floor, so the quoted
headroom is not overstated. With `T₀ = 300 K`, `k_B T₀ ln2 = 2.87 zJ`:

* **Landauer limit (black solid).** Erasing/overwriting one bit costs at
  least `k_B T ln2`; a `b`-bit multiply involves ~`b²` such irreversible bit
  operations:
  ```
  e_Landauer = b² · k_B T₀ ln2 = 2² × 2.87 zJ = 11.48 zJ/MAC
  ```
  (Cross-check: at `b = 5` this gives 71.8 zJ, the value quoted in the
  reference methodology.)

* **Thermodynamic limit (cyan dashed).** Holding an *analog* value at
  `SNR = 2^{2b}` above thermal noise costs
  ```
  e_thermo = k_B T₀ ln(SNR) = 2b · k_B T₀ ln2 = 2(2) × 2.87 zJ = 11.48 zJ/MAC
  ```
  (Cross-check: at `b = 5` this gives 28.7 zJ, again matching the reference.)

* **⚠ At `b = 2` the two bounds are *exactly* equal**, since `b² = 2b` there.
  This is algebra, not coincidence or a plotting bug: the erasure-counting
  bound and the analog-SNR bound cross at 2 bit, and the figure therefore
  shows **one** line carrying both identities (black solid with cyan dashes
  on top, labelled `Landauer = thermodynamic limit`). For reference, at
  `b = 2.4` they would separate by 20 % (16.5 vs 13.8 zJ) and at `b = 5` by
  2.5× (71.8 vs 28.7 zJ). Both scripts detect the equality automatically
  (`limits_coincide` in the plot-data npz) and fall back to two separate
  labelled lines whenever `B_STAR ≠ 2`.

**Headroom.** The measured `0.59 fJ/MAC` sits ≈ 120× *below* an H100 and
≈ 5.1×10⁴ *above* the room-temperature floor.

---

## 7. The two measured points

Wired link, raw decode, no twin/post correction. Both taken from the
**`25 dB` panel** of each run (the `15 dB` panel does not meet the accuracy
criterion and is not plotted).

**Procedure.** (i) Fix the criterion `RMSE < 0.0625`. (ii) A closed-loop
tuner lowers the transmit power until the criterion is just barely met —
the cheapest operating point that still returns correct-enough answers.
(iii) Take `n = 3` independent captures at that fixed power; the criterion
held in all of them.

| | `N = 4096` | `N = 65536` |
|---|---|---|
| run | `..._20260810_011915_N4096` | `..._20260810_002043_N65536` |
| `P_x` at mixer | −62.81 dBm (0.524 nW) | −63.41 dBm (0.457 nW) |
| `T_ip` | 1.918 ms | 30.683 ms |
| `e1` | 0.613 fJ | 0.534 fJ |
| `e2` | 0.366 fJ | 0.023 fJ |
| `e3` | 0.488 fJ | 0.031 fJ |
| **`e_ip`** | **1.467 fJ/MAC** | **0.588 fJ/MAC** |
| vs H100 | **47.7×** cheaper | **119×** cheaper |
| RMSE (mean ± 1 s.d., n=3) | 0.0609 ± 0.0003 | 0.0561 ± 0.0006 |
| ENOB (mean ± 1 s.d., n=3) | 2.478 ± 0.006 bit | 2.575 ± 0.015 bit |
| `std[y]` | 0.3395 | 0.3343 |

**No error bars are drawn on the energy axis.** The repeats were taken at
*fixed* `P_x`, so `e_ip` has no direct repeat-to-repeat scatter; converting
the SNR/ENOB wander into an equivalent power would be a modelling choice,
not a measurement. The repeat statistics live in the RMSE/ENOB rows above.

**On the `N = 4096` run.** It replaces a legacy pre-power-fix acquisition
(raw RMSE 0.0795, LO underdriven) that never met the criterion. With the
peak-aware TX-power fix the LO sits at its −3 dBm target, `P_x` moved from
−58.15 to −62.81 dBm, and the point now falls on the model curve.

---

## 8. File and array reference

### 8.1 Raw measurement files — `data/raw/<run>/gr_fig3c_ip_scatter.npz`

Both runs have a leading axis of length 2 = the two SNR panels
(`labels = ['15 dB', '25 dB']`); the figure uses index 1.

| field | shape | meaning |
|---|---|---|
| `labels`, `targets` | (2,) | panel names / target SNR in dB |
| `vec_N` | scalar | inner-product length `N` |
| `fft_len`, `cp_len`, `gap0`, `gap1`, `frame_len`, `fs_hz` | scalar | waveform geometry (→ `T_ip`, §3.2) |
| `n_data`, `n_pilot` | scalar | 200 data slots, 26 pilot slots |
| `p_rf_dbm_tx` | (2,) | TX-port RF power, dBm (→ `P_x` after the 30-dB pad) |
| `p_lo_dbm_tx` | (2,) | LO drive, dBm (−3 dBm target) |
| `c_true_norm` | (2,200) | true normalised inner products `y` |
| `c_hat` | (2,200) | decoded values `ŷ` |
| `slot_c_true` | (226,) | un-normalised true values, all slots |
| `rmse`, `rmse_mean`, `rmse_sd`, `rmse_reps` | (2,), (2,3) | accuracy, per repeat |
| `enob`, `enob_mean`, `enob_sd`, `enob_reps` | (2,), (2,3) | `log2(std_y/RMSE)` |
| `std_y` | (2,) | spread of the true answers (§5.3) |
| `snr3_db`, `snr3_db_reps` | (2,), (2,3) | 3-bin SNR of the output tone |
| `G_hat`, `drift_rad_per_slot` | (2,) | link gain / phase drift (→ the 8 MACs of `e3`) |
| `n_repeats` | (2,) | 3 |
| `meta_json` | str | full acquisition settings, incl. `rf_atten_db = 30` |
| `tune_history_json` | str | closed-loop power-tuning trace |

Companion PNGs in each run folder: `gr_fig3c_ip_scatter.png` (scatter of
decoded vs. true) and `..._enob.png` (ENOB ± 1 s.d. for both panels).

### 8.2 Plot data — `data/fig5_plot_data.npz`

Everything drawn on the canvas, for re-plotting without the raw data.

| field | meaning |
|---|---|
| `N_curve` (600,) | x samples, 30 … 1e5, log-spaced |
| `e_ip_curve`, `e1_curve`, `e2_curve`, `e3_curve` | the four curves, J/MAC |
| `h100_line`, `e1_floor`, `N_bend` | GPU reference, radio floor, bend location |
| `e_landauer`, `e_thermo`, `b_star`, `limits_coincide` | bottom-panel floor, its 2-bit label, and whether the two bounds are equal |
| `points_N`, `points_e_ip` | the two ▼ markers |
| `points_e1/e2/e3`, `points_px_dbm`, `points_px_W`, `points_T_ip_s` | their breakdown |
| `points_rmse_mean/sd`, `points_enob_mean/sd`, `points_std_y`, `points_n_repeats` | accuracy stats |
| `points_speedup_vs_h100`, `points_run` | 47.7× / 119×, source run names |
| `xlim`, `ylim_top`, `ylim_bottom`, `xlabel`, `ylabel` | axis setup |
| `meta_json` | all benchmark constants + methodology string |

`data/fig5_plot_data_curves.csv` holds the same four curves as plain text
for Origin / Igor / gnuplot.

---

## 9. Reproducing and restyling the figure

```bash
# full pipeline: raw .npz  ->  figure (PDF/SVG/PNG) + plot-data .npz + .csv
python code/fig5_energy_scaling.py

# restyle for the manuscript, from the plot data alone
python code/replot_from_npz.py --font "Arial Nova" --width 3.4 --out figures/single_col
python code/replot_from_npz.py --no-bottom        # top panel only
```

`fig5_energy_scaling.py` asserts every published number against the raw
data (`CHECKS`); if an assertion fires, the data and the figure have drifted
apart and the script stops rather than drawing a stale number.

**Vector output.** PDF and SVG are true vector. `pdf.fonttype = 42` embeds
editable TrueType and `svg.fonttype = 'none'` keeps SVG labels as live
`<text>`, so both open as fully editable objects in Illustrator or Inkscape.
Use the PDF for LaTeX (`\includegraphics{fig5_energy_scaling.pdf}`).

**Fonts.** The scripts request Arial and fall back to DejaVu Sans with a
`findfont` warning if Arial is absent — harmless, but for camera-ready
output run on a machine that has the font, or pass `--font`.

Requirements: Python ≥ 3.9, `numpy`, `matplotlib`. No instrument, network or
proprietary library is needed to rebuild the figure.
