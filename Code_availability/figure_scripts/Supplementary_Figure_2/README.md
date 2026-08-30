# Fig. 5 of the response letter (= Supplementary Figure 2) — c7_analog_layer_scatter

Per-layer multiplication-fidelity scatter of the simulated three-pass fully
analog GTSRB client. One panel per analog pass: the chain's actual layer
output (mixer envelope compression at the planned drive, measured
per-element jitter, link noise of the archived budget) scattered against
the ideal matrix–vector product of that layer's own input, plus the same
outputs with every noise source switched off (orange). Used as **Fig. 5 of
the response letter** ("New Supplementary Note for Comment 7", Reviewer 1)
and as **Supplementary Figure 2** of the revised SI (second figure
environment of `Supp_M\Supplementary_Information.tex`, label
`fig:activation_sim`, shown with paragraph S5.8 of Supplementary Note 5).

## Chain

```
..\fully_analog_simulation\           (shared simulation archive)
    weights\ckpt_ZH_L3.npz      fielded GTSRB L=3 checkpoint
    data\gtsrb_roi_32x32.npz    GTSRB test photographs (first 40 used)
    results\link_budget.json    archived drive ladder + tone SNRs (gate)
        |
        v
make_c7_layer_scatter.py                      (gated pure-numpy replotter)
    pure numpy + matplotlib port of the archived device physics
    (plan_Z / ed_video / selfmix from comb_analog_sim.py), seeded
    default_rng(0); ASSERTS its planned drive ladder against the archived
    link_budget.json to < 0.05 dB per pass BEFORE anything is drawn
        |
        v
reproduced\c7_analog_layer_scatter.pdf/.png   (the figure)
reproduced\c7_layer_scatter_numbers.json      (annotation numbers)
```

## Files

| file | role |
|---|---|
| `make_c7_layer_scatter_original_recovered.py.txt` | verbatim copy of the original generator (`Response_Letter_Reproductivity\Reviewer1_Comment7\b_activation_block_note\make_c7_layer_scatter.py`) |
| `make_c7_layer_scatter.py` | runnable copy, modified **only** in path handling (inputs from `..\fully_analog_simulation\{weights,data,results}`, all outputs to `reproduced\` so the recovered originals are never overwritten); physics, constants, seed and gate untouched |
| `c7_layer_scatter_numbers.json` | recovered original numbers file (ladder −10.00/−34.27/−58.48 dBm; relRMSE 0.539/0.065/0.256; deterministic-only 0.535/0.017/0.000) |
| `README_source_note.md` | README of the source folder (note: it says "Supplementary Figure 3"; in the current SI the figure is Supplementary Figure 2). Its `Run:` line names the obsolete 4-panel `make_c7_activation_figs.py`, which is **not** used by the current letter and is not ported here |
| `published_c7_analog_layer_scatter.pdf/.png` | the published figure, copied from the live figures tree |
| `a\ b\ c\` | per-panel notes (pass 1/2/3); all three panels come from the one seeded chain run |
| `reproduced\` | output of the verification rerun (see below) |

## Constants-block provenance

The replotter hard-codes the device constants instead of importing the
archive; every one was cross-checked against the shared simulation archive:

| constant | value | archive source |
|---|---|---|
| `K_ED` | 208.7 /V | `comb_analog_sim.py` (`K_ED = 208.7`, measured ED scaling) and `link_budget.json` `settings.k_ED_per_V` |
| `CL_PASS` | 7.47 dB | `comb_analog_sim.py` (`CL_PASS = 7.47`) and `settings.cl_pass_dB` |
| `BOOST_DB` | 18.5 dB | `comb_analog_sim.py` (measured passive voltage boost at 400 MHz) and `settings.boost_dB` |
| `R_V` | 3.5 kOhm | `comb_analog_sim.py` (`R_V = 3500.0`) and `settings.R_v_ohm` |
| `IL_STEP` | 1 dB step-down loss | `settings.step_down_loss_dB = 1.0` |
| `P_N` | −73.6 dBm per-bin floor | `comb_analog_sim.py` (`P_N_DBM = -73.6`, "as published") |
| `NOISE_C` | 27.0 | `comb_analog_sim.py` (calibrated constant of the R² noise term) |
| `P3` | −7.53 dBm | `comb_analog_sim.py` (`P3_DBM = P_COMP − PAPR_DB`, PAPR-shifted compression point) |
| `S_ELEM` | 0.06 | `comb_analog_sim.py` (`S_ELEM_NET = 0.06`, representative measured per-element jitter, range 0.03–0.15) |

The compression law `g = (1 + |env|²/(0.5·P3))^{-1/2}`, the jitter
injection `y·(1 + S_ELEM·N(0,1))` and the link-SNR law
`SNR = NOISE_C·P/(D·P_N)` are line-for-line the archived
`comb_analog_sim.py` forms.

## How to re-run

```
cd fig5_c7_analog_layer_scatter
python make_c7_layer_scatter.py
```

Pure `numpy` + `matplotlib` (Agg backend), no other dependencies; the
`..\fully_analog_simulation\` folder must sit next to this one.
Deterministic: `np.random.default_rng(0)` for both the plan and the chain.
The script aborts (`AssertionError: ladder mismatch`) if its planned drive
ladder deviates from the archived `link_budget.json` by ≥ 0.05 dB on any
pass.

## Verification results (rerun 2026-08-29; Python 3.13.14, numpy 2.5.1, matplotlib 3.11.0)

Link-budget gate **passed**; console output of the rerun:

```
ladder: ['-10.00 dBm', '-34.27 dBm', '-58.48 dBm'] SNR: [43.04, 32.57, 8.36]
pass 1: drive -10.0 dBm, SNR 43.0 dB, g=0.144, relRMSE=0.539 (deterministic 0.535)
pass 2: drive -34.3 dBm, SNR 32.6 dB, g=0.992, relRMSE=0.065 (deterministic 0.017)
pass 3: drive -58.5 dBm, SNR 8.4 dB, g=1.047, relRMSE=0.256 (deterministic 0.000)
```

| artefact | MD5 | vs. published |
|---|---|---|
| `published_c7_analog_layer_scatter.pdf` | `25eea40074e3938948b04017660ef55e` | identical in both live trees (`figures\`, `Supp_M\figures\`; the Supp_M tree carries only the PDF) |
| `published_c7_analog_layer_scatter.png` | `9a7368ba33db246bb48ef1f27202c007` | from `figures\` |
| `reproduced\c7_analog_layer_scatter.png` | `9a7368ba33db246bb48ef1f27202c007` | **byte-identical** to published; PIL pixel compare 0 / 1,627,155 pixels differ (2127×765 RGBA) |
| `reproduced\c7_analog_layer_scatter.pdf` | `83af9679ca9e740b302a851112249080` | same size (233,732 B) and byte-identical to the published PDF after stripping only the `/CreationDate` and `/ModDate` timestamps; PyMuPDF 200-dpi renders differ in 0 / 723,180 pixels |
| `reproduced\c7_layer_scatter_numbers.json` | `34c9a4f460be69017952fa0cb57203b8` | **byte-identical** to the recovered original `c7_layer_scatter_numbers.json` |
| `make_c7_layer_scatter_original_recovered.py.txt` | `5e056609cf9db0de4b9fb440ea6a96b6` | identical to the source `make_c7_layer_scatter.py` |
| `README_source_note.md` | `3b0a0ab986f016ad25c08f599e31b993` | identical to the source `README.md` |

**Numbers vs. figure and SI text (paragraph S5.8 + the figure caption of
`Supp_M\Supplementary_Information.tex`):** JSON ladder
−10.0 / −34.27 / −58.48 dBm and SNR 43.04 / 32.57 / 8.36 dB round exactly to
the panel annotations and the S5.8 text (−10.0 dBm 43.0 dB; −34.3 dBm
32.6 dB; −58.5 dBm 8.4 dB; caption "43.0 → 32.6 → 8.4 dB"; Sec. S5.6 quotes
the un-rounded "−10.0 → −34.27 → −58.48 dBm"); relRMSE
0.539 / 0.065 / 0.256 → the quoted 0.54 / 0.07 / 0.26; deterministic-only
0.535 / 0.017 / 0.000 → the quoted 0.54 / 0.02 ("then 0.00"). The
−73.6-dBm floor quoted in S5.8 is the `P_N` constant above. All match.
