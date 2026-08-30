# Fig. 1 — Wireless in-physics inference and its bottleneck

File used by the tex: `Data_availability/Figure_1/fig1_preview_v3.pdf` (copy in this folder).
All panels are concept/simulation — no measured data points (measurements are in Figs. 2–4).

**2026-08-28 revision (author):** `fig1c_theory.py` edited in place — the in-panel
annotation "same mixer — hardware unchanged" was removed (4 lines deleted, nothing else;
the caption now says "evaluated on the same unmodified hardware" instead) — and the
composite was rebuilt. Script + outputs refreshed here (`d\`, `e\`, and the root copies;
`a\fig1a_system_v5.pdf` keeps the original 2026-07-15 render wrap). **Re-executed from
the archived files on 2026-08-29: regenerated `fig1c_theory.png` and the recomposed
`fig1_preview_v3.png` are both pixel-identical to the published ones
(0 / 2,060,998 and 0 / 6,628,868 differing pixels).**

Composition: `fig1_compose_v3.py` assembles the 180 × 66 mm figure from three
columns: `a\fig1a_system_v5.png` (Blender render, 0–62 mm),
`b\fig1b_motivation.png` (panels b+c, 65–121 mm),
`d\fig1c_theory.png` (panels d+e, 124–180 mm).

| Panel | Content | Source |
|---|---|---|
| a | concept schematic | Blender render (`a\`) |
| b | IF power vs LO drive of the calibrated mixer model | simulation, `fig1b_motivation.py` (`b\`) |
| c | computing accuracy vs energy per MAC (optimal window) | simulation, same script (`c\`) |
| d | diode-ring transfer law vs ideal multiplier | analytic, `fig1c_theory.py` (`d\`) |
| e | model–hardware error 16.53 → 1.30 dB | calibration numbers, same script (`e\`) |

Reproduce (verified pixel-identical):
`b> python fig1b_motivation.py`, `d> python fig1c_theory.py`,
then here: `python fig1_compose_v3.py`.
