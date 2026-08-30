# Fig. 2j — model RMSE vs vector length (three tiers)

- Plot code: `fig2_v3_twin.py`, section "k" (values hard-coded in the script).
- Data: `panel_k_values.csv` (tabulated copy of those values).

| N | ideal / physics-only / full twin (dB) | source record |
|---|---|---|
| 1 | 16.53 / 3.94 / 1.30 | scalar summary.json (`..\b` training-run record) |
| 2 | 18.71 / 3.37 / 1.91 | `vector_run_summary.json` (`..\c`) |
| 4 | 23.12 / 3.39 / 1.67 | idem |
| 8 | 21.61 / 2.81 / 1.55 | idem |
| 4096 | 10.35 / 1.80 / 0.79 | 4096DT summary.json (`..\f` training-run record) |

All values are recomputable from the twin_predictions_N*.npz files in panels b–f.
`vector_run_summary.json` (copy here) is the renamed N248 run summary.
