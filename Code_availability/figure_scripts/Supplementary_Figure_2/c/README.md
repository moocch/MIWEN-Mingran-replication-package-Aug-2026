# Panel c — pass 3 (128 -> 43)

Analog pass 3 of the simulated three-pass fully analog GTSRB client:
drive **-58.5 dBm** (exactly -58.48 dBm), per-tone SNR **8.4 dB**
(archived link budget,
`..\..\fully_analog_simulation\results\link_budget.json`,
GTSRB row 3). Deepest unpowered pass shown: deterministic part is
essentially exact (noise-off deviation 0.00) but the low tone SNR makes
random scatter the largest of the chain: rel. RMSE 0.26 — the
multiplication degrading with depth exactly as the link budget predicts.

This panel is not generated separately: all three panels come from one
seeded chain simulation. See `..\README.md` and run
`..\make_c7_layer_scatter.py`; inputs come from the shared archive at
`..\..\fully_analog_simulation\`.
