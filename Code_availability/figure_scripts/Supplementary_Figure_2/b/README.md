# Panel b — pass 2 (128 -> 128)

Analog pass 2 of the simulated three-pass fully analog GTSRB client:
drive **-34.3 dBm** (exactly -34.27 dBm), per-tone SNR **32.6 dB**
(archived link budget,
`..\..\fully_analog_simulation\results\link_budget.json`,
GTSRB row 2). The envelope compression has vanished down the unpowered
ladder (noise-off deviation 0.02) and the random link-noise scatter now
dominates: rel. RMSE 0.07.

This panel is not generated separately: all three panels come from one
seeded chain simulation. See `..\README.md` and run
`..\make_c7_layer_scatter.py`; inputs come from the shared archive at
`..\..\fully_analog_simulation\`.
