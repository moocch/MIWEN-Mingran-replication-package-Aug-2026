# Panel a — pass 1 (3072 -> 128)

Analog pass 1 of the simulated three-pass fully analog GTSRB client:
drive **-10.0 dBm**, per-tone SNR **43.0 dB** (archived link budget,
`..\..\fully_analog_simulation\results\link_budget.json`,
GTSRB row 1). Deviation is dominated by the deterministic mixer envelope
compression of the hottest drive: rel. RMSE 0.54, of which 0.54 survives
with all noise switched off (a fixed map absorbed by hardware-aware
training, not a noise process).

This panel is not generated separately: all three panels come from one
seeded chain simulation. See `..\README.md` and run
`..\make_c7_layer_scatter.py`; inputs come from the shared archive at
`..\..\fully_analog_simulation\`.
