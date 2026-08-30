# Panel (b) — Parallel frequency-multiplexed broadcast (MIWEN, this work)

Panel (b) of `c4_parallel_schematic_v2.pdf`: the parallel
frequency-multiplexed broadcast — `N_p` orthogonal subcarriers, one per
weight, all transmitted within one waveform period
`T_inf = 1/Delta_f = N_p / B`, with the deployment scaling line
(10 MHz -> 10 s, 400 MHz -> 250 ms, 10 GHz -> 10 ms for 10^8 params).

There is no separate per-panel script or data: both panels are drawn by the
single analytic matplotlib script `make_c4_figs_v2.py` at the folder root,
which writes the whole figure to `../reproduced/c4_parallel_schematic_v2.pdf`.
