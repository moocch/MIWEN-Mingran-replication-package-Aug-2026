# Panel (a) — Sequential weight download (model underlying the reviewer's estimate)

Panel (a) of `c4_parallel_schematic_v2.pdf`: the sequential-download model —
weights `w_1 ... w_Np` transmitted one after another, taking
`T_seq = b * N_p / R_serial` (tens of seconds for 10^8 params at 100 Mbit/s).

There is no separate per-panel script or data: both panels are drawn by the
single analytic matplotlib script `make_c4_figs_v2.py` at the folder root,
which writes the whole figure to `../reproduced/c4_parallel_schematic_v2.pdf`.
