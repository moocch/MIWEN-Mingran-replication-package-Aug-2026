# Panel (a) — Single-inference latency vs. bandwidth

Panel (a) of `c4_latency_bandwidth_v2.pdf`: single-inference latency
`T_inf = N_p / B` vs. occupied RF bandwidth (log-log), with curves for
5.3e5 (this work, GTSRB), 1e7, 1e8, and 1e9 parameters, real-time (<10 ms)
and interactive (<100 ms) bands, and red triangles at the five Table
bandwidths (10 s / 1 s / 250 ms / 50 ms / 10 ms for 1e8 params).

There is no separate per-panel script or data: both panels are drawn by the
single analytic matplotlib script `make_c4_figs_v2.py` at the folder root,
which writes the whole figure to `../reproduced/c4_latency_bandwidth_v2.pdf`.
