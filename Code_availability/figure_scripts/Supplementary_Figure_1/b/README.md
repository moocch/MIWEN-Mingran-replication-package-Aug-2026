# Panel (b) — Inference throughput vs. bandwidth

Panel (b) of `c4_latency_bandwidth_v2.pdf`: inference throughput
`R_inf = B / N_p` vs. occupied RF bandwidth (log-log), same four parameter
counts, the >30 inf/s video-rate band, and red triangles at the five Table
bandwidths (0.1 / 1 / 4 / 20 / 100 inf/s for 1e8 params).

There is no separate per-panel script or data: both panels are drawn by the
single analytic matplotlib script `make_c4_figs_v2.py` at the folder root,
which writes the whole figure to `../reproduced/c4_latency_bandwidth_v2.pdf`.
