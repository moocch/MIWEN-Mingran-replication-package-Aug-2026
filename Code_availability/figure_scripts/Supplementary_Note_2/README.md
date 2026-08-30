# Supplementary Note 2 — inference latency and bandwidth scaling

Verifies every number of **Supplementary Note 2** of
`Supp_M/Supplementary_Information.tex`
(lines 388–446: Eq. S5, Supplementary Table 3 `tab:latency`, Supplementary
Figure 1 `fig:latency_bw`).

**No data, no script, no npz.** Every Note-2 number is pure algebra from the
single scaling law of Eq. (S5),

```
T_inf = sum_l N_l / B = N_p / B          R_inf = B / N_p
```

with `N_p = 1e8` parameters (the reviewer's example). Nothing is measured
except the benchtop bandwidth itself: `B = 10 MHz` is the campaign sample
rate, archived as raw measured bytes in
`../note3_client_energy/raw/<run>/gr_fig3c_ip_scatter.npz` (field
`fs_hz = 1e7` in both runs; asserted by `../note3_client_energy/verify_note3.py`).

## The algebra, worked row by row (Supplementary Table 3)

| Occupied `B` | `T_inf = 1e8 / B` | quoted | `R_inf = B / 1e8` | quoted |
|---|---|---|---|---|
| 10 MHz  | 1e8 / 1e7  = 10 s     | 10.0 s | 0.10 /s | 0.10 inf/s |
| 100 MHz | 1e8 / 1e8  = 1 s      | 1.00 s | 1.00 /s | 1.00 inf/s |
| 400 MHz | 1e8 / 4e8  = 0.25 s   | 250 ms | 4.0 /s  | 4.0 inf/s  |
| 2 GHz   | 1e8 / 2e9  = 0.05 s   | 50 ms  | 20 /s   | 20 inf/s   |
| 10 GHz  | 1e8 / 1e10 = 0.01 s   | 10 ms  | 100 /s  | 100 inf/s  |

The prose sentence of Note 2 ("a 1e8-parameter inference takes 10 s at the
10-MHz benchtop bandwidth, 250 ms on a single 5G NR mmWave carrier, and
10 ms at 802.11ay and 6G D-band bandwidths") is rows 1, 3 and 5 of the same
table.

## Machine-checked chains elsewhere in this archive

Both display objects of Note 2 are already reproduced, with scripts and
MD5-verified outputs, by their own archive folders — this README only points
at them:

1. **Supplementary Table 3** (`tab:latency`, also the response letter's
   `tab:c4_latency`) → `../../Code_availability/figure_scripts/Supplementary_Table_3/`
   (`make_latency_table.py` regenerates every cell *and the published
   rounding/formatting* from `T_inf = N_p/B`; output
   `reproduced/latency_table.md`).
2. **Supplementary Figure 1** (`fig:latency_bw`,
   `c4_latency_bandwidth_v2.pdf`, also Response Letter Figure 2) →
   `../../fig2_c4_latency_bandwidth/` (`make_c4_figs_v2.py`, matplotlib only,
   no data inputs; output `reproduced/c4_latency_bandwidth_v2.pdf`; that
   folder's README also MD5-matches the SI's figure file to the letter's).
   The figure's "this work" curve anchor (ten-layer GTSRB network,
   3072·128 + 8·128·128 + 128·43 = 529,792 ≈ 5.3e5 weights) is derived in
   that folder's README.

## Remaining Note-2 statements (definitions, not numbers)

* Eq. (S5) itself and the claim that computation occurs *during* each
  waveform period (architecture description, main text Fig. 1).
* "The benchtop bandwidth is a convenience choice rather than an
  architectural constraint" — a statement about the model's dependence on
  per-tone powers and comb geometry only; the supporting measured evidence
  (identical frequency-domain geometry and identical per-MAC airtime,
  117 ns, at both `N`) is asserted from raw bytes by
  `../note3_client_energy/verify_note3.py`.

Nothing in Note 2 requires further raw data: all five bandwidth rows are
representative standards (5G NR FR1/FR2, 802.11ad/ay, 6G D-band), not
measurements.
