# Response Letter Figure 2 / Supplementary Figure 1 — Latency and throughput vs. occupied bandwidth (`c4_latency_bandwidth_v2.pdf`)

## Purpose
Second display figure of the response to **Reviewer 1, Comment 4** (the
latency-bottleneck concern), included in the letter at
`Main.tex` line 349
(`\includegraphics{figures/c4_latency_bandwidth_v2.pdf}`, `fig:c4_latency_bw`).
Panel (a): single-inference latency `T_inf = N_p / B` vs. occupied RF
bandwidth; panel (b): inference throughput `R_inf = B / N_p`. Curves for
5.3e5 (this work, GTSRB network), 1e7, 1e8, and 1e9 parameters; red triangles
mark the 1e8-parameter case at the five bandwidths of Table `tab:c4_latency`.

**This figure is also Supplementary Figure 1 of the Supplementary
Information**: it is the first figure environment of
`Supp_M/Supplementary_Information.tex` (line 435,
`fig:latency_bw`, with `\figurename` = "Supplementary Figure"). The SI's copy
at `Supp_M/figures/c4_latency_bandwidth_v2.pdf` is
**byte-identical** to the letter's copy (MD5 verified below).

## Generation chain (purely analytic — no data inputs)
```
make_c4_figs_v2.py  --(matplotlib only, no data files)-->  reproduced/c4_latency_bandwidth_v2.pdf
```
Every curve and marker follows from the single scaling law `T_inf = N_p / B`
(Supplementary Note 2, Eq. S5); no measured data enter. The annotated marker
values (10 s / 1 s / 250 ms / 50 ms / 10 ms; 0.1 / 1 / 4 / 20 / 100 inf/s)
match the letter's Table `tab:c4_latency` and Supplementary Table 3 row for
row. The "this work" curve is anchored to the ten-layer GTSRB network,
3072*128 + 8*128*128 + 128*43 = 529,792 ~ 5.3e5 complex weights.

The same script also draws the letter's first Comment-4 figure
(`c4_parallel_schematic_v2.pdf`), so a run leaves **both** PDFs in
`reproduced/`; only `c4_latency_bandwidth_v2.pdf` is this folder's target
(the other figure is archived in `../fig1_c4_parallel_schematic/`).

## Files
| File | Role |
|---|---|
| `make_c4_figs_v2.py` | runnable script; writes into the local `reproduced/` subfolder (never touches the live `figures/` tree) |
| `make_c4_figs_v2_original_recovered.py.txt` | original script recovered verbatim from the session that generated the published figure (2026-07-17, session 88419e0a). Kept as `.txt` **on purpose**: its hard-coded output directory is the live `figures/` tree. Provenance record only. |
| `published_c4_latency_bandwidth_v2.pdf` | byte-for-byte copy of the live published figure |
| `reproduced/` | output of the re-run (both Comment-4 PDFs) |
| `a/`, `b/` | panel pointers (both panels come from the single script above) |

## Provenance and MD5s
| Item | Path | MD5 |
|---|---|---|
| Live published figure (letter) | `figures/c4_latency_bandwidth_v2.pdf` | `fe0e5b12bd1af204464b90b515a35e87` |
| SI copy (Supplementary Figure 1) | `Supp_M/figures/c4_latency_bandwidth_v2.pdf` | `fe0e5b12bd1af204464b90b515a35e87` (byte-identical) |
| Archived copy (old archive) | `Response_Letter_Reproductivity/Reviewer1_Comment4/b_c4_latency_bandwidth/published_c4_latency_bandwidth_v2.pdf` | `fe0e5b12bd1af204464b90b515a35e87` |
| `published_c4_latency_bandwidth_v2.pdf` (this folder, copied from the live tree) | — | `fe0e5b12bd1af204464b90b515a35e87` |
| `make_c4_figs_v2.py` (from old archive, identical in its `a_` and `b_` subfolders) | — | `49136b76851accfce771cc2ed478c6ee` |
| `make_c4_figs_v2_original_recovered.py.txt` | — | `f68dccf118beb627f5351b78b8e95b57` |
| `reproduced/c4_latency_bandwidth_v2.pdf` (this re-run) | — | `a19600acdf4d34fea0d205294c64fce7` |

## How to re-run
```
cd fig2_c4_latency_bandwidth
python make_c4_figs_v2.py        # writes reproduced/c4_latency_bandwidth_v2.pdf (+ the fig1 PDF)
```
Requirements: Python 3 with numpy and matplotlib (headless Agg backend).
Verified here with Python 3.13.14, matplotlib 3.11.0, numpy 2.5.1.

## Verification result (2026-08-29)
- `published_c4_latency_bandwidth_v2.pdf` is **byte-identical (MD5 match)** to
  the live `figures/c4_latency_bandwidth_v2.pdf`, to the SI's
  `Supp_M/figures/c4_latency_bandwidth_v2.pdf` (Supplementary Figure 1), and
  to the old archive's copy — all four files share MD5
  `fe0e5b12bd1af204464b90b515a35e87`.
- Re-run output vs. published, both rendered at **200 dpi** with PyMuPDF 1.28.0
  (2496 x 995 px, RGB): **0 of 2,483,520 pixels differ** (max channel delta 0)
  -> **pixel-identical**.
- Byte sizes identical (40,971 bytes both); the MD5s of published vs.
  reproduced differ only because matplotlib embeds a creation timestamp in the
  PDF metadata. This matches the old archive's 2026-07-27 pixel-identical audit.
