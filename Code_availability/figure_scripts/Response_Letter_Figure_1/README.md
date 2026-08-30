# Response Letter Figure 1 — Sequential download vs. parallel frequency-multiplexed broadcast (`c4_parallel_schematic_v2.pdf`)

## Purpose
This is the first display figure of the response to **Reviewer 1, Comment 4**
(the latency-bottleneck concern). It is included in the letter at
`Main.tex` line 302
(`\includegraphics{figures/c4_parallel_schematic_v2.pdf}`) and illustrates why
MIWEN has no weight-download latency term: panel (a) shows the sequential
weight-download model underlying the reviewer's estimate; panel (b) shows the
parallel frequency-multiplexed broadcast of this work, where all `N_p` weights
occupy orthogonal subcarriers of one waveform period `T_inf = N_p / B`.

## Generation chain (purely analytic — no data inputs)
```
make_c4_figs_v2.py  --(matplotlib only, no data files)-->  reproduced/c4_parallel_schematic_v2.pdf
```
The figure is a pure matplotlib schematic: every element (boxes, stripes,
arrows, annotated numbers) is drawn analytically in the script. The annotated
numbers (10^8 params: 10 MHz -> 10 s, 400 MHz -> 250 ms, 10 GHz -> 10 ms)
follow from the single scaling law `T_inf = N_p / B` (Supplementary Note 2,
Eq. S5) and match the letter's Table `tab:c4_latency`.

The same script also draws the letter's second Comment-4 figure
(`c4_latency_bandwidth_v2.pdf`), so a run leaves **both** PDFs in
`reproduced/`; only `c4_parallel_schematic_v2.pdf` is this folder's target
(the other figure is archived in `../fig2_c4_latency_bandwidth/`).

## Files
| File | Role |
|---|---|
| `make_c4_figs_v2.py` | runnable script; writes into the local `reproduced/` subfolder (never touches the live `figures/` tree) |
| `make_c4_figs_v2_original_recovered.py.txt` | original script recovered verbatim from the session that generated the published figure (2026-07-17, session 88419e0a). Kept as `.txt` **on purpose**: its hard-coded output directory is the live `figures/` tree, so executing it would overwrite the published figure. Provenance record only. |
| `published_c4_parallel_schematic_v2.pdf` | byte-for-byte copy of the live published figure |
| `reproduced/` | output of the re-run (both Comment-4 PDFs) |
| `a/`, `b/` | panel pointers (both panels come from the single script above) |

## Provenance and MD5s
| Item | Path | MD5 |
|---|---|---|
| Live published figure | `figures/c4_parallel_schematic_v2.pdf` | `fd9d1b76978c6fe3bf224ea86fbdcaa5` |
| Archived copy (old archive) | `Response_Letter_Reproductivity/Reviewer1_Comment4/a_c4_parallel_schematic/published_c4_parallel_schematic_v2.pdf` | `fd9d1b76978c6fe3bf224ea86fbdcaa5` |
| `published_c4_parallel_schematic_v2.pdf` (this folder, copied from the live tree) | — | `fd9d1b76978c6fe3bf224ea86fbdcaa5` |
| `make_c4_figs_v2.py` (from old archive, identical in its `a_` and `b_` subfolders) | — | `49136b76851accfce771cc2ed478c6ee` |
| `make_c4_figs_v2_original_recovered.py.txt` | — | `f68dccf118beb627f5351b78b8e95b57` |
| `reproduced/c4_parallel_schematic_v2.pdf` (this re-run) | — | `c74c4d8e7049cd0a7e1792138d0dc80b` |

## How to re-run
```
cd fig1_c4_parallel_schematic
python make_c4_figs_v2.py        # writes reproduced/c4_parallel_schematic_v2.pdf (+ the fig2 PDF)
```
Requirements: Python 3 with numpy and matplotlib (headless Agg backend).
Verified here with Python 3.13.14, matplotlib 3.11.0, numpy 2.5.1.

## Verification result (2026-08-29)
- `published_c4_parallel_schematic_v2.pdf` is **byte-identical (MD5 match)** to
  the live `figures/c4_parallel_schematic_v2.pdf` and to the old archive's copy.
- Re-run output vs. published, both rendered at **200 dpi** with PyMuPDF 1.28.0
  (1807 x 903 px, RGB): **0 of 1,631,721 pixels differ** (max channel delta 0)
  -> **pixel-identical**.
- Byte sizes identical (54,381 bytes both); the MD5s of published vs.
  reproduced differ only because matplotlib embeds a creation timestamp in the
  PDF metadata. This matches the old archive's 2026-07-27 pixel-identical audit.
