# Supplementary Table 3 / Response Letter Table `tab:c4_latency` — Single-inference latency for a 10^8-parameter network

## Purpose
The latency/throughput table shown twice:
1. **Response letter** (Reviewer 1, Comment 4): `Main.tex`,
   `\label{tab:c4_latency}` (line 332; table body lines 337–341).
2. **Supplementary Information, Supplementary Table 3**:
   `Supp_M/Supplementary_Information.tex`,
   `\label{tab:latency}` (line 421; table body lines 426–430). It is the third
   of the SI's six table environments and `\tablename` is renewed to
   "Supplementary Table", so it renders as Supplementary Table 3.

## Generation chain (purely analytic — no data inputs)
```
make_latency_table.py  --(T_inf = N_p/B, R_inf = B/N_p, N_p = 1e8)-->  reproduced/latency_table.md
```
Every numeric cell follows from the single scaling law of Supplementary
Note 2, Eq. (S5), with N_p = 1e8 parameters (the reviewer's example). The
script also reproduces the published rounding/formatting exactly
(10.0 s, 1.00 s, 250 ms, 50 ms, 10 ms; 0.10, 1.00, 4.0, 20, 100 inf/s).

## Files
| File | MD5 |
|---|---|
| `make_latency_table.py` (copied from `Response_Letter_Reproductivity/Reviewer1_Comment4/c_latency_throughput_table/make_latency_table.py`, byte-identical) | `f697551a2f0e483dd9c90470bd24e9dc` |
| `reproduced/latency_table.md` (output of this re-run) | `1791d215afd4c766582e992fba186dc5` |

## How to re-run
```
cd Code_availability/figure_scripts/Supplementary_Table_3
python make_latency_table.py     # writes reproduced/latency_table.md and prints the rows
```
Requirements: any Python 3 (standard library only).
Verified here with Python 3.13.14.

## Cell-for-cell verification (2026-08-29)
The reproduced table was compared cell for cell against **both** LaTeX
sources. The two LaTeX table bodies are themselves identical to each other.

| # | B (script output) | Standard (script output) | T_inf (script) | R_inf (script) | Main.tex `tab:c4_latency` (line) | SI Supp. Table 3 `tab:latency` (line) | Match |
|---|---|---|---|---|---|---|---|
| 1 | 10 MHz  | this work (benchtop demonstration) | 10.0 s | 0.10 inf/s | `10~MHz & this work (benchtop demonstration) & 10.0~s & 0.10~inf/s` (337) | same (426) | ✓ all 4 cells |
| 2 | 100 MHz | single 5G NR FR1 carrier           | 1.00 s | 1.00 inf/s | `100~MHz & single 5G NR FR1 carrier & 1.00~s & 1.00~inf/s` (338) | same (427) | ✓ all 4 cells |
| 3 | 400 MHz | single 5G NR FR2 (mmWave) carrier  | 250 ms | 4.0 inf/s  | `400~MHz & single 5G NR FR2 (mmWave) carrier & 250~ms & 4.0~inf/s` (339) | same (428) | ✓ all 4 cells |
| 4 | 2 GHz   | 802.11ad/WiGig channel             | 50 ms  | 20 inf/s   | `2~GHz & 802.11ad/WiGig channel & 50~ms & 20~inf/s` (340) | same (429) | ✓ all 4 cells |
| 5 | 10 GHz  | 802.11ay / 6G D-band               | 10 ms  | 100 inf/s  | `10~GHz & 802.11ay / 6G D-band & 10~ms & 100~inf/s` (341) | same (430) | ✓ all 4 cells |

**Result: all 5 rows x 4 columns = 20 cells match both the response letter's
`tab:c4_latency` and Supplementary Table 3 exactly** (values, units, and
formatting, including the `~` non-breaking spaces reproduced in
`reproduced/latency_table.md`). No mismatches.

The 10 MHz benchtop entry is consistent with the manuscript's platform
description ("All channels run at 10 MS/s baseband"). The 25 MHz used in the
energy note (Comment 6) is a different quantity — the representative occupied
bandwidth for the energy normalization — and does not appear in this table.
