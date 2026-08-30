# -*- coding: utf-8 -*-
"""
Generates every numeric entry of the response-letter latency table
(tab:c4_latency) from the scaling law of Supplementary Note 2, Eq. (S5):
T_inf = N_p / B and R_inf = B / N_p, with N_p = 1e8 parameters.
Output: reproduced/latency_table.md
"""

# --- replication-package path resolution (see Code_availability/_paths.py) ---
import sys as _sys, pathlib as _pl
for _p in _pl.Path(__file__).resolve().parents:
    if _p.name == "Code_availability":
        _sys.path.insert(0, str(_p)); break
from _paths import data_dir as _data_dir
# ---------------------------------------------------------------------------

import os

HERE = str(_data_dir(__file__))
OUT = os.path.join(HERE, "reproduced")
os.makedirs(OUT, exist_ok=True)

NP = 1e8                                   # parameters (reviewer's example)
ROWS = [
    (10e6, "this work (benchtop demonstration)"),
    (100e6, "single 5G NR FR1 carrier"),
    (400e6, "single 5G NR FR2 (mmWave) carrier"),
    (2e9, "802.11ad/WiGig channel"),
    (10e9, "802.11ay / 6G D-band"),
]


def fmt_time(t):
    """Match the letter's table formatting: 10.0~s, 1.00~s, 250~ms, 50~ms, 10~ms."""
    if t >= 10:
        return f"{t:.1f}~s"
    if t >= 1:
        return f"{t:.2f}~s"
    return f"{t*1e3:.0f}~ms"


def fmt_thr(r):
    """Match the letter's table formatting: 0.10, 1.00, 4.0, 20, 100."""
    if r < 2:
        return f"{r:.2f}"
    if r < 10:
        return f"{r:.1f}"
    return f"{r:.0f}"


lines = ["| Occupied B | Representative standard | T_inf = N_p/B | R_inf = B/N_p |",
         "|---|---|---|---|"]
print(f"T_inf = N_p / B with N_p = {NP:.0e} parameters\n")
for B, std in ROWS:
    t = NP / B
    r = B / NP
    bt = f"{B/1e6:.0f}~MHz" if B < 1e9 else f"{B/1e9:.0f}~GHz"
    row = f"| {bt} | {std} | {fmt_time(t)} | {fmt_thr(r)}~inf/s |"
    lines.append(row)
    print(f"  B = {B/1e6:8.0f} MHz  ({std:38s}) : "
          f"T_inf = {fmt_time(t):8s}  R_inf = {fmt_thr(r)} inf/s")

with open(os.path.join(OUT, "latency_table.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print("\nwritten:", os.path.join(OUT, "latency_table.md"))
