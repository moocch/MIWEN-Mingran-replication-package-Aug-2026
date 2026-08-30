#!/usr/bin/env python3
"""Regenerate battery_digital_preds_extract.npz from the 55-MB battery_slim.npz.

The full as-run battery archive (battery_slim.npz, ~55 MB: per-layer
magnitudes for all 10 chunks) is deliberately NOT copied into this
reproducibility folder; it stays at its source (see README.md).  This
script extracts only the four 1,200-element arrays that the paired
McNemar statistics of Supplementary Note 6 need, so that
verify_note6.py can recompute them from bytes archived here.

Usage:
    python extract_digital_preds.py [path/to/battery_slim.npz]

Default source path: the curated provenance package
    ../../../../202607_MIWEN_Manuscript/V2/GTSRB_inference/share_20260712/battery_slim.npz
(relative to this file; falls back to the absolute path below).
"""

# --- replication-package path resolution (see Code_availability/_paths.py) ---
import sys as _sys, pathlib as _pl
for _p in _pl.Path(__file__).resolve().parents:
    if _p.name == "Code_availability":
        _sys.path.insert(0, str(_p)); break
from _paths import data_dir as _data_dir
# ---------------------------------------------------------------------------

from pathlib import Path
import sys

import numpy as np

HERE = _data_dir(__file__)
DEFAULT_ABS = (r"E:\archive\Manuscript"
               r"\Overleaf\202607_MIWEN_Manuscript\V2\GTSRB_inference"
               r"\share_20260712\battery_slim.npz")
OUT = HERE / "battery_digital_preds_extract.npz"

PROVENANCE = (
    "Extracted 2026-08-29 from battery_slim.npz (GTSRB_inference/"
    "share_20260712, curated 2026-08-27 from QPG-MIT/MIWEN_Mingran "
    "branch handoff/rung3-session, commit 7fcc06a46d): keys preds "
    "(as-run measured battery 2026-08-05, 98.83%), digital_preds "
    "(same ns25 weights as exact linear digital MAC, 98.42%), labels, "
    "img_index. Per-layer magnitude arrays (the other ~55 MB) omitted.")


def main() -> int:
    if len(sys.argv) > 1:
        src = Path(sys.argv[1])
    else:
        rel = (HERE / ".." / ".." / ".." / ".." / "202607_MIWEN_Manuscript"
               / "V2" / "GTSRB_inference" / "share_20260712"
               / "battery_slim.npz")
        src = rel if rel.exists() else Path(DEFAULT_ABS)
    if not src.exists():
        print(f"source not found: {src}")
        return 1
    d = np.load(src, allow_pickle=True)
    np.savez_compressed(
        OUT,
        preds=d["preds"].astype(np.int64),
        digital_preds=d["digital_preds"].astype(np.int64),
        labels=d["labels"].astype(np.int64),
        img_index=d["img_index"].astype(np.int64),
        provenance=PROVENANCE)
    # regeneration must be byte-consistent in content with what it wrote
    z = np.load(OUT, allow_pickle=True)
    for k in ("preds", "digital_preds", "labels", "img_index"):
        assert np.array_equal(z[k], d[k]), k
    print(f"wrote {OUT.name}: 4 x {len(d['preds'])} int64 arrays")
    return 0


if __name__ == "__main__":
    sys.exit(main())
