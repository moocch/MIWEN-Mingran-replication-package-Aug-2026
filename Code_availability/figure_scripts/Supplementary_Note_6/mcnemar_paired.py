#!/usr/bin/env python3
"""Paired McNemar comparison between two eval runs (per-image, exact test).

Compares two npz artifacts produced by 4_gtsrb_confusion_mlpN.py
(keys: preds, labels, img_index, optional keep_mask) on the images both
runs measured. Reports the discordant counts, exact conditional
two-sided McNemar p-value, and per-run accuracies on the common subset.

The test conditions on discordant pairs only (b = A correct & B wrong,
c = A wrong & B correct), so per-image difficulty shared by both runs
cancels; under H0 (no difference) b ~ Binomial(b+c, 1/2). The two-sided
p-value doubles the smaller tail and is capped at 1.

Usage:
    python3 mcnemar_paired.py runA.npz runB.npz [--json report.json]
"""
from __future__ import annotations

import argparse
import json
import math
import sys

import numpy as np


def mcnemar_exact(b: int, c: int) -> float:
    """Exact conditional two-sided McNemar p-value from discordant counts."""
    if b < 0 or c < 0:
        raise ValueError(f"discordant counts must be >= 0, got b={b}, c={c}")
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2.0**n
    return min(1.0, 2.0 * tail)


def paired_discordants(
    preds_a: np.ndarray,
    preds_b: np.ndarray,
    labels_a: np.ndarray,
    labels_b: np.ndarray | None = None,
    img_index_a: np.ndarray | None = None,
    img_index_b: np.ndarray | None = None,
    keep_a: np.ndarray | None = None,
    keep_b: np.ndarray | None = None,
) -> tuple[int, int, int, float, float]:
    """Align two runs and count discordant predictions.

    Returns (b, c, n_common, acc_a, acc_b) where b counts images A got
    right and B got wrong, c the reverse, over the aligned common subset.
    When img_index arrays are given, runs are joined on image identity
    (intersection); otherwise arrays must be positionally aligned and of
    equal length. keep masks (from dropped-chunk handling) filter each
    run before alignment. Ground-truth labels must agree on every common
    image — a mismatch means the runs used different datasets.
    """
    preds_a = np.asarray(preds_a)
    preds_b = np.asarray(preds_b)
    labels_a = np.asarray(labels_a)
    labels_b = labels_a if labels_b is None else np.asarray(labels_b)

    if keep_a is not None:
        sel = np.asarray(keep_a, dtype=bool)
        preds_a, lab_a = preds_a[sel], labels_a[sel]
        idx_a = None if img_index_a is None else np.asarray(img_index_a)[sel]
    else:
        lab_a, idx_a = labels_a, (None if img_index_a is None else np.asarray(img_index_a))
    if keep_b is not None:
        sel = np.asarray(keep_b, dtype=bool)
        preds_b, lab_b = preds_b[sel], labels_b[sel]
        idx_b = None if img_index_b is None else np.asarray(img_index_b)[sel]
    else:
        lab_b, idx_b = labels_b, (None if img_index_b is None else np.asarray(img_index_b))

    if idx_a is not None and idx_b is not None:
        common, pos_a, pos_b = np.intersect1d(idx_a, idx_b, return_indices=True)
        if common.size == 0:
            raise ValueError("runs share no common img_index values")
        preds_a, lab_a = preds_a[pos_a], lab_a[pos_a]
        preds_b, lab_b = preds_b[pos_b], lab_b[pos_b]
    elif len(preds_a) != len(preds_b):
        raise ValueError(
            "runs have different lengths and no img_index to align on: "
            f"{len(preds_a)} vs {len(preds_b)}"
        )

    if not np.array_equal(lab_a, lab_b):
        raise ValueError("ground-truth labels disagree on common images")

    ok_a = preds_a == lab_a
    ok_b = preds_b == lab_b
    b = int(np.sum(ok_a & ~ok_b))
    c = int(np.sum(~ok_a & ok_b))
    n = int(len(preds_a))
    return b, c, n, float(np.mean(ok_a)), float(np.mean(ok_b))


def compare_runs(npz_a: str, npz_b: str, alpha: float = 0.05) -> dict:
    """Full paired comparison report between two eval-run npz files."""
    da = np.load(npz_a, allow_pickle=True)
    db = np.load(npz_b, allow_pickle=True)

    def _keep(d):
        # The eval script saves preds/labels/img_index ALREADY keep-filtered
        # while keep_mask keeps its full n_test length — in that schema the
        # mask must not be applied again (it would misindex or double-drop).
        k = d.get("keep_mask")
        if k is not None and len(k) != len(d["preds"]):
            return None
        return k

    b, c, n, acc_a, acc_b = paired_discordants(
        da["preds"], db["preds"], da["labels"], labels_b=db["labels"],
        img_index_a=da.get("img_index"), img_index_b=db.get("img_index"),
        keep_a=_keep(da), keep_b=_keep(db),
    )
    p = mcnemar_exact(b, c)
    return {
        "run_a": npz_a,
        "run_b": npz_b,
        "n_common": n,
        "acc_a": acc_a,
        "acc_b": acc_b,
        "delta_acc": acc_a - acc_b,
        "b_a_only_correct": b,
        "c_b_only_correct": c,
        "b": b,
        "c": c,
        "p_value": p,
        "significant_at_05": bool(p < alpha),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("npz_a", help="first run npz (e.g. run7_hil.npz)")
    ap.add_argument("npz_b", help="second run npz (e.g. run4 baseline)")
    ap.add_argument("--json", help="also write the report to this path")
    args = ap.parse_args(argv)

    rep = compare_runs(args.npz_a, args.npz_b)
    print(
        f"n_common={rep['n_common']}  "
        f"acc_a={rep['acc_a']:.4f}  acc_b={rep['acc_b']:.4f}  "
        f"delta={rep['delta_acc']:+.4f}"
    )
    print(
        f"discordants: A-only-correct b={rep['b']}, B-only-correct c={rep['c']}  "
        f"exact McNemar p={rep['p_value']:.5f}"
        f"  ({'significant' if rep['significant_at_05'] else 'NOT significant'} at 0.05)"
    )
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rep, f, indent=2)
        print(f"report written to {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
