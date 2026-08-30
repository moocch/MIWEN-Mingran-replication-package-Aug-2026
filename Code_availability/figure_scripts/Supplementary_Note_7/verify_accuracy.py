#!/usr/bin/env python3
"""Recompute the reported N=600 result from the raw prediction files.

    python verify_accuracy.py

Ground truth: 08_frozen_inputs_and_labels/battery_frozen_slim.npz
(`img_index` -> `labels`), the frozen 1200-image GTSRB battery used by
every benchmark in the campaign. The reported result uses the first 600
of that battery, run as two disjoint 300-image segments with the SAME
frozen weights and the SAME frozen per-layer calibration.
"""

# --- replication-package path resolution (see Code_availability/_paths.py) ---
import sys as _sys, pathlib as _pl
for _p in _pl.Path(__file__).resolve().parents:
    if _p.name == "Code_availability":
        _sys.path.insert(0, str(_p)); break
from _paths import data_dir as _data_dir
# ---------------------------------------------------------------------------

import collections
import json
from pathlib import Path

import numpy as np

HERE = _data_dir(__file__)
RES = HERE / "05_hardware_result_0dBm_N600"

b = np.load(HERE / "08_frozen_inputs_and_labels" / "battery_frozen_slim.npz",
            allow_pickle=True)
LAB = dict(zip(b["img_index"].tolist(), b["labels"].tolist()))

CLEAN = ["serial_nn_clean_0_150_20260823.npz",
         "serial_nn_clean_150_215_20260823.npz",
         "serial_nn_clean_215_300_20260823.npz",
         "serial_nn_clean_275_300_20260823.npz",
         "serial_nn_clean_300_600_20260823.npz"]
TWIN = ["serial_nn_twin_0_300_20260823.npz",
        "serial_nn_twin_300_600_20260823.npz"]


def load(path):
    """Concatenate the incrementally-saved chunk predictions of one run."""
    z = np.load(path, allow_pickle=True)
    ks = sorted((k for k in z.keys() if k.startswith("chunk")),
                key=lambda s: int(s[5:].split("_")[0]))
    p = np.concatenate([z[k] for k in ks])
    return p, z["sel"][:len(p)], json.loads(str(z["meta_json"]))


def arm(name, folder, files):
    P, S, wset, pset = [], [], set(), set()
    for f in files:
        pr, se, m = load(RES / folder / f)
        P.append(pr)
        S.append(se)
        wset.add(m["weights"])
        pset.add(tuple(m["power"]))
    P, S = np.concatenate(P), np.concatenate(S)
    assert len(set(S.tolist())) == len(S), "duplicate images across files!"
    y = np.array([LAB[i] for i in S])
    ok = int((P == y).sum())
    top = collections.Counter(P.tolist()).most_common(1)[0]
    print(f"  {name:26s} N={len(P)}  correct={ok:3d}  acc={100*ok/len(P):6.2f}%")
    print(f"  {'':26s} weights={sorted(wset)}  power={sorted(pset)}")
    print(f"  {'':26s} most-predicted class {top[0]} x{top[1]}\n")
    return ok, len(P), S


print("REPORTED RESULT — hardware, (0,0) dBm, N=600\n")
c_ok, c_n, c_s = arm("clean (ideal-mult. train)", "clean_arm", CLEAN)
t_ok, t_n, t_s = arm("twin (hardware-aware)", "twin_arm_hw_aware", TWIN)

assert np.array_equal(np.sort(c_s), np.sort(t_s)), "arms saw different images"
bat = np.load(HERE / "08_frozen_inputs_and_labels" / "battery_random1200_idx.npy")
print(f"  both arms ran the identical image set = battery[0:600] : "
      f"{np.array_equal(np.sort(c_s), np.sort(bat[:600]))}")
print(f"  (frozen battery holds {len(bat)} images; 600 used)\n")

print("Digital comparators, pinned BEFORE the hardware runs")
d = np.load(HERE / "06_digital_comparators" / "digital_pins_seg300600.npz",
            allow_pickle=True)
y2 = np.array([LAB[i] for i in d["sel"]])
pin1 = json.load(open(HERE / "06_digital_comparators" / "serial_predictions.json"))
seg1 = {"twin_under_twin": pin1["twin_under_twin_n300_frame"],
        "clean_under_twin": pin1["clean_under_twin_n300"]}
for k in ("clean_ideal", "twin_under_twin", "clean_under_twin"):
    n2 = int((d[k] == y2).sum())
    if k in seg1:
        n1 = round(seg1[k] / 100 * 300)
        print(f"  {k:18s} seg1 {n1:3d}/300 (pinned {seg1[k]:.2f}%)  "
              f"seg2 {n2:3d}/300 ({100*n2/300:.2f}%)  "
              f"pooled {n1+n2:3d}/600 = {100*(n1+n2)/600:.2f}%")
    else:
        print(f"  {k:18s} seg1  —  (audit: 99.50% on the battery frame)      "
              f"seg2 {n2:3d}/300 ({100*n2/300:.2f}%)")
