"""Build gtsrb_roi_32x32_test.npz from the already-extracted mirror folder
(German-Traffic-Signs-Dataset-GTSRB-master), mirroring
fig4_v4/build_gtsrb_test_cache.py exactly, then gate-check against the
archived battery labels."""
import csv
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(r"E:\archive\Manuscript"
            r"\Overleaf\202607_MIWEN_Manuscript")
SRC = ROOT / "fig4_v5" / "data" / "German-Traffic-Signs-Dataset-GTSRB-master"
OUT = ROOT / "fig4_v5" / "data" / "gtsrb_roi_32x32_test.npz"
IMG = 32

rows = list(csv.DictReader(open(SRC / "Test.csv")))
X = np.empty((len(rows), IMG, IMG, 3), np.uint8)
y = np.empty(len(rows), np.int64)
for i, r in enumerate(rows):
    im = Image.open(SRC / r["Path"]).convert("RGB")
    im = im.crop((int(r["Roi.X1"]), int(r["Roi.Y1"]),
                  int(r["Roi.X2"]), int(r["Roi.Y2"])))
    X[i] = np.asarray(im.resize((IMG, IMG), Image.BILINEAR), np.uint8)
    y[i] = int(r["ClassId"])
np.savez_compressed(OUT, Xte=X, yte=y)
print(f"wrote {OUT}  N={len(rows)}")

# gate 1: labels of the frozen battery must match yte[img_index]
d = np.load(ROOT / "V2" / "hardware_aware_training_v2"
            / "08_frozen_inputs_and_labels" / "battery_frozen_slim.npz")
img_index, labels = d["img_index"], d["labels"]
ok = int((y[img_index] == labels).sum())
print(f"gate1 battery-label match: {ok}/{len(labels)}")
assert ok == len(labels), "battery labels mismatch"
print("GATE PASSED")
