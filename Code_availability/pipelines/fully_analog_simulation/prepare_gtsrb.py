# -*- coding: utf-8 -*-
"""prepare_gtsrb.py -- rebuild data/gtsrb_roi_32x32.npz from the official
GTSRB archives, exactly as used in the main text.

    python3 prepare_gtsrb.py /path/to/GTSRB_Final_Training_Images \
                             /path/to/GTSRB_Final_Test_Images \
                             /path/to/GT-final_test.csv

Each image is cropped to its annotated region of interest, resized to
32 x 32 RGB and flattened to 3072 floats in [0, 1]; labels are the 43
class indices.  The resulting file is what comb_analog_sim.load_data()
expects (keys: Xtr, ytr, Xte, yte).
"""
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def load_split(root, csv_rows, base):
    X, y = [], []
    for r in csv_rows:
        p = Path(base) / r["Filename"] if "Filename" in r else None
        im = Image.open(p).convert("RGB")
        box = (int(r["Roi.X1"]), int(r["Roi.Y1"]),
               int(r["Roi.X2"]), int(r["Roi.Y2"]))
        im = im.crop(box).resize((32, 32), Image.BILINEAR)
        X.append(np.asarray(im, np.float32).reshape(-1) / 255.0)
        y.append(int(r["ClassId"]))
    return np.stack(X), np.asarray(y, np.int32)


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f, delimiter=";"))


if __name__ == "__main__":
    train_root, test_root, test_csv = sys.argv[1:4]
    Xtr, ytr = [], []
    for cls_dir in sorted(Path(train_root).glob("*")):
        if not cls_dir.is_dir():
            continue
        rows = read_csv(next(cls_dir.glob("GT-*.csv")))
        Xc, yc = load_split(None, rows, cls_dir)
        Xtr.append(Xc); ytr.append(yc)
    Xtr = np.concatenate(Xtr); ytr = np.concatenate(ytr)
    Xte, yte = load_split(None, read_csv(test_csv), test_root)
    np.savez_compressed("data/gtsrb_roi_32x32.npz",
                        Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte)
    print("wrote data/gtsrb_roi_32x32.npz", Xtr.shape, Xte.shape)
