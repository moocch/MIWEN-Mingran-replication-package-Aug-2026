"""
prep_fig5_assets.py — build data/input_glyphs.npz for fig5_analog_v2.py
=======================================================================
Inputs (both are existing project assets; no new measurements):
  * ../fig4_v6/data/comb_assets.npz  -> photo_u8 (GTSRB battery STOP photo,
    the same photograph shown in Fig. 4a)
  * mnist.npz from the fully-analog simulation archive
    (V2/fully analog/files (1).zip -> miwen_fully_analog_archive.zip ->
    code/data/mnist.npz) -> one test digit, the same cache the simulation
    trains and tests on.  Pass its local path as argv[1].

Output: data/input_glyphs.npz  {digit_u8 (28x28), photo_u8 (HxWx3)}
"""

# --- replication-package path resolution (see Code_availability/_paths.py) ---
import sys as _sys, pathlib as _pl
for _p in _pl.Path(__file__).resolve().parents:
    if _p.name == "Code_availability":
        _sys.path.insert(0, str(_p)); break
from _paths import data_dir as _data_dir
# ---------------------------------------------------------------------------

import os
import sys

import numpy as np

OUT = str(_data_dir(__file__))
D4 = os.path.join(os.path.dirname(OUT), "fig4_v6", "data")

photo = np.load(os.path.join(D4, "comb_assets.npz"))["photo_u8"]

mnist_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    OUT, "data", "mnist.npz")
z = np.load(mnist_path)
Xte, yte = z["Xte"], z["yte"]
# a clean, thick test '3' (deterministic pick: the class-3 test digit at
# the 75th percentile of ink mass)
idx3 = np.flatnonzero(yte == 3)
mass = Xte[idx3].reshape(len(idx3), -1).astype(np.float64).sum(1)
pick = idx3[np.argsort(mass)[int(0.75 * len(idx3))]]
digit = Xte[pick].reshape(28, 28).astype(np.uint8)

np.savez_compressed(os.path.join(OUT, "data", "input_glyphs.npz"),
                    digit_u8=digit, photo_u8=photo,
                    digit_test_index=np.int64(pick))
print(f"wrote data/input_glyphs.npz (digit = test index {pick}, "
      f"label {int(yte[pick])}; photo {photo.shape})")
