"""Train/test contamination audit for gtsrb_roi_32x32.npz.

GTSRB's official benchmark split puts different physical sign TRACKS in
train vs test (30-frame tracks per instance). If our cache honors it, no
test image should have a near-zero-distance neighbor in train. Checks:
(1) set sizes vs official; (2) exact-duplicate hashes; (3) per-test-image
nearest-neighbor L2 distance into train, benchmarked against the
within-train track-neighbor distance distribution (what leakage looks like).
Audits the full eval frames (0-599) plus a 1000-image random test sample.
"""
import hashlib, sys
import numpy as np

z = np.load("gtsrb_roi_32x32.npz", allow_pickle=True)
Xtr, ytr, Xte, yte = z["Xtr"], z["ytr"], z["Xte"], z["yte"]
print(f"sizes: train {Xtr.shape} test {Xte.shape} "
      f"(official GTSRB: 39209 / 12630)")

htr = {hashlib.sha1(a.tobytes()).hexdigest() for a in Xtr}
dup = sum(1 for a in Xte if hashlib.sha1(a.tobytes()).hexdigest() in htr)
print(f"exact u8 duplicates test-in-train: {dup} / {len(Xte)}")

import torch
dev = "cuda" if torch.cuda.is_available() else "cpu"
T = torch.tensor(Xtr.reshape(len(Xtr), -1).astype(np.float32) / 255.0,
                 device=dev)
Tn = (T * T).sum(1)

def nn_dist(Q):
    Qt = torch.tensor(Q.reshape(len(Q), -1).astype(np.float32) / 255.0,
                      device=dev)
    out = []
    for c0 in range(0, len(Qt), 256):
        q = Qt[c0:c0 + 256]
        d2 = (q * q).sum(1)[:, None] + Tn[None, :] - 2.0 * (q @ T.T)
        out.append(torch.sqrt(torch.clamp(d2.min(1).values, min=0)).cpu())
    return torch.cat(out).numpy() / np.sqrt(3072)   # per-pixel RMS distance

rng = np.random.RandomState(0)
eval600 = nn_dist(Xte[:600])
sample = nn_dist(Xte[rng.choice(len(Xte), 1000, replace=False)])
# leakage benchmark: distance of a train image to its own track neighbors
# (adjacent images in the train file are track-ordered in GTSRB)
tri = rng.choice(len(Xtr) - 1, 2000, replace=False)
a = Xtr[tri].reshape(2000, -1).astype(np.float32) / 255.0
b = Xtr[tri + 1].reshape(2000, -1).astype(np.float32) / 255.0
same_track = np.sqrt(((a - b) ** 2).sum(1)) / np.sqrt(3072)
st = same_track[ytr[tri] == ytr[tri + 1]]         # same-class adjacent pairs

for name, d in (("eval frames 0-599", eval600), ("random 1000 test", sample)):
    print(f"{name}: NN-into-train per-pixel RMS dist "
          f"min {d.min():.4f} p1 {np.percentile(d,1):.4f} "
          f"median {np.median(d):.4f}")
print(f"track-neighbor benchmark (what leakage looks like): "
      f"median {np.median(st):.4f} p90 {np.percentile(st,90):.4f}")
n_below = int((eval600 < np.percentile(st, 90)).sum())
print(f"eval images with NN-dist below track-neighbor p90: {n_below} / 600")
