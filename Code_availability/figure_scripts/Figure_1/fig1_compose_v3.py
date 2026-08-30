"""
fig1_compose_v3.py
==================
v3 of the full Figure 1 composite — does NOT overwrite fig1_preview.png (v1).

Panel (a) is the self-contained Blender render fig1a_system_v5.png
(labels are baked into the render, so no matplotlib labeling pass is
needed).  Panels (b,c) and (d,e) are the existing matplotlib panels.

Figure geometry (same as v1):  180 mm x 66 mm @ 600 dpi
  (a)   0 -  62 mm   fig1a_system_v5.png   (Blender, 62:66 aspect)
  (b,c) 65 - 121 mm  fig1b_motivation.png
  (d,e) 124 - 180 mm fig1c_theory.png

Outputs (new files only):
  fig1_preview_v3.png        600 dpi full-width composite
  fig1_preview_v3_small.png  1/3-size quick-look
  fig1a_system_v5.pdf        vector-wrapped panel (a) for LaTeX includes

Usage:  python fig1_compose_v3.py
"""

# --- replication-package path resolution (see Code_availability/_paths.py) ---
import sys as _sys, pathlib as _pl
for _p in _pl.Path(__file__).resolve().parents:
    if _p.name == "Code_availability":
        _sys.path.insert(0, str(_p)); break
from _paths import data_dir as _data_dir
# ---------------------------------------------------------------------------


import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

OUT = str(_data_dir(__file__))
MM = 1 / 25.4
DPI = 600
mm2px = DPI / 25.4

# ------------------------------------------------- panel (a) PDF wrapper ----
# wrap the high-res PNG in a 62 x 66 mm PDF so LaTeX can include it just
# like the b/c panel PDFs (fig1_include.tex style)
img = Image.open(os.path.join(OUT, "fig1a_system_v5.png")).convert("RGB")
arr = np.asarray(img)
fig = plt.figure(figsize=(62 * MM, 66 * MM))
ax = fig.add_axes([0, 0, 1, 1])
ax.imshow(arr, interpolation="lanczos")
ax.axis("off")
fig.savefig(os.path.join(OUT, "fig1a_system_v5.pdf"), dpi=DPI)
plt.close(fig)
print("wrote fig1a_system_v5.pdf")

# ------------------------------------------------------------- composite ----
W_TOT, H_TOT = int(round(180 * mm2px)), int(round(66 * mm2px))
canvas = Image.new("RGB", (W_TOT, H_TOT), (255, 255, 255))


def paste(path, x_mm, w_mm):
    im = Image.open(os.path.join(OUT, path)).convert("RGB")
    target_w = int(round(w_mm * mm2px))
    target_h = H_TOT
    im = im.resize((target_w, target_h), Image.LANCZOS)
    canvas.paste(im, (int(round(x_mm * mm2px)), 0))


paste("fig1a_system_v5.png", 0, 62)
paste("fig1b_motivation.png", 65, 56)
paste("fig1c_theory.png", 124, 56)

canvas.save(os.path.join(OUT, "fig1_preview_v3.png"), dpi=(DPI, DPI))
canvas.resize((W_TOT // 3, H_TOT // 3), Image.LANCZOS).save(
    os.path.join(OUT, "fig1_preview_v3_small.png"))
print("wrote fig1_preview_v3.png (600 dpi) + fig1_preview_v3_small.png")

# PDF wrapper of the full composite — this is what main_PANS.tex includes
arr2 = np.asarray(canvas)
fig = plt.figure(figsize=(180 * MM, 66 * MM))
ax = fig.add_axes([0, 0, 1, 1])
ax.imshow(arr2, interpolation="lanczos")
ax.axis("off")
fig.savefig(os.path.join(OUT, "fig1_preview_v3.pdf"), dpi=DPI)
plt.close(fig)
print("wrote fig1_preview_v3.pdf")
