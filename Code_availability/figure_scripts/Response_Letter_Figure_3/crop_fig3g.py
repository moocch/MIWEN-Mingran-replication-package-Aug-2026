"""Reproduce the response-letter Fig. 3 crop (panel g of manuscript Fig. 3).

Adapted from crop_fig3g_original_recovered.py.txt (same crop box). Paths are
made relative to this folder so the script is runnable from the archive:
  input : source_from_manuscript_fig3_preview.png  (copy of the frozen
          manuscript preview fig3_v12_preview.png, 4367x2409)
  output: reproduced/fig3g_energy_panel.png

Usage:  python crop_fig3g.py
Requires: Pillow (PIL).
"""

# --- replication-package path resolution (see Code_availability/_paths.py) ---
import sys as _sys, pathlib as _pl
for _p in _pl.Path(__file__).resolve().parents:
    if _p.name == "Code_availability":
        _sys.path.insert(0, str(_p)); break
from _paths import data_dir as _data_dir
# ---------------------------------------------------------------------------


import os

from PIL import Image

HERE = str(_data_dir(__file__))
SRC = os.path.join(HERE, "source_from_manuscript_fig3_preview.png")
DST = os.path.join(HERE, "reproduced", "fig3g_energy_panel.png")

os.makedirs(os.path.dirname(DST), exist_ok=True)

im = Image.open(SRC)
print("source size:", im.size)
w, h = im.size
# panel g occupies the right-hand column, full height
box = (int(w * 0.706), 0, w, h)
im.crop(box).save(DST)
print("cropped box:", box, "->", DST)
