# Response Letter Fig. 3 — fig3g_energy_panel.png (client-energy scaling panel)

## Purpose

The response letter's Fig. 3 (`figures/fig3g_energy_panel.png`) is **panel g of
manuscript Fig. 3** — the client-energy scaling panel. It is not an
independently generated figure: it is a rectangular crop of the frozen
manuscript figure preview `fig3_v12_preview.png` (4367 x 2409 px), taking the
right-hand column that contains panel g.

This folder documents that **crop step end-to-end**: the exact source image,
the exact crop script, the published output, and a byte-level verification that
re-running the crop reproduces the published file.

## Upstream chain (not duplicated here)

The full data-to-figure chain for the panel itself is deliberately **not**
duplicated in this folder. It lives, pixel-verified, in the frozen manuscript
reproducibility archive at:

    fig3\

which contains the Fig. 3 generator `fig3_v12_payoff.py` (producing
`fig3_v12_preview.png`) and, under `fig3\g\` (the `fig5_energy_package`), the
panel-g energy package with the raw measurement-campaign `.npz` files and all
verification asserts. For the panel's own data chain, consult that archive;
this folder only covers the crop from the frozen preview to the response-letter
figure.

## Contents

| File | Role |
|---|---|
| `source_from_manuscript_fig3_preview.png` | Copy of the frozen manuscript preview `fig3\fig3_v12_preview.png` (4367 x 2409) |
| `published_fig3g_energy_panel.png` | Copy of the live published crop `figures\fig3g_energy_panel.png` used by the response letter |
| `crop_fig3g_original_recovered.py.txt` | The original crop script as recovered (absolute paths; crop box `(int(w*0.706), 0, w, h)`) |
| `crop_fig3g.py` | Runnable adaptation: reads `source_from_manuscript_fig3_preview.png` in this folder, writes `reproduced/fig3g_energy_panel.png` |
| `reproduced/fig3g_energy_panel.png` | Output of running `crop_fig3g.py` here |

Note: the recovered original script read the preview from an earlier
archive path; that file is byte-identical (same MD5) to the frozen copy
used here, so the two paths refer to the same image.

## How to re-run

    cd "Response_Letter&Supplementary_Reproductivity\fig3_fig3g_energy_panel"
    python crop_fig3g.py

Requires Python with Pillow (`pip install pillow`). The script prints the
source size and crop box `(3083, 0, 4367, 2409)` and writes
`reproduced/fig3g_energy_panel.png` (1284 x 2409). Then compare MD5s of
`reproduced/fig3g_energy_panel.png` and `published_fig3g_energy_panel.png`.

## Verification results (2026-08-29)

MD5 checksums:

| File | MD5 |
|---|---|
| `source_from_manuscript_fig3_preview.png` (this folder) | `d6972bff73ed63c7ad3170636c2058e1` |
| Frozen archive `fig3\fig3_v12_preview.png` | `d6972bff73ed63c7ad3170636c2058e1` |
| `V2_Manu\fig3\fig3_v12_preview.png` (path in the recovered script) | `d6972bff73ed63c7ad3170636c2058e1` |
| `published_fig3g_energy_panel.png` (this folder) | `24fa39ba4eeb4dc6d5c621addecdd6ee` |
| Live `figures\fig3g_energy_panel.png` | `24fa39ba4eeb4dc6d5c621addecdd6ee` |
| `reproduced/fig3g_energy_panel.png` (output of `crop_fig3g.py`) | `24fa39ba4eeb4dc6d5c621addecdd6ee` |

Outcomes:

- Source preview in this folder is **byte-identical** to the frozen manuscript
  archive's copy (and to the V2 copy the original script read).
- Published copy in this folder is **byte-identical** to the live
  response-letter figure.
- Re-running `crop_fig3g.py` reproduces the published crop **byte-for-byte**
  (identical MD5), so no pixel-level comparison was necessary.

## Update 2026-08-31 — green CNN inference overlay

Manuscript Fig. 3g gained the green GTSRB-CNN inference overlay (four layer
circles + full-CNN star, 15.7 fJ per real MAC, 4.5× below H100; see
`../Figure_3/README.md` and `../../pipelines/energy_budget_nn/`), so the
manuscript preview, the response-letter figure, and this folder's chain were
regenerated. The crop step is unchanged (same script, same crop box
`(3083, 0, 4367, 2409)`).

Verification (2026-08-31) — the MD5 table above documents the 2026-08-29
pre-overlay version; the current files are:

| File | MD5 |
|---|---|
| `source_from_manuscript_fig3_preview.png` = frozen archive `fig3\fig3_v12_preview.png` = `Data_availability/Figure_3/fig3_v12_preview.png` | `2f07993ccd3bb9b82e08383079e0c3ac` |
| `published_fig3g_energy_panel.png` = live `figures\fig3g_energy_panel.png` | `6e1a7a9763d853120603f27e4d5042a2` |
| `reproduced/fig3g_energy_panel.png` (output of `crop_fig3g.py`, re-run 2026-08-31) | `6e1a7a9763d853120603f27e4d5042a2` |

Re-running `crop_fig3g.py` again reproduces the published crop
**byte-for-byte**.
