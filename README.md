# Machine intelligence on wireless edge networks — data and code

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22196371.svg)](https://doi.org/10.5281/zenodo.22196371)

Data and code underlying the figures, tables and reported numbers of

> **Machine intelligence on wireless edge networks**  
> Mingran Jia\*, Sri Krishna Vadlamani\*, Kfir Sulimany\*, Jonathan Morag,  
> Zhihui Gao, Hanfeng Wang, Tingjun Chen, Dirk Englund  
> \*These authors contributed equally to this work.  
>  
> Research Laboratory of Electronics, Massachusetts Institute of Technology,
> Cambridge, MA 02139, USA  
> Department of Electrical and Computer Engineering, Duke University,
> Durham, NC 27708, USA  
>  
> Correspondence: englund@mit.edu

---

## 1. Layout

The package is split into the two trees a *Data availability* and a *Code availability*
statement refer to:

```
Data_availability/          measured and simulated data
  Figure_1/ ... Figure_5/          source data behind each main-text figure
  Supplementary_Figure_1/  _2/     source data behind each supplementary figure
  Supplementary_Table_1/ ... _6/   source data behind each supplementary table
  Supplementary_Note_1/ ... _8/    data behind the numeric claims of each note
  Response_Letter_Figure_1/ _3/ _4/   items that appear only in the response letter
  raw/                             upstream acquisition and training campaigns
    scalar_PIML_calibration/         43x43 scalar mixer sweep and PIML twin fit
    twin_calibration_N4096/          N = 4,096 digital-twin calibration
    vector_heatmaps_N2N4N8/          N = 2/4/8 vector inner-product heatmaps
    vector_heatmaps_N4096/           N = 4,096 vector inner-product heatmaps
    ip_scatter_sweep/                N = 65,536 / 4,096 inner-product campaigns
    comb_GTSRB_campaign/             frequency-comb GTSRB inference battery
    serial_hardware_campaign/        serial hardware-aware training and capture
    energy_package/                  client-side energy model and its inputs
    energy_budget_nn/                GTSRB-CNN client-energy audit (Fig. 3g overlay)
    fully_analog_simulation/         fully analog cascade: code, weights, results
    Figure_1/ ... Figure_5/          per-panel material specific to one figure

Code_availability/          acquisition, analysis and figure-generation code
  figure_scripts/                  one folder per figure, table and note,
                                   mirroring Data_availability/ exactly
  pipelines/                       the upstream campaigns, mirroring
                                   Data_availability/raw/ exactly
  _paths.py                        resolves a script's data folder (see below)
```

**The two trees mirror each other.** `Code_availability/figure_scripts/Figure_3/`
holds the code whose data is in `Data_availability/Figure_3/`;
`Code_availability/pipelines/energy_package/` pairs with
`Data_availability/raw/energy_package/`. Every script obtains its own data folder
from `_paths.data_dir(__file__)`, so scripts read their inputs — and write their
outputs — into the matching folder of the data tree. Nothing needs to be
configured or edited to run them from a fresh clone.

---

## 2. Index

### Main text

| Figure | Figure script | Source data | Upstream campaigns |
|---|---|---|---|
| **Fig. 1** — concept and theory | `figure_scripts/Figure_1/fig1_compose_v3.py` | `Figure_1/` | `raw/scalar_PIML_calibration/`, `raw/Figure_1/` |
| **Fig. 2** — digital twin (a–k) | `figure_scripts/Figure_2/fig2_v3_twin.py` | `Figure_2/`, `Figure_3/` | `raw/vector_heatmaps_N2N4N8/`, `raw/vector_heatmaps_N4096/`, `raw/twin_calibration_N4096/` |
| **Fig. 3** — payoff and energy (a–g) | `figure_scripts/Figure_3/fig3_v12_payoff.py` | `Figure_3/` | `raw/ip_scatter_sweep/`, `raw/energy_package/`, `raw/energy_budget_nn/`, `raw/twin_calibration_N4096/` |
| **Fig. 4** — CNN on a passive mixer (a–e) | `figure_scripts/Figure_4/fig4_v6.py` | `Figure_4/` | `raw/comb_GTSRB_campaign/`, `raw/serial_hardware_campaign/` |
| **Fig. 5** — fully analog cascade (a–d) | `figure_scripts/Figure_5/fig5_analog_v3.py` | `Figure_5/` | `raw/fully_analog_simulation/` |

Asset-preparation scripts live beside the figure script they feed:
`figure_scripts/Figure_4/prep_comb_assets.py`, `prep_fig4_assets.py`,
`fig4_verify_package.py`, `build_cache_from_master.py` and
`figure_scripts/Figure_5/prep_fig5_assets.py`.

### Supplementary information

| Item | Code | Data |
|---|---|---|
| Supplementary Figure 1 — latency and bandwidth | `figure_scripts/Supplementary_Figure_1/` | `Supplementary_Figure_1/` |
| Supplementary Figure 2 — analog layer scatter | `figure_scripts/Supplementary_Figure_2/` | `Supplementary_Figure_2/` |
| Supplementary Tables 1–6 | `figure_scripts/Supplementary_Table_<n>/` | `Supplementary_Table_<n>/` |
| Numeric claims of Supplementary Notes 1–8 | `figure_scripts/Supplementary_Note_<n>/` | `Supplementary_Note_<n>/` |

Supplementary Table 3 and Supplementary Note 2 are closed-form computations with
no input data, so they appear only under `Code_availability/`.

### Response letter

Three display items appear only in the response to the reviewers: Letter Fig. 1
(parallel-broadcast schematic), Letter Fig. 3 (a crop of main Fig. 3g) and
Letter Fig. 4 (main Fig. 5 reproduced verbatim). They are kept separate, under
`Response_Letter_Figure_1/`, `_3/` and `_4/` in both trees.

Each folder carries its own `README.md` describing the provenance of the item.

---

## 3. Requirements

* **Python 3** (tested on 3.13) with `numpy` 2.5.1, `matplotlib` 3.11.0 and `Pillow`
* **JAX (CPU)** — only for the fully analog simulation (Fig. 5)
* **Git LFS** — see §4

All simulations are CPU-only and use fixed seeds; no GPU is required.

To regenerate a main-text figure:

```bash
cd Code_availability/figure_scripts/Figure_3
python fig3_v12_payoff.py        # writes into Data_availability/Figure_3/
```

A small number of acquisition scripts under `Code_availability/pipelines/` drove
instruments over GPIB/USRP and cannot run without that hardware; they are
included as the record of how the measurements were taken. Likewise
`figure_scripts/Figure_4/build_cache_from_master.py` rebuilds the GTSRB cache
from the public GTSRB distribution, which is not redistributed here — its
`<archive>` path constant must be pointed at a local copy of that dataset.

---

## 4. Large files (Git LFS)

`Data_availability/raw/fully_analog_simulation/data/gtsrb_roi_32x32.npz` (127 MB)
exceeds GitHub's 100 MB per-file limit and is stored with
[Git LFS](https://git-lfs.com). Everything else is in plain git.

Install Git LFS before cloning, or run `git lfs pull` afterwards; without it the
file arrives as a small text pointer and the fully analog simulation will not
run. It can also be rebuilt from the public GTSRB distribution with
`Code_availability/pipelines/fully_analog_simulation/prepare_gtsrb.py`.

**If you obtained this package from the Zenodo archive rather than by cloning
GitHub, this one file is a 134-byte LFS pointer, not the data.** GitHub's
automatically generated release archives do not resolve Git LFS objects, so the
archived snapshot carries the pointer in its place. Obtain the file either by
cloning the GitHub repository with Git LFS installed, or by rebuilding it from
the public GTSRB distribution with the `prepare_gtsrb.py` script named above.
It is the only affected file — every other file in the archive is complete.

---

## 5. License

* **Code** (`*.py`) — [MIT](LICENSE)
* **Data, figures and documentation** (`*.npz`, `*.npy`, `*.pt`, `*.json`, `*.csv`,
  `*.png`, `*.pdf`, `*.svg`, `*.md`) — [CC BY 4.0](LICENSE-DATA)

The GTSRB and MNIST caches redistributed here are derived from their original
public distributions and remain subject to those datasets' own terms.
