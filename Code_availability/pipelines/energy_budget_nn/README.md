# energy_budget_nn — client-side energy audit of the GTSRB CNN (Fig. 3g green overlay)

**What it is:** the client-energy audit that prices the fielded comb-encoded GTSRB CNN in the
paper's Eq.-(energy) / Sec. II D convention (e1 transmit at the mixer plane, 6 ADC conversions
+ 8 digital MACs = 14 pJ read-out fee per answer, eta_radio = 10%, H100 reference 70 fJ per
real MAC), after first reproducing the two Sec.-II-D primitive operating points exactly. Its
JSON export carries `comb_nn.fig3g_overlay_points` — the four per-layer points and the
full-CNN aggregate drawn as the **green overlay of manuscript Fig. 3g**.

**Origin:** the rung3-session audit of the lab repository
(`QPG-MIT/MIWEN_Mingran` @ `handoff/rung3-session`, `analysis/energy_budget_nn.py`), archived
here scoped to the comb encoding — the encoding priced in Fig. 3g; every comb number is
unchanged from that audit.

| item | role |
|---|---|
| `energy_budget_nn.py` | the audit: Sec.-II-D reproduction and the comb CNN at (P_LO, P_RF) = (−3, −65) dBm; exports the JSON. Every constant is tagged [paper] / [repo] / [derived] in the source |
| `../../../Data_availability/raw/energy_budget_nn/energy_budget_nn_results.json` | the JSON export (working copy read by the Fig. 3 script: `Data_availability/Figure_3/`) |

## Key results (all frozen in the JSON and asserted by the Fig. 3 figure script)

- **Comb CNN** at the as-fielded (−3, −65) dBm: 1,628 OFDM symbols + 3.1% sync = 2.8366272 s
  airtime per image for 28,847,616 real MACs (98 ns per real MAC) and 31,659 read-out answers →
  e1 = 9.0 nJ + fee = 443.2 nJ = **452 nJ per image = 15.7 fJ per real MAC, 4.5× below H100**,
  98% read-out-fee dominated. Per layer (same flat e1 + 14 pJ/(4N)): conv1 N = 75 → 46.98 fJ,
  dense2 N = 128 → 27.65 fJ, conv2 N = 800 → 4.69 fJ, dense1 N = 1600 → 2.50 fJ; the aggregate
  sits at the answer-averaged N = 28,847,616 / (4 × 31,659) ≈ 228.
- Ledger boundary (Sec. II D conventions): the client pays the data comb only — the −3-dBm
  weight broadcast is base-station-side — and DAC synthesis, static transceiver power, and the
  laboratory SDR's capture budget are excluded.

## Runnability note

`energy_budget_nn.py` derives the comb symbol counts by importing the fielded frame planner
(`plan_layer` from the campaign codebase, `share_20260712/4_gtsrb_confusion_mlpN.py` in the lab
repository), so it is archived here as provenance rather than as a standalone-runnable script.
Every derived primitive the manuscript needs (airtime, MAC and answer counts, per-layer D) is
frozen in the JSON; the Fig. 3 figure script recomputes the overlay from those primitives and
**asserts** the exported point values, so the figure is reproducible from this package alone.
