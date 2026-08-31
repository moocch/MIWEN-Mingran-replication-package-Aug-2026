#!/usr/bin/env python3
"""Client-side energy budget for the GTSRB CNN, in the convention of the
paper's Sec. II D / Eq. (5) — and, first, a reproduction of Sec. D itself.

Eq. (5) convention (benchmark methodology of ref [31]):
    e_ip = e1 + e2 + e3
    e1 = Px * T_air / (4N * eta_radio)   # TX: power AT THE MIXER, radio eff 10%
    e2 = 6 * e_ADC / 4N                  # 6 ADC conversions per answer
    e3 = 8 * e_dig / 4N                  # 8 digital MACs per answer
    eta_radio = 0.10, e_ADC = e_dig = 1 pJ, answer = one complex inner product
    (= 4N real MACs). Component-level, input-referred: no DAC, no static
    transceiver power, no SDR wall-plug — same exclusions as the paper.

Every constant is tagged [paper] (stated in the 2026-08-31 draft),
[repo] (from committed code/logs), or [derived] (computed here).

Run directly for the console summary; `make_energy_report.py` imports the
compute functions below and typesets the same numbers into a PDF.
"""
import math
import sys
import importlib.util
from pathlib import Path

# ---------------------------------------------------------------- constants
ETA_RADIO = 0.10          # [paper] Methods G
E_ADC = 1e-12             # [paper] 1 pJ per conversion
E_DIG = 1e-12             # [paper] 1 pJ per digital MAC
FEE_PER_ANSWER = 6 * E_ADC + 8 * E_DIG   # [paper] "fixed 14 pJ per answer"
H100 = 70e-15             # [paper] H100 arithmetic energy, fJ/real MAC

FS = 10e6                 # [repo] miwen_serial_frozen_reference.py / core
L, CP = 16384, 512        # [repo] comb FFT length and cyclic prefix
T_SYM = (L + CP) / FS     # [derived] 1.6896 ms per OFDM symbol
T_SLOT = 32 / FS          # [repo] serial slot: 32 samples = 3.2 us

T_PER_RMAC = 117e-9       # [paper] "every real MAC receives the same 117 ns"
PX_65536 = 0.46e-9        # [paper] "0.46 nW at the mixer for N = 65,536"

P_COMB_DBM = -65.0        # [repo] data-comb power INSERTED at the mixer RF
                          # port — the demonstrational operating point of the
                          # frozen-battery runs (as physically delivered)
P_SER_DBM = -24.0         # [repo] serial_nn_runner_m10m24.py POWER=(-24,-10)
                          # (RF, LO); unpadded chain -> commanded ~ device

SYNC_FRAC = 1 / 32        # [assumption] 1 sync symbol per 32 payload symbols

# r3plus geometry [repo: r35_r3plus_*_hw.npz / cost_anatomy.py]
# (name, D per inner product, H outputs, vectors per image)
LAYERS = [("conv1", 75, 32, 784), ("conv2", 800, 64, 100),
          ("dense1", 1600, 128, 1), ("dense2", 128, 43, 1)]


def dbm(p_dbm):           # dBm -> W
    return 1e-3 * 10 ** (p_dbm / 10.0)


# ============================================================ 1. Sec. D
def sec_d():
    """Reproduce the two measured operating points of Sec. II D."""
    rows = []
    for N, px in ((4096, None), (65536, PX_65536)):
        fourN = 4 * N
        fee = FEE_PER_ANSWER / fourN
        if px is None:
            # Px not stated for N=4096; back it out from the printed 1.47 fJ
            e1 = 1.47e-15 - fee
            px = e1 * ETA_RADIO / T_PER_RMAC
            tag = "derived from 1.47 fJ"
        else:
            e1 = px * T_PER_RMAC / ETA_RADIO
            tag = "paper"
        rows.append(dict(N=N, px=px, tag=tag, e1=e1, fee=fee, eip=e1 + fee))
    e1_flat = PX_65536 * T_PER_RMAC / ETA_RADIO
    return dict(rows=rows,
                n_cross=FEE_PER_ANSWER / (4 * (H100 - e1_flat)),
                n_bend=FEE_PER_ANSWER / (4 * e1_flat),
                vs_h100=[H100 / 1.47e-15, H100 / 0.59e-15])


# ==================================================== 2. comb CNN, (-3,-65)
def _load_core():
    here = Path(__file__).resolve().parent.parent / "share_20260712"
    sys.path.insert(0, str(here))
    spec = importlib.util.spec_from_file_location(
        "core", here / "4_gtsrb_confusion_mlpN.py")
    core = importlib.util.module_from_spec(spec)
    sys.modules["core"] = core
    spec.loader.exec_module(core)
    return core


def comb_nn():
    """Comb CNN at device-plane (LO,RF)=(-3,-65): exact symbol counts from
    the fielded frame planner (same call as miwen_overhead_ratio.py)."""
    core = _load_core()

    def comb_layer(D, H, Nvec):
        for g in range(1, 17):
            try:
                p = core.plan_layer(math.ceil(D / g), H, L, 6, 0.5, 64)
                return Nvec * p.R * g, p.K, p.R, g
            except Exception:
                continue
        raise RuntimeError(f"{D}->{H}")

    layers, tot_sym = [], 0
    for name, D, H, Nvec in LAYERS:
        s, K, R, g = comb_layer(D, H, Nvec)
        tot_sym += s
        layers.append(dict(name=name, D=D, H=H, Nvec=Nvec, sym=s, K=K, R=R,
                           g=g, cmac=D * H * Nvec, answers=H * Nvec,
                           fee_rmac=FEE_PER_ANSWER / (4 * D)))
    cmac = sum(l["cmac"] for l in layers)
    ans = sum(l["answers"] for l in layers)
    rmac = 4 * cmac
    t_air = tot_sym * T_SYM * (1 + SYNC_FRAC)
    e1 = dbm(P_COMB_DBM) * t_air / ETA_RADIO
    fee = ans * FEE_PER_ANSWER
    return dict(layers=layers, tot_sym=tot_sym, cmac=cmac, answers=ans,
                rmac=rmac, t_air=t_air, e1_img=e1, fee_img=fee,
                tot_img=e1 + fee, tot_rmac=(e1 + fee) / rmac,
                vs_h100=H100 / ((e1 + fee) / rmac))


# ================================================== 3. serial CNN, (-10,-24)
def serial_nn(comb):
    """Serial CNN at (LO,RF)=(-10,-24): one complex product per 3.2-us slot.
    Readout fee uses the Sec.-D benchmark convention (6 conversions + 8
    digital MACs per answer), i.e. it ASSUMES the slow coherent
    integrate-and-dump readout at the row rate — not the as-fielded
    per-sample SDR capture (reported separately)."""
    rmac, ans = comb["rmac"], comb["answers"]
    t_air = comb["cmac"] * T_SLOT
    e1 = dbm(P_SER_DBM) * t_air / ETA_RADIO
    fee = ans * FEE_PER_ANSWER
    return dict(t_air=t_air, e1_img=e1, fee_img=fee, tot_img=e1 + fee,
                tot_rmac=(e1 + fee) / rmac,
                vs_h100=((e1 + fee) / rmac) / H100,
                e2_fielded_rmac=2 * 32 * E_ADC / 4)


# ---------------------------------------------------------------- export
def export_json(path):
    """Figure-ready numbers (for the paper's Fig. 3g overlay and tables)."""
    import json
    d, c = sec_d(), comb_nn()
    s = serial_nn(c)
    rm = c["rmac"]
    out = dict(
        convention=dict(eta_radio=ETA_RADIO, e_adc_J=E_ADC, e_dig_J=E_DIG,
                        fee_per_answer_J=FEE_PER_ANSWER, h100_J_per_realmac=H100,
                        t_per_realmac_s=T_PER_RMAC,
                        model_curve="e_fJ(N) = e1_flat_fJ + 14000/(4N)",
                        e1_flat_fJ=PX_65536 * T_PER_RMAC / ETA_RADIO * 1e15),
        sec_d_reproduction=[
            dict(N=r["N"], px_nW=r["px"] * 1e9, px_source=r["tag"],
                 e1_fJ=r["e1"] * 1e15, fee_fJ=r["fee"] * 1e15,
                 total_fJ_per_realmac=r["eip"] * 1e15) for r in d["rows"]],
        comb_nn=dict(
            operating_point_dbm=dict(lo=-3.0, rf=P_COMB_DBM),
            airtime_per_image_s=c["t_air"], real_macs_per_image=c["rmac"],
            answers_per_image=c["answers"],
            e1_per_image_nJ=c["e1_img"] * 1e9, fee_per_image_nJ=c["fee_img"] * 1e9,
            total_per_image_nJ=c["tot_img"] * 1e9,
            total_fJ_per_realmac=c["tot_rmac"] * 1e15,
            vs_h100=c["vs_h100"],
            fig3g_overlay_points=[
                dict(label=l["name"], N=l["D"],
                     total_fJ_per_realmac=round(
                         l["fee_rmac"] * 1e15 + c["e1_img"] / rm * 1e15, 2))
                for l in c["layers"]] + [
                dict(label="network_aggregate",
                     N=round(rm / (4 * c["answers"])),
                     total_fJ_per_realmac=round(c["tot_rmac"] * 1e15, 2))]),
        serial_nn=dict(
            operating_point_dbm=dict(lo=-10.0, rf=P_SER_DBM),
            airtime_per_image_s=s["t_air"],
            e1_per_image_uJ=s["e1_img"] * 1e6,
            total_pJ_per_realmac=s["tot_rmac"] * 1e12,
            vs_h100_above=s["vs_h100"],
            readout_convention="row-rate integrate-and-dump (6 conv + 8 ops "
                               "per answer); as-fielded per-sample capture "
                               "adds ~16 pJ/realMAC"),
        notes=["Client-side ledger only: LO/weight broadcast, DAC synthesis, "
               "static transceiver power and SDR wall-plug excluded "
               "(Sec. II D conventions).",
               "Comb energy is image-independent by construction (frame-RMS "
               "drive normalization); serial e1 varies per image (frozen "
               "per-layer drive scales) and carries no error bar here."])
    Path(path).write_text(json.dumps(out, indent=1))
    print(f"wrote {path}")


# ---------------------------------------------------------------- console
def main():
    d = sec_d()
    print("=" * 72)
    print("1. REPRODUCE Sec. II D (resolved inner products, 25-dB points)")
    print("=" * 72)
    for r in d["rows"]:
        print(f"N={r['N']:6d}: Px={r['px']*1e9:.3f} nW [{r['tag']}]  "
              f"e1={r['e1']*1e15:.3f} fJ  fee={r['fee']*1e15:.3f} fJ  "
              f"e_ip={r['eip']*1e15:.2f} fJ/realMAC")
    print(f"GPU crossover N ~ {d['n_cross']:.0f}   [paper: 'N >~ 50']")
    print(f"radio-floor bend N ~ {d['n_bend']:.0f}  [paper: 'N ~ 6.5e3']")
    print(f"vs H100: {d['vs_h100'][0]:.0f}x and {d['vs_h100'][1]:.0f}x "
          f"[paper: 48x and 119x]")
    # Open item for the original generator: at FS=10 MS/s with L=4N the
    # payload alone gives (L+CP)/(4N) = 103 ns/realMAC; the stated 117 ns
    # implies ~13% further overhead (sync amortization?). Taken as given.

    c = comb_nn()
    print()
    print("=" * 72)
    print(f"2. COMB CNN at (LO,RF) = (-3, {P_COMB_DBM:.0f}) dBm at the mixer")
    print("   [repo] the frozen-battery operating point as delivered")
    print("=" * 72)
    print(f"{'layer':7s} {'sym/img':>8s} {'K':>4s} {'R':>3s} {'g':>2s} "
          f"{'cMAC/img':>10s} {'answers':>8s} {'fee/rMAC':>9s}")
    for l in c["layers"]:
        print(f"{l['name']:7s} {l['sym']:8d} {l['K']:4d} {l['R']:3d} "
              f"{l['g']:2d} {l['cmac']:10d} {l['answers']:8d} "
              f"{l['fee_rmac']*1e15:7.1f}fJ")
    print(f"\nairtime/image     {c['t_air']:8.2f} s   ({c['tot_sym']} "
          f"symbols + {SYNC_FRAC:.1%} sync)")
    print(f"real MACs/image   {c['rmac']/1e6:8.2f} M   "
          f"answers/image {c['answers']}")
    print(f"e1 (TX)           {c['e1_img']*1e9:8.2f} nJ/img "
          f"= {c['e1_img']/c['rmac']*1e15:6.2f} fJ/realMAC")
    print(f"readout fee       {c['fee_img']*1e9:8.2f} nJ/img "
          f"= {c['fee_img']/c['rmac']*1e15:6.2f} fJ/realMAC")
    print(f"TOTAL             {c['tot_img']*1e9:8.2f} nJ/img "
          f"= {c['tot_rmac']*1e15:6.2f} fJ/realMAC "
          f"({c['vs_h100']:.1f}x below H100)")

    s = serial_nn(c)
    print()
    print("=" * 72)
    print(f"3. SERIAL CNN at (LO,RF) = (-10, {P_SER_DBM:.0f}) dBm")
    print("   [repo] serial_nn_runner_m10m24.py POWER=(-24,-10) (RF,LO)")
    print("=" * 72)
    print(f"airtime/image     {s['t_air']:8.2f} s   "
          f"({c['cmac']} slots x 3.2 us)")
    print(f"e1 (TX)           {s['e1_img']*1e6:8.2f} uJ/img "
          f"= {s['e1_img']/c['rmac']*1e12:6.2f} pJ/realMAC")
    print(f"readout fee       {s['fee_img']*1e9:8.2f} nJ/img "
          f"= {s['fee_img']/c['rmac']*1e15:6.2f} fJ/realMAC")
    print(f"TOTAL             {s['tot_img']*1e6:8.2f} uJ/img "
          f"= {s['tot_rmac']*1e12:6.2f} pJ/realMAC "
          f"({s['vs_h100']:.0f}x ABOVE H100)")
    print(f"[as-fielded readout instead: "
          f"{s['e2_fielded_rmac']*1e12:.1f} pJ/realMAC "
          f"(64 conversions per product) — dominated by e1 anyway]")


if __name__ == "__main__":
    main()
    export_json(Path(__file__).resolve().parent / "energy_budget_nn_results.json")
