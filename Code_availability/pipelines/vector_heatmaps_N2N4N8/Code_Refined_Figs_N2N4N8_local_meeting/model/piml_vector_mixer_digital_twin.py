from __future__ import annotations

# --- replication-package path resolution (see Code_availability/_paths.py) ---
import sys as _sys, pathlib as _pl
for _p in _pl.Path(__file__).resolve().parents:
    if _p.name == "Code_availability":
        _sys.path.insert(0, str(_p)); break
from _paths import data_dir as _data_dir
# ---------------------------------------------------------------------------


import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def _import_n1_blocks():
    here = _data_dir(__file__)
    candidates = [
        here / ".." / ".." / "PIML_Figures" / "PIML_Figures" / "model",
        here / ".." / ".." / "PIML_Figures" / "model",
        here.parent.parent / "PIML_Figures" / "PIML_Figures" / "model",
    ]
    for c in candidates:
        if (c / "piml_mixer_digital_twin.py").is_file():
            sys.path.insert(0, str(c.resolve()))
            from piml_mixer_digital_twin import PhysicsMixer, ResidualNN
            return PhysicsMixer, ResidualNN
    raise FileNotFoundError(
        "Could not find piml_mixer_digital_twin.py from N=1 bundle. "
        "Pass --n1-source-dir manually or place the bundle one level above."
    )


PhysicsMixerN1, _ = _import_n1_blocks()


@dataclass
class VectorDataset:
    N:        int
    ip_sq:    float
    papr_lo:  float
    papr_rf:  float
    p_lo_dbm: np.ndarray
    p_rf_dbm: np.ndarray
    p_if_w:   np.ndarray
    name:     str = ""


def _papr_db_from_complex(vec: np.ndarray, df_tone_hz: float = 1.0e6,
                          fs_hz: float = 10e6, n_period: int = 8) -> float:
    vec = np.asarray(vec, dtype=np.complex128).reshape(-1)
    N = vec.shape[0]
    if N == 1:
        return 0.0
    offsets = (np.arange(N) - (N - 1) / 2.0) * df_tone_hz
    period = 1.0 / abs(np.gcd.reduce(
        np.round(offsets[offsets != 0] / 1e3).astype(int) * 1000
    )) if (offsets != 0).any() else 1.0 / df_tone_hz
    T = max(period * n_period, 8.0 / fs_hz)
    n = int(np.ceil(T * fs_hz))
    t = np.arange(n) / fs_hz
    s = np.zeros(n, dtype=np.complex128)
    for v, f in zip(vec, offsets):
        s += v * np.exp(2j * np.pi * f * t)
    rms2 = float(np.mean(np.abs(s) ** 2))
    peak2 = float(np.max(np.abs(s) ** 2))
    if rms2 <= 0:
        return 0.0
    return 10.0 * math.log10(peak2 / rms2)


def load_vector_npz(path: str | Path) -> VectorDataset:
    path = Path(path)
    d = np.load(path, allow_pickle=True)
    meta = json.loads(str(d["meta_json"]))

    vec_a = np.asarray(d["vec_a"], dtype=np.complex128)
    vec_b = np.asarray(d["vec_b"], dtype=np.complex128)
    N = int(meta.get("vec_N", vec_a.shape[0]))
    ip_sq = float(meta.get(
        "ip_magnitude_squared", float(np.abs(np.vdot(vec_a, vec_b)) ** 2)
    ))
    df_tone_hz = float(meta.get("df_tone_hz", 1.0e6))

    papr_lo = float(meta.get("papr_a_after_db",
                             meta.get("papr_a_before_db", float("nan"))))
    papr_rf = float(meta.get("papr_b_after_db",
                             meta.get("papr_b_before_db", float("nan"))))
    if not math.isfinite(papr_lo):
        papr_lo = _papr_db_from_complex(vec_a, df_tone_hz=df_tone_hz)
    if not math.isfinite(papr_rf):
        papr_rf = _papr_db_from_complex(vec_b, df_tone_hz=df_tone_hz)

    p_lo = d["p_lo_dbm_grid"].astype(np.float64)
    p_rf = d["p_rf_dbm_grid"].astype(np.float64)
    if_v = d["if_amp_uv_mean"].astype(np.float64) * 1e-6
    if_w = np.maximum(if_v ** 2 / 50.0, 1e-15)

    return VectorDataset(
        N=N, ip_sq=ip_sq, papr_lo=papr_lo, papr_rf=papr_rf,
        p_lo_dbm=p_lo, p_rf_dbm=p_rf, p_if_w=if_w, name=path.name,
    )


class VectorPhysicsMixer(nn.Module):

    C_BOUNDS = (-1.5, 1.5)

    @staticmethod
    def _to_raw(value: float, lo: float, hi: float) -> float:
        eps = 1e-3 * (hi - lo)
        v = min(max(value, lo + eps), hi - eps)
        u = (v - lo) / (hi - lo)
        return math.log(u / (1.0 - u))

    def _bounded(self, raw: torch.Tensor) -> torch.Tensor:
        lo, hi = self.C_BOUNDS
        return lo + (hi - lo) * torch.sigmoid(raw)

    def __init__(self, n1_physics: PhysicsMixerN1,
                 c_papr_lo_init: float = 0.0,
                 c_papr_rf_init: float = 0.0) -> None:
        super().__init__()
        self.n1 = n1_physics
        for p in self.n1.parameters():
            p.requires_grad = False
        self._raw_c_lo = nn.Parameter(
            torch.tensor(self._to_raw(c_papr_lo_init, *self.C_BOUNDS))
        )
        self._raw_c_rf = nn.Parameter(
            torch.tensor(self._to_raw(c_papr_rf_init, *self.C_BOUNDS))
        )

    @property
    def c_papr_lo(self) -> torch.Tensor:
        return self._bounded(self._raw_c_lo)

    @property
    def c_papr_rf(self) -> torch.Tensor:
        return self._bounded(self._raw_c_rf)

    def forward(self,
                p_lo_dbm: torch.Tensor,
                p_rf_dbm: torch.Tensor,
                ip_sq: torch.Tensor,
                papr_lo_db: torch.Tensor,
                papr_rf_db: torch.Tensor) -> torch.Tensor:
        p_lo_eff = p_lo_dbm + self.c_papr_lo * papr_lo_db
        p_rf_eff = p_rf_dbm + self.c_papr_rf * papr_rf_db

        n1 = self.n1
        p_rf_w_eff = 1e-3 * torch.pow(10.0, p_rf_eff / 10.0)
        p_rf_comp_w = 1e-3 * torch.pow(10.0, n1.P_RF_comp_dBm / 10.0)

        alpha_lo = n1._alpha_lo(p_lo_eff)
        c_rf = torch.pow(torch.clamp(p_rf_w_eff / p_rf_comp_w, min=1e-30),
                         n1.beta_RF)
        p_rf_compressed = p_rf_w_eff * torch.pow(1.0 + c_rf, -1.0 / n1.beta_RF)

        g_max = torch.pow(10.0, n1.G_max_db / 10.0)
        signal_w = g_max * alpha_lo * p_rf_compressed * ip_sq

        p_rf_w = 1e-3 * torch.pow(10.0, p_rf_dbm / 10.0)
        leak = torch.pow(10.0, n1.leak_dB / 10.0)
        p_noise_w = 1e-3 * torch.pow(10.0, n1.P_noise_dBm / 10.0)

        return signal_w + leak * p_rf_w + p_noise_w

    def vector_params(self) -> dict:
        with torch.no_grad():
            return {
                "c_papr_lo": float(self.c_papr_lo),
                "c_papr_rf": float(self.c_papr_rf),
            }


class VectorResidualNN(nn.Module):

    def __init__(self,
                 hidden: tuple[int, ...] = (24, 24),
                 max_db: float = 4.0,
                 p_norm_scale: float = 30.0,
                 ipsq_norm_scale: float = 2.0,
                 papr_norm_scale: float = 6.0,
                 logN_norm_scale: float = 4.0) -> None:
        super().__init__()
        self.max_db = max_db
        self.p_norm = p_norm_scale
        self.ipsq_norm = ipsq_norm_scale
        self.papr_norm = papr_norm_scale
        self.logN_norm = logN_norm_scale

        layers: list[nn.Module] = []
        last = 6
        for h in hidden:
            layers.append(nn.Linear(last, h))
            layers.append(nn.Tanh())
            last = h
        layers.append(nn.Linear(last, 1))
        self.net = nn.Sequential(*layers)
        with torch.no_grad():
            last_lin = self.net[-1]
            last_lin.weight.mul_(0.01)
            last_lin.bias.zero_()

    def _features(self, p_lo_dbm, p_rf_dbm, ip_sq, N, papr_lo_db, papr_rf_db):
        target_shape = p_lo_dbm.shape
        ones = torch.ones(target_shape, dtype=p_lo_dbm.dtype,
                          device=p_lo_dbm.device)
        ip_sq_b   = torch.as_tensor(ip_sq, dtype=p_lo_dbm.dtype,
                                    device=p_lo_dbm.device) * ones
        N_b       = torch.as_tensor(N, dtype=p_lo_dbm.dtype,
                                    device=p_lo_dbm.device) * ones
        papr_lo_b = torch.as_tensor(papr_lo_db, dtype=p_lo_dbm.dtype,
                                    device=p_lo_dbm.device) * ones
        papr_rf_b = torch.as_tensor(papr_rf_db, dtype=p_lo_dbm.dtype,
                                    device=p_lo_dbm.device) * ones
        log10_ip = torch.log10(torch.clamp(ip_sq_b, min=1e-12))
        logN = torch.log2(torch.clamp(N_b, min=1.0))
        return torch.stack([
            p_lo_dbm   / self.p_norm,
            p_rf_dbm   / self.p_norm,
            log10_ip   / self.ipsq_norm,
            logN       / self.logN_norm,
            papr_lo_b  / self.papr_norm,
            papr_rf_b  / self.papr_norm,
        ], dim=-1)

    def forward(self, p_lo_dbm, p_rf_dbm, ip_sq, N, papr_lo_db, papr_rf_db):
        x = self._features(p_lo_dbm, p_rf_dbm, ip_sq, N, papr_lo_db, papr_rf_db)
        raw = self.net(x).squeeze(-1)
        return self.max_db * torch.tanh(raw)


class VectorPIMLMixer(nn.Module):
    def __init__(self,
                 vector_physics: VectorPhysicsMixer,
                 residual: VectorResidualNN) -> None:
        super().__init__()
        self.physics = vector_physics
        self.residual = residual

    def predict_log10_w(self, p_lo_dbm, p_rf_dbm,
                        ip_sq, N, papr_lo_db, papr_rf_db,
                        use_nn: bool = True) -> torch.Tensor:
        p_phys = self.physics(p_lo_dbm, p_rf_dbm, ip_sq, papr_lo_db, papr_rf_db)
        log10_phys = torch.log10(torch.clamp(p_phys, min=1e-30))
        if not use_nn:
            return log10_phys
        delta_db = self.residual(p_lo_dbm, p_rf_dbm, ip_sq, N, papr_lo_db, papr_rf_db)
        return log10_phys + delta_db / 10.0

    def predict_w(self, *args, **kwargs) -> torch.Tensor:
        return torch.pow(10.0, self.predict_log10_w(*args, **kwargs))


def _per_dataset_tensors(ds: VectorDataset, device):
    P_LO_2d, P_RF_2d = np.meshgrid(ds.p_lo_dbm, ds.p_rf_dbm, indexing="ij")
    return dict(
        p_lo     = torch.tensor(P_LO_2d, dtype=torch.float32, device=device),
        p_rf     = torch.tensor(P_RF_2d, dtype=torch.float32, device=device),
        ip_sq    = torch.tensor(ds.ip_sq, dtype=torch.float32, device=device),
        N        = torch.tensor(float(ds.N), dtype=torch.float32, device=device),
        papr_lo  = torch.tensor(ds.papr_lo, dtype=torch.float32, device=device),
        papr_rf  = torch.tensor(ds.papr_rf, dtype=torch.float32, device=device),
        log10_y  = torch.tensor(np.log10(ds.p_if_w), dtype=torch.float32,
                                device=device),
    )


def compute_loss(model: VectorPIMLMixer,
                 tensors_per_ds: Sequence[dict],
                 *,
                 use_nn: bool,
                 lam_nn: float = 1e-4,
                 lam_delta: float = 5e-3,
                 lam_smooth: float = 1e-2) -> tuple[torch.Tensor, dict]:
    mse_terms = []
    deltas: list[torch.Tensor] = []
    for t in tensors_per_ds:
        log10_pred = model.predict_log10_w(
            t["p_lo"], t["p_rf"], t["ip_sq"], t["N"],
            t["papr_lo"], t["papr_rf"], use_nn=use_nn,
        )
        err = log10_pred - t["log10_y"]
        mse_terms.append(torch.mean(err ** 2))
        if use_nn:
            deltas.append(model.residual(
                t["p_lo"], t["p_rf"], t["ip_sq"], t["N"],
                t["papr_lo"], t["papr_rf"],
            ))
    mse = torch.stack(mse_terms).mean()
    diag = {"mse_log10": float(mse.detach())}
    total = mse

    if use_nn and deltas:
        reg_delta_terms = []
        reg_smooth_terms = []
        for delta in deltas:
            reg_delta_terms.append(torch.mean(delta ** 2))
            d2_lo = delta[2:, :] - 2 * delta[1:-1, :] + delta[:-2, :]
            d2_rf = delta[:, 2:] - 2 * delta[:, 1:-1] + delta[:, :-2]
            reg_smooth_terms.append(torch.mean(d2_lo ** 2) + torch.mean(d2_rf ** 2))
        reg_delta = torch.stack(reg_delta_terms).mean()
        reg_smooth = torch.stack(reg_smooth_terms).mean()
        reg_nn = sum((p ** 2).sum() for p in model.residual.parameters())

        if lam_delta > 0:
            total = total + lam_delta * reg_delta
            diag["reg_delta"] = float(reg_delta.detach())
        if lam_smooth > 0:
            total = total + lam_smooth * reg_smooth
            diag["reg_smooth"] = float(reg_smooth.detach())
        if lam_nn > 0:
            total = total + lam_nn * reg_nn
            diag["reg_nn"] = float(reg_nn.detach())

    diag["total"] = float(total.detach())
    return total, diag


def fit_papr_lbfgs(model: VectorPIMLMixer,
                   tensors_per_ds: Sequence[dict],
                   max_iter: int = 200) -> float:
    for p in model.residual.parameters():
        p.requires_grad = False
    for p in model.physics.n1.parameters():
        p.requires_grad = False
    for p in (model.physics._raw_c_lo, model.physics._raw_c_rf):
        p.requires_grad = True

    opt = torch.optim.LBFGS(
        [model.physics._raw_c_lo, model.physics._raw_c_rf],
        lr=0.5, max_iter=max_iter,
        tolerance_grad=1e-9, tolerance_change=1e-12,
        line_search_fn="strong_wolfe",
    )

    def closure():
        opt.zero_grad()
        loss, _ = compute_loss(model, tensors_per_ds, use_nn=False,
                               lam_nn=0.0, lam_delta=0.0, lam_smooth=0.0)
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        _, diag = compute_loss(model, tensors_per_ds, use_nn=False,
                               lam_nn=0.0, lam_delta=0.0, lam_smooth=0.0)
    return float(diag["mse_log10"])


def train_stage(model, tensors_per_ds, *, optimizer, n_iter, use_nn, label,
                log_every=200, **loss_kwargs) -> list[dict]:
    history = []
    for it in range(n_iter):
        optimizer.zero_grad()
        loss, diag = compute_loss(model, tensors_per_ds, use_nn=use_nn,
                                  **loss_kwargs)
        loss.backward()
        optimizer.step()
        if it == 0 or (it + 1) % log_every == 0 or it == n_iter - 1:
            mse = diag["mse_log10"]
            rmse_db = 10.0 * math.sqrt(mse)
            rec = {"stage": label, "iter": it + 1, "rmse_db": rmse_db, **diag}
            history.append(rec)
            print(f"  [{label}] iter {it+1:5d}  log-MSE={mse:.5f}  "
                  f"RMSE={rmse_db:.3f} dB  total={diag['total']:.5f}")
    return history


def train_vector_piml(model: VectorPIMLMixer,
                      tensors_per_ds: Sequence[dict],
                      *,
                      n_iter_nn:    int = 5000,
                      n_iter_joint: int = 2500,
                      lr_nn:        float = 5e-3,
                      lr_joint_papr:float = 5e-4,
                      lr_joint_nn:  float = 1e-3,
                      lam_nn:       float = 1e-4,
                      lam_delta:    float = 5e-3,
                      lam_smooth:   float = 1e-2) -> list[dict]:
    history: list[dict] = []

    print("\n=== Stage 2: residual NN training (PAPR + physics frozen) ===")
    for p in model.physics.parameters():
        p.requires_grad = False
    for p in model.residual.parameters():
        p.requires_grad = True
    opt2 = torch.optim.Adam(model.residual.parameters(), lr=lr_nn)
    history += train_stage(model, tensors_per_ds,
                           optimizer=opt2, n_iter=n_iter_nn, use_nn=True,
                           label="nn", lam_nn=lam_nn, lam_delta=lam_delta,
                           lam_smooth=lam_smooth)

    print("\n=== Stage 3: joint fine-tune (c_papr + NN; N=1 physics frozen) ===")
    for p in model.physics.n1.parameters():
        p.requires_grad = False
    for p in (model.physics._raw_c_lo, model.physics._raw_c_rf):
        p.requires_grad = True
    for p in model.residual.parameters():
        p.requires_grad = True
    opt3 = torch.optim.Adam([
        {"params": [model.physics._raw_c_lo, model.physics._raw_c_rf],
         "lr": lr_joint_papr},
        {"params": model.residual.parameters(), "lr": lr_joint_nn},
    ])
    history += train_stage(model, tensors_per_ds,
                           optimizer=opt3, n_iter=n_iter_joint, use_nn=True,
                           label="joint", lam_nn=lam_nn, lam_delta=lam_delta,
                           lam_smooth=lam_smooth)
    return history


def load_n1_trained_physics(n1_pt: Path, n1_summary: Path) -> PhysicsMixerN1:
    summary = json.loads(n1_summary.read_text())
    p = summary["physics_params"]
    pm = PhysicsMixerN1(
        G_max_db      = p["G_max_dB"],
        P_LO_sat_dBm  = p["P_LO_sat_dBm"],
        beta_LO       = p["beta_LO"],
        P_RF_comp_dBm = p["P_RF_comp_dBm"],
        beta_RF       = p["beta_RF"],
        P_noise_dBm   = p["P_noise_dBm"],
        leak_dB       = p["leak_dB"],
        lo_shape      = p["lo_shape"],
    )

    if n1_pt.is_file():
        ckpt = torch.load(n1_pt, map_location="cpu", weights_only=False)
        pm_ckpt = PhysicsMixerN1(lo_shape=p["lo_shape"])
        pm_ckpt.load_state_dict(ckpt["physics_state"])
        ckpt_params = pm_ckpt.physical_params()
        for k in ("G_max_dB", "P_LO_sat_dBm", "beta_LO", "P_RF_comp_dBm",
                  "beta_RF", "P_noise_dBm", "leak_dB"):
            if abs(ckpt_params[k] - p[k]) > 0.02:
                print(f"  warning: summary {k}={p[k]} differs from .pt {ckpt_params[k]}")

    for q in pm.parameters():
        q.requires_grad = False
    pm.eval()
    return pm


def evaluate_per_dataset(models_per_ds: dict,
                         datasets: Sequence[VectorDataset],
                         tensors_per_ds: Sequence[dict]) -> dict:
    out: dict = {}
    for ds, t in zip(datasets, tensors_per_ds):
        model = models_per_ds[ds.N]
        model.eval()
        with torch.no_grad():
            p_lo_w = 1e-3 * 10 ** (t["p_lo"].cpu().numpy() / 10.0)
            p_rf_w = 1e-3 * 10 ** (t["p_rf"].cpu().numpy() / 10.0)
            ideal_w = p_lo_w * p_rf_w * ds.ip_sq / 1e-3

            saved_c_lo = float(model.physics._raw_c_lo.detach())
            saved_c_rf = float(model.physics._raw_c_rf.detach())
            model.physics._raw_c_lo.data.fill_(
                float(model.physics._to_raw(0.0, *model.physics.C_BOUNDS))
            )
            model.physics._raw_c_rf.data.fill_(
                float(model.physics._to_raw(0.0, *model.physics.C_BOUNDS))
            )
            naive_w = model.physics(
                t["p_lo"], t["p_rf"], t["ip_sq"], t["papr_lo"], t["papr_rf"]
            ).cpu().numpy()
            model.physics._raw_c_lo.data.fill_(saved_c_lo)
            model.physics._raw_c_rf.data.fill_(saved_c_rf)

            phys_w = model.physics(
                t["p_lo"], t["p_rf"], t["ip_sq"], t["papr_lo"], t["papr_rf"]
            ).cpu().numpy()

            log10_full = model.predict_log10_w(
                t["p_lo"], t["p_rf"], t["ip_sq"], t["N"],
                t["papr_lo"], t["papr_rf"], use_nn=True
            ).cpu().numpy()
            full_w = 10.0 ** log10_full

            delta_db = model.residual(
                t["p_lo"], t["p_rf"], t["ip_sq"], t["N"],
                t["papr_lo"], t["papr_rf"]
            ).cpu().numpy()

            out[ds.N] = dict(
                p_lo_dbm=ds.p_lo_dbm, p_rf_dbm=ds.p_rf_dbm,
                p_if_meas_w=ds.p_if_w,
                p_if_ideal_w=ideal_w,
                p_if_naive_w=naive_w,
                p_if_phys_w=phys_w,
                p_if_full_w=full_w,
                delta_db=delta_db,
                ip_sq=ds.ip_sq, papr_lo=ds.papr_lo, papr_rf=ds.papr_rf,
            )
    return out


def metrics_db(pred_w, meas_w) -> dict:
    p = 10.0 * np.log10(np.maximum(pred_w, 1e-30)) + 30.0
    m = 10.0 * np.log10(np.maximum(meas_w, 1e-30)) + 30.0
    err = (p - m).ravel()
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae  = float(np.mean(np.abs(err)))
    bias = float(np.mean(err))
    p95  = float(np.percentile(np.abs(err), 95))
    within_2 = float(np.mean(np.abs(err) <= 2.0) * 100.0)
    within_1 = float(np.mean(np.abs(err) <= 1.0) * 100.0)
    return dict(RMSE_dB=rmse, MAE_dB=mae, Bias_dB=bias, p95_abs_err_dB=p95,
                within_1dB_pct=within_1, within_2dB_pct=within_2)


def plot_per_N_heatmaps(eval_out: dict, out_png: Path):
    Ns = sorted(eval_out.keys())
    fig, ax = plt.subplots(len(Ns), 6, figsize=(22, 4.0 * len(Ns)),
                           squeeze=False)
    for row, N in enumerate(Ns):
        d = eval_out[N]
        meas, ideal = d["p_if_meas_w"], d["p_if_ideal_w"]
        naive, phys, full = d["p_if_naive_w"], d["p_if_phys_w"], d["p_if_full_w"]
        delta = d["delta_db"]
        extent = [d["p_rf_dbm"][0], d["p_rf_dbm"][-1],
                  d["p_lo_dbm"][0], d["p_lo_dbm"][-1]]
        vmin = max(np.nanmin(meas), 1e-12)
        vmax = np.nanmax(meas)
        norm = LogNorm(vmin=vmin, vmax=vmax)

        labels = [
            ("Measured (N9020A)",                       meas,  norm),
            ("Ideal: P_LO·P_RF·|<a,b>|²",                ideal, LogNorm(
                vmin=max(np.nanmin(ideal), 1e-18),
                vmax=max(np.nanmax(ideal), 1e-12))),
            ("Naive: frozen N=1 phys · |<a,b>|²",         naive, norm),
            ("Vector physics  (+ PAPR shift)",          phys,  norm),
            ("Full PIML (physics + NN)",                full,  norm),
        ]
        for col, (title, img, nm) in enumerate(labels):
            im = ax[row, col].imshow(img, origin="lower", aspect="auto",
                                     extent=extent, norm=nm, cmap="viridis")
            ax[row, col].set_title(f"N={N}  --  {title}", fontsize=10)
            ax[row, col].set_xlabel("RF dBm"); ax[row, col].set_ylabel("LO dBm")
            fig.colorbar(im, ax=ax[row, col], fraction=0.046, pad=0.04)

        absd = max(0.5, float(np.nanmax(np.abs(delta))))
        im = ax[row, 5].imshow(delta, origin="lower", aspect="auto",
                               extent=extent, cmap="coolwarm",
                               vmin=-absd, vmax=+absd)
        ax[row, 5].set_title(f"N={N}  --  NN correction (dB)", fontsize=10)
        ax[row, 5].set_xlabel("RF dBm"); ax[row, 5].set_ylabel("LO dBm")
        fig.colorbar(im, ax=ax[row, 5], fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def plot_residuals(eval_out: dict, out_png: Path):
    Ns = sorted(eval_out.keys())
    fig, ax = plt.subplots(len(Ns), 3, figsize=(13, 3.6 * len(Ns)), squeeze=False)
    for row, N in enumerate(Ns):
        d = eval_out[N]
        extent = [d["p_rf_dbm"][0], d["p_rf_dbm"][-1],
                  d["p_lo_dbm"][0], d["p_lo_dbm"][-1]]
        def res(p): return 10.0 * (np.log10(np.maximum(p, 1e-30))
                                   - np.log10(np.maximum(d["p_if_meas_w"], 1e-30)))
        triples = [
            ("Ideal - meas (dB)",        res(d["p_if_ideal_w"]), 25.0),
            ("Naive (N1 phys+|<a,b>|²) - meas (dB)",  res(d["p_if_naive_w"]), 6.0),
            ("Full PIML - meas (dB)",   res(d["p_if_full_w"]),  3.0),
        ]
        for col, (title, img, vmax_abs) in enumerate(triples):
            v = max(vmax_abs, float(np.nanmax(np.abs(img))))
            im = ax[row, col].imshow(img, origin="lower", aspect="auto",
                                     extent=extent, cmap="coolwarm",
                                     vmin=-v, vmax=+v)
            ax[row, col].set_title(f"N={N}  --  {title}", fontsize=10)
            ax[row, col].set_xlabel("RF dBm"); ax[row, col].set_ylabel("LO dBm")
            fig.colorbar(im, ax=ax[row, col], fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def plot_slices(eval_out: dict, out_png: Path):
    Ns = sorted(eval_out.keys())
    fig, ax = plt.subplots(len(Ns), 2, figsize=(13, 4.0 * len(Ns)), squeeze=False)
    for row, N in enumerate(Ns):
        d = eval_out[N]
        p_lo = d["p_lo_dbm"]; p_rf = d["p_rf_dbm"]
        meas, full = d["p_if_meas_w"], d["p_if_full_w"]
        rf_targets = [-50.0, -20.0, 0.0, +10.0]
        for prf in rf_targets:
            j = int(np.argmin(np.abs(p_rf - prf)))
            ax[row, 0].plot(p_lo, 10 * np.log10(np.maximum(meas[:, j], 1e-30)) + 30,
                            "o", ms=3, label=f"meas RF={prf:+.0f}")
            ax[row, 0].plot(p_lo, 10 * np.log10(np.maximum(full[:, j], 1e-30)) + 30,
                            "-", alpha=0.8)
        ax[row, 0].set_title(f"N={N}  --  IF vs LO (lines = PIML)"); 
        ax[row, 0].set_xlabel("LO dBm"); ax[row, 0].set_ylabel("IF dBm")
        ax[row, 0].grid(True, alpha=0.3); ax[row, 0].legend(fontsize=7, ncols=2)

        lo_targets = [-30.0, -10.0, 0.0, +10.0]
        for plo in lo_targets:
            i = int(np.argmin(np.abs(p_lo - plo)))
            ax[row, 1].plot(p_rf, 10 * np.log10(np.maximum(meas[i, :], 1e-30)) + 30,
                            "o", ms=3, label=f"meas LO={plo:+.0f}")
            ax[row, 1].plot(p_rf, 10 * np.log10(np.maximum(full[i, :], 1e-30)) + 30,
                            "-", alpha=0.8)
        ax[row, 1].set_title(f"N={N}  --  IF vs RF (lines = PIML)")
        ax[row, 1].set_xlabel("RF dBm"); ax[row, 1].set_ylabel("IF dBm")
        ax[row, 1].grid(True, alpha=0.3); ax[row, 1].legend(fontsize=7, ncols=2)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def plot_scatter(eval_out: dict, out_png: Path):
    Ns = sorted(eval_out.keys())
    fig, ax = plt.subplots(1, len(Ns), figsize=(5 * len(Ns), 5), squeeze=False)
    for col, N in enumerate(Ns):
        d = eval_out[N]
        meas_db = 10 * np.log10(np.maximum(d["p_if_meas_w"], 1e-30)).ravel() + 30
        ideal_db = 10 * np.log10(np.maximum(d["p_if_ideal_w"], 1e-30)).ravel() + 30
        naive_db = 10 * np.log10(np.maximum(d["p_if_naive_w"], 1e-30)).ravel() + 30
        full_db  = 10 * np.log10(np.maximum(d["p_if_full_w"], 1e-30)).ravel() + 30

        ax[0, col].scatter(meas_db, ideal_db, s=4, alpha=0.4, label="ideal")
        ax[0, col].scatter(meas_db, naive_db, s=4, alpha=0.5, label="N=1 phys + |<a,b>|²")
        ax[0, col].scatter(meas_db, full_db, s=4, alpha=0.7, label="full PIML")
        lim = [min(meas_db.min(), full_db.min()), max(meas_db.max(), full_db.max())]
        ax[0, col].plot(lim, lim, "k--", lw=1, label="y = x")
        ax[0, col].set_xlim(lim); ax[0, col].set_ylim(lim)
        ax[0, col].set_xlabel("Measured IF (dBm)")
        ax[0, col].set_ylabel("Predicted IF (dBm)")
        ax[0, col].set_title(f"N={N}, |<a,b>|²={d['ip_sq']:.3g}")
        ax[0, col].legend(fontsize=8); ax[0, col].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n1-pt",      required=True, type=Path,
                   help="N=1 trained checkpoint piml_mixer.pt")
    p.add_argument("--n1-summary", required=True, type=Path,
                   help="N=1 trained summary.json (used for the human-readable physics)")
    p.add_argument("--data-N2", type=Path, default=None)
    p.add_argument("--data-N4", type=Path, default=None)
    p.add_argument("--data-N8", type=Path, default=None)
    p.add_argument("--outdir",  type=Path, required=True)
    p.add_argument("--device",  default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--seed",    type=int, default=0)
    p.add_argument("--n-iter-nn",    type=int, default=5000)
    p.add_argument("--n-iter-joint", type=int, default=2500)
    p.add_argument("--lr-nn",        type=float, default=5e-3)
    p.add_argument("--lr-joint-papr",type=float, default=5e-4)
    p.add_argument("--lr-joint-nn",  type=float, default=1e-3)
    p.add_argument("--lam-nn",     type=float, default=1e-4)
    p.add_argument("--lam-delta",  type=float, default=5e-3)
    p.add_argument("--lam-smooth", type=float, default=1e-2)
    p.add_argument("--nn-max-db",  type=float, default=4.0)
    p.add_argument("--nn-hidden",  type=int, nargs="+", default=[24, 24])
    p.add_argument("--per-N", action="store_true",
                   help="Train a SEPARATE residual NN per dataset (each NN sees "
                        "only one N's measurements). Per-N gives lower RMSE at the "
                        "cost of no generalization across N. Default: one joint NN.")
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(args.device)

    outdir = args.outdir.resolve(); outdir.mkdir(parents=True, exist_ok=True)

    print("=== Loading N=1 trained physics ===")
    n1_physics = load_n1_trained_physics(args.n1_pt, args.n1_summary)
    print(f"  N=1 params: {n1_physics.physical_params()}")

    datasets: list[VectorDataset] = []
    for path, N_expect in [(args.data_N2, 2), (args.data_N4, 4), (args.data_N8, 8)]:
        if path is None:
            continue
        ds = load_vector_npz(path)
        if ds.N != N_expect:
            print(f"  warning: file {path} has N={ds.N}, expected {N_expect}")
        print(f"  loaded N={ds.N}  |<a,b>|^2={ds.ip_sq:.4f}  "
              f"PAPR(LO)={ds.papr_lo:.2f} dB  PAPR(RF)={ds.papr_rf:.2f} dB  "
              f"grid={ds.p_if_w.shape}")
        datasets.append(ds)
    if not datasets:
        raise SystemExit("No datasets supplied; need at least one --data-N{2,4,8}")
    tensors_per_ds = [_per_dataset_tensors(ds, device) for ds in datasets]

    vec_phys = VectorPhysicsMixer(n1_physics).to(device)
    vec_nn = VectorResidualNN(hidden=tuple(args.nn_hidden),
                              max_db=args.nn_max_db).to(device)
    model = VectorPIMLMixer(vec_phys, vec_nn).to(device)

    print("\n=== Stage 1: PAPR-shift fit (L-BFGS) ===")
    mse1 = fit_papr_lbfgs(model, tensors_per_ds, max_iter=200)
    rmse1 = 10.0 * math.sqrt(mse1)
    with torch.no_grad():
        c_lo_val = float(model.physics.c_papr_lo.detach())
        c_rf_val = float(model.physics.c_papr_rf.detach())
    print(f"  after PAPR fit:  RMSE = {rmse1:.3f} dB,  "
          f"c_papr_lo = {c_lo_val:+.4f},  c_papr_rf = {c_rf_val:+.4f}")

    if args.per_N:
        print("\n=== Per-N training mode: one residual NN per dataset ===")
        per_N_nns = {}
        history = []
        for ds, t in zip(datasets, tensors_per_ds):
            print(f"\n  --- Training residual NN for N={ds.N} ---")
            nn_k = VectorResidualNN(hidden=tuple(args.nn_hidden),
                                    max_db=args.nn_max_db).to(device)
            model_k = VectorPIMLMixer(vec_phys, nn_k).to(device)
            history += train_vector_piml(
                model_k, [t],
                n_iter_nn=args.n_iter_nn, n_iter_joint=args.n_iter_joint,
                lr_nn=args.lr_nn, lr_joint_papr=args.lr_joint_papr,
                lr_joint_nn=args.lr_joint_nn,
                lam_nn=args.lam_nn, lam_delta=args.lam_delta, lam_smooth=args.lam_smooth,
            )
            per_N_nns[ds.N] = nn_k
        models_per_ds = {ds.N: VectorPIMLMixer(vec_phys, per_N_nns[ds.N]).to(device)
                         for ds in datasets}
    else:
        history = train_vector_piml(model, tensors_per_ds,
            n_iter_nn=args.n_iter_nn, n_iter_joint=args.n_iter_joint,
            lr_nn=args.lr_nn, lr_joint_papr=args.lr_joint_papr,
            lr_joint_nn=args.lr_joint_nn,
            lam_nn=args.lam_nn, lam_delta=args.lam_delta, lam_smooth=args.lam_smooth,
        )
        models_per_ds = {ds.N: model for ds in datasets}

    print("\n=== Final evaluation ===")
    out = evaluate_per_dataset(models_per_ds, datasets, tensors_per_ds)
    metrics = {}
    for N in sorted(out.keys()):
        d = out[N]
        m_ideal = metrics_db(d["p_if_ideal_w"], d["p_if_meas_w"])
        m_naive = metrics_db(d["p_if_naive_w"], d["p_if_meas_w"])
        m_phys  = metrics_db(d["p_if_phys_w"], d["p_if_meas_w"])
        m_full  = metrics_db(d["p_if_full_w"], d["p_if_meas_w"])
        metrics[N] = dict(ideal=m_ideal, naive=m_naive, phys=m_phys, full=m_full)
        print(f"  N={N}")
        for label, m in [("ideal", m_ideal), ("naive (N=1 phys + |<a,b>|²)", m_naive),
                         ("vector physics (+ PAPR)", m_phys), ("full PIML", m_full)]:
            print(f"     {label:<32}  RMSE = {m['RMSE_dB']:6.3f} dB    "
                  f"bias = {m['Bias_dB']:+6.3f} dB    "
                  f"|err|<=2dB = {m['within_2dB_pct']:5.1f}%")

    plot_per_N_heatmaps(out, outdir / "fig1_heatmaps.png")
    plot_residuals(out,     outdir / "fig2_residuals.png")
    plot_slices(out,        outdir / "fig3_slices.png")
    plot_scatter(out,       outdir / "fig4_scatter.png")

    for N, d in out.items():
        np.savez_compressed(outdir / f"twin_predictions_N{N}.npz",
            p_lo_dbm     = d["p_lo_dbm"],
            p_rf_dbm     = d["p_rf_dbm"],
            p_if_meas_w  = d["p_if_meas_w"],
            p_if_ideal_w = d["p_if_ideal_w"],
            p_if_naive_w = d["p_if_naive_w"],
            p_if_phys_w  = d["p_if_phys_w"],
            p_if_full_w  = d["p_if_full_w"],
            delta_db     = d["delta_db"],
            ip_sq        = d["ip_sq"],
            papr_lo      = d["papr_lo"],
            papr_rf      = d["papr_rf"],
            N            = N,
        )

    if args.per_N:
        residual_states = {
            f"residual_state_N{ds.N}": models_per_ds[ds.N].residual.state_dict()
            for ds in datasets
        }
        torch.save({
            "vector_physics_state": vec_phys.state_dict(),
            "n1_physics_params":    n1_physics.physical_params(),
            "vector_papr_params":   vec_phys.vector_params(),
            "residual_hidden":      list(args.nn_hidden),
            "residual_max_db":      args.nn_max_db,
            "trained_on_N":         [ds.N for ds in datasets],
            "per_N":                True,
            **residual_states,
        }, outdir / "piml_vector_mixer.pt")
    else:
        torch.save({
            "vector_physics_state": model.physics.state_dict(),
            "residual_state":       model.residual.state_dict(),
            "n1_physics_params":    n1_physics.physical_params(),
            "vector_papr_params":   model.physics.vector_params(),
            "residual_hidden":      list(args.nn_hidden),
            "residual_max_db":      args.nn_max_db,
            "trained_on_N":         [ds.N for ds in datasets],
            "per_N":                False,
        }, outdir / "piml_vector_mixer.pt")

    summary = {
        "n1_physics_params":   n1_physics.physical_params(),
        "vector_papr_params":  model.physics.vector_params(),
        "trained_on": [
            {"N": ds.N, "ip_sq": ds.ip_sq,
             "papr_lo_db": ds.papr_lo, "papr_rf_db": ds.papr_rf,
             "file": ds.name}
            for ds in datasets
        ],
        "metrics_db": {str(N): metrics[N] for N in metrics},
        "config": vars(args),
    }
    summary["config"] = {k: (str(v) if isinstance(v, Path) else v)
                         for k, v in summary["config"].items()}
    with (outdir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nWrote outputs to {outdir}")
    for name in ["fig1_heatmaps.png", "fig2_residuals.png", "fig3_slices.png",
                 "fig4_scatter.png", "piml_vector_mixer.pt", "summary.json"]:
        print(f"  - {name}")


if __name__ == "__main__":
    main()
