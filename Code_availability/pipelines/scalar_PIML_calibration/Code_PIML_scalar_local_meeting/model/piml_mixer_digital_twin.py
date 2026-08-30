from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


@dataclass
class Heatmap:
    p_lo_dbm: np.ndarray
    p_rf_dbm: np.ndarray
    p_if_w:   np.ndarray


def load_heatmap_npz(path: str | Path) -> Heatmap:
    data = np.load(path)
    p_lo = data["p_lo_dbm_grid"].astype(np.float64)
    p_rf = data["p_rf_dbm_grid"].astype(np.float64)
    if_v = data["if_amp_uv_mean"].astype(np.float64) * 1e-6
    if_w = (if_v ** 2) / 50.0
    if_w = np.maximum(if_w, 1e-15)
    return Heatmap(p_lo_dbm=p_lo, p_rf_dbm=p_rf, p_if_w=if_w)


class PhysicsMixer(nn.Module):

    BOUNDS = {
        "G_max_db":      (-30.0,   5.0),
        "P_LO_sat_dBm":  (-30.0,  20.0),
        "beta_LO":       (  0.3,   6.0),
        "P_RF_comp_dBm": (-15.0,  25.0),
        "beta_RF":       (  0.5,  20.0),
        "P_noise_dBm":   (-95.0, -30.0),
        "leak_dB":       (-110.0, -30.0),
    }

    @staticmethod
    def _to_raw(value: float, lo: float, hi: float) -> float:
        eps = 1e-3 * (hi - lo)
        v = min(max(value, lo + eps), hi - eps)
        u = (v - lo) / (hi - lo)
        return math.log(u / (1.0 - u))

    def _bounded(self, raw: torch.Tensor, name: str) -> torch.Tensor:
        lo, hi = self.BOUNDS[name]
        return lo + (hi - lo) * torch.sigmoid(raw)

    def __init__(
        self,
        G_max_db: float = -3.0,
        P_LO_sat_dBm: float = 0.0,
        beta_LO: float = 1.0,
        P_RF_comp_dBm: float = 5.0,
        beta_RF: float = 2.0,
        P_noise_dBm: float = -73.0,
        leak_dB: float = -80.0,
        lo_shape: str = "weibull",
    ) -> None:
        super().__init__()
        if lo_shape not in ("hill", "weibull"):
            raise ValueError(f"lo_shape must be 'hill' or 'weibull', got {lo_shape!r}")
        self.lo_shape = lo_shape

        b = self.BOUNDS
        self._raw_G        = nn.Parameter(torch.tensor(self._to_raw(G_max_db,      *b["G_max_db"])))
        self._raw_Psat     = nn.Parameter(torch.tensor(self._to_raw(P_LO_sat_dBm,  *b["P_LO_sat_dBm"])))
        self._raw_betaLO   = nn.Parameter(torch.tensor(self._to_raw(beta_LO,       *b["beta_LO"])))
        self._raw_PRFcomp  = nn.Parameter(torch.tensor(self._to_raw(P_RF_comp_dBm, *b["P_RF_comp_dBm"])))
        self._raw_betaRF   = nn.Parameter(torch.tensor(self._to_raw(beta_RF,       *b["beta_RF"])))
        self._raw_Pnoise   = nn.Parameter(torch.tensor(self._to_raw(P_noise_dBm,   *b["P_noise_dBm"])))
        self._raw_leak     = nn.Parameter(torch.tensor(self._to_raw(leak_dB,       *b["leak_dB"])))

    @property
    def G_max_db(self):       return self._bounded(self._raw_G,       "G_max_db")
    @property
    def P_LO_sat_dBm(self):   return self._bounded(self._raw_Psat,    "P_LO_sat_dBm")
    @property
    def beta_LO(self):        return self._bounded(self._raw_betaLO,  "beta_LO")
    @property
    def P_RF_comp_dBm(self):  return self._bounded(self._raw_PRFcomp, "P_RF_comp_dBm")
    @property
    def beta_RF(self):        return self._bounded(self._raw_betaRF,  "beta_RF")
    @property
    def P_noise_dBm(self):    return self._bounded(self._raw_Pnoise,  "P_noise_dBm")
    @property
    def leak_dB(self):        return self._bounded(self._raw_leak,    "leak_dB")

    def _alpha_lo(self, p_lo_dbm: torch.Tensor) -> torch.Tensor:
        log10_x = (p_lo_dbm - self.P_LO_sat_dBm) / 10.0
        log10_x = torch.clamp(log10_x, min=-30.0, max=5.0)
        if self.lo_shape == "weibull":
            log10_xpow = self.beta_LO * log10_x
            log10_xpow = torch.clamp(log10_xpow, min=-30.0, max=10.0)
            x_pow = torch.pow(10.0, log10_xpow)
            return -torch.expm1(-x_pow)
        else:
            log10_xpow = self.beta_LO * log10_x
            log10_xpow = torch.clamp(log10_xpow, min=-30.0, max=30.0)
            x_pow = torch.pow(10.0, log10_xpow)
            return x_pow / (1.0 + x_pow)

    def forward(self, p_lo_dbm: torch.Tensor, p_rf_dbm: torch.Tensor) -> torch.Tensor:
        p_rf_w = 1e-3 * torch.pow(10.0, p_rf_dbm / 10.0)
        p_rf_comp_w = 1e-3 * torch.pow(10.0, self.P_RF_comp_dBm / 10.0)

        alpha_lo = self._alpha_lo(p_lo_dbm)

        c = torch.pow(torch.clamp(p_rf_w / p_rf_comp_w, min=1e-30),
                      self.beta_RF)
        p_rf_eff = p_rf_w * torch.pow(1.0 + c, -1.0 / self.beta_RF)

        g_max = torch.pow(10.0, self.G_max_db / 10.0)
        leak = torch.pow(10.0, self.leak_dB / 10.0)
        p_noise_w = 1e-3 * torch.pow(10.0, self.P_noise_dBm / 10.0)

        p_if_w = g_max * alpha_lo * p_rf_eff + leak * p_rf_w + p_noise_w
        return p_if_w

    def physical_params(self) -> dict:
        with torch.no_grad():
            return {
                "lo_shape": self.lo_shape,
                "G_max_dB": float(self.G_max_db),
                "conversion_loss_dB": -float(self.G_max_db),
                "P_LO_sat_dBm": float(self.P_LO_sat_dBm),
                "beta_LO": float(self.beta_LO),
                "P_RF_comp_dBm": float(self.P_RF_comp_dBm),
                "beta_RF": float(self.beta_RF),
                "P_noise_dBm": float(self.P_noise_dBm),
                "leak_dB": float(self.leak_dB),
            }


class ResidualNN(nn.Module):

    def __init__(
        self,
        hidden: tuple[int, ...] = (16, 16),
        max_db: float = 2.0,
        p_norm_scale: float = 30.0,
    ) -> None:
        super().__init__()
        self.max_db = max_db
        self.p_norm_scale = p_norm_scale

        layers: list[nn.Module] = []
        last = 2
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

    def forward(self, p_lo_dbm: torch.Tensor, p_rf_dbm: torch.Tensor) -> torch.Tensor:
        x = torch.stack([p_lo_dbm / self.p_norm_scale,
                         p_rf_dbm / self.p_norm_scale], dim=-1)
        raw = self.net(x).squeeze(-1)
        return self.max_db * torch.tanh(raw)


class PIMLMixer(nn.Module):
    def __init__(self, physics: PhysicsMixer, residual: ResidualNN) -> None:
        super().__init__()
        self.physics = physics
        self.residual = residual

    def predict_log10_w(self, p_lo_dbm: torch.Tensor, p_rf_dbm: torch.Tensor,
                        use_nn: bool = True) -> torch.Tensor:
        p_phys_w = self.physics(p_lo_dbm, p_rf_dbm)
        log10_phys = torch.log10(torch.clamp(p_phys_w, min=1e-30))
        if not use_nn:
            return log10_phys
        delta_db = self.residual(p_lo_dbm, p_rf_dbm)
        return log10_phys + delta_db / 10.0

    def predict_w(self, p_lo_dbm, p_rf_dbm, use_nn: bool = True) -> torch.Tensor:
        return torch.pow(10.0, self.predict_log10_w(p_lo_dbm, p_rf_dbm, use_nn=use_nn))


def compute_loss(
    model: PIMLMixer,
    P_LO: torch.Tensor,
    P_RF: torch.Tensor,
    log10_meas: torch.Tensor,
    *,
    use_nn: bool = True,
    lam_nn: float = 1e-4,
    lam_delta: float = 1e-3,
    lam_smooth: float = 1e-3,
) -> tuple[torch.Tensor, dict]:
    log10_pred = model.predict_log10_w(P_LO, P_RF, use_nn=use_nn)
    err = log10_pred - log10_meas
    mse = torch.mean(err ** 2)

    diag = {"mse_log10": float(mse.detach())}
    total = mse

    if use_nn and (lam_nn > 0 or lam_delta > 0 or lam_smooth > 0):
        delta = model.residual(P_LO, P_RF)

        if lam_delta > 0:
            reg_delta = torch.mean(delta ** 2)
            total = total + lam_delta * reg_delta
            diag["reg_delta"] = float(reg_delta.detach())

        if lam_smooth > 0:
            d2_lo = delta[2:, :] - 2 * delta[1:-1, :] + delta[:-2, :]
            d2_rf = delta[:, 2:] - 2 * delta[:, 1:-1] + delta[:, :-2]
            reg_smooth = torch.mean(d2_lo ** 2) + torch.mean(d2_rf ** 2)
            total = total + lam_smooth * reg_smooth
            diag["reg_smooth"] = float(reg_smooth.detach())

        if lam_nn > 0:
            reg_nn = sum((p ** 2).sum() for p in model.residual.net.parameters())
            total = total + lam_nn * reg_nn
            diag["reg_nn"] = float(reg_nn.detach())

    diag["total"] = float(total.detach())
    return total, diag


def train_stage(
    model: PIMLMixer,
    P_LO: torch.Tensor,
    P_RF: torch.Tensor,
    log10_meas: torch.Tensor,
    *,
    optimizer: torch.optim.Optimizer,
    n_iter: int,
    use_nn: bool,
    label: str,
    log_every: int = 100,
    **loss_kwargs,
) -> list[dict]:
    history = []
    for it in range(n_iter):
        optimizer.zero_grad()
        loss, diag = compute_loss(model, P_LO, P_RF, log10_meas,
                                  use_nn=use_nn, **loss_kwargs)
        loss.backward()
        optimizer.step()
        if it == 0 or (it + 1) % log_every == 0 or it == n_iter - 1:
            rec = {"stage": label, "iter": it + 1, **diag}
            history.append(rec)
            mse = diag["mse_log10"]
            rmse_db = 10.0 * math.sqrt(mse)
            print(f"  [{label}] iter {it+1:5d}  log-MSE={mse:.5f}  "
                  f"RMSE={rmse_db:.3f} dB  total={diag['total']:.5f}")
    return history


def _fit_physics_lbfgs(
    model: PIMLMixer,
    P_LO: torch.Tensor,
    P_RF: torch.Tensor,
    log10_meas: torch.Tensor,
    *,
    max_iter: int = 200,
) -> float:
    for p in model.residual.parameters():
        p.requires_grad = False
    for p in model.physics.parameters():
        p.requires_grad = True

    opt = torch.optim.LBFGS(model.physics.parameters(),
                            lr=0.5,
                            max_iter=max_iter,
                            tolerance_grad=1e-9,
                            tolerance_change=1e-12,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss, _ = compute_loss(model, P_LO, P_RF, log10_meas, use_nn=False,
                               lam_nn=0.0, lam_delta=0.0, lam_smooth=0.0)
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        loss, diag = compute_loss(model, P_LO, P_RF, log10_meas, use_nn=False,
                                  lam_nn=0.0, lam_delta=0.0, lam_smooth=0.0)
    return float(diag["mse_log10"])


def _physics_multi_start(
    P_LO: torch.Tensor,
    P_RF: torch.Tensor,
    log10_meas: torch.Tensor,
    nn_max_db: float,
    nn_hidden: tuple[int, ...],
    n_starts: int = 16,
    seed: int = 0,
    lo_shape: str = "weibull",
) -> tuple[PIMLMixer, float]:
    rng = np.random.default_rng(seed)

    G_grid        = [-1.0, -3.0, -5.0, -7.0]
    P_LO_sat_grid = [-5.0,  0.0,  3.0,  7.0]
    beta_LO_grid  = [ 0.7,  1.0,  1.3,  1.8]
    P_noise_grid  = [-68.0, -73.0, -78.0]

    combos: list[tuple] = []
    for g in G_grid:
        for psat in P_LO_sat_grid:
            for b in beta_LO_grid:
                for pn in P_noise_grid:
                    combos.append((g, psat, b, pn))
    rng.shuffle(combos)
    combos = combos[:n_starts]

    best_mse = math.inf
    best_state = None
    best_params = None

    print(f"\n=== Stage 1: physics-only fit (multi-start L-BFGS, "
          f"{len(combos)} starts, lo_shape={lo_shape}) ===")
    for k, (g, psat, b, pn) in enumerate(combos, 1):
        physics = PhysicsMixer(
            G_max_db=g,
            P_LO_sat_dBm=psat,
            beta_LO=b,
            P_RF_comp_dBm=5.0,
            beta_RF=2.0,
            P_noise_dBm=pn,
            leak_dB=-80.0,
            lo_shape=lo_shape,
        ).to(P_LO.device)
        residual = ResidualNN(hidden=tuple(nn_hidden),
                              max_db=nn_max_db).to(P_LO.device)
        model_k = PIMLMixer(physics, residual).to(P_LO.device)

        try:
            mse_k = _fit_physics_lbfgs(model_k, P_LO, P_RF, log10_meas)
        except Exception as e:
            print(f"  start {k:2d}: FAILED ({e})")
            continue

        rmse_db = 10.0 * math.sqrt(mse_k)
        marker = " *" if mse_k < best_mse else ""
        print(f"  start {k:2d} init=(G={g:+.1f}dB, Psat={psat:+.1f}, "
              f"betaLO={b:.2f}, Pn={pn:+.0f}) "
              f"-> RMSE={rmse_db:.3f} dB{marker}")
        if mse_k < best_mse:
            best_mse = mse_k
            best_state = {kk: v.clone() for kk, v in model_k.state_dict().items()}
            best_params = model_k.physics.physical_params()

    if best_state is None:
        raise RuntimeError("All physics multi-start runs failed.")

    physics = PhysicsMixer(lo_shape=lo_shape).to(P_LO.device)
    residual = ResidualNN(hidden=tuple(nn_hidden),
                          max_db=nn_max_db).to(P_LO.device)
    best_model = PIMLMixer(physics, residual).to(P_LO.device)
    best_model.load_state_dict(best_state)
    print(f"  --> best physics RMSE = {10.0 * math.sqrt(best_mse):.3f} dB")
    print(f"  --> best params: {best_params}")
    return best_model, best_mse


def train_piml(
    model: PIMLMixer,
    P_LO: torch.Tensor,
    P_RF: torch.Tensor,
    log10_meas: torch.Tensor,
    *,
    n_iter_nn:    int = 4000,
    n_iter_joint: int = 2000,
    lr_nn: float = 5e-3,
    lr_joint_phys: float = 5e-4,
    lr_joint_nn: float = 1e-3,
    lam_nn: float = 1e-4,
    lam_delta: float = 5e-3,
    lam_smooth: float = 1e-2,
) -> list[dict]:
    history: list[dict] = []

    print("\n=== Stage 2: residual NN training (physics frozen) ===")
    for p in model.physics.parameters():
        p.requires_grad = False
    for p in model.residual.parameters():
        p.requires_grad = True
    opt2 = torch.optim.Adam(model.residual.parameters(), lr=lr_nn)
    history += train_stage(
        model, P_LO, P_RF, log10_meas,
        optimizer=opt2, n_iter=n_iter_nn, use_nn=True,
        label="nn",
        lam_nn=lam_nn, lam_delta=lam_delta, lam_smooth=lam_smooth,
    )

    print("\n=== Stage 3: joint fine-tune ===")
    for p in model.physics.parameters():
        p.requires_grad = True
    for p in model.residual.parameters():
        p.requires_grad = True
    opt3 = torch.optim.Adam([
        {"params": model.physics.parameters(),  "lr": lr_joint_phys},
        {"params": model.residual.parameters(), "lr": lr_joint_nn},
    ])
    history += train_stage(
        model, P_LO, P_RF, log10_meas,
        optimizer=opt3, n_iter=n_iter_joint, use_nn=True,
        label="joint",
        lam_nn=lam_nn, lam_delta=lam_delta, lam_smooth=lam_smooth,
    )
    return history


def plot_comparison(
    p_lo: np.ndarray, p_rf: np.ndarray,
    p_if_meas_w: np.ndarray,
    p_if_phys_w: np.ndarray,
    p_if_full_w: np.ndarray,
    delta_db: np.ndarray,
    out_png: Path,
) -> None:
    fig, ax = plt.subplots(2, 2, figsize=(13, 10))

    extent = [p_rf[0], p_rf[-1], p_lo[0], p_lo[-1]]
    vmin = max(np.nanmin(p_if_meas_w), 1e-12)
    vmax = np.nanmax(p_if_meas_w)
    norm = LogNorm(vmin=vmin, vmax=vmax)

    im0 = ax[0, 0].imshow(p_if_meas_w, origin="lower", aspect="auto",
                          extent=extent, norm=norm, cmap="viridis")
    ax[0, 0].set_title("Measured IF power (W)")
    fig.colorbar(im0, ax=ax[0, 0])

    im1 = ax[0, 1].imshow(p_if_full_w, origin="lower", aspect="auto",
                          extent=extent, norm=norm, cmap="viridis")
    ax[0, 1].set_title("Digital twin (physics + NN) IF power (W)")
    fig.colorbar(im1, ax=ax[0, 1])

    err_db = 10.0 * (np.log10(np.maximum(p_if_full_w, 1e-30))
                     - np.log10(np.maximum(p_if_meas_w, 1e-30)))
    abs_lim = max(2.0, float(np.nanmax(np.abs(err_db))))
    im2 = ax[1, 0].imshow(err_db, origin="lower", aspect="auto",
                          extent=extent, cmap="coolwarm",
                          vmin=-abs_lim, vmax=abs_lim)
    ax[1, 0].set_title("Twin - measured (dB)")
    fig.colorbar(im2, ax=ax[1, 0])

    abs_d = max(0.5, float(np.nanmax(np.abs(delta_db))))
    im3 = ax[1, 1].imshow(delta_db, origin="lower", aspect="auto",
                          extent=extent, cmap="coolwarm",
                          vmin=-abs_d, vmax=abs_d)
    ax[1, 1].set_title("NN correction delta (dB)")
    fig.colorbar(im3, ax=ax[1, 1])

    for a in ax.ravel():
        a.set_xlabel("RF port power (dBm)")
        a.set_ylabel("LO port power (dBm)")

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_slices(p_lo, p_rf, p_if_meas_w, p_if_full_w, p_if_phys_w, out_png: Path):
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))

    rf_targets = [-50.0, -20.0, 0.0, 10.0]
    for prf in rf_targets:
        j = int(np.argmin(np.abs(p_rf - prf)))
        ax[0].plot(p_lo, 10 * np.log10(p_if_meas_w[:, j] / 1e-3),
                   "o", ms=3, label=f"meas RF={prf:+.0f} dBm")
        ax[0].plot(p_lo, 10 * np.log10(p_if_full_w[:, j] / 1e-3),
                   "-", alpha=0.8)
    ax[0].set_xlabel("LO power (dBm)")
    ax[0].set_ylabel("IF power (dBm)")
    ax[0].set_title("IF vs LO (markers = meas, lines = twin)")
    ax[0].legend(fontsize=8, ncols=2)
    ax[0].grid(True, alpha=0.3)

    lo_targets = [-30.0, -10.0, 0.0, 10.0]
    for plo in lo_targets:
        i = int(np.argmin(np.abs(p_lo - plo)))
        ax[1].plot(p_rf, 10 * np.log10(p_if_meas_w[i, :] / 1e-3),
                   "o", ms=3, label=f"meas LO={plo:+.0f} dBm")
        ax[1].plot(p_rf, 10 * np.log10(p_if_full_w[i, :] / 1e-3),
                   "-", alpha=0.8)
    ax[1].set_xlabel("RF power (dBm)")
    ax[1].set_ylabel("IF power (dBm)")
    ax[1].set_title("IF vs RF (markers = meas, lines = twin)")
    ax[1].legend(fontsize=8, ncols=2)
    ax[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", required=True,
                   help="Path to the station heatmap NPZ (e.g. heatmap_2db.npz)")
    p.add_argument("--outdir", default="piml_run")
    p.add_argument("--device", default="cpu",
                   choices=["cpu", "cuda"])
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--n-physics-starts", type=int, default=16)
    p.add_argument("--n-iter-nn",    type=int, default=4000)
    p.add_argument("--n-iter-joint", type=int, default=2000)
    p.add_argument("--lr-nn",   type=float, default=5e-3)
    p.add_argument("--lr-joint-phys", type=float, default=5e-4)
    p.add_argument("--lr-joint-nn",   type=float, default=1e-3)

    p.add_argument("--lam-nn",     type=float, default=1e-4)
    p.add_argument("--lam-delta",  type=float, default=5e-3)
    p.add_argument("--lam-smooth", type=float, default=1e-2)
    p.add_argument("--nn-max-db",  type=float, default=4.0)
    p.add_argument("--nn-hidden",  type=int, nargs="+", default=[24, 24])
    p.add_argument("--lo-shape", choices=["weibull", "hill"], default="weibull",
                   help="LO turn-on shape: 'weibull' (default) is alpha = 1 - "
                        "exp(-(P_LO/P_LO_sat)^beta_LO), giving a power-law rise "
                        "and a tight exponential saturation; 'hill' = "
                        "x^b/(1+x^b) has the same rise but a power-law tail "
                        "that under-saturates at high LO drive.")

    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    hm = load_heatmap_npz(args.npz)
    print(f"Loaded heatmap : {hm.p_if_w.shape}  "
          f"LO {hm.p_lo_dbm.min():+.0f}..{hm.p_lo_dbm.max():+.0f} dBm  "
          f"RF {hm.p_rf_dbm.min():+.0f}..{hm.p_rf_dbm.max():+.0f} dBm")

    P_LO_2d, P_RF_2d = np.meshgrid(hm.p_lo_dbm, hm.p_rf_dbm, indexing="ij")
    P_LO_t = torch.tensor(P_LO_2d, dtype=torch.float32, device=device)
    P_RF_t = torch.tensor(P_RF_2d, dtype=torch.float32, device=device)
    log10_meas_t = torch.tensor(np.log10(hm.p_if_w),
                                dtype=torch.float32, device=device)

    model, _ = _physics_multi_start(
        P_LO_t, P_RF_t, log10_meas_t,
        nn_max_db=args.nn_max_db,
        nn_hidden=tuple(args.nn_hidden),
        n_starts=args.n_physics_starts,
        seed=args.seed,
        lo_shape=args.lo_shape,
    )

    history = train_piml(
        model, P_LO_t, P_RF_t, log10_meas_t,
        n_iter_nn=args.n_iter_nn,
        n_iter_joint=args.n_iter_joint,
        lr_nn=args.lr_nn,
        lr_joint_phys=args.lr_joint_phys, lr_joint_nn=args.lr_joint_nn,
        lam_nn=args.lam_nn, lam_delta=args.lam_delta, lam_smooth=args.lam_smooth,
    )

    model.eval()
    with torch.no_grad():
        p_phys_w = model.physics(P_LO_t, P_RF_t).cpu().numpy()
        log10_full = model.predict_log10_w(P_LO_t, P_RF_t, use_nn=True).cpu().numpy()
        delta_db = model.residual(P_LO_t, P_RF_t).cpu().numpy()
    p_full_w = 10.0 ** log10_full

    err_db_phys = 10.0 * (np.log10(np.maximum(p_phys_w, 1e-30))
                          - np.log10(np.maximum(hm.p_if_w, 1e-30)))
    err_db_full = 10.0 * (np.log10(np.maximum(p_full_w, 1e-30))
                          - np.log10(np.maximum(hm.p_if_w, 1e-30)))
    rmse_phys_db = float(np.sqrt(np.mean(err_db_phys ** 2)))
    rmse_full_db = float(np.sqrt(np.mean(err_db_full ** 2)))
    mae_full_db = float(np.mean(np.abs(err_db_full)))

    print("\n=== Final metrics (in dB error space) ===")
    print(f"  RMSE (physics only)       = {rmse_phys_db:.3f} dB")
    print(f"  RMSE (physics + residual) = {rmse_full_db:.3f} dB")
    print(f"  MAE  (physics + residual) = {mae_full_db:.3f} dB")

    plot_comparison(hm.p_lo_dbm, hm.p_rf_dbm, hm.p_if_w,
                    p_phys_w, p_full_w, delta_db,
                    outdir / "comparison_heatmaps.png")
    plot_slices(hm.p_lo_dbm, hm.p_rf_dbm, hm.p_if_w,
                p_full_w, p_phys_w,
                outdir / "comparison_slices.png")

    np.savez(outdir / "twin_predictions.npz",
             p_lo_dbm=hm.p_lo_dbm,
             p_rf_dbm=hm.p_rf_dbm,
             p_if_meas_w=hm.p_if_w,
             p_if_phys_w=p_phys_w,
             p_if_full_w=p_full_w,
             delta_db=delta_db)

    torch.save({
        "physics_state": model.physics.state_dict(),
        "residual_state": model.residual.state_dict(),
        "physics_params_human": model.physics.physical_params(),
        "residual_max_db": model.residual.max_db,
        "residual_hidden": list(args.nn_hidden),
    }, outdir / "piml_mixer.pt")

    summary = {
        "input_npz": str(Path(args.npz).resolve()),
        "physics_params": model.physics.physical_params(),
        "metrics_db": {
            "rmse_physics_only": rmse_phys_db,
            "rmse_physics_plus_nn": rmse_full_db,
            "mae_physics_plus_nn": mae_full_db,
        },
        "config": vars(args),
    }
    with (outdir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nWrote outputs to: {outdir}")
    for name in ["comparison_heatmaps.png", "comparison_slices.png",
                 "twin_predictions.npz", "piml_mixer.pt", "summary.json"]:
        print(f"  - {name}")


if __name__ == "__main__":
    main()
