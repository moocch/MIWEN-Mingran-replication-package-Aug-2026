"""
analog_physics.py — Device-level physics for the fully-analog MIWEN chain.
=========================================================================
Every block is a differentiable JAX function so the whole RF chain can be
trained through ("hardware-tailored" training, MIWEN Sec. 2.3.2), but the
transfer functions come from device equations with parameters anchored to
datasheets / published measurements:

  * Diode-ring double-balanced mixer — exact ring equation (MIWEN Eq. 5 /
    Methods 4.3) with diode ideality n; output transformer/network factor g_x
    CALIBRATED so the switching-mode conversion loss reproduces the
    Mini-Circuits ZEM-4300+ spec (6.65 dB typ @ LO +7 dBm) — the same mixer
    on the group's bench.  Published ring mixers: 5.0–8.2 dB;
    ideal switching floor 3.92 dB.
  * Zero-bias Schottky full-wave pair (Skyworks SMS7630 SPICE params:
    Is = 5 uA, n = 1.05, Rs = 20 ohm) with an optional passive input match
    (50 ohm -> r_hi, published low-power rectifier practice), harvested two
    ways:
      - detector(): DC/video component  -> "abs" activation
      - doubler():  2nd harmonic        -> "square" activation
    Solved per carrier phase by Newton iterations on the exact Shockley +
    series-R + load equation (branch-correct: reverse conduction kept, so the
    small-signal limit is the physical square-law detection regime).
    Anchors: SMS7630 rectifier 31% RF-DC @ -20 dBm (boosted match, best
    published), 62% @ -10 dBm, 5-18% typical; ZBR ~ 5.1 kOhm.
    Doubler anchors: resistive-doubler theory floor 6 dB; HMC189A 13 dB typ.
  * Band-pass "filtration" across time bins — fixed FIR (FBAR-like decaying
    cosine impulse response), pass-band insertion loss 1.5 dB (air-gap FBAR
    duplexer filters: 1.2–1.5 dB measured; production FBAR < 1–1.5 dB).
  * Interconnect: 0.3 dB per hop (PCB trace + connector, engineering value).
  * Johnson–Nyquist noise 4kTRB injected at every port; every lossy passive
    element re-injects thermal noise so the floor never drops below kTB.

No amplifier anywhere in the default chain.  An optional literature LNA
(15.5 dB gain / 1.68 dB NF / 1.05 mW CMOS @ 2.4 GHz) can be inserted once
for comparison.
"""
from __future__ import annotations
import numpy as np
import jax
import jax.numpy as jnp

# ----------------------------------------------------------------- constants
KB = 1.380649e-23
QE = 1.602176634e-19
T0 = 290.0
VT = KB * T0 / QE                       # 24.99 mV

# SMS7630 (Skyworks datasheet SPICE table)
D_IS = 5.0e-6                           # saturation current [A]
D_N  = 1.05                             # ideality
D_RS = 20.0                             # series resistance [ohm]
NVT  = D_N * VT                         # 26.2 mV

R0     = 50.0                           # system impedance
TAU    = 100e-9                         # time-bin duration (10 Mbin/s)
B_BIN  = 1.0 / TAU                      # per-bin noise bandwidth = 10 MHz
SIG_TH = float(np.sqrt(4 * KB * T0 * R0 * B_BIN))   # 2.83 uV rms

# losses (voltage factors), literature-anchored
IL_FILT_DB = 1.5                        # FBAR BPF insertion loss
IL_IC_DB   = 0.3                        # interconnect per hop
A_FILT = 10 ** (-IL_FILT_DB / 20)
A_IC   = 10 ** (-IL_IC_DB / 20)

A_BAL_DET = 10 ** (-0.6 / 20)           # detector/doubler input balun loss
A_MATCH   = 10 ** (-0.5 / 20)           # matching-network efficiency

# mixer output transformer/network factor: set by calibrate_mixer()
G_X_DEFAULT = 1.464

# optional LNA (single insertion, comparison only): measured CMOS design,
# 15.5 dB gain, NF 1.68 dB, 1.05 mW @ 2.4 GHz
LNA_GAIN_DB = 15.5
LNA_NF_DB   = 1.68

# ------------------------------------------------------------------- mixer
def ring_core(vlo, vrf):
    """Exact diode-ring transfer (MIWEN Eq. 5/58 with ideality n).
    Small signal: vlo*vrf/(4 n VT).  Large |vlo|: +/- vrf/2 (commutation)."""
    a = vlo / NVT
    b = vrf / NVT
    return vrf / 2 + (NVT / 2) * (jnp.logaddexp(a, -b) - jnp.logaddexp(a, b))


def mixer(vlo, vrf, key, g_x=G_X_DEFAULT, add_noise=True):
    """Double-balanced ring mixer, bin level. Ports referenced to R0 = 50 ohm.
    Johnson noise enters at both ports and at the output."""
    if add_noise:
        k1, k2, k3 = jax.random.split(key, 3)
        vlo = vlo + SIG_TH * jax.random.normal(k1, vlo.shape)
        vrf = vrf + SIG_TH * jax.random.normal(k2, vrf.shape)
        out = g_x * ring_core(vlo, vrf)
        out = out + SIG_TH * jax.random.normal(k3, out.shape)
    else:
        out = g_x * ring_core(vlo, vrf)
    return out


# ------------------------------------------- Schottky pair (activation core)
def _series_diode_current(vs, r_tot, n_iter=18):
    """Solve  vs = i*r_tot + u,  i = Is*(exp(u/nVT)-1)  for junction voltage
    u.  Fixed-count Newton on u (smooth, jit/grad friendly).  Returns i."""
    u0_fwd = NVT * jnp.log1p(jnp.maximum(vs, 0.0) / (D_IS * r_tot))
    u = jnp.where(vs > 0, jnp.minimum(vs, u0_fwd), vs)
    u = jnp.clip(u, -2.0, 0.6)
    for _ in range(n_iter):
        e  = jnp.exp(u / NVT)
        i  = D_IS * (e - 1.0)
        g  = i * r_tot + u - vs
        gp = D_IS * r_tot * e / NVT + 1.0
        u  = jnp.clip(u - g / gp, -2.0, 0.6)
    return D_IS * (jnp.exp(u / NVT) - 1.0)


_PHASES = jnp.linspace(0, 2 * jnp.pi, 32, endpoint=False)


def _fullwave_current(v_bin, r_hi, r_load):
    """Full-wave SMS7630 pair (centre-tapped balun, 0.6 dB): branch currents
    from the exact Shockley equation for +v and -v drives (reverse conduction
    kept -> the small-signal limit is the physical square-law regime, not an
    ideal rectifier).  A passive input match steps 50 ohm up to r_hi
    (voltage boost sqrt(r_hi/50) x 0.5 dB network loss), as in published
    low-power rectifier designs.  Returns total load current i(phi)."""
    m = jnp.sqrt(r_hi / R0) * A_MATCH
    r_tot = r_hi + D_RS + r_load
    vs = A_BAL_DET * m * v_bin[..., None] * jnp.cos(_PHASES)
    return _series_diode_current(vs, r_tot) + _series_diode_current(-vs, r_tot)


def _detector_exact(v_bin, r_hi, r_load):
    i = _fullwave_current(v_bin, r_hi, r_load)
    return jnp.mean(i, axis=-1) * r_load


def _doubler_exact(v_bin, r_hi, r_load):
    i = _fullwave_current(v_bin, r_hi, r_load)
    c2 = 2.0 * jnp.mean(i * jnp.cos(2 * _PHASES), axis=-1) * r_load
    return jnp.abs(c2)


# Cached transfer tables: the exact solver tabulated on a dense grid (768
# knots, 0.1 uV .. 3 V); training / Monte-Carlo use differentiable linear
# interpolation (jnp.interp) -> same physics, ~50x faster.  Both devices are
# even functions of the bin value, so |v| indexing is exact.
_TABLES: dict = {}


def _get_table(kind, r_hi, r_load):
    tkey = (kind, round(float(r_hi), 2), round(float(r_load), 2))
    if tkey not in _TABLES:
        # Force eager evaluation even if first touched inside a jit trace:
        # the cached table must be a concrete array, never a tracer.
        with jax.ensure_compile_time_eval():
            grid = np.concatenate([[0.0], np.logspace(-7, np.log10(3.0), 767)])
            f = _detector_exact if kind == "rect" else _doubler_exact
            vals = np.asarray(f(jnp.asarray(grid, jnp.float32), r_hi, r_load),
                              np.float32)
            _TABLES[tkey] = (jnp.asarray(grid, jnp.float32),
                             jnp.asarray(vals))
    return _TABLES[tkey]


def detector(v_bin, key, r_hi=R0, r_load=R0, add_noise=True, exact=False):
    """SMS7630 envelope detector ("abs" activation): DC (video) component of
    the full-wave pair current into the load.  Sign-insensitive.  Square-law
    below the junction knee, linear-rectifier above."""
    if exact:
        v_out = _detector_exact(v_bin, r_hi, r_load)
    else:
        g, vals = _get_table("rect", r_hi, r_load)
        v_out = jnp.interp(jnp.abs(v_bin), g, vals)
    if add_noise:
        v_out = v_out + SIG_TH * jax.random.normal(key, v_out.shape)
    return v_out


def doubler(v_bin, key, r_hi=R0, r_load=R0, add_noise=True, exact=False):
    """Same full-wave pair harvested at the 2nd harmonic = passive frequency
    doubler ("square" activation); a band-pass at 2 f0 selects it (that
    filter's 1.5 dB IL is applied by the caller's filter stage).  Server-side
    carrier agility folds the x2 frequency back into the IF plan."""
    if exact:
        out = _doubler_exact(v_bin, r_hi, r_load)
    else:
        g, vals = _get_table("sq", r_hi, r_load)
        out = jnp.interp(jnp.abs(v_bin), g, vals)
    if add_noise:
        out = out + SIG_TH * jax.random.normal(key, out.shape)
    return out


# ------------------------------------------------------------------- filter
def make_filter_kernel(n_taps=31, decay=8.0, period=6.0):
    """FBAR-like band-pass impulse response sampled at the bin rate:
    exponentially decaying cosine (high-Q resonator ring-down) -> couples
    neighbouring time bins (MIWEN 'filtration').  Peak |H| normalised, then
    scaled to the pass-band insertion loss (1.5 dB)."""
    k = np.arange(n_taps) - n_taps // 2
    h = np.exp(-np.abs(k) / decay) * np.cos(2 * np.pi * k / period)
    H = np.fft.rfft(h, 4096)
    h = h / np.abs(H).max()
    return jnp.asarray(A_FILT * h, jnp.float32)


def bin_filter(v, h, key, add_noise=True):
    """Circular convolution across bins + passive-loss thermal noise."""
    n = v.shape[-1]
    hp = jnp.roll(jnp.pad(h, (0, n - h.shape[0])), -(h.shape[0] // 2))
    Hf = jnp.fft.rfft(hp)
    out = jnp.fft.irfft(jnp.fft.rfft(v, axis=-1) * Hf, n=n, axis=-1)
    if add_noise:
        out = out + SIG_TH * float(np.sqrt(max(1 - A_FILT ** 2, 0.0))) * \
              jax.random.normal(key, out.shape)
    return out


def interconnect(v, key, add_noise=True):
    out = A_IC * v
    if add_noise:
        out = out + SIG_TH * float(np.sqrt(1 - A_IC ** 2)) * \
              jax.random.normal(key, out.shape)
    return out


def lna(v, key, add_noise=True):
    g = 10 ** (LNA_GAIN_DB / 20)
    f = 10 ** (LNA_NF_DB / 10)
    if add_noise:
        v = v + SIG_TH * float(np.sqrt(f - 1)) * jax.random.normal(key, v.shape)
    return g * v


# --------------------------------------------- mixer calibration (waveform)
def calibrate_mixer(target_cl_db=6.65, lo_dbm=7.0, verbose=True):
    """Waveform-level switching-mode conversion-loss calibration.
    One sinusoidal RF tone + strong LO through ring_core; the IF component is
    extracted by FFT.  g_x is chosen so CL(+7 dBm LO) == ZEM-4300+ (6.65 dB).
    Returns g_x and the CL-vs-LO-power curve of the calibrated model."""
    n = 4096
    k_rf, k_lo = 400, 290
    k_if = k_rf - k_lo
    a_rf = np.sqrt(2 * R0 * 10 ** ((-30 - 30) / 10))     # RF at -30 dBm
    lo_range = np.arange(-8, 14.1, 1.0)
    cl = []
    tt = np.arange(n)
    for p in lo_range:
        a_lo = np.sqrt(2 * R0 * 10 ** ((p - 30) / 10))
        vlo = a_lo * np.cos(2 * np.pi * k_lo * tt / n)
        vrf = a_rf * np.cos(2 * np.pi * k_rf * tt / n)
        y = np.asarray(ring_core(jnp.asarray(vlo), jnp.asarray(vrf)))
        Y = np.fft.rfft(y) / n * 2
        a_if = np.abs(Y[k_if])
        cl.append(-20 * np.log10(a_if / a_rf))
    cl = np.asarray(cl)
    cl_at = float(np.interp(lo_dbm, lo_range, cl))
    gx_db = cl_at - target_cl_db
    g_x = 10 ** (gx_db / 20)
    if verbose:
        print(f"[cal] raw ring CL @ LO {lo_dbm:+.0f} dBm = {cl_at:.2f} dB -> "
              f"g_x = {g_x:.3f} ({gx_db:+.2f} dB transformer/network factor) "
              f"to match ZEM-4300+ {target_cl_db} dB")
    return float(g_x), lo_range, cl - gx_db


# ----------------------------------------------------- multiplication window
def mult_window_scan(g_x, w_grid=None, x_rms=0.005):
    """Effective multiplicative gain and model error of the ring in
    four-quadrant multiplication mode vs weight (LO-port) drive.  The
    activation probe is kept small (5 mV rms) so the NRMSE isolates the
    weight-port compression (a large probe adds its own ~(x/nVT)^2/12
    distortion floor: at 20 mV rms that alone is ~5 %)."""
    rng = np.random.default_rng(0)
    if w_grid is None:
        w_grid = np.logspace(np.log10(2e-3), np.log10(0.6), 25)
    x = rng.standard_normal(4096); x /= np.sqrt(np.mean(x ** 2))
    w = rng.standard_normal(4096); w /= np.sqrt(np.mean(w ** 2))
    gains, errs = [], []
    for wr in w_grid:
        y = np.asarray(ring_core(jnp.asarray(w * wr), jnp.asarray(x * x_rms))) * g_x
        p = (w * wr) * (x * x_rms)
        g = np.sum(y * p) / np.sum(p ** 2)
        r = y / g - p
        gains.append(g)
        errs.append(np.sqrt(np.mean(r ** 2)) / np.sqrt(np.mean(p ** 2)))
    return w_grid, np.asarray(gains), np.asarray(errs)


if __name__ == "__main__":
    gx, lo_r, cl = calibrate_mixer()
    v1, v2 = 0.003, 0.004
    y = float(ring_core(jnp.array(v1), jnp.array(v2)))
    print(f"[test] ring small-signal: {y*1e6:.3f} uV vs v1*v2/(4nVT) = "
          f"{v1*v2/(4*NVT)*1e6:.3f} uV")
    y = float(ring_core(jnp.array(1.0), jnp.array(0.01)))
    print(f"[test] ring commutation: {y:.4f} (expect ~ +0.005 = vrf/2)")
    key = jax.random.PRNGKey(0)
    print("[test] detector V-transfer (dB) for r_hi = 50 / 1k / 5k:")
    for a in (0.001, 0.003, 0.01, 0.03, 0.1, 0.3):
        row = [f"{20*np.log10(max(float(detector(jnp.array(a), key, r_hi=rh, add_noise=False))/a,1e-15)):+7.1f}"
               for rh in (50., 1000., 5000.)]
        print(f"   A={a*1e3:6.1f} mV: " + "  ".join(row))
    print("[test] doubler CL (dB) for r_hi = 50 / 1k / 5k:")
    for a in (0.01, 0.03, 0.1, 0.3):
        row = []
        for rh in (50., 1000., 5000.):
            v = float(doubler(jnp.array(a), key, r_hi=rh, add_noise=False))
            row.append(f"{10*np.log10(max((a/max(v,1e-30))**2,1e-30)):6.1f}")
        print(f"   A={a*1e3:6.1f} mV: " + "  ".join(row))
    print(f"[test] sigma_th = {SIG_TH*1e6:.2f} uV, nVT = {NVT*1e3:.1f} mV")
