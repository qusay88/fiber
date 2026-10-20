"""
Fiber-optic vs conventional media in a Networked Control System (NCS)
Plant: Furuta (rotary inverted) pendulum, QUBE-Servo-2-like parameters.

Run:  python fiber_furuta_ncs.py            (all experiments, saves figures in ./results)
      python fiber_furuta_ncs.py --quick    (fewer Monte-Carlo runs)
"""
import math, heapq, argparse, os, json
import numpy as np
from scipy.linalg import solve_discrete_are, solve_continuous_are
from scipy.signal import cont2discrete
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =====================================================================
# 1) PLANT: Furuta pendulum (alpha = 0 is UPRIGHT)
# =====================================================================
class P:  # parameters (SI)
    Mr, Lr = 0.095, 0.085          # arm mass, length
    Mp, Lp = 0.024, 0.129          # pendulum mass, length
    Rm, km = 8.4, 0.042            # motor resistance, torque/back-emf const
    Dr, Dp = 1e-3, 5e-5            # viscous friction arm / pendulum
    g = 9.81
    Vmax = 10.0                    # actuator saturation [V]

Jr = P.Mr * P.Lr**2 / 3.0          # arm inertia about its axis
lp = P.Lp / 2.0
Jp = P.Mp * P.Lp**2 / 12.0
a_ = Jr + P.Mp * P.Lr**2
b_ = P.Mp * P.Lr * lp
c_ = P.Mp * lp**2 + Jp
mgl = P.Mp * P.g * lp


def params(mp=1.0, lp=1.0, jr=1.0, fr=1.0, km=1.0):
    """Plant parameter set (a, b, c, mgl, Dr, Dp, km, Rm) with multiplicative perturbations."""
    Mp, Lp = P.Mp * mp, P.Lp * lp
    l_p = Lp / 2.0
    Jp_ = Mp * Lp**2 / 12.0
    a = Jr * jr + Mp * P.Lr**2
    b = Mp * P.Lr * l_p
    c = Mp * l_p**2 + Jp_
    return (a, b, c, Mp * P.g * l_p, P.Dr * fr, P.Dp * fr, P.km * km, P.Rm)


NOM = params()


def f(x, V, par=None):
    """x=[theta, alpha, dtheta, dalpha]; V = motor voltage; par = plant parameter tuple."""
    a, b, c_pp, mgl_, Dr, Dp, km_, Rm = NOM if par is None else par
    th, al, dth, dal = x
    s, c = math.sin(al), math.cos(al)
    tau = km_ * (V - km_ * dth) / Rm
    m11 = a + c_pp * s * s
    m12 = b * c
    m22 = c_pp
    r1 = tau - Dr * dth - 2 * c_pp * s * c * dal * dth + b * s * dal * dal
    r2 = -Dp * dal + c_pp * s * c * dth * dth + mgl_ * s
    det = m11 * m22 - m12 * m12
    ddth = (m22 * r1 - m12 * r2) / det
    ddal = (-m12 * r1 + m11 * r2) / det
    return np.array([dth, dal, ddth, ddal])


def rk4(x, V, dt, par=None):
    k1 = f(x, V, par)
    k2 = f(x + 0.5 * dt * k1, V, par)
    k3 = f(x + 0.5 * dt * k2, V, par)
    k4 = f(x + dt * k3, V, par)
    return x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


def linearize_upright():
    """Numerical Jacobian at the upright equilibrium."""
    x0 = np.zeros(4); eps = 1e-6
    A = np.zeros((4, 4))
    for i in range(4):
        d = np.zeros(4); d[i] = eps
        A[:, i] = (f(x0 + d, 0) - f(x0 - d, 0)) / (2 * eps)
    B = ((f(x0, eps) - f(x0, -eps)) / (2 * eps)).reshape(4, 1)
    return A, B


# =====================================================================
# 2) CONTROLLER: energy swing-up + LQR balance (designed in discrete time)
# =====================================================================
def design_lqr(Tc, Q=None, R=None):
    A, B = linearize_upright()
    Ad, Bd, *_ = cont2discrete((A, B, np.eye(4), np.zeros((4, 1))), Tc)
    Q = np.diag([5.0, 80.0, 0.5, 2.0]) if Q is None else Q
    R = np.array([[1.0]]) if R is None else R
    Pm = solve_discrete_are(Ad, Bd, Q, R)
    K = np.linalg.solve(R + Bd.T @ Pm @ Bd, Bd.T @ Pm @ Ad)
    return Ad, Bd, K


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class Controller:
    def __init__(self, K, Ad=None, Bd=None, Tc=0.005, predict=False, nn=None):
        self.K = K
        self.Ad, self.Bd, self.Tc, self.predict, self.nn = Ad, Bd, Tc, predict, nn
        self.hist = []                       # past commands (for the predictor)
        self.mode = "swing"
        self.k_e, self.k_th, self.k_dth = 20000.0, 2.0, 0.1   # swing-up gains (near bang-bang)

    def _predict(self, y, age, est_ca):
        """Propagate the delayed measurement through the linear model using past commands."""
        n = int(round((age + est_ca) / self.Tc)); n = max(0, min(n, 8))
        x = y.copy()
        for i in range(n):
            j = -(n - i)
            u = self.hist[j] if len(self.hist) >= -j else 0.0
            x = self.Ad @ x + (self.Bd * u).flatten()
        return x

    def _predict_nl(self, y, age, est_ca):
        """Nonlinear prediction (RK4, 1 ms steps) through the past commands (works in swing-up too)."""
        n = int(round((age + est_ca) / self.Tc)); n = max(0, min(n, 10))
        x = np.array(y, float); sub = int(round(self.Tc / 0.001))
        for i in range(n):
            j = -(n - i)
            u = self.hist[j] if len(self.hist) >= -j else 0.0
            for _ in range(sub):
                x = rk4(x, u, 0.001)
        return x

    def __call__(self, y, age=0.0, est_ca=0.0):
        if self.predict and self.nn is None:
            y = self._predict_nl(y, age, est_ca)
        th, al, dth, dal = y[0], wrap(y[1]), y[2], y[3]
        y = np.array([th, al, dth, dal])
        if self.mode == "swing" and abs(al) < 0.35 and abs(dal) < 8.0:
            self.mode = "balance"
        if self.mode == "balance" and abs(al) > 0.9:   # fell -> re-swing
            self.mode = "swing"
        if self.mode == "balance":
            if self.nn is not None:
                u = float(self.nn(y))
            else:
                u = -float((self.K @ y).item())
            u = float(np.clip(u, -P.Vmax, P.Vmax))
            self.hist.append(u); self.hist = self.hist[-12:]
            return u
        else:
            E = 0.5 * c_ * dal**2 + mgl * (math.cos(al) - 1.0)   # 0 at upright
            u = -self.k_e * (0.0 - E) * np.sign(dal * math.cos(al) + 1e-9)
            u = u - self.k_th * th - self.k_dth * dth
        u = float(np.clip(u, -P.Vmax, P.Vmax))
        self.hist.append(u); self.hist = self.hist[-12:]
        return u



# ---------------------------------------------------------------------
# Neural balancing controller (behaviour cloning of LQR) - numpy MLP
# ---------------------------------------------------------------------
class MLP:
    def __init__(self, sizes=(4, 32, 32, 1), seed=0):
        r = np.random.default_rng(seed)
        self.W = [r.normal(0, 1 / math.sqrt(a), (a, b)) for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(b) for b in sizes[1:]]
        self.mu = np.zeros(4); self.sd = np.ones(4)

    def forward(self, X, cache=False):
        H = [(X - self.mu) / self.sd]
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            Z = H[-1] @ W + b
            H.append(np.tanh(Z) if i < len(self.W) - 1 else Z)
        return (H if cache else H[-1])

    def __call__(self, y):
        return float(self.forward(np.asarray(y, float)[None, :])[0, 0])

    def fit(self, X, Y, iters=3000, lr=3e-3, batch=512):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        mW = [np.zeros_like(w) for w in self.W]; vW = [np.zeros_like(w) for w in self.W]
        mb = [np.zeros_like(b) for b in self.b]; vb = [np.zeros_like(b) for b in self.b]
        rng = np.random.default_rng(1); Xf, Yf = X, Y
        for it in range(1, iters + 1):
            idx = rng.integers(0, len(Xf), batch); X, Y = Xf[idx], Yf[idx]
            H = self.forward(X, cache=True)
            G = 2 * (H[-1] - Y) / len(X)
            for i in reversed(range(len(self.W))):
                gW = H[i].T @ G; gb = G.sum(0)
                if i > 0:
                    G = (G @ self.W[i].T) * (1 - H[i] ** 2)
                for (m_, v_, g, p) in ((mW, vW, gW, self.W), (mb, vb, gb, self.b)):
                    m_[i] = 0.9 * m_[i] + 0.1 * g; v_[i] = 0.999 * v_[i] + 0.001 * g * g
                    p[i] -= lr * (m_[i] / (1 - 0.9 ** it)) / (np.sqrt(v_[i] / (1 - 0.999 ** it)) + 1e-8)
        return float(np.mean((self.forward(Xf) - Yf) ** 2))


_NN_CACHE = {}


def train_nn_controller(K, seed=0, n=20000):
    r = np.random.default_rng(seed)
    X = np.column_stack([r.uniform(-1.0, 1.0, n), r.uniform(-0.6, 0.6, n),
                         r.uniform(-4, 4, n), r.uniform(-8, 8, n)])
    Y = np.clip(-(X @ K.flatten()), -P.Vmax, P.Vmax)[:, None]
    net = MLP(seed=seed); mse = net.fit(X, Y)
    return net, mse


def make_controller(kind, K, Ad, Bd, Tc):
    if kind == "lqr":
        return Controller(K, Ad, Bd, Tc)
    if kind == "pred":
        return Controller(K, Ad, Bd, Tc, predict=True)
    if kind == "nn":
        key = tuple(np.round(K.flatten(), 6))
        if key not in _NN_CACHE:
            _NN_CACHE[key] = train_nn_controller(K)[0]
        return Controller(K, Ad, Bd, Tc, nn=_NN_CACHE[key])
    raise ValueError(kind)

# =====================================================================
# 3) CHANNEL MODELS (one-way sensor->controller and controller->actuator)
# =====================================================================
# Channel parameters. Anchors (see README / paper references):
#  - propagation ~5 us/km in fiber (n~1.47) and ~5.2 us/km in Cat5 (VF~0.64)
#  - store-and-forward switch hop ~120 us (1500 B @ 100 Mb/s) + ~15 us jitter/hop
#  - wired real-time Ethernet: cycle <100 us, jitter ~us; Wi-Fi: ms-level latency/jitter,
#    loss hidden by retransmissions but latency grows (exponential backoff)
# The absolute values are ENGINEERING ASSUMPTIONS built on those anchors: replace them
# with your own measurements (ping/iperf/OWAMP) before publishing.
US = 1e-6
MEDIA = {
    # base delay [s], jitter scale [s] (exponential), loss prob, sensor-noise std [rad]
    "Fiber":  dict(base=25 * US,   jitter=2 * US,    loss=0.0,  noise=0.0005),  # 1 hop, ~1 km
    "Copper": dict(base=500 * US,  jitter=60 * US,   loss=1e-3, noise=0.0020),  # ~4 switch hops, EMI
    "WiFi":   dict(base=4000 * US, jitter=3000 * US, loss=0.03, noise=0.0020),
}


def medium_for_distance(kind, km):
    """Distance model: fiber = 1 hop + 5 us/km; copper = 1 switch hop per 100 m (Ethernet limit)."""
    if kind == "Fiber":
        return dict(base=25 * US + 5 * US * km, jitter=2 * US, loss=0.0, noise=0.0005)
    hops = max(1, math.ceil(km * 10))
    return dict(base=hops * 120 * US + 5.2 * US * km, jitter=hops * 15 * US,
                loss=min(0.05, 1e-4 * hops), noise=0.0020)


class Channel:
    def __init__(self, base, jitter, loss, rng):
        self.base, self.jitter, self.loss, self.rng = base, jitter, loss, rng
        self.q = []          # heap of (arrival_time, seq, payload)
        self.seq = 0

    def send(self, t, payload):
        if self.rng.random() < self.loss:
            return                                   # packet lost
        d = self.base + (self.rng.exponential(self.jitter) if self.jitter > 0 else 0.0)
        heapq.heappush(self.q, (t + d, self.seq, payload))
        self.seq += 1

    def receive(self, t):
        """Return list of payloads that arrived up to time t (in arrival order)."""
        out = []
        while self.q and self.q[0][0] <= t:
            out.append(heapq.heappop(self.q)[2])
        return out


# =====================================================================
# 4) CLOSED-LOOP NCS SIMULATION
# =====================================================================
def simulate(medium, Tc=0.005, T=6.0, dt=0.001, seed=0, x0=None, K=None,
             const_delay=None, loss=None, jitter=None, noise=None, ctrl_kind="lqr", est_scale=1.0, T_end_check=None, plant_scales=None, clock_offset=0.0, clock_drift=0.0):
    """medium: key in MEDIA or dict. const_delay: extra fixed delay each way [s]."""
    rng = np.random.default_rng(seed)
    m = dict(MEDIA[medium]) if isinstance(medium, str) else dict(medium)
    if const_delay is not None:
        m.update(base=const_delay, jitter=0.0)
    if loss is not None:   m["loss"] = loss
    if jitter is not None: m["jitter"] = jitter
    if noise is not None:  m["noise"] = noise
    Ad, Bd, K0 = design_lqr(Tc)
    K = K0 if K is None else K
    ctrl = make_controller(ctrl_kind, K, Ad, Bd, Tc)
    par_plant = params(**plant_scales) if plant_scales else None
    sc = Channel(m["base"], m["jitter"], m["loss"], rng)   # sensor -> controller
    ca = Channel(m["base"], m["jitter"], m["loss"], rng)   # controller -> actuator
    x = np.array([0.0, math.pi, 0.0, 0.0]) if x0 is None else np.array(x0, float)
    x[1] += rng.normal(0, 0.02)                            # random initial offset
    n = int(T / dt); ns = int(round(Tc / dt))
    y_last = x.copy(); u_applied = 0.0
    log = np.zeros((n, 6))
    for i in range(n):
        t = i * dt
        if i % ns == 0:                                    # sensor samples & sends
            y = x + rng.normal(0, m["noise"], 4) * np.array([1, 1, 20, 20])
            sc.send(t, (t, y))
        arr = sc.receive(t)                                # controller side
        if arr:
            t_pkt, y_last = max(arr, key=lambda p: p[0])   # newest packet wins
            age_meas = max(0.0, (t - t_pkt) + clock_offset + clock_drift * t)
            u_new = ctrl(y_last, age=age_meas, est_ca=est_scale * (m["base"] + m["jitter"]))
            ca.send(t, u_new)
        arr_u = ca.receive(t)                              # actuator side
        if arr_u:
            u_applied = arr_u[-1]                          # hold last (ZOH)
        x = rk4(x, u_applied, dt, par_plant)
        log[i] = [t, x[0], wrap(x[1]), x[2], x[3], u_applied]
        if abs(x[0]) > 8 * math.pi:                        # arm over-rotation -> abort
            log = log[:i + 1]; break
    return log


# =====================================================================
# 5) METRICS
# =====================================================================
def metrics(log, tol=math.radians(5)):
    t, al, u = log[:, 0], log[:, 2], log[:, 5]
    up = np.abs(al) < tol
    # success: balanced (|alpha|<tol) over the final 1.5 s
    tail = t > (t[-1] - 1.5)
    success = bool(len(t) > 1 and t[-1] > 5.9 and up[tail].all())
    t_up = np.nan
    if success:
        bad = np.where(~up)[0]
        t_up = t[bad[-1] + 1] if len(bad) else 0.0      # last time it entered the band
    bal = t > (t_up if success else t[-1])
    iae = np.trapezoid(np.abs(al[tail]), t[tail]) if success else np.nan
    eff = np.sqrt(np.mean(u[tail] ** 2)) if success else np.nan
    return dict(success=success, t_settle=t_up, iae=iae, u_rms=eff)


# =====================================================================
# 6) EXPERIMENTS
# =====================================================================
def exp_baseline(outdir):
    log = simulate({"base": 0, "jitter": 0, "loss": 0, "noise": 0.0}, seed=1)
    fig, ax = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    ax[0].plot(log[:, 0], np.degrees(log[:, 2])); ax[0].set_ylabel("alpha [deg]")
    ax[1].plot(log[:, 0], np.degrees(log[:, 1])); ax[1].set_ylabel("theta [deg]")
    ax[2].plot(log[:, 0], log[:, 5]); ax[2].set_ylabel("V [volt]"); ax[2].set_xlabel("t [s]")
    fig.suptitle("Baseline (ideal network): swing-up then LQR balance")
    fig.tight_layout(); fig.savefig(f"{outdir}/01_baseline.png", dpi=150); plt.close(fig)
    print("baseline:", metrics(log))


def exp_media(outdir, runs):
    res = {}
    for name in MEDIA:
        rows = [metrics(simulate(name, seed=s)) for s in range(runs)]
        succ = np.mean([r["success"] for r in rows])
        def m(k):
            v = [r[k] for r in rows if r["success"]]
            return (float(np.mean(v)), float(np.std(v))) if v else (np.nan, np.nan)
        res[name] = dict(success=succ, t_settle=m("t_settle"), iae=m("iae"), u_rms=m("u_rms"))
        print(name, res[name])
    names = list(res)
    fig, ax = plt.subplots(1, 4, figsize=(14, 3.6))
    ax[0].bar(names, [100 * res[n]["success"] for n in names]); ax[0].set_title("Success rate [%]")
    for k, a, ttl in [("t_settle", ax[1], "Time to stabilise [s]"),
                      ("iae", ax[2], "IAE of alpha (last 1.5 s)"),
                      ("u_rms", ax[3], "RMS voltage [V]")]:
        a.bar(names, [res[n][k][0] for n in names], yerr=[res[n][k][1] for n in names], capsize=4)
        a.set_title(ttl)
    fig.tight_layout(); fig.savefig(f"{outdir}/02_media_comparison.png", dpi=150); plt.close(fig)
    json.dump(res, open(f"{outdir}/media_results.json", "w"), indent=2, default=float)
    return res


def exp_traces(outdir):
    fig, ax = plt.subplots(len(MEDIA), 1, figsize=(8, 7), sharex=True)
    for a, name in zip(ax, MEDIA):
        log = simulate(name, seed=3)
        a.plot(log[:, 0], np.degrees(log[:, 2])); a.set_ylabel(f"{name}\nalpha [deg]")
        a.set_ylim(-190, 190)
    ax[-1].set_xlabel("t [s]"); fig.tight_layout()
    fig.savefig(f"{outdir}/03_time_traces.png", dpi=150); plt.close(fig)


def exp_linear_delay_margin(outdir, Tc=0.005, max_steps=40):
    """Analytic: x[k+1]=Ad x[k]+Bd u[k-d],  u=-K x  -> spectral radius of augmented system."""
    Ad, Bd, K = design_lqr(Tc)
    rho = []
    for d in range(0, max_steps + 1):
        n = 4
        Z = np.zeros((n + d, n + d))
        Z[:n, :n] = Ad
        if d == 0:
            Z[:n, :n] = Ad - Bd @ K
        else:
            Z[:n, n + d - 1:n + d] = Bd                    # u delayed d steps
            Z[n, :n] = -K.flatten()                        # new u enters the buffer
            for i in range(1, d):
                Z[n + i, n + i - 1] = 1.0
        rho.append(max(abs(np.linalg.eigvals(Z))))
    rho = np.array(rho)
    unstable = np.where(rho >= 1)[0]
    d_max = (unstable[0] - 1) if len(unstable) else max_steps
    plt.figure(figsize=(7, 4))
    plt.plot(np.arange(len(rho)) * Tc * 1e3, rho, "o-"); plt.axhline(1, color="r", ls="--")
    plt.xlabel("total loop delay [ms]"); plt.ylabel("spectral radius")
    plt.title(f"Linear delay margin = {d_max * Tc * 1e3:.0f} ms"); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(f"{outdir}/04_linear_delay_margin.png", dpi=150); plt.close()
    print(f"linear delay margin (LQR, Tc={Tc*1e3:.0f} ms): {d_max*Tc*1e3:.0f} ms")
    return d_max * Tc


def exp_sweeps(outdir, runs):
    def sweep(values, **kw):
        out = []
        for v in values:
            ok = [metrics(simulate("Fiber", seed=s, **{kw["key"]: v}))["success"] for s in range(runs)]
            out.append(100 * np.mean(ok))
        return out
    delays = np.array([0, 2, 5, 8, 12, 16, 20, 25, 30]) * 1e-3
    jit = np.array([0, 1, 2, 4, 6, 8, 12]) * 1e-3
    loss = np.array([0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4])
    s_d = sweep(delays, key="const_delay"); s_j = sweep(jit, key="jitter"); s_l = sweep(loss, key="loss")
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
    ax[0].plot(delays * 1e3, s_d, "o-"); ax[0].set_xlabel("fixed one-way delay [ms]")
    ax[1].plot(jit * 1e3, s_j, "o-");   ax[1].set_xlabel("jitter (mean) [ms]")
    ax[2].plot(loss * 100, s_l, "o-");  ax[2].set_xlabel("packet loss [%]")
    for a in ax: a.set_ylabel("success rate [%]"); a.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{outdir}/05_sweeps.png", dpi=150); plt.close(fig)
    print("delay sweep", dict(zip((delays * 1e3).tolist(), s_d)))
    print("jitter sweep", dict(zip((jit * 1e3).tolist(), s_j)))
    print("loss sweep", dict(zip(loss.tolist(), s_l)))



def exp_distance(outdir, runs):
    kms = [0.1, 0.5, 1, 2, 3, 5, 8, 12]
    res = {}
    for kind in ("Fiber", "Copper"):
        sr, ts = [], []
        for km in kms:
            ms = [metrics(simulate(medium_for_distance(kind, km), seed=s)) for s in range(runs)]
            sr.append(100 * np.mean([m["success"] for m in ms]))
            v = [m["t_settle"] for m in ms if m["success"]]
            ts.append(np.mean(v) if v else np.nan)
        res[kind] = dict(success=sr, t_settle=ts)
        print("distance", kind, dict(zip(kms, sr)))
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for kind in res:
        ax[0].plot(kms, res[kind]["success"], "o-", label=kind)
        ax[1].plot(kms, res[kind]["t_settle"], "o-", label=kind)
    ax[0].set_ylabel("success rate [%]"); ax[1].set_ylabel("time to stabilise [s]")
    for a in ax: a.set_xlabel("link length [km]"); a.grid(alpha=.3); a.legend()
    fig.tight_layout(); fig.savefig(f"{outdir}/06_distance.png", dpi=150); plt.close(fig)
    json.dump(res, open(f"{outdir}/distance_results.json", "w"), indent=2, default=float)


# ---------------------------------------------------------------------
# Controller comparison experiments (LQR vs delay-compensated vs NN-imitation)
# ---------------------------------------------------------------------
def _sr(kind, runs, **kw):
    ms = [metrics(simulate(seed=s, ctrl_kind=kind, **kw)) for s in range(runs)]
    ok = [m for m in ms if m["success"]]
    return (100 * len(ok) / runs, float(np.mean([m["t_settle"] for m in ok])) if ok else float("nan"))


def exp_ctrl_delay(outdir, runs=10):
    delays = [0, 5, 8, 12, 16, 20, 25, 30, 40]
    res = {k: [_sr(k, runs, medium="Fiber", const_delay=d * 1e-3)[0] for d in delays] for k in ("lqr", "pred")}
    json.dump(dict(delays_ms=delays, **res), open(f"{outdir}/ctrl_delay.json", "w"))
    plt.figure(figsize=(6.5, 4))
    for k, lab in (("lqr", "LQR (no compensation)"), ("pred", "LQR + nonlinear predictor")):
        plt.plot(delays, res[k], "o-", label=lab)
    plt.xlabel("fixed one-way delay [ms]"); plt.ylabel("success rate [%]"); plt.grid(alpha=.3); plt.legend()
    plt.tight_layout(); plt.savefig(f"{outdir}/07_ctrl_delay.png", dpi=150); plt.close()
    print("ctrl delay", delays, res)


def exp_ctrl_distance(outdir, runs=10):
    kms = [1, 3, 5, 8, 12, 16, 20]
    res = {}
    for k in ("lqr", "pred"):
        res[k] = [_sr(k, runs, medium=medium_for_distance("Copper", km))[0] for km in kms]
    json.dump(dict(kms=kms, **res), open(f"{outdir}/ctrl_distance.json", "w"))
    plt.figure(figsize=(6.5, 4))
    for k, lab in (("lqr", "Copper + LQR"), ("pred", "Copper + predictor")):
        plt.plot(kms, res[k], "o-", label=lab)
    plt.xlabel("link length [km]"); plt.ylabel("success rate [%]"); plt.grid(alpha=.3); plt.legend()
    plt.tight_layout(); plt.savefig(f"{outdir}/08_ctrl_distance.png", dpi=150); plt.close()
    print("ctrl distance", kms, res)


def exp_ctrl_media(outdir, runs=20):
    res = {}
    for name in MEDIA:
        for k in ("lqr", "pred"):
            res[f"{name}-{k}"] = _sr(k, runs, medium=name)
    json.dump(res, open(f"{outdir}/ctrl_media.json", "w"))
    print("ctrl media", res)


def exp_ctrl_robust(outdir, runs=10):
    """Predictor with wrong delay estimate (scale on estimated controller->actuator delay), 16 ms fixed delay."""
    scales = [0.0, 0.5, 1.0, 1.5, 2.0]
    res = [_sr("pred", runs, medium="Fiber", const_delay=16e-3, est_scale=sc)[0] for sc in scales]
    json.dump(dict(scales=scales, success=res), open(f"{outdir}/ctrl_robust.json", "w"))
    print("est_scale robustness", scales, res)


def exp_nn_balance(outdir, runs=10):
    """Balance-only (start near upright): LQR vs NN imitation vs predictor, delay sweep."""
    delays = [0, 8, 12, 16, 20, 25, 30]
    res = {k: [_sr(k, runs, medium="Fiber", const_delay=d * 1e-3, x0=[0, 0.15, 0, 0])[0] for d in delays]
           for k in ("lqr", "nn", "pred")}
    json.dump(dict(delays_ms=delays, **res), open(f"{outdir}/balance_only.json", "w"))
    print("balance-only", delays, res)


# ---------------------------------------------------------------------
# Robustness: model mismatch and clock errors
# ---------------------------------------------------------------------
def exp_mismatch_single(outdir, runs=8, names=("mp", "lp", "km", "fr"), tag="a"):
    """One parameter at a time (plant only; controller/predictor keep nominal values)."""
    factors = [0.8, 0.9, 1.1, 1.2]
    res = {}
    for nme in names:
        res[nme] = dict(
            pred16=[_sr("pred", runs, medium="Fiber", const_delay=16e-3, plant_scales={nme: fct})[0] for fct in factors],
            lqr5=[_sr("lqr", runs, medium="Fiber", const_delay=5e-3, plant_scales={nme: fct})[0] for fct in factors])
        print("mismatch", nme, res[nme], flush=True)
    json.dump(dict(factors=factors, res=res), open(f"{outdir}/mismatch_single_{tag}.json", "w"))


def exp_mismatch_random(outdir, level=0.1, n=20, delay=16e-3, kind="pred"):
    """All parameters perturbed at once, uniform in [1-level, 1+level]."""
    r = np.random.default_rng(123)
    ok = 0
    for s_ in range(n):
        sc = {k: float(r.uniform(1 - level, 1 + level)) for k in ("mp", "lp", "jr", "fr", "km")}
        m = metrics(simulate("Fiber", seed=s_, const_delay=delay, ctrl_kind=kind, plant_scales=sc))
        ok += m["success"]
    print("random mismatch", level, kind, delay, 100 * ok / n, flush=True)
    return 100 * ok / n


def exp_clock(outdir, runs=8):
    offs = [-10, -5, -2, 0, 2, 5, 10, 20]
    res = [_sr("pred", runs, medium="Fiber", const_delay=16e-3, clock_offset=o * 1e-3)[0] for o in offs]
    drifts = [0, 100, 1000, 10000]      # ppm (10 000 ppm = 1 %)
    resd = [_sr("pred", runs, medium="Fiber", const_delay=16e-3, clock_drift=d * 1e-6)[0] for d in drifts]
    json.dump(dict(offsets_ms=offs, offset_success=res, drift_ppm=drifts, drift_success=resd),
              open(f"{outdir}/clock.json", "w"))
    print("clock offset", offs, res, "drift", drifts, resd, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    out = "results"; os.makedirs(out, exist_ok=True)
    runs = 6 if args.quick else 30
    exp_baseline(out)
    exp_linear_delay_margin(out)
    exp_media(out, runs)
    exp_traces(out)
    exp_sweeps(out, max(4, runs // 3))
    exp_distance(out, max(4, runs // 3))
    exp_ctrl_delay(out); exp_ctrl_distance(out); exp_ctrl_media(out); exp_ctrl_robust(out); exp_nn_balance(out)
    print("done -> ./results")
