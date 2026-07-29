"""Exemplar: the density IS an object — draw it as a field, not an outline.

Ornstein-Uhlenbeck ensemble dX = -theta X dt + sigma dW from a point mass
at x0. The marginal p_t(x) contracts to the stationary law N(0, s^2/2th);
the sample paths never settle. Three layers, three ink levels:

  RasterField      p_t(x) as a wash-ramped field (the claim's object)
  family, n=28     WEIGHT_BG sample paths (the population, one texture)
  actor, n=1       WEIGHT_ACTOR + width_profile + casing (one member,
                   still wandering after the density has converged)

This program is the worked exemplar for the channels the corpus was not
using: RasterField, the wash ramp, the background-family/foreground-actor
weight hierarchy, width_profile taper, casing, frame clipping.
"""

import numpy as np

from figlib.gates import Checks
from figlib.scene import Callout, Curve, MathLabel, Point, RasterField, Scene, Vector
from figlib.style import WEIGHT_ACTOR, WEIGHT_BG, WEIGHT_CONTENT, Role
from figlib.theme import RISO

CLAIM = (
    "An OU ensemble's marginal p_t(x) contracts onto the stationary law "
    "N(0, sigma^2/2theta) while every individual sample path keeps "
    "fluctuating at full stationary amplitude — the density converges, "
    "the paths do not."
)

THEME = RISO

PARAMS = {
    "theta": 1.0,
    "sigma": 1.0,
    "x0": 2.2,
    "T": 4.0,
    "n_steps": 480,
    "n_paths": 28,
    "seed": 7,
    "raster_nt": 260,           # field columns (t)
    "raster_nx": 220,           # field rows (x)
    # the raster starts here, not at 0: at t = 0 the marginal is a point
    # mass narrower than one grid cell — a spike the raster cannot carry
    # honestly (its column would integrate to ~0). The x0 dot and the
    # envelope pinching to a point say what happens on [0, t0].
    "raster_t0": 0.02,
    "field_x": (-2.3, 2.75),    # raster extent in x
    "xlim": (-0.06, 4.42),
    "ylim": (-2.72, 2.92),
    "wash_max": 0.6,            # peak wash strength of the density ramp
}


def _mean(p, t):
    return p["x0"] * np.exp(-p["theta"] * t)


def _var(p, t):
    return p["sigma"] ** 2 / (2 * p["theta"]) * (1.0 - np.exp(-2 * p["theta"] * t))


def compute(p):
    T, n = p["T"], p["n_steps"]
    dt = T / n
    t = np.linspace(0.0, T, n + 1)

    # Euler-Maruyama ensemble, seeded: the drawing is reproducible and the
    # gate can re-derive every path from the same recursion.
    rng = np.random.default_rng(p["seed"])
    noise = rng.standard_normal((p["n_paths"], n))
    X = np.empty((p["n_paths"], n + 1))
    X[:, 0] = p["x0"]
    for k in range(n):
        X[:, k + 1] = (X[:, k] - p["theta"] * X[:, k] * dt
                       + p["sigma"] * np.sqrt(dt) * noise[:, k])

    # the actor: the member farthest from the mean at the end — the one
    # whose wandering the converged density most flagrantly fails to predict
    actor = int(np.argmax(np.abs(X[:, -1] - _mean(p, T))))

    # analytic marginal on the raster grid; row 0 = TOP of extent
    ts = np.linspace(p["raster_t0"], T, p["raster_nt"])
    xs = np.linspace(p["field_x"][1], p["field_x"][0], p["raster_nx"])
    m, v = _mean(p, ts), np.maximum(_var(p, ts), 1e-8)
    P = (np.exp(-((xs[:, None] - m[None, :]) ** 2) / (2 * v[None, :]))
         / np.sqrt(2 * np.pi * v[None, :]))

    return {"t": t, "X": X, "actor": actor, "ts": ts, "xs": xs, "P": P,
            "sig_inf": p["sigma"] / np.sqrt(2 * p["theta"]), "p": p}


def build(g):
    p = g["p"]
    s = Scene(xlim=p["xlim"], ylim=p["ylim"], height_px=390.0, clip="frame")
    t, X = g["t"], g["X"]
    two_sig = 2.0 * g["sig_inf"]

    # layer 1 — the field: p_t(x) through the wash ramp (paper at zero
    # density, so the field needs no boundary of its own). The ramp
    # saturates at the STATIONARY peak, not the array max: the early
    # near-point-mass columns are ~5x denser and would otherwise own the
    # whole dynamic range, flattening the object the claim is about to a
    # barely-tinted band. Denser-than-stationary honestly clips to full ink.
    stat_peak = 1.0 / np.sqrt(2.0 * np.pi * g["sig_inf"] ** 2)
    s.add(RasterField(g["P"], extent=(p["raster_t0"], p["T"], *p["field_x"]),
                      # small strength floor: the zero-density corners tint
                      # faintly instead of printing a paper-colored rectangle
                      # that fights the paper's own gradient
                      ramp=lambda u: RISO.wash(
                          0, strength=0.04 + (p["wash_max"] - 0.04) * u),
                      vmin=0.0, vmax=stat_peak, interp=True))

    # layer 2 — the population: one texture, not 28 individuals
    for i in range(X.shape[0]):
        if i == g["actor"]:
            continue
        s.add(Curve(np.column_stack([t, X[i]]), role=Role.CONTENT,
                    width_scale=WEIGHT_BG, opacity=0.5))

    # layer 3 — the envelope m(t) +- 2 sigma(t): the density's own summary
    for sign in (+1.0, -1.0):
        env = _mean(p, t) + sign * 2.0 * np.sqrt(_var(p, t))
        s.add(Curve(np.column_stack([t, env]), role=Role.ACCENT1,
                    width_scale=WEIGHT_CONTENT))
    for sign in (+1.0, -1.0):     # the stationary band the envelope approaches
        s.add(Curve(np.array([[0.0, sign * two_sig], [p["T"], sign * two_sig]]),
                    role=Role.CONSTRUCTION, width_scale=0.9))

    # layer 4 — the actor: heaviest ink, tapering up as it outlives the
    # density's convergence, cased so it reads OVER the family
    s.add(Curve(np.column_stack([t, X[g["actor"]]]), role=Role.ACCENT2,
                width_scale=WEIGHT_ACTOR, width_profile=(0.55, 0.8, 1.0, 1.1),
                casing=True))

    s.add(Point((0.0, p["x0"]), role=Role.CONTENT, radius_scale=1.1))
    s.add(MathLabel(r"x_0", (0.07, p["x0"]), ha="left", va="bottom",
                    offset_px=(2, -4), size_pt=10))

    s.add(MathLabel(r"dX_t = -\theta X_t\,dt + \sigma\,dW_t",
                    (2.62, 2.55), ha="center", va="center", size_pt=11,
                    role=Role.ANNOTATION, halo=True))
    s.add(Callout(r"p_t(x)", anchor=(3.55, 1.95), target=(2.6, 0.42)))
    s.add(MathLabel(r"m(t) \pm 2\sigma(t)", (1.02, 1.72), ha="left",
                    va="bottom", size_pt=10, role=Role.ANNOTATION, halo=True,
                    color=RISO.ink(Role.ACCENT1).color))
    s.add(MathLabel(r"\pm 2\sigma_\infty,\quad \sigma_\infty^2 = \sigma^2/2\theta",
                    (3.3, -two_sig), ha="center", va="top", offset_px=(0, 6),
                    size_pt=10, role=Role.ANNOTATION, halo=True))

    # time arrow under the field
    s.add(Vector((0.0, -2.52), (p["T"], -2.52), role=Role.ANNOTATION,
                 width_scale=0.8))
    s.add(MathLabel(r"t", (p["T"], -2.52), ha="left", va="center",
                    offset_px=(6, 0), size_pt=11))
    return s


def assertions(g):
    p = g["p"]
    c = Checks()
    t, X = g["t"], g["X"]
    T = p["T"]

    # (1) the analytic layer solves its own moment ODEs: dm = -theta m,
    # dv = sigma^2 - 2 theta v (finite differences on the drawn samples)
    ts = g["ts"]
    m, v = _mean(p, ts), _var(p, ts)
    # interior points only: np.gradient's one-sided endpoint stencils are
    # O(dt) and would need a tolerance loose enough to hide a real defect
    dm = np.gradient(m, ts)[1:-1]
    dv = np.gradient(v, ts)[1:-1]
    c.check(float(np.max(np.abs(dm + p["theta"] * m[1:-1]))) < 5e-3,
            "mean curve does not solve dm/dt = -theta m")
    c.check(float(np.max(np.abs(
        dv - (p["sigma"] ** 2 - 2 * p["theta"] * v[1:-1])))) < 5e-3,
            "variance curve does not solve dv/dt = sigma^2 - 2 theta v")

    # (2) the raster is an honest density: every drawn column carries its
    # mass (truncation to the drawn x-range loses < 1%), and its peak row
    # sits on m(t) to within one grid step
    xs = g["xs"]
    dx = abs(xs[1] - xs[0])
    mass = np.trapezoid(g["P"][::-1], xs[::-1], axis=0)
    c.check(float(mass.min()) > 0.99,
            f"raster column mass {mass.min():.4f} — drawn x-range truncates the density")
    peak_x = xs[np.argmax(g["P"], axis=0)]
    c.check(float(np.max(np.abs(peak_x - m))) <= dx + 1e-12,
            "raster ridge is off the analytic mean")

    # (3) the ensemble matches its own marginal at T (seeded, so exact)
    n = X.shape[0]
    sig_T = np.sqrt(_var(p, T))
    c.check(abs(float(X[:, -1].mean()) - _mean(p, T)) < 4 * sig_T / np.sqrt(n),
            "ensemble mean at T outside 4 standard errors of m(T)")
    ratio = float(X[:, -1].var(ddof=1)) / _var(p, T)
    c.check(0.4 < ratio < 2.2,
            f"ensemble variance ratio {ratio:.2f} at T incompatible with v(T)")

    # (4) the claim's other half: paths do NOT settle — over the last unit
    # of time (density converged) the actor still moves at O(sigma_inf)
    tail = t >= T - 1.0
    swing = float(X[g["actor"], tail].max() - X[g["actor"], tail].min())
    c.check(swing > g["sig_inf"],
            f"actor swings only {swing:.2f} in the last unit of time — "
            f"'paths keep fluctuating' would be a lie at this seed")

    # (5) every drawn path starts at the point mass and follows the same
    # EM recursion the gate re-derives (actor included — it IS a member)
    c.check(bool(np.all(X[:, 0] == p["x0"])), "a path does not start at x0")
    dt = T / p["n_steps"]
    drift = X[:, 1:] - X[:, :-1] + p["theta"] * X[:, :-1] * dt
    z = drift / (p["sigma"] * np.sqrt(dt))
    c.check(float(np.abs(z).max()) < 6.0,
            "path increments incompatible with the claimed SDE discretization")
    c.done()
