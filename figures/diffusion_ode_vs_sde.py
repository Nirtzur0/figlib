"""Generation in a diffusion model: reverse SDE vs probability-flow ODE.

The mechanism the figure argues: both samplers integrate the SAME score
field, so they share every intermediate marginal — stochastic paths
jitter inside the transport, the ODE follows it deterministically, and
both land in the same two-mode data distribution from the same noise.
"""

import numpy as np

from figlib.format import WIDE
from figlib.geometry import (Mixture1D, mixture_pdf, mixture_score, ou_marginal,
                             pf_ode_paths, reverse_sde_paths, sample_mixture)
from figlib.scene import Curve, FilledCurve, MathLabel, Point, Scene, Vector
from figlib.style import Role

CLAIM = (
    "Starting from the same draws of pure noise, the jittery reverse-time SDE "
    "paths and the smooth deterministic probability-flow ODE paths of a "
    "diffusion model transport the standard-normal prior into the same "
    "two-mode data distribution, because both integrate the same score field "
    "and share the same marginals at every intermediate time."
)

FORMAT = WIDE

PARAMS = {
    "T": 2.5,
    "mix": {"weights": [0.35, 0.65], "means": [-1.8, 1.5], "sigmas": [0.15, 0.15]},
    "x_T_draws": [-1.9, -1.1, -0.45, 0.2, 0.9, 1.7],
    "n_steps_draw": 350,
    "n_steps_ode": 600,
    "n_ensemble": 2000,
    "seed": 19,
}


def compute(p):
    m0 = Mixture1D(*(np.array(p["mix"][k], dtype=float) for k in ("weights", "means", "sigmas")))
    T = p["T"]
    rng = np.random.default_rng(p["seed"])
    x_T = np.array(p["x_T_draws"], dtype=float)

    ts_ode, ode = pf_ode_paths(m0, x_T, T, p["n_steps_ode"])
    ts_sde, sde = reverse_sde_paths(m0, x_T, T, p["n_steps_draw"], rng)

    # large ensemble for the numerical gate (marginal invariance is a
    # distributional claim; six drawn paths can't certify it)
    big_xT = sample_mixture(ou_marginal(m0, T), p["n_ensemble"], rng)
    _, big_sde = reverse_sde_paths(m0, big_xT, T, p["n_steps_draw"], rng)
    _, big_ode = pf_ode_paths(m0, big_xT, T, p["n_steps_ode"])

    xs = np.linspace(-2.7, 2.6, 400)
    return {
        "m0": m0, "T": T,
        "ts_ode": ts_ode, "ode": ode,
        "ts_sde": ts_sde, "sde": sde,
        "big_sde": big_sde, "big_ode": big_ode,
        "xs": xs,
        "pdf_T": mixture_pdf(ou_marginal(m0, T), xs),
        "pdf_0": mixture_pdf(m0, xs),
    }


def _u(t, T):
    """Plot coordinate: generation runs left (noise) to right (data)."""
    return T - t


def build(g):
    T = g["T"]
    m0 = g["m0"]
    s = Scene(height_px=470, xlim=(-0.85, T + 0.85), ylim=(-3.25, 3.0))

    # time arrow along the bottom
    y0 = -2.95
    s.add(Vector((0.0, y0), (T, y0), role=Role.ANNOTATION, width_scale=0.9))
    s.add(MathLabel(r"\text{generation time}\; (t: T \to 0)", (T / 2, y0),
                    ha="center", va="top", offset_px=(0, 8), role=Role.ANNOTATION))

    # marginal densities, drawn sideways: prior at the left edge, data at the right
    # each lobe scaled to the same visual width (marginals are shape, not
    # a shared density axis)
    prior = np.column_stack([-g["pdf_T"] * (0.55 / g["pdf_T"].max()), g["xs"]])
    data = np.column_stack([T + g["pdf_0"] * (0.55 / g["pdf_0"].max()), g["xs"]])
    s.add(FilledCurve(np.vstack([[[0, g["xs"][0]]], prior, [[0, g["xs"][-1]]]]),
                      role=Role.MUTED, opacity=0.25, outline=True))
    s.add(FilledCurve(np.vstack([[[T, g["xs"][0]]], data, [[T, g["xs"][-1]]]]),
                      role=Role.MUTED, opacity=0.25, outline=True))
    s.add(MathLabel(r"p_T = \mathcal{N}(0,1)", (-0.30, 2.55), ha="center", va="bottom",
                    role=Role.ANNOTATION))
    s.add(MathLabel(r"p_0", (T + 0.35, 2.3), ha="center", va="bottom", role=Role.ANNOTATION))

    # SDE paths: jittery, thin, semi-transparent — a few suffice; more is fog
    for row in g["sde"][:4]:
        pts = np.column_stack([_u(g["ts_sde"], T), row])
        s.add(Curve(pts, role=Role.ACCENT2, width_scale=0.5, opacity=0.55))
    # ODE paths: smooth, on top
    for row in g["ode"]:
        pts = np.column_stack([_u(g["ts_ode"], T), row])
        s.add(Curve(pts, role=Role.ACCENT1, width_scale=0.9))

    # shared launch points; endpoints marked per method
    for x0 in g["ode"][:, 0]:
        s.add(Point((0.0, x0), role=Role.CONTENT, radius_scale=0.8))
    for row in g["ode"]:
        s.add(Point((T, row[-1]), role=Role.ACCENT1, radius_scale=0.8))
    for row in g["sde"][:4]:
        s.add(Point((T, row[-1]), role=Role.ACCENT2, radius_scale=0.8))

    # an SDE ensemble's endpoints, as a strip inside the data lobe: the
    # marginal claim shown, not asserted
    for x_end in g["big_sde"][:70, -1]:
        s.add(Point((T + 0.10, x_end), role=Role.ACCENT2, radius_scale=0.38))

    # method labels anchored to representative paths
    s.add(MathLabel(r"\text{reverse SDE:}\;\; dx = [-x - 2\nabla_x \log p_t]\,dt + \sqrt{2}\,d\bar{w}",
                    (0.35, 2.62), ha="left", va="bottom", role=Role.ACCENT2))
    s.add(MathLabel(r"\text{probability flow ODE:}\;\; \dot{x} = -x - \nabla_x \log p_t",
                    (0.35, -2.72), ha="left", va="top", role=Role.ACCENT1))

    return s


def assertions(g):
    m0, T = g["m0"], g["T"]
    # 1. the drawn ODE paths really solve the ODE: re-integrate at double
    #    resolution and demand pathwise agreement (Richardson check)
    ode = g["ode"]
    n = ode.shape[1] - 1
    _, fine = pf_ode_paths(m0, ode[:, 0], T, 2 * n)
    err = np.max(np.abs(fine[:, ::2] - ode))
    assert err < 1e-3, f"ODE solution not converged: pathwise error {err:.2e}"

    # 2. both big ensembles reproduce the data mixture (weights and means)
    for name, end in (("sde", g["big_sde"][:, -1]), ("ode", g["big_ode"][:, -1])):
        left = end < 0
        assert abs(left.mean() - m0.weights[0]) < 0.04, f"{name}: mode weights wrong"
        assert abs(end[left].mean() - m0.means[0]) < 0.08, f"{name}: left mode off"
        assert abs(end[~left].mean() - m0.means[1]) < 0.08, f"{name}: right mode off"

    # 3. marginal invariance at intermediate time: SDE ensemble quantiles
    #    match the analytic marginal
    j = g["big_sde"].shape[1] // 2
    t_mid = T - T * j / (g["big_sde"].shape[1] - 1)
    m_mid = ou_marginal(m0, t_mid)
    xs = np.linspace(-4, 4, 4000)
    cdf = np.cumsum(mixture_pdf(m_mid, xs))
    cdf /= cdf[-1]
    for q in (0.25, 0.5, 0.75):
        analytic = xs[np.searchsorted(cdf, q)]
        empirical = np.quantile(g["big_sde"][:, j], q)
        assert abs(analytic - empirical) < 0.12, (
            f"marginal mismatch at t={t_mid:.2f}, q={q}: {analytic:.3f} vs {empirical:.3f}")
