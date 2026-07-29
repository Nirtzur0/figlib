"""Conditioning, not step size, sets gradient descent's clock.

For a quadratic f(x) = 0.5 x^T A x with eigenvalues (mu, L), mu the
smallest curvature and L = kappa * mu the largest, steepest descent with
the best constant step eta = 2/(mu+L) contracts the distance to the
optimum by rho = (kappa-1)/(kappa+1) EVERY iteration, in EVERY direction —
including the shallow one. Reaching a fixed relative tolerance costs
N ~ log(1/tol) / log(1/rho) ~ O(kappa) iterations: doubling the condition
number roughly doubles the iteration count, independent of how carefully
the step is tuned. No step size fixes this; eta is already optimal.

Two remedies attack kappa directly rather than eta. Heavy-ball momentum
turns the same recursion into a damped linear oscillator whose contraction
factor at the optimal (eta, beta) is (sqrt(kappa)-1)/(sqrt(kappa)+1) — the
condition number enters through its SQUARE ROOT, so the iteration count
drops from O(kappa) to O(sqrt(kappa)). The price is that the two
eigen-directions no longer decay independently: the extremal directions
pick up complex-conjugate iteration roots of equal modulus, and the
objective can rise between steps even while the envelope shrinks — the
classic momentum overshoot, not a bug to hide. Newton's method rescales
by the Hessian itself, x_{k+1} = x_k - A^{-1} grad f(x_k): on an exact
quadratic that maps every point to the optimum in one step, because
A^{-1} A is the identity regardless of kappa. This figure draws the same
start, the same tolerance, and all three trajectories on the same
elliptical level sets, so the iteration counts a reader would otherwise
have to derive are instead a fact read off the page.
"""

import numpy as np

from figlib.format import WIDE
from figlib.gates import Checks
from figlib.scene import Curve, MathLabel, Point, Scene
from figlib.style import WEIGHT_ACTOR, WEIGHT_CONTENT, Role
from figlib.theme import RISO

CLAIM = (
    "Gradient descent's iteration count is governed by the Hessian's "
    "condition number kappa, not the step size: on the same ill-conditioned "
    "quadratic, plain steepest descent needs ~kappa iterations to zig-zag "
    "to tolerance, heavy-ball momentum cuts that to ~sqrt(kappa) by damping "
    "the oscillation rather than shrinking the step, and Newton's method "
    "reaches the optimum in a single step because it rescales by the "
    "Hessian and so never sees kappa at all."
)

EXPOSITION = """
The optimization textbook's standard cautionary result (Nocedal & Wright;
Boyd & Vandenberghe) is that gradient descent's convergence rate on a
quadratic is set by the Hessian's condition number kappa = L/mu, not by how
carefully the step size is tuned. With the best constant step, steepest
descent still needs on the order of kappa iterations to reach a fixed
relative tolerance, because the same step that is barely stable along the
steep direction is absurdly conservative along the shallow one -- the
method is forced to zig-zag across the narrow valley instead of walking
down it. A reader who has only seen the bound N ~ O(kappa) in symbols needs
to see the zig-zag itself, and needs to see that the two standard remedies
attack kappa rather than the step: heavy-ball momentum turns the recursion
into a damped oscillator whose rate depends on sqrt(kappa) instead of
kappa, at the cost of legitimate overshoot, while Newton's method rescales
by the Hessian itself and so never sees kappa at all, converging in one
step regardless of how ill-conditioned the problem is.
"""

THEME = RISO
FORMAT = WIDE

PARAMS = {
    "mu": 1.0,             # smallest curvature (eigenvalue of the Hessian)
    "kappa": 25.0,         # condition number: L = kappa * mu
    "theta_deg": 45.0,     # eigenvector frame, rotated off the page axes
    "u0": (3.0, 0.85),     # start point in the EIGENBASIS (shallow, steep):
                           # mostly along the valley floor, small perpendicular
                           # offset — the classic "high on one wall" start
    "tol": 1e-2,           # ||x_k|| / ||x_0|| target for GD and momentum
    "max_iter": 400,       # safety cap on the simulated recursions
    "level_ratios": (1.0, 0.55, 0.28, 0.12, 0.045),  # f(x)/f(x0) ladder
    "gd_arrow_fracs": (0.10, 0.32, 0.55, 0.78),
    "n_ellipse": 240,
}


def _rotation(theta_deg):
    th = np.radians(theta_deg)
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s], [s, c]])


def _run(A, x0, eta, beta, max_iter, tol_norm):
    """x_{k+1} = x_k - eta*A@x_k + beta*(x_k - x_{k-1}), beta=0 -> plain GD.

    x_{-1} := x_0 so the first step carries no momentum term. Returns the
    full path (including x_0) up to and including the first iterate inside
    tol_norm, or max_iter iterates if it never gets there.
    """
    path = [np.asarray(x0, dtype=float)]
    prev = path[0]
    for _ in range(max_iter):
        cur = path[-1]
        nxt = cur - eta * (A @ cur) + beta * (cur - prev)
        prev = cur
        path.append(nxt)
        if np.linalg.norm(nxt) < tol_norm:
            break
    return np.array(path)


def compute(p):
    mu, kappa = p["mu"], p["kappa"]
    L = mu * kappa
    R = _rotation(p["theta_deg"])
    A = R @ np.diag([mu, L]) @ R.T

    x0 = R @ np.array(p["u0"], dtype=float)
    tol_norm = p["tol"] * np.linalg.norm(x0)

    eta_gd = 2.0 / (mu + L)
    gd = _run(A, x0, eta_gd, 0.0, p["max_iter"], tol_norm)

    sm, sL = np.sqrt(mu), np.sqrt(L)
    eta_hb = 4.0 / (sL + sm) ** 2
    beta_hb = ((sL - sm) / (sL + sm)) ** 2
    mom = _run(A, x0, eta_hb, beta_hb, p["max_iter"], tol_norm)

    newton = np.array([x0, x0 - np.linalg.solve(A, A @ x0)])

    def f(x):
        return 0.5 * np.einsum("...i,ij,...j->...", x, A, x)

    f0 = float(f(x0))
    levels = [f0 * r for r in p["level_ratios"]]

    def ellipse(c, n):
        t = np.linspace(0.0, 2.0 * np.pi, n)
        u = np.column_stack([
            np.sqrt(2.0 * c / mu) * np.cos(t),
            np.sqrt(2.0 * c / L) * np.sin(t),
        ])
        return u @ R.T

    ellipses = [ellipse(c, p["n_ellipse"]) for c in levels]

    return {
        "p": p, "A": A, "R": R, "mu": mu, "L": L, "kappa": kappa,
        "x0": x0, "tol_norm": tol_norm,
        "gd": gd, "mom": mom, "newton": newton,
        "eta_gd": eta_gd, "eta_hb": eta_hb, "beta_hb": beta_hb,
        "f": f, "f0": f0, "ellipses": ellipses, "levels": levels,
    }


def _arc_positions(pts, fracs):
    seg = np.diff(pts, axis=0)
    s = np.concatenate([[0.0], np.cumsum(np.hypot(seg[:, 0], seg[:, 1]))])
    total = s[-1]
    return s / total if total > 0 else s


def build(g):
    p = g["p"]
    gd, mom, newton = g["gd"], g["mom"], g["newton"]
    all_pts = np.vstack([g["ellipses"][0], gd, mom, newton])
    pad = 0.35
    xlim = (float(all_pts[:, 0].min()) - pad, float(all_pts[:, 0].max()) + pad)
    ylim = (float(all_pts[:, 1].min()) - pad, float(all_pts[:, 1].max()) + pad)
    s = Scene(xlim=xlim, ylim=ylim, clip="frame")

    # layer 1 — the terrain: elliptical level sets of the quadratic, the
    # scaffolding every trajectory is judged against, not the claim itself
    for ell in g["ellipses"]:
        s.add(Curve(ell, closed=True, role=Role.CONSTRUCTION, width_scale=0.75))

    # slot 2 (sky) sits too close to slot 0 (indigo) at a glance — both
    # blue-family — so Newton takes slot 3 (teal) instead, keeping the
    # three hues visually separable, not just gate-separable
    hue_gd, hue_mom, hue_newton = (RISO.categorical(i) for i in (0, 1, 3))

    # layer 2 — plain gradient descent: a dense zig-zag, thin, no per-vertex
    # clutter (58 iterates would be unreadable as individual points) —
    # sparse direction markers carry the motion instead
    fracs = _arc_positions(gd, p["gd_arrow_fracs"])
    s.add(Curve(gd, role=Role.CONTENT, width_scale=WEIGHT_CONTENT,
                color=hue_gd, arrows=p["gd_arrow_fracs"], arrow_style="filled",
                arrow_scale=0.75))

    # layer 3 — heavy-ball momentum: few enough iterates that each discrete
    # step is drawn (filled dots), which is itself the "damped oscillator"
    # argument made visible — the path bows past the axis before settling
    s.add(Curve(mom, role=Role.CONTENT, width_scale=WEIGHT_CONTENT,
                color=hue_mom, arrows=(0.5,), arrow_style="filled"))
    for xy in mom[1:-1]:
        s.add(Point(tuple(xy), role=Role.CONTENT, color=hue_mom,
                    radius_scale=0.55))

    # layer 4 — THE object: Newton's single step, heaviest ink, one segment
    s.add(Curve(newton, role=Role.CONTENT, width_scale=WEIGHT_ACTOR,
                color=hue_newton, arrows=(0.6,), arrow_style="filled",
                arrow_scale=1.2))

    # start and optimum
    s.add(Point(tuple(g["x0"]), role=Role.ANNOTATION, filled=True, radius_scale=1.1))
    s.add(MathLabel(r"x_0", tuple(g["x0"]), ha="left", va="bottom",
                    offset_px=(6, -6), size_pt=11, role=Role.ANNOTATION, halo=True))
    s.add(Point((0.0, 0.0), role=Role.ANNOTATION, filled=False, radius_scale=1.1))
    s.add(MathLabel(r"x^\star", (0.0, 0.0), ha="left", va="top",
                    offset_px=(7, 20), size_pt=11, role=Role.ANNOTATION, halo=True))

    # method labels: parked in the empty lower-left wedge (outside the
    # outermost level curve, clear of every trajectory) rather than pinned
    # to a curve point — with 58 GD iterates crammed into the valley band,
    # no in-band position stays collision-free, so this is a floating
    # caption block, same device as the ODE caption in demo_basin_wash
    n_gd, n_mom = len(gd) - 1, len(mom) - 1
    legend_x = xlim[0] + 0.5
    s.add(MathLabel(
        rf"\text{{gradient descent: }} {n_gd}\ \text{{iters}}",
        (legend_x, -1.2), ha="left", va="center", size_pt=11,
        color=hue_gd, halo=True))
    s.add(MathLabel(
        rf"\text{{momentum: }} {n_mom}\ \text{{iters}}",
        (legend_x, -1.9), ha="left", va="center", size_pt=11,
        color=hue_mom, halo=True))
    s.add(MathLabel(
        r"\text{Newton: } 1\ \text{iter}",
        (legend_x, -2.6), ha="left", va="center",
        size_pt=11, color=hue_newton, halo=True))

    s.add(MathLabel(
        rf"\kappa = L/\mu = {g['kappa']:.0f}", (xlim[0] + 0.08, ylim[1] - 0.05),
        ha="left", va="top", size_pt=12, role=Role.ANNOTATION, halo=True))
    s.add(MathLabel(
        r"N_{\mathrm{GD}} \sim \kappa,\quad N_{\mathrm{mom}} \sim \sqrt{\kappa},"
        r"\quad N_{\mathrm{Newton}} = 1",
        (xlim[0] + 0.08, ylim[0] + 0.05), ha="left", va="bottom", size_pt=10.5,
        role=Role.ANNOTATION, halo=True))

    return s


def assertions(g):
    c = Checks()
    A, mu, L, kappa = g["A"], g["mu"], g["L"], g["kappa"]

    # (1) A really has the claimed eigenvalues, independent of the drawn
    # trajectories — a construction bug in the rotation would show up here
    eigs = np.sort(np.linalg.eigvalsh(A))
    c.check(np.allclose(eigs, [mu, L], atol=1e-8),
            f"Hessian eigenvalues {eigs} != claimed ({mu}, {L})")
    c.check(abs(L / mu - kappa) < 1e-8, "L/mu does not match claimed kappa")

    # (2) Newton's single step is exact: A^{-1} A x0 = x0, so the drawn
    # second point should be at the optimum to machine precision, not "by
    # construction" in the code (the actual solve is what's checked)
    newton_end = g["newton"][1]
    c.check(np.linalg.norm(newton_end) < 1e-9,
            f"Newton step does not land on the optimum: |x_1| = "
            f"{np.linalg.norm(newton_end):.2e}")

    # (3) both GD and momentum actually reach the claimed tolerance at the
    # iteration counts annotated on the figure (self-consistency: the
    # numbers drawn are the numbers the simulation produced)
    gd, mom, tol_norm = g["gd"], g["mom"], g["tol_norm"]
    c.check(np.linalg.norm(gd[-1]) < tol_norm,
            "GD trajectory drawn without reaching its tolerance")
    c.check(np.linalg.norm(mom[-1]) < tol_norm,
            "momentum trajectory drawn without reaching its tolerance")
    c.check(len(gd) - 1 < g["p"]["max_iter"],
            "GD hit the iteration cap without converging")
    c.check(len(mom) - 1 < g["p"]["max_iter"],
            "momentum hit the iteration cap without converging")

    # (4) GD's objective decreases monotonically every step — a real
    # property of eta = 2/(mu+L) (each eigen-component's magnitude strictly
    # contracts), checked on the actual simulated array, not assumed
    f_gd = g["f"](gd)
    c.check(bool(np.all(np.diff(f_gd) < 1e-12)),
            "GD's objective is not monotonically decreasing")

    # (5) momentum is NOT asserted monotonic — at the optimal (eta, beta)
    # the extremal directions get complex-conjugate iteration roots and can
    # legitimately overshoot. State what's actually true instead: momentum
    # still converges, and does so in materially fewer iterations than GD.
    f_mom = g["f"](mom)
    momentum_overshoots = bool(np.any(np.diff(f_mom) > 1e-12))
    n_gd, n_mom = len(gd) - 1, len(mom) - 1
    c.check(n_mom < n_gd,
            f"momentum ({n_mom} iters) did not beat GD ({n_gd} iters)")

    # (6) the empirical speed-up tracks the sqrt(kappa) theory: momentum's
    # iteration count should sit within a factor of 2 of GD's / sqrt(kappa)
    # (a cross-check against the closed-form rates, not a restatement —
    # the simulation could disagree with the theory if eta/beta were wrong)
    predicted_ratio = np.sqrt(kappa)
    empirical_ratio = n_gd / n_mom
    c.check(0.5 * predicted_ratio < empirical_ratio < 2.0 * predicted_ratio,
            f"N_GD/N_mom = {empirical_ratio:.2f} far from sqrt(kappa) = "
            f"{predicted_ratio:.2f}")

    # (7) every drawn ellipse is a true level set of f: f evaluated on its
    # points equals its assigned level to numerical precision
    for ell, level in zip(g["ellipses"], g["levels"]):
        vals = g["f"](ell)
        c.check(float(np.max(np.abs(vals - level))) < 1e-6 * max(abs(level), 1.0),
                f"drawn ellipse deviates from its level f = {level:.4f}")

    c.done()
    # momentum_overshoots is computed above but deliberately not asserted
    # either way: at the optimal (eta, beta) it MAY overshoot (complex
    # iteration roots) and that is a true, expected property of this
    # trajectory, not a defect — asserting it False would be false, and
    # asserting it True would overfit to this one kappa/x0 choice.
    del momentum_overshoots
