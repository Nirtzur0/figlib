"""Mobius transformations classified by the trace of their matrix.

Four panels, one construction run four times: conjugate M to its normal
form w -> m w by sending its fixed points to 0 and infinity, read the
multiplier m off the matrix trace, draw the invariant orbit family in
normal coordinates, pull it back. Only the fixed-point structure and the
value of m change between panels; the pipeline (matrix -> tau^2 -> m ->
orbit -> pullback) is identical.
"""

import numpy as np

from figlib.figure import Figure, Panel
from figlib.format import WIDE
from figlib.gates import Checks
from figlib.scene import Callout, Curve, MathLabel, Point, Scene
from figlib.style import WEIGHT_ACTOR, WEIGHT_BG, Role
from figlib.theme import RISO

CLAIM = (
    "The normalized trace of a Mobius transformation's matrix, "
    "tau^2 = tr(M)^2 / det(M), predicts its invariant orbit geometry "
    "outright: tau^2 in [0,4) draws closed circles around two centers "
    "(elliptic), tau^2 = 4 draws a family tangent at one point "
    "(parabolic), real tau^2 > 4 draws arcs flowing source to sink "
    "(hyperbolic), and non-real tau^2 draws spirals winding around both "
    "poles (loxodromic) -- no case-by-case argument, one number."
)

EXPOSITION = """
Needham's classification of Mobius transformations (Visual Complex
Analysis, ch. 3, "Classification of Mobius Transformations") turns on a
single invariant. Conjugate M to its normal form by sending its fixed
points to 0 and infinity -- F M F^{-1} becomes the elementary map
w -> m w for a complex multiplier m -- and Needham shows the whole
qualitative behaviour of M is read off m alone: |m| = 1 rotates,
m real =/= 1 flows radially, m off the unit circle and off the positive
reals spirals, m = 1 is the one degenerate case with a single fixed
point. What ties this back to M itself, rather than to the coordinates
F happened to choose, is that m is pinned down by tau^2 = tr(M)^2 /
det(M) = m + 1/m + 2 -- the trace-squared-over-determinant is exactly
the quantity invariant under conjugation and rescaling the matrix, so it
depends on M and nothing else. The reader is left, after that one
equation, to reconstruct four qualitatively different pictures from
memory. This figure is the picture the algebra earns but the book leaves
to the reader's imagination: the same four-step construction --
factor the matrix, solve for m from tau^2, draw the orbit family in
normal coordinates, pull it back through F^{-1} -- run once per panel,
so that "trace alone decides the geometry" is something the page shows
rather than the algebra merely implies.
"""

THEME = RISO
FORMAT = WIDE

PARAMS = {
    "lim": 2.2,
    # two-fixed-point classes: fixed points are +-1 BY CONSTRUCTION of the
    # matrix (a,b,c,d) = (1+m, 1-m, 1-m, 1+m); only m differs per class.
    "elliptic": dict(
        m=complex(np.exp(1j * np.deg2rad(70.0))),
        s0s=(-0.95, -0.50, 0.50, 0.95), n_t=420,
    ),
    "hyperbolic": dict(
        m=2.4 + 0.0j,
        phi0=tuple(np.pi * f for f in
                    (0.15, 0.4, 0.6, 0.85, 1.15, 1.4, 1.6, 1.85)),
        T=6.4, n_t=480,
    ),
    "loxodromic": dict(
        m=1.7 * complex(np.exp(1j * np.deg2rad(50.0))),
        phi0=tuple(np.pi * f for f in
                    (0.15, 0.4, 0.6, 0.85, 1.15, 1.4, 1.6, 1.85)),
        T=4.3, n_t=560,
    ),
    # single (double) fixed point at p=0, tangent direction along t.
    "parabolic": dict(
        p=0.0 + 0.0j, t=1.0 + 0.0j,
        offsets=(-1.95, -1.05, -0.55, 0.55, 1.05, 1.95),
        u_max=6.5, n_t=420,
    ),
    "probe_ts": (-2.3, -0.7, 0.4, 1.6),   # invariance-check probe times
    "n_iter_attract": 40,                 # iterations to certify attraction
}


# --- the shared algebra ------------------------------------------------

def mobius(z, a, b, c, d):
    return (a * z + b) / (c * z + d)


def _fixed_points(a, b, c, d):
    """Solve c z^2 + (d-a) z - b = 0."""
    A, B, C = c, d - a, -b
    disc = B ** 2 - 4 * A * C
    r = np.sqrt(disc + 0j)
    return (-B + r) / (2 * A), (-B - r) / (2 * A)


def _trace2(a, b, c, d):
    return (a + d) ** 2 / (a * d - b * c)


def _multiplier_from_trace2(tau2, branch_hint):
    """Invert tau^2 = m + 1/m + 2 for m, picking the root on the same side
    of the unit circle / real axis as branch_hint (the two roots are
    m, 1/m -- either labels the same conjugacy class, but only one
    matches the multiplier the figure actually draws)."""
    disc = (tau2 - 2) ** 2 - 4
    root = np.sqrt(disc + 0j)
    m1 = ((tau2 - 2) + root) / 2
    m2 = ((tau2 - 2) - root) / 2
    d1 = abs(m1 - branch_hint)
    d2 = abs(m2 - branch_hint)
    return m1 if d1 <= d2 else m2


def _classify(tau2, tol=1e-6):
    if abs(tau2.imag) < tol:
        t = tau2.real
        if abs(t - 4.0) < tol:
            return "parabolic"
        if 0.0 <= t < 4.0:
            return "elliptic"
        if t > 4.0:
            return "hyperbolic"
    return "loxodromic"


def _F(z):
    """Conjugation sending the two-fixed-point normal form's poles
    xi_plus=1 -> 0, xi_minus=-1 -> infinity."""
    return (z - 1) / (z + 1)


def _zmap(w):
    """Inverse of _F: z = (1+w)/(1-w)."""
    return (1 + w) / (1 - w)


def _two_fixed_family(m, phi0s, s0s, t_range, n_t):
    """Orbit curves of w -> m*w (fixed pts +1 -> 0, -1 -> infinity),
    parametrized so t=1 is EXACTLY one application of M: with
    m = r*e^{i theta}, w(t) = w0 * m^t, so z(t+1) = M(z(t)) identically."""
    ln_r, theta = np.log(abs(m)), np.angle(m)
    t = np.linspace(t_range[0], t_range[1], n_t)
    curves = []
    for phi0 in phi0s:
        for s0 in s0s:
            s = s0 + t * ln_r
            phi = phi0 + t * theta
            w = np.exp(s + 1j * phi)
            z = _zmap(w)
            curves.append((t, np.column_stack([z.real, z.imag])))
    return curves


def _parabolic_family(p, t, offsets, u_max, n_t):
    """Orbit curves of w -> w+t pulled back through z = p + 1/w: circles
    tangent to direction t at p, plus the single degenerate line (o=0),
    all parametrized so u -> u + |t| is exactly one application of M."""
    that = t / abs(t)
    n_hat = 1j * that
    u = np.linspace(-u_max, u_max, n_t)
    curves = []
    for o in offsets:
        w = o * n_hat + u * that
        z = p + 1 / w
        curves.append((u, np.column_stack([z.real, z.imag])))
    zline = p + u * that
    curves.append((u, np.column_stack([zline.real, zline.imag])))
    return curves


# --- compute -------------------------------------------------------------

def compute(p):
    lim = p["lim"]
    # top strip left blank on purpose: the drawn geometry never reaches it,
    # so the title/tau^2 text has an ink-free band instead of fighting the
    # orbit family for space (architecture.md: axis titles sit outside the
    # geometry extents by design).
    xlim = (-lim, lim)
    ylim = (-lim, lim + 0.62)
    out = {"xlim": xlim, "ylim": ylim, "lim": lim}

    # elliptic: family varies s0 (nested Apollonian loops), phi0 fixed
    ep = p["elliptic"]
    a, b, c, d = 1 + ep["m"], 1 - ep["m"], 1 - ep["m"], 1 + ep["m"]
    tau2 = _trace2(a, b, c, d)
    m_rec = _multiplier_from_trace2(tau2, ep["m"])
    theta = np.angle(m_rec)
    curves = _two_fixed_family(m_rec, phi0s=(0.0,), s0s=ep["s0s"],
                                t_range=(0.0, 2 * np.pi / abs(theta)),
                                n_t=ep["n_t"])
    out["elliptic"] = dict(a=a, b=b, c=c, d=d, tau2=tau2, m=m_rec,
                            xi=_fixed_points(a, b, c, d), curves=curves,
                            kind="two_fixed", label=r"\xi_{+}", label2=r"\xi_{-}")

    # hyperbolic: family varies phi0 (circles through both poles), s0 fixed
    hp = p["hyperbolic"]
    a, b, c, d = 1 + hp["m"], 1 - hp["m"], 1 - hp["m"], 1 + hp["m"]
    tau2 = _trace2(a, b, c, d)
    m_rec = _multiplier_from_trace2(tau2, hp["m"])
    curves = _two_fixed_family(m_rec, phi0s=hp["phi0"], s0s=(0.0,),
                                t_range=(-hp["T"], hp["T"]), n_t=hp["n_t"])
    out["hyperbolic"] = dict(a=a, b=b, c=c, d=d, tau2=tau2, m=m_rec,
                              xi=_fixed_points(a, b, c, d), curves=curves,
                              kind="two_fixed", label=r"\xi_{+}", label2=r"\xi_{-}")

    # loxodromic: same shape of family as hyperbolic, m off both axes
    lp = p["loxodromic"]
    a, b, c, d = 1 + lp["m"], 1 - lp["m"], 1 - lp["m"], 1 + lp["m"]
    tau2 = _trace2(a, b, c, d)
    m_rec = _multiplier_from_trace2(tau2, lp["m"])
    curves = _two_fixed_family(m_rec, phi0s=lp["phi0"], s0s=(0.0,),
                                t_range=(-lp["T"], lp["T"]), n_t=lp["n_t"])
    out["loxodromic"] = dict(a=a, b=b, c=c, d=d, tau2=tau2, m=m_rec,
                              xi=_fixed_points(a, b, c, d), curves=curves,
                              kind="two_fixed", label=r"\xi_{+}", label2=r"\xi_{-}")

    # parabolic: single double fixed point, translation normal form
    pp = p["parabolic"]
    pt, tt = pp["p"], pp["t"]
    a = 1 + pt * tt
    b = -pt ** 2 * tt
    c = tt
    d = 1 - pt * tt
    tau2 = _trace2(a, b, c, d)
    xi1, xi2 = _fixed_points(a, b, c, d)
    curves = _parabolic_family(pt, tt, pp["offsets"], pp["u_max"], pp["n_t"])
    out["parabolic"] = dict(a=a, b=b, c=c, d=d, tau2=tau2, t=tt, p=pt,
                             xi=(xi1, xi2), curves=curves,
                             kind="parabolic", label=r"\xi")

    return out


# --- build -----------------------------------------------------------

_TITLES = {
    "elliptic": r"\mathbf{Elliptic}",
    "parabolic": r"\mathbf{Parabolic}",
    "hyperbolic": r"\mathbf{Hyperbolic}",
    "loxodromic": r"\mathbf{Loxodromic}",
}


def _tau2_label(tau2, tol=1e-6):
    if abs(tau2.imag) < tol:
        return rf"\tau^2 = {tau2.real:.2f}"
    sign = "+" if tau2.imag >= 0 else "-"
    return rf"\tau^2 = {tau2.real:.2f} {sign} {abs(tau2.imag):.2f}i"


def _panel(name, g, xlim, ylim):
    s = Scene(xlim=xlim, ylim=ylim, clip="frame")

    # one member of the family is THE orbit (WEIGHT_ACTOR, no arrowhead
    # crowding); the rest are the population it belongs to (WEIGHT_BG).
    n = len(g["curves"])
    actor = n // 2
    for k, (_, pts) in enumerate(g["curves"]):
        if k == actor:
            s.add(Curve(pts, role=Role.CONTENT, width_scale=WEIGHT_ACTOR,
                        arrows=(0.55,), arrow_scale=1.2))
        else:
            s.add(Curve(pts, role=Role.CONTENT, width_scale=WEIGHT_BG,
                        arrows=(0.55,), arrow_scale=0.85, opacity=0.85))

    # fixed points: a Callout (boxed label + leader) rather than a bare
    # MathLabel, because a boxed callout is exempt from the label-on-ink
    # check regardless of theme transparency -- a haloed label is not
    # (the halo casing only paints, and so only earns the exemption, on an
    # opaque ground; this corpus's figures are groundless by default).
    if g["kind"] == "two_fixed":
        xi_p, xi_m = g["xi"]
        s.add(Point((xi_p.real, xi_p.imag), role=Role.CONTENT, radius_scale=1.1))
        s.add(Point((xi_m.real, xi_m.imag), role=Role.CONTENT, radius_scale=1.1))
        s.add(Callout(g["label"], (xi_p.real + 0.55, xi_p.imag + 0.62),
                     (xi_p.real, xi_p.imag), size_pt=11))
        s.add(Callout(g["label2"], (xi_m.real - 0.55, xi_m.imag - 0.62),
                     (xi_m.real, xi_m.imag), size_pt=11))
    else:
        xi1, _ = g["xi"]
        s.add(Point((xi1.real, xi1.imag), role=Role.CONTENT, radius_scale=1.3))
        s.add(Callout(g["label"], (xi1.real - 0.75, xi1.imag + 0.85),
                     (xi1.real, xi1.imag), size_pt=11))

    # title / tau^2: anchored in the blank top strip, no curve ink to
    # collide with regardless of how the family fills the rest of the panel
    s.add(MathLabel(_TITLES[name], (xlim[0], ylim[1]), ha="left", va="top",
                    offset_px=(6, 8), size_pt=12))
    s.add(MathLabel(_tau2_label(g["tau2"]), (xlim[0], ylim[1]), ha="left",
                    va="top", offset_px=(6, 30), size_pt=10,
                    role=Role.ANNOTATION))
    return s


def build(g):
    xlim, ylim = g["xlim"], g["ylim"]
    order = ["elliptic", "parabolic", "hyperbolic", "loxodromic"]
    panels = [Panel(_panel(name, g[name], xlim, ylim), tag=None) for name in order]
    return Figure(panels=panels, grid=(2, 2), gap_px=40.0)


# --- assertions --------------------------------------------------------

def _check_two_fixed(c, name, g, probe_ts):
    a, b, c_, d = g["a"], g["b"], g["c"], g["d"]
    tau2, m = g["tau2"], g["m"]
    expected = {"elliptic": "elliptic", "hyperbolic": "hyperbolic",
                "loxodromic": "loxodromic"}[name]
    c.check(_classify(tau2) == expected,
            f"{name}: tau^2={tau2!r} classified as {_classify(tau2)!r}, "
            f"not {expected!r}")

    xi_p, xi_m = g["xi"]
    c.check(min(abs(xi_p - 1.0), abs(xi_p - (-1.0))) < 1e-8 and
            min(abs(xi_m - 1.0), abs(xi_m - (-1.0))) < 1e-8 and
            abs(xi_p - xi_m) > 1e-6,
            f"{name}: fixed points {xi_p!r}, {xi_m!r} are not {{+1,-1}}")

    ln_r, theta = np.log(abs(m)), np.angle(m)

    # invariance: for each drawn family member, M(z(t)) == z(t+1) for
    # probe times strictly inside the drawn range -- this uses the ACTUAL
    # matrix (a,b,c,d), not the multiplier, so it is a genuine check that
    # the drawn curve is invariant under M rather than a restatement of
    # how it was built.
    for t_arr, pts in g["curves"]:
        tmin, tmax = t_arr[0], t_arr[-1]
        # recover (phi0, s0) for this member from its first sample
        s_first = np.log(abs(_F(pts[0, 0] + 1j * pts[0, 1])))
        phi_first = np.angle(_F(pts[0, 0] + 1j * pts[0, 1]))
        for t0 in probe_ts:
            if not (tmin < t0 < tmax - 1.0):
                continue
            s0 = s_first - t_arr[0] * ln_r
            phi0 = phi_first - t_arr[0] * theta
            z0 = _zmap(np.exp(s0 + t0 * ln_r + 1j * (phi0 + t0 * theta)))
            z1 = _zmap(np.exp(s0 + (t0 + 1) * ln_r + 1j * (phi0 + (t0 + 1) * theta)))
            lhs = mobius(z0, a, b, c_, d)
            c.check(abs(lhs - z1) < 1e-7,
                    f"{name}: M(z(t)) != z(t+1) at t={t0} "
                    f"(|diff|={abs(lhs - z1):.2e})")


def assertions(g):
    c = Checks()
    probe_ts = list(PARAMS["probe_ts"])

    for name in ("elliptic", "hyperbolic", "loxodromic"):
        _check_two_fixed(c, name, g[name], probe_ts)

    # elliptic: neutral rotation -- |m|=1 must preserve |F(z)| along every
    # orbit exactly (the Apollonian level it sits on), and iterating M
    # must stay bounded away from BOTH fixed points (no attraction)
    eg = g["elliptic"]
    rng = np.random.default_rng(3)
    for _ in range(6):
        z0 = complex(rng.uniform(-1.8, 1.8), rng.uniform(-1.8, 1.8))
        if abs(z0 - 1) < 0.05 or abs(z0 + 1) < 0.05:
            continue
        r0 = abs(_F(z0))
        z = z0
        min_d = min(abs(z0 - 1), abs(z0 + 1))
        for _ in range(PARAMS["n_iter_attract"]):
            z = mobius(z, eg["a"], eg["b"], eg["c"], eg["d"])
            c.check(abs(abs(_F(z)) - r0) < 1e-6,
                    "elliptic: |F(M^k(z))| drifted -- not a level-preserving rotation")
            min_d = min(min_d, abs(z - 1), abs(z + 1))
        c.check(min_d > 0.02, "elliptic: an orbit crept up on a fixed point")

    # hyperbolic + loxodromic: forward iteration must actually converge on
    # the fixed point predicted by |m| (attracting = the one at F-image 0
    # if |m|<1, else infinity i.e. xi_minus), and diverge from the other
    for name in ("hyperbolic", "loxodromic"):
        gg = g[name]
        attracting = 1.0 if abs(gg["m"]) < 1 else -1.0
        repelling = -attracting
        rng = np.random.default_rng(11)
        for _ in range(4):
            z = complex(rng.uniform(-1.8, 1.8), rng.uniform(-1.8, 1.8))
            if min(abs(z - 1), abs(z + 1)) < 0.05:
                continue
            for _ in range(PARAMS["n_iter_attract"]):
                z = mobius(z, gg["a"], gg["b"], gg["c"], gg["d"])
            c.check(abs(z - attracting) < 1e-4,
                    f"{name}: forward orbit did not converge to the "
                    f"attracting fixed point {attracting} (landed {z!r})")
            c.check(abs(z - repelling) > 0.5,
                    f"{name}: forward orbit did not move away from the "
                    f"repelling fixed point {repelling}")

    # parabolic: tau^2 == 4 exactly (up to fp), single double fixed point,
    # and M(z(u)) == z(u + |t|) on the drawn family -- the horocycle
    # invariance, checked against the actual matrix
    pg = g["parabolic"]
    c.check(_classify(pg["tau2"]) == "parabolic",
            f"parabolic: tau^2={pg['tau2']!r} not classified parabolic")
    xi1, xi2 = pg["xi"]
    c.check(abs(xi1 - xi2) < 1e-8 and abs(xi1 - pg["p"]) < 1e-8,
            f"parabolic: fixed points {xi1!r},{xi2!r} are not a double root at {pg['p']!r}")
    step = abs(pg["t"])
    for u_arr, pts in pg["curves"]:
        umin, umax = u_arr[0], u_arr[-1]
        for u0 in probe_ts:
            if not (umin < u0 < umax - step):
                continue
            that = pg["t"] / abs(pg["t"])
            n_hat = 1j * that
            # recover this member's offset from its construction order is
            # unnecessary: reconstruct w(u) generically via z - p = 1/w,
            # using the FIRST sample to fix the offset consistently.
            w_first = 1 / (pts[0, 0] + 1j * pts[0, 1] - pg["p"]) if abs(pts[0, 0] + 1j*pts[0,1] - pg["p"]) > 1e-12 else None
            if w_first is None:
                continue
            o = ((w_first - u_arr[0] * that) / n_hat).real
            w0 = o * n_hat + u0 * that
            w1 = o * n_hat + (u0 + step) * that
            z0 = pg["p"] + 1 / w0
            z1 = pg["p"] + 1 / w1
            lhs = mobius(z0, pg["a"], pg["b"], pg["c"], pg["d"])
            c.check(abs(lhs - z1) < 1e-6,
                    f"parabolic: M(z(u)) != z(u+|t|) at u={u0} "
                    f"(|diff|={abs(lhs - z1):.2e})")

    c.done()
