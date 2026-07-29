"""VCA Figure [12]: the full conformal grid of Omega = z + 1/z (p. 528).

Uniform flow past the unit disc, no circulation. Psi = Im Omega gives the
streamlines (solid, hollow markers pointing downstream); Phi = Re Omega
gives the equipotentials (dashed, the book's convention). Both families
are drawn at the SAME level spacing k, so conformality of Omega away
from its critical points makes the grid a mesh of near-squares — cell
side k/|Omega'| in both directions. The grid degenerates exactly at the
stagnation points z = +-1, where Omega'(z) = 1 - 1/z^2 = 0 and the map
ceases to be conformal.
"""

import numpy as np

from figlib.builders import stagnation_points, stream_function_lines
from figlib.format import WIDE
from figlib.gates import Checks
from figlib.scene import Curve, FilledCurve, MathLabel, Point, RightAngleMark, Scene
from figlib.style import Role
from figlib.theme import RISO

CLAIM = (
    "The level curves of Psi = Im Omega and Phi = Re Omega for "
    "Omega = z + 1/z form an orthogonal grid of near-squares that flows "
    "around the disc and breaks down only at the stagnation points +-1, "
    "where the map ceases to be conformal."
)

FORMAT = WIDE
THEME = RISO

PARAMS = {
    "xlim": (-4.2, 4.2),
    "ylim": (-2.1, 2.1),
    "dk": 0.3,                  # ONE spacing for both ladders: DPsi = DPhi
    "psi_kmax": 6,              # Psi levels k*dk, k in [-6, 6]  (13 streamlines)
    "phi_kmax": 13,             # Phi levels k*dk, k in [-13, 13] (27 arcs)
    "n": 720,                   # contour grid resolution per axis
    "label_psi_level": 0.9,     # streamline that carries the Psi label
    "label_psi_x": 2.85,
    "label_phi_level": -2.7,    # equipotential that carries the Phi label
    "label_phi_y": 1.49,
    "square_levels": (1.2, 1.8),  # (Psi, Phi) crossing for the right-angle mark
}


# --- the analytic layer: Omega = z + 1/z, coded as two independent scalars ---


def _psi(x, y):
    r2 = x * x + y * y
    return y * (1.0 - 1.0 / r2)


def _phi(x, y):
    r2 = x * x + y * y
    return x * (1.0 + 1.0 / r2)


def _grad_psi(x, y):
    r2 = x * x + y * y
    r4 = r2 * r2
    return 2.0 * x * y / r4, 1.0 - 1.0 / r2 + 2.0 * y * y / r4


def _grad_phi(x, y):
    r2 = x * x + y * y
    r4 = r2 * r2
    return 1.0 + 1.0 / r2 - 2.0 * x * x / r4, -2.0 * x * y / r4


def _vel(P):
    """conj(Omega'(z)) as a plane field: the velocity of the flow."""
    z = P[:, 0] + 1j * P[:, 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        W = 1.0 - 1.0 / (z * z)
    return np.column_stack([W.real, -W.imag])


def _bisect(f, lo, hi, tol=1e-12):
    flo = f(lo)
    for _ in range(90):
        mid = 0.5 * (lo + hi)
        if f(mid) * flo <= 0.0:
            hi = mid
        else:
            lo, flo = mid, f(mid)
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def _crossing(psi_level, phi_level, x0, y0):
    """Newton solve Psi = psi_level, Phi = phi_level from (x0, y0)."""
    x, y = x0, y0
    for _ in range(60):
        F = np.array([_psi(x, y) - psi_level, _phi(x, y) - phi_level])
        if np.hypot(*F) < 1e-13:
            break
        px, py = _grad_psi(x, y)
        qx, qy = _grad_phi(x, y)
        dx, dy = np.linalg.solve(np.array([[px, py], [qx, qy]]), -F)
        x, y = x + dx, y + dy
    return x, y


def _unit(v):
    return v / np.hypot(v[0], v[1])


def compute(p):
    k = p["dk"]
    psi_levels = [i * k for i in range(-p["psi_kmax"], p["psi_kmax"] + 1)]
    phi_levels = [i * k for i in range(-p["phi_kmax"], p["phi_kmax"] + 1)]
    inside = lambda X, Y: X * X + Y * Y < 1.0

    streams = stream_function_lines(
        _psi, p["xlim"], p["ylim"], psi_levels, n=p["n"], mask=inside,
        arrows=(0.30, 0.72), role=Role.CONTENT, width_scale=0.8,
        arrow_scale=0.75)
    equis = stream_function_lines(
        _phi, p["xlim"], p["ylim"], phi_levels, n=p["n"], mask=inside,
        arrows=(), role=Role.CONTENT, width_scale=0.5, dash="4 3.5")

    stag = sorted(stagnation_points(_vel, p["xlim"], p["ylim"], grid_n=96))

    # label anchors, solved onto their curves
    y_psi = _bisect(lambda y: _psi(p["label_psi_x"], y) - p["label_psi_level"],
                    0.3, 2.0)
    x_phi = _bisect(lambda x: _phi(x, p["label_phi_y"]) - p["label_phi_level"],
                    -4.0, -1.5)

    # a generic crossing far from the disc, for the right-angle mark
    sq_psi, sq_phi = p["square_levels"]
    cx, cy = _crossing(sq_psi, sq_phi, 1.7, 1.3)
    px, py = _grad_psi(cx, cy)
    qx, qy = _grad_phi(cx, cy)
    d1 = _unit(np.array([py, -px]))          # streamline tangent (downstream)
    d2 = _unit(np.array([-qy, qx]))          # equipotential tangent
    if d1[0] < 0:
        d1 = -d1
    if d2[1] < 0:
        d2 = -d2

    return {"streams": streams, "equis": equis, "stag": stag,
            "psi_levels": psi_levels, "phi_levels": phi_levels,
            "label_psi_pt": (p["label_psi_x"], y_psi),
            "label_phi_pt": (x_phi, p["label_phi_y"]),
            "square_pt": (cx, cy), "square_dirs": (d1, d2), "p": p}


def build(g):
    p = g["p"]
    s = Scene(xlim=p["xlim"], ylim=p["ylim"])

    # dashed family under, solid family over: the book's subordination
    s.add(*g["equis"])
    s.add(*g["streams"])

    # the obstacle: near-black disc over the contour ends at its boundary
    th = np.linspace(0.0, 2.0 * np.pi, 240)
    circle = np.column_stack([np.cos(th), np.sin(th)])
    s.add(FilledCurve(circle, role=Role.CONTENT, opacity=1.0, outline=False))
    s.add(Curve(circle, closed=True, role=Role.CONTENT, width_scale=1.0))

    # stagnation points: where Omega' = 0 and the grid degenerates
    for x, y in g["stag"]:
        s.add(Point((x, y), filled=True, role=Role.ACCENT2, radius_scale=1.0))
    s.add(MathLabel(r"\Omega'(\pm 1) = 0", (2.55, -0.17), ha="center", va="center",
                    size_pt=10, role=Role.ANNOTATION, halo=True))

    # the map, named at the inflow (the widest cell, under the axis)
    s.add(MathLabel(r"\Omega = z + 1/z", (-2.62, -0.20), ha="center",
                    va="center", size_pt=12, role=Role.ANNOTATION, halo=True))

    # one label per family, on its curve, offset perpendicular
    s.add(MathLabel(r"\Psi = \mathrm{Im}\,\Omega", g["label_psi_pt"],
                    ha="center", va="bottom", offset_px=(0, -9),
                    size_pt=10, role=Role.ANNOTATION, halo=True))
    s.add(MathLabel(r"\Phi = \mathrm{Re}\,\Omega", g["label_phi_pt"],
                    ha="right", va="center", offset_px=(-8, 0),
                    size_pt=10, role=Role.ANNOTATION, halo=True))

    # the theorem at one generic crossing: the families meet at right angles
    (d1, d2) = g["square_dirs"]
    s.add(RightAngleMark(g["square_pt"], tuple(d1), tuple(d2), size=0.14))

    # the equal-spacing admission: same ladder step for both families
    s.add(MathLabel(r"\Delta\Psi = \Delta\Phi", (-4.05, -1.76), ha="left",
                    va="center", size_pt=9.5, role=Role.ANNOTATION, halo=True))
    return s


# --- gates -------------------------------------------------------------------


def _sampled_tangents(curves, stride=5, stencil=3):
    """(midpoints, unit chord tangents) over every polyline, wide-stencil."""
    mids, tans = [], []
    for cv in curves:
        pts = cv.pts
        if len(pts) < 2 * stencil + 3:
            continue
        idx = np.arange(stencil, len(pts) - stencil, stride)
        a, b = pts[idx - stencil], pts[idx + stencil]
        t = b - a
        L = np.hypot(t[:, 0], t[:, 1])
        keep = L > 1e-9
        mids.append(0.5 * (a + b)[keep])
        tans.append(t[keep] / L[keep, None])
    return np.vstack(mids), np.vstack(tans)


def _far_from_degeneracy(P, r_min=1.3, stag_excl=0.35):
    r = np.hypot(P[:, 0], P[:, 1])
    dp = np.hypot(P[:, 0] - 1.0, P[:, 1])
    dm = np.hypot(P[:, 0] + 1.0, P[:, 1])
    return (r > r_min) & (dp > stag_excl) & (dm > stag_excl)


def _angle_to_field_deg(mids, tans, field_dir):
    """Unsigned line-angle (deg) between drawn tangents and an analytic
    direction field, evaluated at the chord midpoints."""
    vx, vy = field_dir(mids[:, 0], mids[:, 1])
    L = np.hypot(vx, vy)
    cross = np.abs(tans[:, 0] * vy / L - tans[:, 1] * vx / L)
    return np.degrees(np.arcsin(np.clip(cross, 0.0, 1.0)))


def assertions(g):
    p = g["p"]
    c = Checks()
    k = p["dk"]

    # (1) every drawn point lies on a level of its own family
    for curves, fn, levels, name in (
            (g["streams"], _psi, g["psi_levels"], "Psi"),
            (g["equis"], _phi, g["phi_levels"], "Phi")):
        ladder = np.asarray(levels, dtype=float)
        span = ladder[-1] - ladder[0]
        worst = 0.0
        for cv in curves:
            vals = fn(cv.pts[:, 0], cv.pts[:, 1])
            err = np.min(np.abs(vals[:, None] - ladder[None, :]), axis=1)
            worst = max(worst, float(err.max()))
        c.check(worst < 2e-3 * span,
                f"{name} point off its level by {worst:.2e} (tol {2e-3 * span:.2e})")

    # every drawn point respects the obstacle
    r_min = min(float(np.hypot(cv.pts[:, 0], cv.pts[:, 1]).min())
                for cv in g["streams"] + g["equis"])
    c.check(r_min > 1.0 - 1e-4, f"drawn point inside the disc: r = {r_min:.6f}")

    # (2a) orthogonality, analytic: grad Psi . grad Phi = 0 on a sample grid
    xs = np.linspace(p["xlim"][0], p["xlim"][1], 161)
    ys = np.linspace(p["ylim"][0], p["ylim"][1], 81)
    X, Y = np.meshgrid(xs, ys)
    out = X * X + Y * Y > 1.05
    px, py = _grad_psi(X[out], Y[out])
    qx, qy = _grad_phi(X[out], Y[out])
    gp, gq = np.hypot(px, py), np.hypot(qx, qy)
    dot = np.abs(px * qx + py * qy) / (gp * gq)
    c.check(float(dot.max()) < 1e-12,
            f"grad Psi . grad Phi = {dot.max():.2e} normalized (tol 1e-12)")
    # the analytic gradients agree with finite differences of the plotted scalars
    h = 1e-6
    fd_px = (_psi(X[out] + h, Y[out]) - _psi(X[out] - h, Y[out])) / (2 * h)
    fd_qy = (_phi(X[out], Y[out] + h) - _phi(X[out], Y[out] - h)) / (2 * h)
    c.check(float(np.max(np.abs(fd_px - px))) < 1e-6
            and float(np.max(np.abs(fd_qy - qy))) < 1e-6,
            "closed-form gradients disagree with finite differences")
    # the near-square mechanism: |grad Psi| = |grad Phi| (= |Omega'|)
    ratio = np.abs(gp / gq - 1.0)
    c.check(float(ratio.max()) < 1e-12,
            f"|grad Psi| != |grad Phi|: rel diff {ratio.max():.2e}")

    # (2b) drawn tangents align with the analytic fields to 1 degree
    sm, st = _sampled_tangents(g["streams"])
    keep = _far_from_degeneracy(sm)
    a_s = _angle_to_field_deg(sm[keep], st[keep],
                              lambda x, y: (lambda g_: (g_[1], -g_[0]))(_grad_psi(x, y)))
    em, et = _sampled_tangents(g["equis"])
    keep_e = _far_from_degeneracy(em)
    a_e = _angle_to_field_deg(em[keep_e], et[keep_e],
                              lambda x, y: (lambda g_: (-g_[1], g_[0]))(_grad_phi(x, y)))
    c.check(float(a_s.max()) < 1.0,
            f"streamline tangent off the velocity field by {a_s.max():.2f} deg")
    c.check(float(a_e.max()) < 1.0,
            f"equipotential tangent off the analytic field by {a_e.max():.2f} deg")

    # (2c) at drawn crossings of the two families, the tangents are
    # perpendicular to 2 degrees (nearest-point pairs, polyline tangents)
    grid_h = (p["xlim"][1] - p["xlim"][0]) / p["n"]
    stencil = 3
    n_cross, worst_perp = 0, 0.0
    for scv in g["streams"]:
        sp = scv.pts
        if len(sp) < 2 * stencil + 3:
            continue
        for ecv in g["equis"]:
            ep = ecv.pts
            if len(ep) < 2 * stencil + 3:
                continue
            ss, ee = sp[::4], ep[::4]
            D = np.hypot(ss[:, None, 0] - ee[None, :, 0],
                         ss[:, None, 1] - ee[None, :, 1])
            i, j = np.unravel_index(np.argmin(D), D.shape)
            if D[i, j] > 2.5 * grid_h:
                continue
            i, j = i * 4, j * 4
            if not (stencil <= i < len(sp) - stencil
                    and stencil <= j < len(ep) - stencil):
                continue
            mid = 0.5 * (sp[i] + ep[j])
            if not _far_from_degeneracy(mid[None, :])[0]:
                continue
            t1 = sp[i + stencil] - sp[i - stencil]
            t2 = ep[j + stencil] - ep[j - stencil]
            cosang = abs(np.dot(t1, t2)) / (np.hypot(*t1) * np.hypot(*t2))
            dev = abs(90.0 - np.degrees(np.arccos(np.clip(cosang, 0.0, 1.0))))
            worst_perp = max(worst_perp, float(dev))
            n_cross += 1
    c.check(n_cross >= 150, f"only {n_cross} family crossings sampled (need 150)")
    c.check(worst_perp < 2.0,
            f"crossing tangents off perpendicular by {worst_perp:.2f} deg "
            f"({n_cross} crossings)")

    # (3) stagnation points: Omega'(+-1) = 0 exactly, field zero there
    c.check(abs(1.0 - 1.0 / (1.0 ** 2)) == 0.0
            and abs(1.0 - 1.0 / ((-1.0) ** 2)) == 0.0,
            "Omega'(+-1) != 0")
    for sx in (-1.0, 1.0):
        v = _vel(np.array([[sx, 0.0]]))[0]
        c.check(float(np.hypot(v[0], v[1])) < 1e-10,
                f"|field| at z = {sx:+.0f} is {np.hypot(v[0], v[1]):.2e}")
    c.check(len(g["stag"]) == 2,
            f"expected 2 stagnation points, found {len(g['stag'])}")
    for (x, y), sx in zip(g["stag"], (-1.0, 1.0)):
        c.check(np.hypot(x - sx, y) < 1e-8,
                f"stagnation point at ({x:.6f}, {y:.6f}), not ({sx:+.0f}, 0)")

    # (4) equal level spacing, and the SAME spacing, for both families
    for levels, name in ((g["psi_levels"], "Psi"), (g["phi_levels"], "Phi")):
        d = np.diff(np.asarray(levels))
        c.check(np.allclose(d, k), f"{name} levels not equally spaced by dk")

    # the right-angle mark sits on both of its levels, edges perpendicular
    cx, cy = g["square_pt"]
    sq_psi, sq_phi = p["square_levels"]
    c.check(abs(_psi(cx, cy) - sq_psi) < 1e-10
            and abs(_phi(cx, cy) - sq_phi) < 1e-10,
            "right-angle mark off its crossing")
    d1, d2 = g["square_dirs"]
    c.check(abs(float(np.dot(d1, d2))) < 1e-12, "right-angle mark edges not perpendicular")

    c.done()
