"""VCA Figure [12]: the full conformal grid of Omega = z + 1/z (p. 528).

Uniform flow past the unit disc, no circulation. Psi = Im Omega gives the
streamlines (solid, hollow markers pointing downstream); Phi = Re Omega
gives the equipotentials (dashed, the book's convention). Both families
are drawn at the SAME level spacing k, so conformality of Omega away
from its critical points makes the grid a mesh of near-squares — cell
side k/|Omega'| in both directions. The grid degenerates exactly at the
stagnation points z = +-1, where Omega'(z) = 1 - 1/z^2 = 0 and the map
ceases to be conformal; the critical level curves Phi = +-2 are drawn
branching through +-1 at 45 degrees, so the degeneracy is shown, not
just marked. A dotted rounded frame bounds the flow window — every
curve either runs to the frame or is clipped by it — and the family
labels sit outside the window, each tied to its own curve by a short
leader tick in the family's line style.

Scope choice: the disc interior stays solid (the book's dual reading —
the dipole grid inside — is a second claim this figure does not make);
the dead space carries the two formulas instead, in paper ink.
"""

from dataclasses import replace

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
    "xlim": (-4.7, 4.7),        # scene extents: side pad bands carry labels
    "window": (-4.2, 4.2, -2.1, 2.1),   # the flow window = the drawn frame
    "corner_r": 0.26,           # rounded-frame corner radius (math units)
    "dk": 0.3,                  # ONE spacing for both ladders: DPsi = DPhi
    "psi_kmax": 6,              # Psi levels k*dk, k in [-6, 6]  (13 streamlines)
    "phi_kmax": 13,             # Phi levels k*dk, k in [-13, 13] (27 arcs)
    "n": 720,                   # contour grid resolution per axis
    "crit_snap": 0.08,          # max gap tip -> stagnation point before snapping
    "label_psi_level": 0.9,     # streamline whose right-edge exit carries Psi
    "label_phi_level": -2.7,    # equipotential whose bottom exit carries Phi
    "square_levels": (0.3, 1.5),  # (Psi, Phi) crossing on the disc shoulder
    "square_seed": (0.9, 0.9),  # Newton start for that crossing
    "square_size": 0.12,        # right-angle mark side (math units)
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


# --- the frame: a rounded window the flow is clipped by ----------------------


def _rounded_rect(x0, x1, y0, y1, r, n=10):
    q = np.linspace(0, np.pi / 2, n)
    arcs = [
        np.column_stack([x1 - r + r * np.cos(q), y1 - r + r * np.sin(q)]),
        np.column_stack([x0 + r - r * np.sin(q), y1 - r + r * np.cos(q)]),
        np.column_stack([x0 + r - r * np.cos(q), y0 + r - r * np.sin(q)]),
        np.column_stack([x1 - r + r * np.sin(q), y0 + r - r * np.cos(q)]),
    ]
    return np.vstack(arcs)


def _rect_gap(pts, x0, x1, y0, y1, r):
    """Distance from each point to the frame's core rect [x0+r,x1-r] x
    [y0+r,y1-r]; a point is inside the rounded window iff gap <= r."""
    ncx = np.clip(pts[:, 0], x0 + r, x1 - r)
    ncy = np.clip(pts[:, 1], y0 + r, y1 - r)
    return np.hypot(pts[:, 0] - ncx, pts[:, 1] - ncy)


def _boundary_point(p_in, p_out, x0, x1, y0, y1, r):
    """The point where the segment p_in -> p_out crosses the rounded-window
    boundary (bisection on the gap function)."""
    a, b = 0.0, 1.0
    for _ in range(40):
        m = 0.5 * (a + b)
        q = p_in + m * (p_out - p_in)
        if _rect_gap(q[None, :], x0, x1, y0, y1, r)[0] <= r:
            a = m
        else:
            b = m
    return p_in + a * (p_out - p_in)


def _trim_to_window(curves, x0, x1, y0, y1, r):
    """Cut every polyline to the rounded window; a curve leaving through a
    corner splits into inside runs, each run's cut end interpolated onto
    the frame arc — trimmed curves touch the drawn border, no stubs."""
    out = []
    for cv in curves:
        pts = cv.pts
        inside = _rect_gap(pts, x0, x1, y0, y1, r) <= r + 1e-9
        if inside.all():
            out.append(cv)
            continue
        idx = np.flatnonzero(inside)
        if idx.size < 2:
            continue
        for run in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
            if len(run) < 2:
                continue
            a, b = int(run[0]), int(run[-1])
            seg = [pts[a:b + 1]]
            if a > 0:
                seg.insert(0, _boundary_point(
                    pts[a], pts[a - 1], x0, x1, y0, y1, r)[None, :])
            if b < len(pts) - 1:
                seg.append(_boundary_point(
                    pts[b], pts[b + 1], x0, x1, y0, y1, r)[None, :])
            out.append(replace(cv, pts=np.vstack(seg), closed=False))
    return out


def compute(p):
    k = p["dk"]
    wx0, wx1, wy0, wy1 = p["window"]
    wxlim, wylim = (wx0, wx1), (wy0, wy1)
    psi_levels = [i * k for i in range(-p["psi_kmax"], p["psi_kmax"] + 1)]
    phi_levels = [i * k for i in range(-p["phi_kmax"], p["phi_kmax"] + 1)]
    inside = lambda X, Y: X * X + Y * Y < 1.0

    streams = stream_function_lines(
        _psi, wxlim, wylim, psi_levels, n=p["n"], mask=inside,
        arrows=(0.30, 0.72), role=Role.CONTENT, width_scale=0.8,
        arrow_scale=0.75)
    equis = stream_function_lines(
        _phi, wxlim, wylim, phi_levels, n=p["n"], mask=inside,
        arrows=(), role=Role.CONTENT, width_scale=0.5, dash="4 3.5")

    # the critical level curves Phi = +-2: through the stagnation points,
    # branching at 45 degrees — the degeneracy of the grid, drawn. Each
    # branch tip stops a mask cell short of +-1; snap it to the exact point
    # (the pre-snap gap is recorded and asserted below).
    crit_raw = stream_function_lines(
        _phi, wxlim, wylim, [-2.0, 2.0], n=p["n"], mask=inside,
        arrows=(), role=Role.CONTENT, width_scale=0.5, dash="1.5 3")
    crit, crit_gaps, crit_touch = [], [], {-1.0: 0, 1.0: 0}
    for cv in crit_raw:
        pts = cv.pts
        sx = 1.0 if float(np.mean(pts[:, 0])) > 0 else -1.0
        target = np.array([sx, 0.0])
        d0 = float(np.hypot(*(pts[0] - target)))
        d1 = float(np.hypot(*(pts[-1] - target)))
        if min(d0, d1) < p["crit_snap"]:
            crit_gaps.append(min(d0, d1))
            crit_touch[sx] += 1
            pts = (np.vstack([target, pts]) if d0 < d1
                   else np.vstack([pts, target]))
        crit.append(replace(cv, pts=pts))

    r = p["corner_r"]
    streams = _trim_to_window(streams, wx0, wx1, wy0, wy1, r)
    equis = _trim_to_window(equis, wx0, wx1, wy0, wy1, r)
    crit = _trim_to_window(crit, wx0, wx1, wy0, wy1, r)

    stag = sorted(stagnation_points(_vel, wxlim, wylim, grid_n=96))

    # label anchors: the exact exit points of the labelled curves.
    # Psi rides a streamline out the RIGHT edge (only streamlines reach it,
    # so the attachment is unambiguous); Phi rides an equipotential out the
    # bottom edge.
    y_psi_exit = _bisect(
        lambda y: _psi(wx1, y) - p["label_psi_level"], 0.4, 1.6)
    x_phi_exit = _bisect(
        lambda x: _phi(x, wy0) - p["label_phi_level"], -3.4, -2.0)
    qxe, qye = _grad_phi(x_phi_exit, wy0)
    phi_dir = _unit(np.array([-qye, qxe]))   # equipotential tangent at exit
    if phi_dir[1] < 0:
        phi_dir = -phi_dir                   # points INTO the window

    # the orthogonality witness: a drawn crossing ON the disc shoulder,
    # where the cells are visibly curved — the surprising place
    sq_psi, sq_phi = p["square_levels"]
    cx, cy = _crossing(sq_psi, sq_phi, *p["square_seed"])
    px, py = _grad_psi(cx, cy)
    qx, qy = _grad_phi(cx, cy)
    d1 = _unit(np.array([py, -px]))          # streamline tangent (downstream)
    d2 = _unit(np.array([-qy, qx]))          # equipotential tangent
    if d1[0] < 0:
        d1 = -d1
    if d2[1] < 0:
        d2 = -d2

    frame = _rounded_rect(wx0, wx1, wy0, wy1, r)

    return {"streams": streams, "equis": equis, "crit": crit,
            "crit_gaps": crit_gaps, "crit_touch": crit_touch,
            "stag": stag, "psi_levels": psi_levels, "phi_levels": phi_levels,
            "psi_exit": (wx1, y_psi_exit), "phi_exit": (x_phi_exit, wy0),
            "phi_exit_dir": phi_dir,
            "square_pt": (cx, cy), "square_dirs": (d1, d2),
            "frame": frame, "p": p}


def build(g):
    p = g["p"]
    wx0, wx1, wy0, wy1 = p["window"]
    s = Scene(xlim=p["xlim"], ylim=(wy0, wy1))

    # dashed family under, solid family over: the book's subordination;
    # the critical curves sit with their family, in a finer dash
    s.add(*g["equis"])
    s.add(*g["crit"])
    s.add(*g["streams"])

    # the obstacle: near-black disc over the contour ends at its boundary
    th = np.linspace(0.0, 2.0 * np.pi, 240)
    circle = np.column_stack([np.cos(th), np.sin(th)])
    s.add(FilledCurve(circle, role=Role.CONTENT, opacity=1.0, outline=False))
    s.add(Curve(circle, closed=True, role=Role.CONTENT, width_scale=1.0))

    # stagnation points: where Omega' = 0 and the grid degenerates
    for x, y in g["stag"]:
        s.add(Point((x, y), filled=True, role=Role.ACCENT2, radius_scale=1.0))

    # the window: dotted rounded frame — ragged contour ends read as
    # clipped by it, a window into the infinite flow
    s.add(Curve(g["frame"], closed=True, role=Role.FRAME))

    # the formulas ride the disc's dead space, in paper ink
    paper = THEME.paper[0]
    s.add(MathLabel(r"\Omega = z + 1/z", (0.0, 0.30), ha="center",
                    va="center", size_pt=11, role=Role.ANNOTATION,
                    color=paper, pin=True))
    s.add(MathLabel(r"\Omega'({\pm}1) = 0", (0.0, -0.32), ha="center",
                    va="center", size_pt=10, role=Role.ANNOTATION,
                    color=paper, pin=True))

    # one label per family, OUTSIDE the window, tied to its own curve by a
    # short leader tick in the family's line style
    ex, ey = g["psi_exit"]
    s.add(Curve(np.array([[ex, ey], [ex + 0.11, ey]]), role=Role.ANNOTATION,
                width_scale=0.9))
    s.add(MathLabel(r"\Psi = \mathrm{Im}\,\Omega", (ex + 0.14, ey),
                    ha="left", va="center", size_pt=10,
                    role=Role.ANNOTATION, pin=True))
    fx, fy = g["phi_exit"]
    td = g["phi_exit_dir"]
    exit_pt = np.array([fx, fy])
    # the tick continues the equipotential through the frame, along its
    # own tangent and in its own (dashed) line style
    s.add(Curve(np.vstack([exit_pt + 0.06 * td, exit_pt - 0.085 * td]),
                role=Role.ANNOTATION, width_scale=1.15, dash="2 1.4"))
    s.add(MathLabel(r"\Phi = \mathrm{Re}\,\Omega",
                    (fx - 0.11 * td[0], fy - 0.11 * td[1]),
                    ha="center", va="top", size_pt=9.5,
                    role=Role.ANNOTATION, pin=True))

    # the equal-spacing admission: same ladder step for both families
    s.add(MathLabel(r"\Delta\Psi = \Delta\Phi", (4.05, wy0 - 0.09),
                    ha="center", va="top", size_pt=9.5, role=Role.ANNOTATION))

    # the theorem at one drawn crossing on the disc shoulder
    (d1, d2) = g["square_dirs"]
    s.add(RightAngleMark(g["square_pt"], tuple(d1), tuple(d2),
                         size=p["square_size"]))
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
    wx0, wx1, wy0, wy1 = p["window"]

    # (1) every drawn point lies on a level of its own family (the snapped
    # critical tips are on-level exactly: Phi(+-1, 0) = +-2)
    for curves, fn, levels, name in (
            (g["streams"], _psi, g["psi_levels"], "Psi"),
            (g["equis"], _phi, g["phi_levels"], "Phi"),
            (g["crit"], _phi, [-2.0, 2.0], "critical Phi")):
        ladder = np.asarray(levels, dtype=float)
        span = ladder[-1] - ladder[0]
        worst = 0.0
        for cv in curves:
            vals = fn(cv.pts[:, 0], cv.pts[:, 1])
            err = np.min(np.abs(vals[:, None] - ladder[None, :]), axis=1)
            worst = max(worst, float(err.max()))
        c.check(worst < 2e-3 * span,
                f"{name} point off its level by {worst:.2e} (tol {2e-3 * span:.2e})")

    all_curves = g["streams"] + g["equis"] + g["crit"]
    # every drawn point respects the obstacle
    r_min = min(float(np.hypot(cv.pts[:, 0], cv.pts[:, 1]).min())
                for cv in all_curves)
    c.check(r_min > 1.0 - 1e-4, f"drawn point inside the disc: r = {r_min:.6f}")
    # ... and the window: nothing pokes past the drawn frame
    worst_gap = max(float(_rect_gap(cv.pts, wx0, wx1, wy0, wy1,
                                    p["corner_r"]).max())
                    for cv in all_curves)
    c.check(worst_gap <= p["corner_r"] + 1e-6,
            f"curve point {worst_gap - p['corner_r']:.2e} past the frame")

    # the critical curves pass through the stagnation points: each of +-1
    # is hit EXACTLY by at least one branch tip, and the pre-snap gap was
    # below the mask resolution — the snap is a repair of one grid cell,
    # not an invention
    for sx in (-1.0, 1.0):
        touched = any(float(np.min(np.hypot(cv.pts[:, 0] - sx,
                                            cv.pts[:, 1]))) < 1e-12
                      for cv in g["crit"])
        c.check(touched and g["crit_touch"][sx] >= 1,
                f"no critical branch through z = {sx:+.0f}")
    c.check(len(g["crit_gaps"]) >= 2
            and max(g["crit_gaps"]) < p["crit_snap"],
            f"critical tip snapped over a gap of {max(g['crit_gaps'], default=np.inf):.3f} "
            f"(tol {p['crit_snap']})")

    # (2a) orthogonality, analytic: grad Psi . grad Phi = 0 on a sample grid
    xs = np.linspace(wx0, wx1, 161)
    ys = np.linspace(wy0, wy1, 81)
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
    em, et = _sampled_tangents(g["equis"] + g["crit"])
    keep_e = _far_from_degeneracy(em)
    a_e = _angle_to_field_deg(em[keep_e], et[keep_e],
                              lambda x, y: (lambda g_: (-g_[1], g_[0]))(_grad_phi(x, y)))
    c.check(float(a_s.max()) < 1.0,
            f"streamline tangent off the velocity field by {a_s.max():.2f} deg")
    c.check(float(a_e.max()) < 1.0,
            f"equipotential tangent off the analytic field by {a_e.max():.2f} deg")

    # (2c) at drawn crossings of the two families, the tangents are
    # perpendicular to 2 degrees (nearest-point pairs, polyline tangents)
    grid_h = (wx1 - wx0) / p["n"]
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

    # the label leader ticks touch their own curves: the exits solve the
    # labelled level equations on the frame
    ex, ey = g["psi_exit"]
    fx, fy = g["phi_exit"]
    c.check(abs(_psi(ex, ey) - p["label_psi_level"]) < 1e-9,
            "Psi label tick misses its streamline exit")
    c.check(abs(_phi(fx, fy) - p["label_phi_level"]) < 1e-9,
            "Phi label tick misses its equipotential exit")

    # the right-angle mark sits on both of its levels, edges perpendicular
    cx, cy = g["square_pt"]
    sq_psi, sq_phi = p["square_levels"]
    c.check(abs(_psi(cx, cy) - sq_psi) < 1e-10
            and abs(_phi(cx, cy) - sq_phi) < 1e-10,
            "right-angle mark off its crossing")
    d1, d2 = g["square_dirs"]
    c.check(abs(float(np.dot(d1, d2))) < 1e-12, "right-angle mark edges not perpendicular")

    c.done()
