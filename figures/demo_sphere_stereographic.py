"""VCA Fig [19] (p.140): stereographic projection from the north pole.

The unit sphere Sigma sits centered on the plane C, which is drawn as
the book draws it: an opaque sheet through the equator, hiding the
southern cap where the sheet is in front and letting it poke out below
the sheet's near edge. From N the ray through a plane point p pierces
Sigma at the image p-hat (chord inside the sphere dashed, exterior rod
solid). A whole line through p maps to a circle through N: as p runs
out along the line (hollow arrows), p-hat climbs the circle into N.
"""

import numpy as np

from figlib.scene import Curve, FilledCurve, MathLabel, Point, Scene
from figlib.sphere3d import (Sphere, anchor, circle_points, drape, occlude,
                             point_on, silhouette, disc, visibility_runs)
from figlib.style import Role
from figlib.surface3d import Camera, as_floor, compose, project
from figlib.format import WIDE
from figlib.theme import RISO, SHADE, Hue

THEME = RISO
FORMAT = WIDE

CLAIM = (
    "Stereographic projection from the north pole N sends a point p of the "
    "plane outside the unit circle to the point p-hat where the ray N->p "
    "pierces the northern hemisphere of the unit sphere; the equator is the "
    "unit circle itself, and a straight line through p maps to a circle "
    "through N, so p -> infinity drives p-hat -> N."
)

PARAMS = {
    "p": (0.55, -1.02),       # in the plane, just outside the unit circle
    "azim": -40.0,
    "elev": 28.0,
    "plane_rot_deg": 18.0,    # plane quad twist vs the screen axes
    "plane_s": (2.5, 1.7),    # extents along the screen-parallel edge
    "plane_t": (1.3, 1.6),    # extents along the depth edge (far, near)
    "line_dir_deg": 200.0,    # world-xy direction of the line through p
    "q_t": 1.6,               # second sample point q = p + q_t * dir
    "line_arrow_ts": (-0.55, 0.7, 2.3),
    "circle_arrow_ts": (-1.6, 3.5),
    "n": 512,
}

BODY = "#c2ae8c"          # flat printed-tan sphere body
BODY_UNDER = "#a8926d"    # the cap seen below the sheet's near edge
# The sheet of C is lit paper, as in the book: it reads by occlusion and
# its rim, not figure-ground contrast (SHADE channel, like 3D facets).
SHEET = Hue("#f4ead0", SHADE)

# painter tags for everything that has to beat the sphere's silhouette
TAG_CAP = 1.03            # southern cap sliver, over the body fill
TAG_AXIS = 1.04           # axis dashes, over the cap, under the sheet
TAG_SHEET = 1.20          # the near sheet: occludes limb + southern cap
TAG_RIM = 1.25            # sheet rim edges on the near side
TAG_EQ_FRONT = 1.30       # the unit circle's front arc, in the sheet
TAG_LINE = 1.35           # the line L, drawn ON the sheet
TAG_STUB = 1.50           # direction arrows
TAG_RAY = 2.00            # projection rods above everything


def stereo(x2: np.ndarray) -> np.ndarray:
    """Inverse stereographic image of plane points (..., 2) on the unit
    sphere: the point where the segment N -> (x, y, 0) meets |q| = 1."""
    x2 = np.asarray(x2, dtype=float)
    r2 = (x2**2).sum(axis=-1, keepdims=True)
    return np.concatenate([2.0 * x2, r2 - 1.0], axis=-1) / (r2 + 1.0)


def _clip_half(poly: np.ndarray, normal2: np.ndarray, sign: float) -> np.ndarray:
    """Sutherland-Hodgman: keep the part of a z=0 polygon with
    sign * (xy . normal2) >= 0."""
    d = sign * (poly[:, :2] @ normal2)
    out = []
    for i in range(len(poly)):
        j = (i + 1) % len(poly)
        if d[i] >= 0:
            out.append(poly[i])
        if d[i] * d[j] < 0:
            t = d[i] / (d[i] - d[j])
            out.append(poly[i] + t * (poly[j] - poly[i]))
    return np.array(out)


def _line_range_in_quad(P2, d2, u, v, s_ext, t_ext):
    """Parameter interval of the line P2 + t*d2 inside the plane quad,
    in the quad's own (u, v) sheet coordinates."""
    lo, hi = -np.inf, np.inf
    for axis, (a0, a1) in ((u[:2], (-s_ext[0], s_ext[1])),
                           (v[:2], (-t_ext[0], t_ext[1]))):
        c0, c1 = float(P2 @ axis), float(d2 @ axis)
        if abs(c1) < 1e-12:
            assert a0 <= c0 <= a1, "line parallel to and outside the sheet"
            continue
        ta, tb = (a0 - c0) / c1, (a1 - c0) / c1
        lo, hi = max(lo, min(ta, tb)), min(hi, max(ta, tb))
    assert lo < hi, "line misses the sheet quad"
    return lo, hi


def _near_sheet_fill(near_poly: np.ndarray, right: np.ndarray,
                     toward: np.ndarray, n_arc: int = 96):
    """The visible sheet as a filled region: the near half-quad with its
    artificial cut edge replaced by the equator's FRONT arc, so the fill
    abuts the sphere exactly along the unit circle and occludes the
    southern cap up to the sheet's near edge.

    Returns (polygon3, (T1, T2)) where T1, T2 are the equator-silhouette
    tangency points the split edge lands on."""
    d = near_poly[:, :2] @ toward[:2]
    cut = np.where(np.abs(d) < 1e-9)[0]
    assert len(cut) == 2, "near sheet clip must produce exactly two cut vertices"
    m = len(near_poly)
    i = next(k for k in cut if (k + 1) % m in cut)   # cut edge k -> k+1
    rolled = np.roll(near_poly, -((i + 1) % m), axis=0)
    cut2, cut1 = rolled[0], rolled[-1]               # walk cut2 ... cut1

    def tangency(vertex):
        s = 1.0 if float(vertex[:2] @ right[:2]) >= 0 else -1.0
        return s * np.array([right[0], right[1], 0.0])

    T1, T2 = tangency(cut1), tangency(cut2)
    th1 = np.arctan2(T1[1], T1[0])
    th2 = np.arctan2(T2[1], T2[0])
    dth = (th2 - th1 + np.pi) % (2.0 * np.pi) - np.pi
    if abs(abs(dth) - np.pi) < 1e-9:                 # antipodal: pick front arc
        for cand in (np.pi, -np.pi):
            mid = th1 + 0.5 * cand
            if np.cos(mid) * toward[0] + np.sin(mid) * toward[1] > 0:
                dth = cand
                break
    th = th1 + np.linspace(0.0, dth, n_arc)
    arc = np.column_stack([np.cos(th), np.sin(th), np.zeros(n_arc)])
    return np.vstack([rolled, arc]), (T1, T2)


def _cap_below_edge(edge3: np.ndarray, cam: Camera, n_arc: int = 96):
    """2D sliver of the sphere's disc on the far side of the sheet's near
    edge — the southern cap the book shows poking out below the sheet."""
    (e0, e1), _ = project(edge3, cam)
    dv = e1 - e0
    a = float(dv @ dv)
    b = 2.0 * float(e0 @ dv)
    c = float(e0 @ e0) - 1.0
    disc_ = b * b - 4.0 * a * c
    assert disc_ > 0, "sheet near edge must cross the sphere's disc"
    s1, s2 = (-b - np.sqrt(disc_)) / (2 * a), (-b + np.sqrt(disc_)) / (2 * a)
    A, B = e0 + s1 * dv, e0 + s2 * dv
    phiA, phiB = np.arctan2(A[1], A[0]), np.arctan2(B[1], B[0])
    nrm = np.array([-dv[1], dv[0]])
    side_center = np.sign(float(-e0 @ nrm))          # which side holds (0,0)
    dphi = (phiB - phiA) % (2.0 * np.pi)
    mid = phiA + 0.5 * dphi
    if np.sign(float((np.array([np.cos(mid), np.sin(mid)]) - e0) @ nrm)) == side_center:
        phiA, phiB = phiB, phiA
        dphi = (phiB - phiA) % (2.0 * np.pi)
    phi = phiA + np.linspace(0.0, dphi, n_arc)
    return np.column_stack([np.cos(phi), np.sin(phi)]), (A, B)


def compute(p):
    cam = Camera(p["azim"], p["elev"])
    right, up, toward = cam.axes()
    sph = Sphere(np.zeros(3), 1.0)

    P2 = np.array(p["p"], dtype=float)
    P = np.array([*P2, 0.0])
    N = np.array([0.0, 0.0, 1.0])
    S = np.array([0.0, 0.0, -1.0])
    # ray N -> p meets |q| = 1 at parameter t = 2 / (|p|^2 + 1)
    t = 2.0 / (P2 @ P2 + 1.0)
    phat = N + t * (P - N)

    # plane quad: a parallelogram twisted plane_rot_deg from the screen axes
    ang = np.arctan2(right[1], right[0]) + np.radians(p["plane_rot_deg"])
    u = np.array([np.cos(ang), np.sin(ang), 0.0])
    v = np.array([u[1], -u[0], 0.0])
    if v[:2] @ toward[:2] < 0:
        v = -v
    (s0, s1), (t0, t1) = p["plane_s"], p["plane_t"]
    quad = np.array([-s0 * u - t0 * v, s1 * u - t0 * v,
                     s1 * u + t1 * v, -s0 * u + t1 * v])
    far_poly = _clip_half(quad, toward[:2], -1.0)
    near_poly = _clip_half(quad, toward[:2], +1.0)
    near_fill, tangencies = _near_sheet_fill(near_poly, right, toward)
    cap_arc, cap_chord = _cap_below_edge(quad[2:4], cam)

    equator = circle_points(sph, np.array([0.0, 0.0, 1.0]), 0.0, n=p["n"])
    eq_runs = visibility_runs(equator, sph, cam)

    # the line L through p and its image circle through N
    psi = np.radians(p["line_dir_deg"])
    d2 = np.array([np.cos(psi), np.sin(psi)])
    t_lo, t_hi = _line_range_in_quad(P2, d2, u, v, (s0, s1), (t0, t1))
    line3 = np.array([[*(P2 + t_lo * d2), 0.0], [*(P2 + t_hi * d2), 0.0]])
    t_star = -float(P2 @ d2)                          # closest approach to O
    line_dist = float(np.linalg.norm(P2 + t_star * d2))
    ts = t_star + 2.0 * np.tan(np.linspace(-np.pi / 2 + 0.03,
                                           np.pi / 2 - 0.03, p["n"]))
    circle_img = stereo(P2[None, :] + ts[:, None] * d2[None, :])
    circle_pts = np.vstack([N, circle_img, N])        # closes through N

    Q2 = P2 + p["q_t"] * d2
    Q = np.array([*Q2, 0.0])
    qhat = stereo(Q2)

    ray_p = np.array([N, P])                          # p's rod: N -> p exactly
    ray_q = np.array([N - 0.07 * (Q - N), Q])         # q's rod pokes past N

    def img_stub(tv):
        q3 = stereo(P2 + tv * d2)
        dq = (stereo(P2 + (tv + 1e-4) * d2) - q3) / 1e-4
        return q3, dq / np.linalg.norm(dq)

    line_stubs = [(np.array([*(P2 + tv * d2), 0.0]), np.array([*d2, 0.0]))
                  for tv in p["line_arrow_ts"]]
    circle_stubs = [img_stub(tv) for tv in p["circle_arrow_ts"]]

    return {"cam": cam, "sph": sph, "P": P, "N": N, "S": S, "phat": phat,
            "Q": Q, "qhat": qhat, "quad": quad,
            "far_poly": far_poly, "near_poly": near_poly,
            "near_fill": near_fill, "tangencies": tangencies,
            "cap_arc": cap_arc, "cap_chord": cap_chord,
            "plane_uv": (u, v), "equator": equator, "eq_runs": eq_runs,
            "line3": line3, "line_d2": d2, "line_dist": line_dist,
            "line_ts": ts, "circle_pts": circle_pts, "circle_img": circle_img,
            "ray_p": ray_p, "ray_q": ray_q,
            "line_stubs": line_stubs, "circle_stubs": circle_stubs,
            "n": p["n"]}


def _rim(g, tag: float, poly: np.ndarray):
    """The sheet's boundary edges (the artificial cut edge through the
    sphere is not stroked)."""
    cam = g["cam"]
    _, _, toward = cam.axes()
    pts2, _ = project(poly, cam)
    d = poly[:, :2] @ toward[:2]
    items = []
    for i in range(len(poly)):
        j = (i + 1) % len(poly)
        if abs(d[i]) < 1e-9 and abs(d[j]) < 1e-9:
            continue
        items.append((tag, Curve(pts2[[i, j]], role=Role.ANNOTATION,
                                 width_scale=0.8)))
    return items


def _stub(g, pt3, dir3, tag, role, length=0.14):
    """A short tangent segment ending at pt3, carrying a hollow direction
    arrow (Needham's open triangles for a moving point)."""
    pts2, _ = project(np.array([pt3 - length * dir3, pt3]), g["cam"])
    return (tag, Curve(pts2, role=role, width_scale=1.0,
                       arrows=(1.0,), arrow_style="hollow", arrow_scale=0.9))


def build(g):
    cam, sph = g["cam"], g["sph"]
    s = Scene()

    # the sheet: full quad behind everything, opaque near part on top
    quad2, _ = project(g["quad"], cam)
    floor = as_floor([(0.0, FilledCurve(quad2, opacity=1.0, outline=False,
                                        color=SHEET))])
    near_fill2, _ = project(g["near_fill"], cam)
    near_sheet = [(TAG_SHEET, FilledCurve(near_fill2, opacity=1.0,
                                          outline=False, color=SHEET))]
    far = _rim(g, -50.0, g["far_poly"])       # hidden where the body covers it
    near = _rim(g, TAG_RIM, g["near_poly"])

    body = disc(sph, cam, color=BODY)
    limb = silhouette(sph, cam, role=Role.CONTENT, width_scale=1.2)
    cap = [(TAG_CAP, FilledCurve(g["cap_arc"], opacity=1.0, outline=False,
                                 color=BODY_UNDER))]

    # the unit circle = the equator, the projection's fixed set; its front
    # arc is the visible sheet-sphere junction, over the sheet fill
    eq = []
    for visible, run in g["eq_runs"]:
        pts2, depth = project(run, cam)
        if visible:
            eq.append((TAG_EQ_FRONT,
                       Curve(pts2, role=Role.ACCENT1, width_scale=1.5)))
        else:
            eq.append((float(depth.mean()),
                       Curve(pts2, role=Role.ACCENT1, dash="dashed",
                             width_scale=1.1, opacity=0.85)))

    # the line L in the sheet and its image circle through N
    line = [(TAG_LINE, c) for _, c in
            occlude(g["line3"], sph, cam, hidden=None, n=g["n"],
                    role=Role.CONTENT, width_scale=1.35)]
    circle = drape(g["circle_pts"], sph, cam, hidden="dashed",
                   role=Role.CONTENT, width_scale=1.15)

    # polar axis: dashed construction, wiped by the sheet where the sheet
    # is in front, re-emerging on the cap sliver (the book's dots near S)
    axis = [(TAG_AXIS, c) for _, c in
            occlude(np.array([g["N"], g["S"]]), sph, cam, n=g["n"],
                    role=Role.ANNOTATION, dash="dashed", width_scale=0.9)]

    # projection rods: chord inside the sphere dashed, exterior solid
    def rod(pts3, width):
        items = []
        for tag, c in occlude(pts3, sph, cam, n=g["n"], role=Role.ACCENT2,
                              width_scale=width):
            items.append((tag if c.dash else TAG_RAY, c))
        return items

    rays = rod(g["ray_p"], 1.4) + rod(g["ray_q"], 1.0)

    stubs = [_stub(g, pt, d3, TAG_LINE + 0.01, Role.CONTENT)
             for pt, d3 in g["line_stubs"]]
    stubs += [_stub(g, pt, d3, TAG_STUB, Role.CONTENT)
              for pt, d3 in g["circle_stubs"]]

    s.items.extend(compose(floor, far, body, limb, cap, eq, circle, axis,
                           near_sheet, near, line, stubs, rays))

    for q, role, r_sc in ((g["N"], Role.CONTENT, 1.0),
                          (g["phat"], Role.ACCENT2, 1.0),
                          (g["P"], Role.ACCENT2, 1.0),
                          (g["qhat"], Role.ACCENT2, 0.7),
                          (g["Q"], Role.ACCENT2, 0.7)):
        s.add(Point(anchor(q, cam), role=role, radius_scale=r_sc))
    # hidden points step down: open dots, like the book's white marks
    s.add(Point(anchor(np.zeros(3), cam), role=Role.ANNOTATION,
                filled=False, radius_scale=0.8))
    s.add(Point(anchor(g["S"], cam), role=Role.ANNOTATION,
                filled=False, radius_scale=0.85))

    s.add(MathLabel(r"N", anchor(g["N"], cam), ha="center", va="bottom",
                    offset_px=(0, -28)))
    s.add(MathLabel(r"S", anchor(g["S"], cam), ha="center", va="top",
                    offset_px=(0, 26), role=Role.ANNOTATION))
    s.add(MathLabel(r"O", anchor(np.zeros(3), cam), ha="right", va="center",
                    offset_px=(-9, 1), role=Role.ANNOTATION))
    s.add(MathLabel(r"p", anchor(g["P"], cam), ha="right", va="top",
                    offset_px=(-11, 8), role=Role.ACCENT2))
    s.add(MathLabel(r"\hat{p}", anchor(g["phat"], cam), ha="left", va="bottom",
                    offset_px=(9, -3), role=Role.ACCENT2))
    # Sigma ON the upper-right body: a sphere point projecting to (.62, .55)
    sig = 0.62 * cam.axes()[0] + 0.55 * cam.axes()[1]
    sig = point_on(g["sph"], sig + np.sqrt(1 - 0.62**2 - 0.55**2) * cam.axes()[2])
    s.add(MathLabel(r"\Sigma", anchor(sig, cam), ha="center", va="center"))
    u, v = g["plane_uv"]
    s.add(MathLabel(r"\mathbb{C}", anchor(-1.6 * u + 0.6 * v, cam),
                    ha="center", va="center", role=Role.ANNOTATION))

    # the book sets UNIT CIRCLE in perspective along the front arc; a
    # single rotation per word tracks the arc's local tangent (set on the
    # right half of the arc, where the line L has fallen away)
    a = np.radians(PARAMS["azim"])
    for word, dth in ((r"\text{UNIT}", 0.50), (r"\text{CIRCLE}", 1.05)):
        th = a + dth
        pos = 1.17 * np.array([np.cos(th), np.sin(th), 0.0])
        tan3 = np.array([-np.sin(th), np.cos(th), 0.0])
        (p2, t2), _ = project(np.array([pos, pos + 0.1 * tan3]), cam)
        ang_deg = np.degrees(np.arctan2(t2[1] - p2[1], t2[0] - p2[0]))
        if not -90 < ang_deg <= 90:
            ang_deg += 180 if ang_deg <= -90 else -180
        s.add(MathLabel(word, (float(p2[0]), float(p2[1])), ha="center",
                        va="center", angle_deg=float(ang_deg),
                        role=Role.ACCENT1, size_pt=9.5))
    return s


def assertions(g):
    N, P, phat = g["N"], g["P"], g["phat"]
    cam = g["cam"]
    _, _, toward = cam.axes()

    # p-hat and q-hat lie on the sphere, on their rays, and invert to p, q
    for A, ahat in ((P, phat), (g["Q"], g["qhat"])):
        assert abs(np.linalg.norm(ahat) - 1.0) < 1e-9, "image not on the sphere"
        assert np.linalg.norm(np.cross(ahat - N, A - N)) < 1e-9, "image off the ray"
        assert np.linalg.norm(A[:2]) > 1.0 and ahat[2] > 0.0
        back = ahat[:2] / (1.0 - ahat[2])
        assert np.linalg.norm(back - A[:2]) < 1e-9, "not the stereographic image"
    # both piercing points face the camera, or their rods read as hidden
    assert phat @ toward > 0 and g["qhat"] @ toward > 0

    # the line stays outside the unit circle, so its image stays northern
    assert g["line_dist"] > 1.0
    assert g["circle_img"][:, 2].min() > 0.0

    # lines map to circles through N: every pointwise-projected image of
    # the line is coplanar with N (the plane spanned by N and the line)
    d3 = np.array([*g["line_d2"], 0.0])
    nrm = np.cross(P - N, d3)
    nrm /= np.linalg.norm(nrm)
    assert np.abs((g["circle_img"] - N) @ nrm).max() < 1e-9, \
        "line image is not the circle through N"
    on_sphere = np.abs(np.linalg.norm(g["circle_img"], axis=1) - 1.0)
    assert on_sphere.max() < 1e-9

    # the sheet's split edge lands exactly on the silhouette: the equator
    # tangency points project onto the limb circle to float precision
    for T in g["tangencies"]:
        (t2,), _ = project(T[None, :], cam)
        assert abs(np.linalg.norm(t2) - 1.0) < 1e-9, \
            "sheet split edge off the silhouette"
    # and the cap sliver's arc is on the limb too
    r_arc = np.linalg.norm(g["cap_arc"], axis=1)
    assert np.abs(r_arc - 1.0).max() < 1e-9

    # the drawn equator split matches the exact dot-product visibility test
    for visible, run in g["eq_runs"]:
        signed = (run - g["sph"].center) @ toward
        if visible:
            assert signed.min() > -1e-9, "visible equator arc dips behind the limb"
        else:
            assert signed.max() < 1e-9, "hidden equator arc pokes past the limb"
