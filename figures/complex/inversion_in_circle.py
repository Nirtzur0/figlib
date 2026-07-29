"""Inversion in a circle: z -> R^2 / z-bar, as a Needham panel pair.

CLAIM lives in the correspondence: circles through the center of inversion
straighten into lines, circles that miss the center stay circles, the
inversion circle itself is fixed pointwise, and a pair of circles crossing
at right angles crosses at right angles again after the map — even though
the map reverses the sense of every angle it preserves.
"""

import numpy as np

from figlib.correspond import Correspondence
from figlib.figure import Connector, Figure, Panel
from figlib.format import WIDE
from figlib.layout import geometry_extents
from figlib.scene import Callout, Curve, MathLabel, Point, RightAngleMark, Scene
from figlib.style import Role
from figlib.theme import RISO

CLAIM = (
    "Inversion z -> R^2/z-bar sends circles through the center of inversion "
    "to lines, circles that miss the center to circles, fixes the circle "
    "|z| = R pointwise, and preserves the angle at which two curves cross in "
    "magnitude while reversing its sense -- so an orthogonal pair of circles "
    "stays orthogonal, a right angle mapping to a right angle of the "
    "opposite hand."
)

EXPOSITION = """
Needham introduces inversion in a circle (Visual Complex Analysis, ch. 3)
as the geometric ancestor of the Mobius transformations: not z -> R^2/z,
which is holomorphic and orientation-preserving, but z -> R^2/z-bar, built
from conjugation -- reflection in the real axis -- composed with the
holomorphic inversion. Conjugation reverses orientation and leaves
Mobius geometry untouched, so the composite is anticonformal: it still
preserves the magnitude of every angle between crossing curves, but flips
its sign. Needham's punchline is that this single map already carries
the whole circle-preserving machinery of Mobius transformations, in its
starkest form: every circle or line (a "circline") maps to a circline,
the circle of inversion |z| = R survives as the fixed locus, and nothing
outside that locus is fixed -- points swap the inside of the circle for
the outside along each ray. The figure asks for the one picture that
makes every clause checkable at once: a family of circles through the
center becoming lines, a family that misses the center staying round, and
a single orthogonal crossing surviving the map with its right angle
intact but its handedness reversed.
"""

THEME = RISO
FORMAT = WIDE

PARAMS = {
    "R": 1.0,
    # a pencil of circles through the center of inversion: (radius, center
    # direction in degrees). Fanned across directions, not stacked on one
    # ray, so the family reads as "circles through a point" rather than a
    # pile that self-occludes the one crossing the figure tracks.
    "through_origin": [(0.55, 0.0), (0.42, 125.0), (0.5, -115.0)],
    "orth_a_index": 0,        # which through-origin circle carries the demo
    # radius is DERIVED from orthogonality; this center puts the crossing at
    # (0.449, 0.541), clear of the inversion circle and every generic family
    # member by the widest margin found by a grid search (0.30 math units)
    "orth_b_center": (1.62, 0.76),
    # circles that miss the center: (center, radius)
    "offaxis_circles": [((0.3, -0.95), 0.42)],
    "samples": 480,
    "eps_phi": 0.02,           # rad, singularity margin at the center
    "image_line_max": 2.3,     # display window: keep the line images finite
    "fit_tol": 1e-6,           # circle-fit residual budget
    "line_tol": 1e-8,          # closed-form line-projection residual budget
    "angle_tol_deg": 0.5,      # finite-difference tangent slack
}


# ---------------------------------------------------------------- geometry

def _full_circle(center, radius, n, phi=None):
    if phi is None:
        phi = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    pts = np.column_stack([center[0] + radius * np.cos(phi),
                            center[1] + radius * np.sin(phi)])
    return phi, pts


def _through_origin_phi(n, eps, theta0=0.0):
    """Angle grid for a circle through the origin with center at direction
    theta0: the long way around, excluding a margin at the one sample that
    would land ON 0 (phi = theta0 + pi), where 1/z-bar is singular."""
    singular = theta0 + np.pi
    psi = np.linspace(eps, 2 * np.pi - eps, n)
    return (singular + psi) % (2 * np.pi)


def _invert(pts, R):
    z = pts[:, 0] + 1j * pts[:, 1]
    w = R ** 2 / np.conj(z)
    return np.column_stack([w.real, w.imag])


def _fit_circle(pts):
    """Kasa algebraic fit: x^2+y^2 = D x + E y + F, linear in D,E,F."""
    x, y = pts[:, 0], pts[:, 1]
    A = np.column_stack([x, y, np.ones_like(x)])
    b = x ** 2 + y ** 2
    D, E, F = np.linalg.lstsq(A, b, rcond=None)[0]
    cx, cy = D / 2.0, E / 2.0
    r = np.sqrt(F + cx ** 2 + cy ** 2)
    resid = np.hypot(x - cx, y - cy) - r
    return (cx, cy), r, resid


def _circle_intersections(c1, r1, c2, r2):
    c1, c2 = np.asarray(c1, float), np.asarray(c2, float)
    d_vec = c2 - c1
    d = np.hypot(*d_vec)
    a_ = (r1 ** 2 - r2 ** 2 + d ** 2) / (2 * d)
    h = np.sqrt(max(r1 ** 2 - a_ ** 2, 0.0))
    p2 = c1 + a_ * d_vec / d
    perp = np.array([-d_vec[1], d_vec[0]]) / d
    return p2 + h * perp, p2 - h * perp


def _nearest_idx(pts, target):
    d = np.hypot(pts[:, 0] - target[0], pts[:, 1] - target[1])
    return int(np.argmin(d))


def _tangent_at(pts, idx):
    n = len(pts)
    v = pts[(idx + 1) % n] - pts[(idx - 1) % n]
    return v / np.linalg.norm(v)


def _signed_angle_deg(u, v):
    cross = u[0] * v[1] - u[1] * v[0]
    dot = u[0] * v[0] + u[1] * v[1]
    return np.degrees(np.arctan2(cross, dot))


def compute(p):
    R, N, eps = p["R"], p["samples"], p["eps_phi"]
    line_max = p["image_line_max"]

    # inversion circle: literally fixed, so drawn from one array in both panels
    _, inv_pts = _full_circle((0.0, 0.0), R, N)
    inv_img = _invert(inv_pts, R)

    # through-origin pencil: each maps to a line perpendicular to the ray
    # from 0 through its center (the tangent at 0 carries straight through);
    # masked to a finite window along that line, plus the closed-form
    # signed distance the image line must sit at, R^2/(2*radius).
    lines = []
    for radius, angle_deg in p["through_origin"]:
        theta0 = np.radians(angle_deg)
        uc = np.array([np.cos(theta0), np.sin(theta0)])       # normal to the line
        tangent_dir = np.array([-uc[1], uc[0]])               # along the line
        center = radius * uc
        phi = _through_origin_phi(N, eps, theta0)
        _, pts = _full_circle(center, radius, N, phi=phi)
        img = _invert(pts, R)
        along = img @ tangent_dir
        mask = np.abs(along) <= line_max
        lines.append({
            "radius": radius, "angle_deg": angle_deg, "center": center,
            "uc": uc, "tangent_dir": tangent_dir, "phi": phi,
            "pts": pts, "img": img, "img_kept": img[mask],
            "proj0": R ** 2 / (2 * radius),
        })
    orth_a = lines[p["orth_a_index"]]

    # off-axis family (misses the center): full circle -> circle image
    offaxis = []
    for center, radius in p["offaxis_circles"]:
        _, pts = _full_circle(center, radius, N)
        img = _invert(pts, R)
        center_fit, r_fit, resid = _fit_circle(img)
        offaxis.append({"center": center, "radius": radius, "pts": pts,
                        "img": img, "center_fit": center_fit, "r_fit": r_fit,
                        "resid": resid})

    # the accent second circle of the orthogonal pair: also misses the
    # center, radius DERIVED from the orthogonality condition
    # |c1 - c2|^2 = r1^2 + r2^2 against orth_a.
    ca, ra = orth_a["center"], orth_a["radius"]
    cb = p["orth_b_center"]
    d2 = (cb[0] - ca[0]) ** 2 + (cb[1] - ca[1]) ** 2
    rb = np.sqrt(d2 - ra ** 2)
    _, orth_b_pts = _full_circle(cb, rb, N)
    orth_b_img = _invert(orth_b_pts, R)
    ob_center_fit, ob_r_fit, ob_resid = _fit_circle(orth_b_img)

    # the two circles cross at two points; take the upper one
    pA, pB = _circle_intersections(ca, ra, cb, rb)
    p_int = pA if pA[1] >= pB[1] else pB
    p_int_img = _invert(p_int[None, :], R)[0]

    # tangents at the crossing, before and after, from the SAME sample
    # index in both curves (finite difference on the parametrized arrays;
    # the image inherits its ordering from the domain, so index i is the
    # same physical point on both sides of the map)
    orth_a_full_phi = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
    _, orth_a_full = _full_circle(ca, ra, N, phi=orth_a_full_phi)
    orth_a_full_img = _invert(orth_a_full, R)
    idx_a = _nearest_idx(orth_a_full, p_int)
    idx_b = _nearest_idx(orth_b_pts, p_int)
    tan_a = _tangent_at(orth_a_full, idx_a)
    tan_b = _tangent_at(orth_b_pts, idx_b)
    tan_a_img = _tangent_at(orth_a_full_img, idx_a)
    tan_b_img = _tangent_at(orth_b_img, idx_b)
    ang_domain = _signed_angle_deg(tan_a, tan_b)
    ang_image = _signed_angle_deg(tan_a_img, tan_b_img)

    def _outward(u, v):
        n = -(u + v)
        norm = np.linalg.norm(n)
        return n / norm if norm > 1e-9 else np.array([u[1], -u[0]])

    out_domain = _outward(tan_a, tan_b)
    out_image = _outward(tan_a_img, tan_b_img)

    return {
        "R": R, "inv_pts": inv_pts, "inv_img": inv_img,
        "lines": lines, "orth_a": orth_a, "offaxis": offaxis,
        "ca": np.array(ca), "ra": ra, "cb": np.array(cb), "rb": rb,
        "orth_b_pts": orth_b_pts, "orth_b_img": orth_b_img,
        "ob_center_fit": ob_center_fit, "ob_r_fit": ob_r_fit, "ob_resid": ob_resid,
        "p_int": p_int, "p_int_img": p_int_img,
        "tan_a": tan_a, "tan_b": tan_b,
        "tan_a_img": tan_a_img, "tan_b_img": tan_b_img,
        "ang_domain": ang_domain, "ang_image": ang_image,
        "out_domain": out_domain, "out_image": out_image,
    }


# ------------------------------------------------------------------- scene

def _panel(g, mapped: bool) -> Scene:
    s = Scene()
    R = g["R"]

    # the fixed locus: same array both panels -> zero residual by the map,
    # not by reuse -- inv_img is invert(inv_pts), asserted equal in compute
    s.add(Curve(g["inv_pts"], closed=True, role=Role.CONSTRUCTION,
                dash="dashed", width_scale=0.9, key="inversion-circle"))
    inv_tag_target = (R * np.cos(np.radians(205)), R * np.sin(np.radians(205)))
    s.add(Callout(r"|z| = R", anchor=(-0.8 * R, -1.05 * R), target=inv_tag_target,
                 size_pt=10, role=Role.ANNOTATION))
    s.add(Point((R, 0.0), role=Role.CONTENT, radius_scale=0.7, key="fixed-1"))
    s.add(MathLabel(r"R", (R, 0.0), ha="left", va="center", offset_px=(9, 0)))
    if not mapped:
        # 0 has no image -- it is the one point inversion cannot fix in
        # place, so it is drawn only as the domain's center of inversion,
        # not repeated (falsely) as a surviving point in the image panel
        s.add(Point((0.0, 0.0), role=Role.ANNOTATION, radius_scale=0.6))
        s.add(MathLabel(r"0", (0.0, 0.0), ha="right", va="top",
                        offset_px=(-6, 8), role=Role.ANNOTATION))

    # generic through-origin family -> lines: plain CONTENT, no correspondence
    for L in g["lines"]:
        if L is g["orth_a"]:
            continue
        pts = L["img_kept"] if mapped else L["pts"]
        s.add(Curve(pts, closed=not mapped, role=Role.CONTENT, width_scale=0.85))

    # generic off-axis family -> circles: plain CONTENT
    for C in g["offaxis"]:
        pts = C["img"] if mapped else C["pts"]
        s.add(Curve(pts, closed=True, role=Role.CONTENT, width_scale=0.85))

    # tracked orthogonal pair: through-origin member (ACCENT1) and
    # off-axis member (ACCENT2), correspondence-keyed across the panels
    a_pts = g["orth_a"]["img_kept"] if mapped else g["orth_a"]["pts"]
    s.add(Curve(a_pts, closed=not mapped, role=Role.ACCENT1, width_scale=1.3,
                key="orth-a"))
    b_pts = g["orth_b_img"] if mapped else g["orth_b_pts"]
    s.add(Curve(b_pts, closed=True, role=Role.ACCENT2, width_scale=1.3,
                key="orth-b"))

    p_int = g["p_int_img"] if mapped else g["p_int"]
    tan_a = g["tan_a_img"] if mapped else g["tan_a"]
    tan_b = g["tan_b_img"] if mapped else g["tan_b"]
    out = g["out_image"] if mapped else g["out_domain"]
    s.add(Point(tuple(p_int), role=Role.CONTENT, radius_scale=0.8, key="orth-point"))
    corner_size = 0.16 * g["orth_a"]["radius"]
    s.add(RightAngleMark(tuple(p_int), tuple(tan_a), tuple(tan_b),
                        size=corner_size, role=Role.ANNOTATION))
    # the corner mark sits where two families cross -- busy ink -- so the
    # reading of the angle is a Callout with a leader into clear space
    # along the outward bisector of the two tangents, not a label pinned
    # to the crossing itself.
    lead = 5.2 * corner_size if not mapped else 3.4 * corner_size
    anchor = tuple(np.array(p_int) + lead * out)
    lab = r"90^\circ_{-}" if mapped else r"90^\circ_{+}"
    s.add(Callout(lab, anchor=anchor, target=tuple(p_int), size_pt=10,
                 role=Role.ANNOTATION, key="orth-corner"))
    return s


CORRESPONDENCE = [
    Correspondence(
        parts=(0, 1),
        varies="the inversion z -> R^2/z-bar applied to every drawn circle",
        # orth-a, orth-b, orth-point move (a circle straightens, the other's
        # radius/center shift, the crossing translates) but that is a
        # position change, deliberately outside the fingerprint -- so they
        # are bound (same object, same role/color/dash both sides) without
        # being listed here. orth-corner is the one keyed group whose LABEL
        # TEXT actually differs ("90deg" vs "90deg (reversed)"), which is
        # the checkable trace of the claim.
        changes=("orth-corner",),
    ),
]


def build(g):
    a, b = _panel(g, mapped=False), _panel(g, mapped=True)
    dx_a = np.ptp(geometry_extents(a)[0])
    dx_b = np.ptp(geometry_extents(b)[0])
    frac_b = dx_b / (dx_a + dx_b)
    return Figure(
        panels=[Panel(a, tag=r"[a]", width_frac=1.0 - frac_b),
                Panel(b, tag=r"[b]", width_frac=frac_b)],
        connectors=[Connector(0, 1, kind="map", label=r"z \mapsto R^2/\bar z",
                             label_below=r"0 \mapsto \infty")],
    )


def assertions(g):
    R = g["R"]

    # the inversion circle is fixed pointwise by the actual map, not by
    # array reuse
    assert np.max(np.abs(g["inv_img"] - g["inv_pts"])) < 1e-9, \
        "the circle |z|=R is not fixed pointwise by z -> R^2/z-bar"

    # every through-origin circle's (finite window of its) image sits on
    # the predicted line: signed distance R^2/(2*radius) along the ray
    # through its center, direction uc -- the general form of Re(w)=R^2/(2a)
    for L in g["lines"]:
        tag = f"radius={L['radius']}, angle={L['angle_deg']}deg"
        assert len(L["img_kept"]) > 20, \
            f"{tag}: display window kept too few points to read as a line"
        dev = np.max(np.abs(L["img_kept"] @ L["uc"] - L["proj0"]))
        assert dev < PARAMS["line_tol"], \
            f"{tag}: image is not the line at distance {L['proj0']:.4f} (max dev {dev:.2e})"

    # every off-axis circle's image really is a circle: fitted-center
    # residual is tiny, not just visually round
    for C in g["offaxis"] + [{"resid": g["ob_resid"], "r_fit": g["ob_r_fit"],
                             "center": "orth-b"}]:
        resid = C["resid"]
        rel = np.max(np.abs(resid)) / C["r_fit"]
        assert rel < PARAMS["fit_tol"], \
            f"image of circle at {C['center']} is not a circle: relative fit residual {rel:.2e}"

    # the marked intersection point really lies on both tracked circles
    assert abs(np.hypot(*(g["p_int"] - g["ca"])) - g["ra"]) < 1e-6
    assert abs(np.hypot(*(g["p_int"] - g["cb"])) - g["rb"]) < 1e-6

    # the tracked pair really crosses at 90 degrees before the map...
    assert abs(abs(g["ang_domain"]) - 90.0) < PARAMS["angle_tol_deg"], \
        f"tracked circles do not cross at 90deg in the domain ({g['ang_domain']:.2f}deg)"
    # ...and at 90 degrees again after it...
    assert abs(abs(g["ang_image"]) - 90.0) < PARAMS["angle_tol_deg"], \
        f"tracked images do not cross at 90deg ({g['ang_image']:.2f}deg)"
    # ...but the map reverses the sense in which the angle is swept
    assert np.sign(g["ang_domain"]) != np.sign(g["ang_image"]), \
        "the map did not reverse the orientation of the crossing angle"
