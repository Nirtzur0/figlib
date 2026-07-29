"""VCA Fig [4] (p.58): the mapping z -> z^n on a polar grid.

Modern twist over Needham's grayscale: rays are hue-coded by angle, so
the reader can SEE which ray went where — the tripling of angles is
carried by color correspondence, not just by the theta/n-theta labels.
"""

import numpy as np

from figlib.format import WIDE
from figlib.scene import AngleMark, Curve, MathLabel, Point, Scene, Vector
from figlib.style import Role
from figlib.theme import RISO

CLAIM = (
    "The mapping z -> z^3 sends each ray at angle theta to the ray at angle "
    "3*theta and each circular arc of radius r to the arc of radius r^3, so a "
    "polar-grid sector opens out into a fan three times as wide while every "
    "grid intersection stays a right angle."
)

EXPOSITION = """
Chapter 2, "Complex Functions as Transformations," section II.1
("Positive Integer Powers"), is where Needham first cashes in the
book's central habit — picture a function as a map that carries the
whole plane somewhere, not as a formula evaluated pointwise. Before he
has named the derivative or conformality, he needs the reader to
already trust two structural facts about z -> z^n: radial lines survive
as radial lines, only rotated by a multiple of their original angle,
and circles survive as circles, only rescaled in radius. Both facts
follow at once from writing z in polar form, z = r e^{i theta}, so that
z^n = r^n e^{in theta} — but stated that way they read as an algebraic
identity, not as a claim about shape. Needham's own figure at this
point is exactly this: the polar grid under z -> z^n, because a grid is
the cheapest object whose distortion makes "angles multiply, radii
don't" into something the eye checks.

This figure reproduces that grid with one modern addition: hue tags
each ray by its angle, so a ray can be tracked to its image by color
rather than by re-deriving n*theta, and the right angle at every grid
crossing survives the map — conformality, a fact Needham has not yet
named, already sitting in the picture.
"""

THEME = RISO

FORMAT = WIDE

PARAMS = {
    "n": 3,
    "theta_max_deg": 72.0,
    "n_rays": 19,
    "radii": [0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 1.0, 1.1, 1.25],
    "samples": 160,
    "w_origin": 3.9,   # where the image panel sits
}

# theta is BOUNDED and monotone here (0 -> theta_max), so it is an ORDERED
# quantity and rides the theme's own ramp. A hand-rolled viridis table used to
# live here, with its top four stops darkened by hand because stock viridis
# runs to #dbc932 (1.69:1 on white). That table was a worse reimplementation
# of theme.ramp: it silently stopped retheming, and the moment the figure got
# a cream paper variant the contrast gate caught two of its stops below the
# 3:1 floor. The theme owns ink-vs-paper contrast; a figure must not.
RAMP_LO = 0.06


def ray_color(i: int, n: int):
    """Hue encodes theta, through the theme's ordered ramp."""
    t = i / max(n - 1, 1)
    return THEME.ramp(RAMP_LO + (1.0 - RAMP_LO) * t)


def arc_gray(r: float, R: float):
    """Lightness encodes radius, dark = large r — the r -> r^n bunching must
    be readable, so the arcs cannot be uniform faint ink."""
    return THEME.surface_shade(1.0 - 0.75 * (r / R))


def compute(p):
    n = p["n"]
    th_max = np.deg2rad(p["theta_max_deg"])
    R = max(p["radii"])
    thetas = np.linspace(0.0, th_max, p["n_rays"])
    rr = np.linspace(0.02, R, p["samples"])
    aa = np.linspace(0.0, th_max, p["samples"])

    rays = [rr[:, None] * np.array([np.cos(t), np.sin(t)]) for t in thetas]
    arcs = [r * np.column_stack([np.cos(aa), np.sin(aa)]) for r in p["radii"]]

    def fmap(pts):
        z = (pts[:, 0] + 1j * pts[:, 1]) ** n
        return np.column_stack([z.real, z.imag])

    return {
        "n": n, "thetas": thetas, "radii": np.array(p["radii"]), "R": R,
        "th_max": th_max,
        "rays": rays, "arcs": arcs,
        "rays_w": [fmap(r) for r in rays],
        "arcs_w": [fmap(a) for a in arcs],
    }


def build(g):
    n = g["n"]
    W = PARAMS["w_origin"]
    R = g["R"]
    Rn = R ** n
    s = Scene()

    def shift(pts):
        out = pts.copy()
        out[:, 0] += W
        return out

    # --- domain panel (z-plane) ---
    s.add(Vector((-0.25, 0.0), (R + 0.42, 0.0), role=Role.MUTED, width_scale=0.8))
    s.add(Vector((0.0, -0.18), (0.0, R + 0.12), role=Role.MUTED, width_scale=0.8))
    for r, arc in zip(g["radii"][:-1], g["arcs"][:-1]):
        if abs(r - 1.0) < 1e-9:
            s.add(Curve(arc, role=Role.CONSTRUCTION, width_scale=1.0))
        else:
            s.add(Curve(arc, role=Role.MUTED, width_scale=0.85, color=arc_gray(r, R)))
    s.add(Curve(g["arcs"][-1], role=Role.CONTENT, width_scale=1.2))  # outer arc bold
    for i, ray in enumerate(g["rays"]):
        s.add(Curve(ray, role=Role.CONTENT, width_scale=0.8, color=ray_color(i, len(g["rays"]))))
    s.add(AngleMark((0, 0), (1, 0), (np.cos(g["th_max"]), np.sin(g["th_max"])), radius=0.17))
    s.add(MathLabel(r"\theta", (0.255 * np.cos(g["th_max"] / 2), 0.255 * np.sin(g["th_max"] / 2)),
                    ha="left", va="center", size_pt=10, role=Role.ANNOTATION,
                    offset_px=(0, 30)))
    s.add(Point((1.0, 0.0), role=Role.CONTENT, radius_scale=0.7))
    s.add(MathLabel(r"1", (1.0, 0.0), ha="center", va="top", offset_px=(0, 8)))
    s.add(MathLabel(r"r", (R, 0.0), ha="center", va="top", offset_px=(6, 8)))
    s.add(MathLabel(r"0", (0.0, 0.0), ha="right", va="top", offset_px=(-5, 6)))

    # --- mapping arrow ---
    ax0, ax1, ay = R + 0.35, R + 1.15, 1.62
    s.add(Vector((ax0, ay), (ax1, ay), role=Role.ANNOTATION))
    s.add(MathLabel(r"z \mapsto z^{n}", ((ax0 + ax1) / 2, ay),
                    ha="center", va="bottom", offset_px=(0, -8), size_pt=13))
    s.add(MathLabel(rf"(\text{{here }} n = {n})", ((ax0 + ax1) / 2, ay),
                    ha="center", va="top", offset_px=(0, 8), size_pt=9, role=Role.ANNOTATION))

    # --- image panel (w-plane) ---
    s.add(Vector((W - Rn - 0.35, 0.0), (W + Rn + 0.45, 0.0), role=Role.MUTED, width_scale=0.8))
    s.add(Vector((W, -0.35), (W, Rn + 0.25), role=Role.MUTED, width_scale=0.8))
    for r, arc in zip(g["radii"][:-1], g["arcs_w"][:-1]):
        if abs(r - 1.0) < 1e-9:
            s.add(Curve(shift(arc), role=Role.CONSTRUCTION, width_scale=1.0))
        else:
            s.add(Curve(shift(arc), role=Role.MUTED, width_scale=0.85, color=arc_gray(r, R)))
    s.add(Curve(shift(g["arcs_w"][-1]), role=Role.CONTENT, width_scale=1.2))
    for i, ray in enumerate(g["rays_w"]):
        s.add(Curve(shift(ray), role=Role.CONTENT, width_scale=0.8, color=ray_color(i, len(g["rays_w"]))))
    th_img = n * g["th_max"]
    s.add(MathLabel(r"n\theta", (W + (Rn + 0.16) * np.cos(th_img / 2), (Rn + 0.16) * np.sin(th_img / 2)),
                    ha="center", va="center", role=Role.ANNOTATION))
    s.add(Point((W + 1.0, 0.0), role=Role.CONTENT, radius_scale=0.7))
    s.add(MathLabel(r"1", (W + 1.0, 0.0), ha="center", va="top", offset_px=(0, 8)))
    s.add(MathLabel(r"r^{n}", (W + Rn, 0.0), ha="center", va="top", offset_px=(8, 8)))
    s.add(MathLabel(r"0", (W, 0.0), ha="right", va="top", offset_px=(-5, 6)))

    return s


def assertions(g):
    n = g["n"]
    # image rays have constant angle exactly n*theta
    for th, ray_w in zip(g["thetas"], g["rays_w"]):
        ang = np.arctan2(ray_w[:, 1], ray_w[:, 0])
        ang = np.unwrap(ang)
        target = ((n * th + np.pi) % (2 * np.pi)) - np.pi
        dev = np.abs(((ang - target + np.pi) % (2 * np.pi)) - np.pi)
        assert dev.max() < 1e-9, f"image of ray {np.rad2deg(th):.0f}deg is not a ray at n*theta"
    # image arcs have constant radius exactly r^n
    for r, arc_w in zip(g["radii"], g["arcs_w"]):
        rad = np.hypot(arc_w[:, 0], arc_w[:, 1])
        assert np.max(np.abs(rad - r ** n)) < 1e-9, f"image of arc r={r} is not radius r^n"
    # conformality: mapped rays meet mapped arcs at right angles.
    # tangents via finite differences at matched parameter points.
    ray = g["rays"][4]
    z = ray[:, 0] + 1j * ray[:, 1]
    w = z ** n
    tang_ray = np.diff(w)
    # arc through each sample point of that ray: tangent i*z * (dw/dz) direction
    dwdz = n * z[:-1] ** (n - 1)
    tang_arc = 1j * z[:-1] * dwdz
    cosang = np.real(tang_ray * np.conj(tang_arc)) / (np.abs(tang_ray) * np.abs(tang_arc))
    assert np.max(np.abs(cosang)) < 1e-3, "grid intersections not orthogonal after mapping"
