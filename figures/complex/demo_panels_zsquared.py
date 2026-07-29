"""z -> z^2 on a polar sector, as a Needham panel pair.

The correspondence channel does the arguing: the ACCENT1 ray and the
ACCENT2 circle keep their hues across the map, so the reader sees the
ray land at twice the angle and the circle contract to the squared
radius, while the dashed unit circle stays put.
"""

import numpy as np

from figlib.correspond import Correspondence
from figlib.figure import Connector, Figure, Panel
from figlib.format import WIDE
from figlib.layout import geometry_extents
from figlib.scene import AngleMark, Curve, MathLabel, Point, Scene
from figlib.style import Role
from figlib.theme import RISO

CLAIM = (
    "The mapping z -> z^2 doubles every angle and squares every modulus: a "
    "polar sector of opening theta_max maps to one of opening 2*theta_max, the "
    "ray at theta_0 lands on the ray at 2*theta_0, the circle of radius "
    "r_0 < 1 contracts to radius r_0^2, and the unit circle maps to itself."
)

EXPOSITION = """
Needham introduces z -> z^n (VCA ch. 2, "Polynomials," section "Positive
Integer Powers," Fig [4]) as the first mapping the reader is asked to
picture rather than compute: writing z = r e^{i theta}, w = r^n e^{in
theta}, so a ray at angle theta lands on the ray at n*theta and a circle
of radius r lands on the circle of radius r^n. Needham's own figure runs
n = 3 on a handful of rays and arcs and leaves the general case to the
reader; this pair specializes to n = 2 and asks the domain and codomain
panels to carry the argument by correspondence rather than assertion --
the same ray, the same circle, tracked by hue across both panels, landing
exactly where the formula says. The unit circle is drawn as the one curve
that does not move, because r = r^2 only at r = 1: it is the fixed set of
the family, and the panel widths are set so it renders at the same size
on both sides, so "maps to itself" is something the eye checks rather
than something the caption asserts.
"""

THEME = RISO
FORMAT = WIDE

PARAMS = {
    "theta_max_deg": 70.0,
    "theta0_deg": 45.0,        # the highlighted ray; image is the vertical ray
    "r0": 0.75,                # the highlighted circle; image radius 0.5625
    "radii": [0.35, 0.55, 0.75, 1.0, 1.2],
    "n_rays": 8,
    "samples": 240,
}


def compute(p):
    th_max = np.deg2rad(p["theta_max_deg"])
    th0 = np.deg2rad(p["theta0_deg"])
    thetas = np.linspace(0.0, th_max, p["n_rays"])
    rr = np.linspace(0.04, max(p["radii"]), p["samples"])
    aa = np.linspace(0.0, th_max, p["samples"])

    def polar(r, th):
        return np.column_stack([r * np.cos(th), r * np.sin(th)])

    def fmap(pts):
        z = (pts[:, 0] + 1j * pts[:, 1]) ** 2
        return np.column_stack([z.real, z.imag])

    rays = [polar(rr, t) for t in thetas]
    arcs = [polar(r, aa) for r in p["radii"]]
    ray0 = polar(rr, th0)
    circ0 = polar(p["r0"], aa)
    g = {
        "th_max": th_max, "th0": th0, "r0": p["r0"],
        "radii": np.array(p["radii"]), "R": max(p["radii"]),
        "thetas": thetas,
        "rays": rays, "arcs": arcs, "ray0": ray0, "circ0": circ0,
        "rays_w": [fmap(r) for r in rays], "arcs_w": [fmap(a) for a in arcs],
        "ray0_w": fmap(ray0), "circ0_w": fmap(circ0),
        "z0": np.array([p["r0"] * np.cos(th0), p["r0"] * np.sin(th0)]),
    }
    g["z0_w"] = fmap(g["z0"][None, :])[0]
    return g


def _panel(g, mapped: bool) -> Scene:
    """Domain (mapped=False) or codomain panel — same grammar both sides."""
    rays = g["rays_w"] if mapped else g["rays"]
    arcs = g["arcs_w"] if mapped else g["arcs"]
    ray0 = g["ray0_w"] if mapped else g["ray0"]
    circ0 = g["circ0_w"] if mapped else g["circ0"]
    z0 = g["z0_w"] if mapped else g["z0"]
    th_open = 2 * g["th_max"] if mapped else g["th_max"]
    r_out = g["R"] ** 2 if mapped else g["R"]
    s = Scene()

    for ray in rays[1:-1]:
        s.add(Curve(ray, role=Role.MUTED, width_scale=0.8))
    for r, arc in zip(g["radii"], arcs):
        if abs(r - 1.0) < 1e-9:
            # the fixed unit circle: r = r^2 only here, so it is the one
            # object the map leaves alone — the anchor of the binding
            s.add(Curve(arc, role=Role.CONSTRUCTION, key="unit-circle"))
        elif abs(r - g["r0"]) > 1e-9:
            s.add(Curve(arc, role=Role.MUTED, width_scale=0.8))
    # sector boundary: the two edge rays and the outer arc
    s.add(Curve(rays[0], role=Role.CONTENT),
          Curve(rays[-1], role=Role.CONTENT),
          Curve(arcs[-1], role=Role.CONTENT, width_scale=1.15))
    # the tracked objects, hue-matched across panels, oriented; marker
    # fractions chosen clear of the marked point z_0
    s.add(Curve(ray0, role=Role.ACCENT1, width_scale=1.2, key="tracked-ray",
                arrows=(0.78,) if mapped else (0.45,)))
    # 0.30: the arrow-on-mark gate found 0.28 landing on the '2\theta_0' label
    s.add(Curve(circ0, role=Role.ACCENT2, width_scale=1.2, arrows=(0.30,),
                key="tracked-circle"))
    s.add(Point(tuple(z0), role=Role.CONTENT, radius_scale=0.9, key="z0"))
    s.add(Point((1.0, 0.0), role=Role.CONTENT, radius_scale=0.7, key="fixed-1"))

    th0 = 2 * g["th0"] if mapped else g["th0"]
    s.add(AngleMark((0, 0), (1, 0), (np.cos(th0), np.sin(th0)),
                    radius=0.16 * r_out, key="tracked-ray"))
    lab = r"2\theta_0" if mapped else r"\theta_0"
    lr = 0.24 * r_out
    s.add(MathLabel(lab, (lr * np.cos(th0 / 2), lr * np.sin(th0 / 2)),
                    ha="left", va="center", size_pt=10, role=Role.ANNOTATION,
                    key="tracked-ray"))
    s.add(MathLabel(r"z_0^2" if mapped else r"z_0", tuple(z0),
                    ha="left", va="bottom", offset_px=(5, -5), key="z0"))
    # haloed: the radius label belongs ON the axis at the circle's foot
    # (contiguity), and at the shared page scale that spot carries grid ink
    s.add(MathLabel(r"r_0^2" if mapped else r"r_0", (circ0[0, 0], 0.0),
                    ha="center", va="top",
                    offset_px=(13, -28) if mapped else (0, 8),
                    role=Role.ACCENT2, halo=True, key="tracked-circle"))
    s.add(MathLabel(r"1", (1.0, 0.0), ha="center", va="top", offset_px=(8, 8),
                    key="fixed-1"))
    return s


CORRESPONDENCE = [
    Correspondence(
        parts=(0, 1),
        varies="the map z -> z^2 applied to every drawn object",
        # what the map moves: the tracked ray doubles its angle, the tracked
        # circle squares its radius, z_0 goes to z_0^2. The unit circle and
        # the point 1 are NOT here — they are the fixed set, and the figure
        # fails if they ever stop matching.
        changes=("tracked-ray", "tracked-circle", "z0"),
    ),
]


def build(g):
    # Slot widths are DERIVED so the two panels land on one page scale.
    # At the hand-picked 0.44/0.56 the unit circle — the one object z^2
    # fixes — was drawn at 302 px in [a] and 181 px in [b]: the figure's
    # own invariant, visibly changing size. Splitting the row in the ratio
    # of the panels' extents puts both at 220 px, so "maps to itself" is
    # something the reader can see rather than take on trust.
    a, b = _panel(g, mapped=False), _panel(g, mapped=True)
    dx_a = np.ptp(geometry_extents(a)[0])
    dx_b = np.ptp(geometry_extents(b)[0])
    frac_b = dx_b / (dx_a + dx_b)
    return Figure(
        panels=[Panel(a, tag=r"[a]", width_frac=1.0 - frac_b),
                Panel(b, tag=r"[b]", width_frac=frac_b)],
        connectors=[Connector(0, 1, kind="map", label=r"z^{2}")],
    )


def assertions(g):
    # every mapped ray is a ray at exactly twice the angle
    for th, ray_w in zip(g["thetas"], g["rays_w"]):
        ang = np.arctan2(ray_w[:, 1], ray_w[:, 0])
        dev = np.abs(((ang - 2 * th + np.pi) % (2 * np.pi)) - np.pi)
        assert dev.max() < 1e-9, f"image of ray {np.rad2deg(th):.0f}deg not at 2*theta"
    # every mapped circle has radius exactly r^2; the unit circle is fixed
    for r, arc_w in zip(g["radii"], g["arcs_w"]):
        rad = np.hypot(arc_w[:, 0], arc_w[:, 1])
        assert np.max(np.abs(rad - r ** 2)) < 1e-9, f"image of circle r={r} not r^2"
    assert np.max(np.abs(np.hypot(*g["circ0_w"].T) - g["r0"] ** 2)) < 1e-9
    # the highlighted ray at 45deg lands on the vertical ray
    assert np.max(np.abs(g["ray0_w"][:, 0])) < 1e-9, "image of the theta_0=45deg ray is not vertical"
    # the tracked point obeys both halves at once
    z0 = g["z0"][0] + 1j * g["z0"][1]
    w0 = g["z0_w"][0] + 1j * g["z0_w"][1]
    assert abs(w0 - z0 ** 2) < 1e-12
    assert abs(abs(w0) - g["r0"] ** 2) < 1e-12 and abs(np.angle(w0) - 2 * g["th0"]) < 1e-12
