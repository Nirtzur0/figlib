"""VCA Fig [19] (p.140): stereographic projection from the north pole.

The unit sphere Sigma sits centered on the plane C (equator = unit
circle). The ray from N through a point p of the plane pierces the
sphere at its stereographic image p-hat; p outside the unit circle
lands on the northern hemisphere, and p -> infinity drives p-hat -> N.
"""

import numpy as np

from figlib.scene import Curve, MathLabel, Point, Scene
from figlib.sphere3d import (Sphere, anchor, circle_on_sphere, circle_points,
                             disc, occlude, point_on, silhouette,
                             visibility_runs)
from figlib.style import Role
from figlib.surface3d import Camera, compose, project
from figlib.format import WIDE
from figlib.theme import RISO

THEME = RISO
FORMAT = WIDE

CLAIM = (
    "Stereographic projection from the north pole N sends a point p of the "
    "plane outside the unit circle to the point p-hat where the ray N->p "
    "pierces the northern hemisphere of the unit sphere; the equator is the "
    "unit circle itself, fixed by the projection."
)

PARAMS = {
    "p": (0.8, -2.0),         # in the plane, |p| > 1
    "azim": -40.0,
    "elev": 28.0,
    "plane_rot_deg": 18.0,    # plane quad twist vs the screen axes
    "plane_s": (2.3, 2.5),    # extents along the screen-parallel edge
    "plane_t": (1.5, 1.7),    # extents along the depth edge (far, near)
    "lat": 0.5,               # a latitude circle on the northern hemisphere
    "n": 512,
}


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


def compute(p):
    cam = Camera(p["azim"], p["elev"])
    right, up, toward = cam.axes()
    sph = Sphere(np.zeros(3), 1.0)

    px, py = p["p"]
    P = np.array([px, py, 0.0])
    N = np.array([0.0, 0.0, 1.0])
    S = np.array([0.0, 0.0, -1.0])
    # ray N -> p meets |q| = 1 at parameter t = 2 / (|p|^2 + 1)
    t = 2.0 / (px * px + py * py + 1.0)
    phat = N + t * (P - N)

    # plane quad: a parallelogram twisted plane_rot_deg from the screen axes,
    # split at the sphere into a far half (behind) and a near half (in front)
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

    equator = circle_points(sph, np.array([0.0, 0.0, 1.0]), 0.0, n=p["n"])
    eq_runs = visibility_runs(equator, sph, cam)

    ray = np.array([N - 0.10 * (P - N), N + 1.06 * (P - N)])
    meridian_normal = np.array([-phat[1], phat[0], 0.0])

    return {"cam": cam, "sph": sph, "P": P, "N": N, "S": S, "phat": phat,
            "far_poly": far_poly, "near_poly": near_poly,
            "plane_uv": (u, v), "equator": equator, "eq_runs": eq_runs,
            "ray": ray, "meridian_normal": meridian_normal,
            "lat": p["lat"], "n": p["n"]}


def _rim(g, tag: float, poly: np.ndarray):
    """The sheet's boundary edges (the artificial cut edge through the
    sphere is not stroked). The plane itself is line art, as in the book."""
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


def build(g):
    cam, sph = g["cam"], g["sph"]
    s = Scene()

    far = _rim(g, -50.0, g["far_poly"])       # hidden where the body covers it
    near = _rim(g, 0.90, g["near_poly"])
    body = disc(sph, cam, color="#c2ae8c")    # flat printed-tan body
    limb = silhouette(sph, cam, role=Role.CONTENT, width_scale=1.2)

    # the unit circle = the equator, the projection's fixed set; its front
    # arc lies IN the sheet, so it draws above the near rim
    eq = []
    for visible, run in g["eq_runs"]:
        pts2, depth = project(run, cam)
        if visible:
            eq.append((max(float(depth.mean()) + 0.04, 0.96),
                       Curve(pts2, role=Role.ACCENT1, width_scale=1.3)))
        else:
            eq.append((float(depth.mean()),
                       Curve(pts2, role=Role.ACCENT1, dash="dashed",
                             width_scale=1.1, opacity=0.85)))

    lat = circle_on_sphere(sph, np.array([0.0, 0.0, 1.0]), g["lat"], cam,
                           n=g["n"], role=Role.ANNOTATION, width_scale=0.9)
    meridian = circle_on_sphere(sph, g["meridian_normal"], 0.0, cam,
                                n=g["n"], role=Role.ANNOTATION, width_scale=0.9)
    # polar axis: dashed throughout (construction grammar), occluded anyway
    axis = occlude(np.array([g["N"], g["S"]]), sph, cam, n=g["n"],
                   role=Role.ANNOTATION, dash="dashed", width_scale=0.9)
    # the ray lives above the plane: lift it past the near sheet's tag
    ray = [(t + 1.0, c) for t, c in
           occlude(g["ray"], sph, cam, n=g["n"], role=Role.ACCENT2,
                   width_scale=1.4)]

    s.items.extend(compose(far, near, body, limb, eq, lat, meridian, axis, ray))

    for q, role in ((g["N"], Role.CONTENT), (g["phat"], Role.ACCENT2),
                    (g["P"], Role.ACCENT2)):
        s.add(Point(anchor(q, cam), role=role))
    s.add(Point(anchor(g["S"], cam), role=Role.ANNOTATION, radius_scale=0.85))

    s.add(MathLabel(r"N", anchor(g["N"], cam), ha="right", va="bottom",
                    offset_px=(-10, 0)))
    s.add(MathLabel(r"S", anchor(g["S"], cam), ha="center", va="top",
                    offset_px=(0, 30), role=Role.ANNOTATION))
    s.add(MathLabel(r"p", anchor(g["P"], cam), ha="right", va="top",
                    offset_px=(-11, 8), role=Role.ACCENT2))
    s.add(MathLabel(r"\hat{p}", anchor(g["phat"], cam), ha="right", va="bottom",
                    offset_px=(-8, -5), role=Role.ACCENT2))
    # Sigma on the upper-right body: a sphere point projecting to (.62, .55)
    sig = 0.62 * cam.axes()[0] + 0.55 * cam.axes()[1]
    sig = point_on(sph, sig + np.sqrt(1 - 0.62**2 - 0.55**2) * cam.axes()[2])
    s.add(MathLabel(r"\Sigma", anchor(sig, cam), ha="center", va="center"))
    u, v = g["plane_uv"]
    s.add(MathLabel(r"\mathbb{C}", anchor(-1.9 * u + 1.0 * v, cam),
                    ha="center", va="center", role=Role.ANNOTATION))
    # name the fixed circle on its front-right arc
    eq_front = anchor(np.array([np.cos(-0.2), np.sin(-0.2), 0.0]), cam)
    s.add(MathLabel(r"|z| = 1", eq_front, ha="left", va="top",
                    offset_px=(12, 12), role=Role.ACCENT1))
    return s


def assertions(g):
    N, P, phat = g["N"], g["P"], g["phat"]
    # p-hat lies on the sphere and on the ray N -> p
    assert abs(np.linalg.norm(phat) - 1.0) < 1e-9, "p-hat not on the sphere"
    assert np.linalg.norm(np.cross(phat - N, P - N)) < 1e-9, "p-hat off the ray"
    # outside the unit circle -> northern hemisphere
    assert np.linalg.norm(P[:2]) > 1.0 and phat[2] > 0.0
    # the inverse projection recovers p: (x, y) / (1 - z)
    p_back = phat[:2] / (1.0 - phat[2])
    assert np.linalg.norm(p_back - P[:2]) < 1e-9, "not the stereographic image"
    # the drawn equator split matches the exact dot-product visibility test
    _, _, toward = g["cam"].axes()
    for visible, run in g["eq_runs"]:
        signed = (run - g["sph"].center) @ toward
        if visible:
            assert signed.min() > -1e-9, "visible equator arc dips behind the limb"
        else:
            assert signed.max() < 1e-9, "hidden equator arc pokes past the limb"
