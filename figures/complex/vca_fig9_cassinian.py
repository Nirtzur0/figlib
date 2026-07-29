"""VCA Fig [9] (p.62): Cassinian curves are the preimages of circles under z^2.

Left: the level curves |z-1||z+1| = k^2 — two eggs, the lemniscate, ovals.
Right: their images under w = z^2: concentric circles |w - 1| = k^2.
Each level keeps its color across the map; the lemniscate (k=1) is
accented and its image is the circle through the origin.
"""

import numpy as np

from figlib.format import WIDE
from figlib.geometry import cassinian_curves
from figlib.scene import AngleMark, Callout, Curve, MathLabel, Point, Scene, Vector
from figlib.style import Role
from figlib.theme import RISO

CLAIM = (
    "The Cassinian curves |z-1||z+1| = k^2 with foci +-1 are exactly the "
    "preimages under w = z^2 of the concentric circles |w-1| = k^2, with the "
    "k=1 lemniscate mapping to the circle through the origin."
)

THEME = RISO
FORMAT = WIDE

PARAMS = {
    "ks": [0.7, 0.85, 1.0, 1.2, 1.45],
    "w_origin": 3.75,
    "z_mark_k": 1.2,
    "z_mark_t": 1.05,   # polar angle of the marked point z
}

# The level k is an ordered quantity -> the theme's ramp channel, floored
# away from the light end so every stroke clears the ink-contrast gate
# (CLEAN.ramp(0.45) is the lightest stop at >= 3:1 on white).
RAMP_LO = 0.45


def _level_color(k: float, ks: list[float]) -> str | None:
    if k == 1.0:
        return None            # the lemniscate is THE object: accent role, no override
    t = (k - min(ks)) / (max(ks) - min(ks))
    return THEME.ramp(RAMP_LO + (1.0 - RAMP_LO) * t)


def compute(p):
    curves = {k: cassinian_curves(k) for k in p["ks"]}
    images = {}
    for k, loops in curves.items():
        segs = []
        for loop in loops:
            z = (loop[:, 0] + 1j * loop[:, 1]) ** 2
            segs.append(np.column_stack([z.real, z.imag]))
        images[k] = segs
    # marked point z on the k=1.2 oval
    k0, t0 = p["z_mark_k"], p["z_mark_t"]
    r0 = np.sqrt(np.cos(2 * t0) + np.sqrt(k0**4 - np.sin(2 * t0) ** 2))
    z0 = r0 * np.exp(1j * t0)
    return {"ks": p["ks"], "curves": curves, "images": images, "z0": z0, "k0": k0}


def build(g):
    W = PARAMS["w_origin"]
    s = Scene()

    def shift(pts):
        out = pts.copy()
        out[:, 0] += W
        return out

    # --- left: Cassinian family ---
    s.add(Vector((-2.05, 0.0), (2.15, 0.0), role=Role.MUTED, width_scale=0.8))
    for k in g["ks"]:
        role = Role.ACCENT2 if k == 1.0 else Role.CONTENT
        wsc = 1.25 if k == 1.0 else 0.95
        for loop in g["curves"][k]:
            s.add(Curve(loop, closed=True, role=role, width_scale=wsc,
                        color=_level_color(k, g["ks"])))
    s.add(Point((-1.0, 0.0), role=Role.CONTENT))
    s.add(Point((1.0, 0.0), role=Role.CONTENT))
    s.add(MathLabel(r"-1", (-1.0, 0.0), ha="center", va="top", offset_px=(0, 9)))
    s.add(MathLabel(r"1", (1.0, 0.0), ha="center", va="top", offset_px=(0, 9)))
    s.add(MathLabel(r"0", (0.0, 0.0), ha="center", va="top", offset_px=(0, 9),
                    role=Role.ANNOTATION))

    # the defining property at a marked point: r1 r2 = k^2
    z0 = g["z0"]
    s.add(Curve(np.array([[-1.0, 0.0], [z0.real, z0.imag]]), role=Role.CONSTRUCTION, width_scale=0.9))
    s.add(Curve(np.array([[1.0, 0.0], [z0.real, z0.imag]]), role=Role.CONSTRUCTION, width_scale=0.9))
    s.add(Point((z0.real, z0.imag), role=Role.CONTENT, radius_scale=0.8))
    s.add(MathLabel(r"z", (z0.real, z0.imag), ha="center", va="bottom", offset_px=(2, -7)))
    m1 = ((z0.real - 1) / 2, z0.imag / 2)
    m2 = ((z0.real + 1) / 2, z0.imag / 2)
    s.add(MathLabel(r"r_1", m1, ha="center", va="bottom", offset_px=(-4, -5),
                    size_pt=10, role=Role.ANNOTATION))
    s.add(MathLabel(r"r_2", m2, ha="left", va="bottom", offset_px=(5, -38),
                    size_pt=10, role=Role.ANNOTATION))
    s.add(MathLabel(r"r_1 r_2 = k^2", (-2.15, 1.42), ha="left", va="center",
                    role=Role.ANNOTATION, halo=True))
    s.add(MathLabel(r"(\text{each curve its own } k)", (-2.15, 1.42), ha="left", va="top",
                    offset_px=(0, 14), size_pt=9, role=Role.ANNOTATION))
    d1 = (z0.real + 1, z0.imag)
    d2 = (z0.real - 1, z0.imag)
    s.add(AngleMark((-1.0, 0.0), (1.0, 0.0), d1, radius=0.22))
    a1 = np.arctan2(d1[1], d1[0])
    s.add(MathLabel(r"\theta_1", (-1.0 + 0.38 * np.cos(a1 / 2), 0.38 * np.sin(a1 / 2)),
                    ha="left", va="center", size_pt=9, role=Role.ANNOTATION))
    s.add(AngleMark((1.0, 0.0), (1.0, 0.0), d2, radius=0.22))
    a2 = np.arctan2(d2[1], d2[0])
    s.add(MathLabel(r"\theta_2", (0.998, 0.0954), offset_px=(-9, 0),
                    ha="left", va="center", size_pt=9, role=Role.ANNOTATION,
                    halo=True))

    # --- mapping arrow ---
    s.add(Vector((1.95, 1.62), (2.7, 1.62), role=Role.ANNOTATION))
    s.add(MathLabel(r"z \mapsto z^2", (2.32, 1.62), ha="center", va="bottom",
                    offset_px=(0, -8), size_pt=13))
    s.add(MathLabel(r"w - 1 = (z-1)(z+1)", (2.32, 1.62), ha="center", va="top",
                    offset_px=(0, 8), size_pt=9, role=Role.ANNOTATION))

    # --- right: concentric circles |w-1| = k^2 ---
    s.add(Vector((W - 1.45, 0.0), (W + 3.45, 0.0), role=Role.MUTED, width_scale=0.8))
    for k in g["ks"]:
        role = Role.ACCENT2 if k == 1.0 else Role.CONTENT
        wsc = 1.25 if k == 1.0 else 0.95
        for seg in g["images"][k]:
            s.add(Curve(shift(seg), closed=True, role=role, width_scale=wsc,
                        color=_level_color(k, g["ks"])))
    s.add(Point((W + 1.0, 0.0), role=Role.CONTENT))
    s.add(MathLabel(r"1", (W + 1.0, 0.0), ha="center", va="top", offset_px=(0, 9)))
    s.add(Point((W, 0.0), role=Role.CONTENT, radius_scale=0.7))
    s.add(MathLabel(r"0", (W, 0.0), ha="right", va="top", offset_px=(-5, 9)))
    # This corner of the w-plane is genuinely oversubscribed: four
    # annotations (center, image point, its radius, its angle) all refer
    # to features packed inside one ring of concentric circles. Rather than
    # nudge each in isolation and push the collision onto its neighbour,
    # the two that cannot find room near their referent get a leader
    # (Callout) into the ink-free band above/below the rings; the two that
    # can stay close (the angle sum, and now the center) keep a plain label.
    s.add(Callout(r"\text{foci} \mapsto \text{center}", anchor=(W + 1.02, -1.19),
                  target=(W + 1.0, 0.0), size_pt=9, role=Role.ANNOTATION))
    w0 = z0 * z0
    s.add(Point((W + w0.real, w0.imag), role=Role.CONTENT, radius_scale=0.8))
    s.add(Callout(r"w = z^2", anchor=(W + 0.03, 1.47),
                  target=(W + w0.real, w0.imag)))
    # w - 1 = (z-1)(z+1): length r1 r2 = k^2, angle theta1 + theta2
    s.add(Curve(np.array([[W + 1.0, 0.0], [W + w0.real, w0.imag]]),
                role=Role.CONSTRUCTION, width_scale=0.9))
    mw = ((1.0 + w0.real) / 2, w0.imag / 2)
    s.add(Callout(r"r_1 r_2 = k^2", anchor=(W + 0.94, 1.2),
                  target=(W + mw[0], mw[1]), size_pt=10, role=Role.ANNOTATION))
    dw = (w0.real - 1.0, w0.imag)
    s.add(AngleMark((W + 1.0, 0.0), (1.0, 0.0), dw, radius=0.30))
    aw = np.arctan2(dw[1], dw[0])
    s.add(MathLabel(r"\theta_1 + \theta_2", (W + 1.1, 0.0954), offset_px=(-26, 0),
                    ha="left", va="center", size_pt=9, role=Role.ANNOTATION,
                    halo=True))

    return s


def assertions(g):
    # every plotted Cassinian point satisfies its defining equation
    for k, loops in g["curves"].items():
        for loop in loops:
            z = loop[:, 0] + 1j * loop[:, 1]
            prod = np.abs(z - 1) * np.abs(z + 1)
            assert np.max(np.abs(prod - k * k)) < 1e-7, f"curve k={k} violates |z-1||z+1|=k^2"
    # every image point lies on the circle |w-1| = k^2
    for k, segs in g["images"].items():
        for seg in segs:
            w = seg[:, 0] + 1j * seg[:, 1]
            assert np.max(np.abs(np.abs(w - 1) - k * k)) < 1e-7, f"image k={k} not on |w-1|=k^2"
    # the lemniscate passes through the origin
    lem = np.vstack(g["curves"][1.0])
    assert np.min(np.hypot(lem[:, 0], lem[:, 1])) < 1e-3, "lemniscate misses the origin"
    # the marked point is on its curve
    z0, k0 = g["z0"], g["k0"]
    assert abs(abs(z0 - 1) * abs(z0 + 1) - k0 * k0) < 1e-9
