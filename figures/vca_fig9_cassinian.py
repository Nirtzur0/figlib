"""VCA Fig [9] (p.62): Cassinian curves are the preimages of circles under z^2.

Left: the level curves |z-1||z+1| = k^2 — two eggs, the lemniscate, ovals.
Right: their images under w = z^2: concentric circles |w - 1| = k^2.
Each level keeps its color across the map; the lemniscate (k=1) is
accented and its image is the circle through the origin.
"""

import numpy as np

from figlib.geometry import cassinian_curves
from figlib.scene import Curve, MathLabel, Point, Scene, Vector
from figlib.style import Role

CLAIM = (
    "The Cassinian curves |z-1||z+1| = k^2 with foci +-1 are exactly the "
    "preimages under w = z^2 of the concentric circles |w-1| = k^2, with the "
    "k=1 lemniscate mapping to the circle through the origin."
)

PARAMS = {
    "ks": [0.7, 0.85, 1.0, 1.2, 1.45],
    "w_origin": 3.75,
    "z_mark_k": 1.2,
    "z_mark_t": 1.05,   # polar angle of the marked point z
}

LEVEL_COLORS = {0.7: "#9db8d9", 0.85: "#6f94c4", 1.0: "#d1495b",
                1.2: "#41669f", 1.45: "#254a80"}


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
                        color=LEVEL_COLORS[k]))
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
    s.add(MathLabel(r"r_2", m2, ha="left", va="bottom", offset_px=(5, -4),
                    size_pt=10, role=Role.ANNOTATION))
    s.add(MathLabel(r"r_1 r_2 = k^2", (-1.55, 1.15), ha="left", va="center",
                    role=Role.ANNOTATION))

    # --- mapping arrow ---
    s.add(Vector((2.25, 1.35), (3.0, 1.35), role=Role.ANNOTATION))
    s.add(MathLabel(r"z \mapsto z^2", (2.62, 1.35), ha="center", va="bottom",
                    offset_px=(0, -8), size_pt=13))

    # --- right: concentric circles |w-1| = k^2 ---
    s.add(Vector((W - 1.45, 0.0), (W + 3.45, 0.0), role=Role.MUTED, width_scale=0.8))
    for k in g["ks"]:
        role = Role.ACCENT2 if k == 1.0 else Role.CONTENT
        wsc = 1.25 if k == 1.0 else 0.95
        for seg in g["images"][k]:
            s.add(Curve(shift(seg), closed=True, role=role, width_scale=wsc,
                        color=LEVEL_COLORS[k]))
    s.add(Point((W + 1.0, 0.0), role=Role.CONTENT))
    s.add(MathLabel(r"1", (W + 1.0, 0.0), ha="center", va="top", offset_px=(0, 9)))
    s.add(Point((W, 0.0), role=Role.CONTENT, filled=False, radius_scale=0.8))
    s.add(MathLabel(r"0", (W, 0.0), ha="right", va="top", offset_px=(-5, 9)))
    w0 = z0 * z0
    s.add(Point((W + w0.real, w0.imag), role=Role.CONTENT, radius_scale=0.8))
    s.add(MathLabel(r"w = z^2", (W + w0.real, w0.imag), ha="left", va="bottom",
                    offset_px=(6, -5)))

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
