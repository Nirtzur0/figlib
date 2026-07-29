"""VCA Fig [30] (p.165): portrait of an elliptic Mobius transformation.

M has fixed points xi_pm = +-1 and multiplier m = e^{i pi/3}. Conjugating
by F(z) = (z-1)/(z+1) sends xi+ -> 0, xi- -> inf, and M becomes the pure
rotation w -> m w. The whole bipolar net is therefore the F-preimage of a
log-polar grid: rays through 0 pull back to the C1 circles through both
fixed points, origin-centred circles pull back to the invariant Apollonian
C2 circles. Cells of the net are conformal squares (Delta s = Delta phi =
pi/6); M advances every cell by exactly two angular steps, so the black
orbit of one cell closes after six applications.

Needham's figure is reproduced up to a 180-degree rotation: with F fixed
as above, m = e^{i pi/3} turns counterclockwise about xi+ (right pole),
clockwise about xi- and downward through the central band -- the book's
(unlabeled) picture is the same flow with the poles swapped.

Layout choice (recorded): the book tilts its axis ~8 degrees as a
genericity cue; this recreation keeps the fixed points on the real axis.
The assertions are on the computed configuration either way, and the
axis-aligned frame is what lets the pi/3 angle arc be read against the
drawn real axis directly. Stated in the readback/judge records.
"""

import numpy as np

from figlib.builders import mapped_grid
from figlib.format import WIDE
from figlib.gates import Checks
from figlib.scene import AngleMark, Callout, Curve, FilledCurve, MathLabel, Scene
from figlib.style import Role
from figlib.theme import RISO

CLAIM = (
    "An elliptic Mobius transformation rotates the whole plane along the "
    "invariant Apollonian circles about its two fixed points, each "
    "application advancing every cell of the bipolar net by the fixed "
    "multiplier angle pi/3 -- so the black orbit of one cell closes after "
    "six steps: M^6 = id."
)

THEME = RISO
FORMAT = WIDE

D = np.pi / 6          # grid step: Delta s = Delta phi (conformal squares)

PARAMS = {
    "m": np.exp(1j * np.pi / 3),
    "n_bands": 7,               # s = -7D .. 7D  ->  7 rings per pole
    "xlim": (-4.6, 4.6),
    "ylim": (-2.56, 2.56),
    "corner_r": 0.28,           # rounded-rect frame radius (math units)
    "r_clamp": 45.0,            # honest crop of the net's passage through inf
    "orbit_i": 4,               # radial band [-3D, -2D] around xi+
    "orbit_js": [11, 1, 3, 5, 7, 9],   # phi steps of pi/3: one M-orbit
    "grey_parity": 1,           # (i + j) % 2 of the washed cells
    # iteration indices M^k, k = 0..5, one per orbit cell in orbit_js
    # order: (x, y, size_pt, on_dark). 0, 1, 5 sit inside their (large)
    # cells; 2 sits just outside its cell's outer edge; 3, 4 flank their
    # radially-compressed cells on the lens side. Pinned: position IS
    # meaning. Per-label style split: a numeral ON a black cell is set in
    # paper ink, no halo (its ground is the fill, and a paper halo would
    # punch a hole in the cell); a numeral BESIDE its cell keeps
    # annotation ink with a halo against the net's curves.
    "orbit_labels": [
        (1.67, -0.28, 9.5, True),   # 0  inside cell (phi 330..360)
        (1.33,  0.55, 9.5, True),   # 1  inside cell (phi 30..60)
        (0.80,  0.86, 9.0, False),  # 2  beside cell (phi 90..120), outward
        (0.38,  0.15, 8.5, False),  # 3  beside cell (phi 150..180), lens
        (0.38, -0.15, 8.5, False),  # 4  beside cell (phi 180..210), lens
        (0.99, -0.56, 9.5, True),   # 5  inside cell (phi 270..300)
    ],
    # pi/3 angle arc at xi+: spans exactly one advance step = two angular
    # cells, from the drawn real axis (phi = 0) to the tangent of the
    # phi = 2D net curve at the pole
    "arc_r": 0.30,
    "arc_ray_len": 0.42,
    "arc_label": (1.68, 0.26),
    # fixed-point callouts: boxes centred in the conformal cell
    # (s, phi) in [D,2D] x [10D,11D] off xi+ (and its 180-degree twin off
    # xi-) -- the one big grey cell near each pole that no C1/C2 curve or
    # arrowhead enters; leaders thread the paper corridor between orbit
    # cells 5 and 0 (and its twin)
    "callouts": [(r"\xi_+", (1.60, -1.42), (1.0, 0.0)),
                 (r"\xi_-", (-1.60, 1.42), (-1.0, 0.0))],
    # motion arrows: (s/D, phi0/D, phi1/D, width_scale, arrow_scale),
    # always sampled with INCREASING phi, so direction = the flow of m.
    # A point's 180-degree twin is (-s, [12 - phi1, 12 - phi0]).
    "arrows": [
        # far field: the flow climbs both outer edges and descends the middle
        (-0.9, 11.85, 12.6, 2.0, 1.30),  # far right, rising quarter-sweep
        ( 0.9, 11.3, 12.2, 2.0, 1.30),   # far left, rising (twin)
        ( 0.75,  0.7,  1.6, 1.7, 1.15),  # top left, cresting rightward
        (-0.75, 10.4, 11.3, 1.7, 1.15),  # bottom right, cresting (twin)
        # mid rings around each pole: CCW about xi+, CW about xi-
        (-1.5,  0.8,  1.6, 1.5, 1.00),   # above xi+, up-left
        ( 1.5, 10.4, 11.2, 1.5, 1.00),   # below xi-, leftward (twin)
        (-1.5, 10.4, 11.2, 1.5, 1.00),   # below xi+, rightward
        ( 1.5,  0.8,  1.6, 1.5, 1.00),   # above xi-, rightward (twin)
        # the orbit band: arrows between black cells on the FAR side only —
        # the near-side (phi ~ pi) members are radially compressed and must
        # stay uncovered so the orbit chain stays countable. The slot at
        # phi in [0.25, 0.80] is ceded to the pi/3 angle mark: an arrow
        # there tips into the arc label, and the arc itself already gives
        # the sense of advance across that gap.
        (-2.5,  2.25, 2.80, 1.1, 0.75),
        (-2.5,  8.25, 8.80, 1.1, 0.75),
        (-2.5, 10.25, 10.80, 1.1, 0.75),
        # the central lens: ONE column of down-arrows either side of the
        # midline (was four columns: read as a translation field)
        ( 0.5,  3.6,  4.6, 1.1, 0.72),
        ( 0.5,  7.4,  8.4, 1.1, 0.72),
        (-0.5,  3.6,  4.6, 1.1, 0.72),
        (-0.5,  7.4,  8.4, 1.1, 0.72),
        # innermost legible ring at each pole
        (-3.5,  2.6,  3.6, 1.0, 0.62),
        (-3.5,  8.6,  9.6, 1.0, 0.62),
        ( 3.5,  8.4,  9.4, 1.0, 0.62),
        ( 3.5,  2.4,  3.4, 1.0, 0.62),
        # xi- collar band (twin ring of the orbit band): counter-circulation
        # documented with the same density as the xi+ side
        ( 2.5, 11.20, 11.75, 1.0, 0.70),
        ( 2.5,  9.20,  9.75, 1.0, 0.70),
        ( 2.5,  1.20,  1.75, 1.0, 0.70),
    ],
}


# --- the maps ---------------------------------------------------------------

def F(z: np.ndarray) -> np.ndarray:
    return (z - 1) / (z + 1)


def zmap(zeta: np.ndarray) -> np.ndarray:
    """F^{-1}(e^zeta): the bipolar net as the image of a log-polar grid."""
    w = np.exp(np.asarray(zeta, dtype=complex))
    with np.errstate(divide="ignore", invalid="ignore"):
        return (1 + w) / (1 - w)


def mobius(z: np.ndarray, m: complex) -> np.ndarray:
    """M with fixed points +-1 and multiplier m, in closed form."""
    return ((1 + m) * z + (1 - m)) / ((1 - m) * z + (1 + m))


# --- sanitizers: the net passes through infinity; the crop admits it --------

def _finite_runs(pts: np.ndarray, rmax: float) -> list[np.ndarray]:
    """Split a polyline at non-finite or far (|z| > rmax) samples."""
    ok = np.isfinite(pts).all(axis=1)
    ok &= np.where(ok, np.hypot(pts[:, 0], pts[:, 1]) <= rmax, False)
    runs, start = [], None
    for k, good in enumerate(ok):
        if good and start is None:
            start = k
        elif not good and start is not None:
            if k - start >= 2:
                runs.append(pts[start:k])
            start = None
    if start is not None and len(pts) - start >= 2:
        runs.append(pts[start:])
    return runs


def _clamp_poly(pts: np.ndarray, rmax: float) -> np.ndarray | None:
    """Drop a cell polygon's far vertices. For the four cells that touch
    infinity the dropped chunk is contiguous (the corner at w = 1), and the
    closing chord lies at |z| ~ rmax -- far outside the frame -- so the fill
    is exact within the crop."""
    ok = np.isfinite(pts).all(axis=1)
    ok &= np.where(ok, np.hypot(pts[:, 0], pts[:, 1]) <= rmax, False)
    kept = pts[ok]
    return kept if len(kept) >= 3 else None


def _rounded_rect(x0, x1, y0, y1, r, n=10) -> np.ndarray:
    q = np.linspace(0, np.pi / 2, n)
    arcs = [
        np.column_stack([x1 - r + r * np.cos(q), y1 - r + r * np.sin(q)]),
        np.column_stack([x0 + r - r * np.sin(q), y1 - r + r * np.cos(q)]),
        np.column_stack([x0 + r - r * np.cos(q), y0 + r - r * np.sin(q)]),
        np.column_stack([x1 - r + r * np.sin(q), y0 + r - r * np.cos(q)]),
    ]
    return np.vstack(arcs)


# --- the program ------------------------------------------------------------

def compute(p):
    n = p["n_bands"]
    u = D * np.arange(-n, n + 1)          # s values (log radii)
    v = D * np.arange(0, 13)              # phi values, 0 .. 2 pi
    mg = mapped_grid(zmap, u, v, samples_per_line=240)

    rc = p["r_clamp"]
    c2 = [(float(s), run) for s, c in zip(u, mg.u_curves)
          for run in _finite_runs(c.pts, rc)]
    c1 = [run for c in mg.v_curves[:-1] for run in _finite_runs(c.pts, rc)]

    x0, x1 = p["xlim"]
    y0, y1 = p["ylim"]

    def in_frame(poly):
        return (poly[:, 0].min() < x1 + 0.1 and poly[:, 0].max() > x0 - 0.1
                and poly[:, 1].min() < y1 + 0.1 and poly[:, 1].max() > y0 - 0.1)

    orbit_set = {(p["orbit_i"], j) for j in p["orbit_js"]}
    grey = []
    for i in range(len(u) - 1):
        for j in range(len(v) - 1):
            if (i + j) % 2 != p["grey_parity"] or (i, j) in orbit_set:
                continue
            poly = _clamp_poly(mg.cell(i, j), rc)
            if poly is not None and in_frame(poly):
                grey.append(poly)

    # the orbit cells are interior: raw boundaries, no clamping needed
    orbit = [mg.cell(p["orbit_i"], j) for j in p["orbit_js"]]

    arrows = []
    for s, f0, f1, wsc, asc in p["arrows"]:
        phi = np.linspace(f0 * D, f1 * D, 40)
        z = zmap(s * D + 1j * phi)
        arrows.append((np.column_stack([z.real, z.imag]), wsc, asc))

    # angle-arc directions from the net itself: tangents at xi+ of the two
    # C1 curves bounding one advance step (phi = 0 and phi = 2D = arg m)
    t0 = zmap(-9.0 + 0.0j) - 1.0
    t1 = zmap(-9.0 + 1j * (2 * D)) - 1.0
    arc_dirs = (t0 / abs(t0), t1 / abs(t1))

    return {"u": u, "v": v, "c2": c2, "c1": c1, "grey": grey,
            "orbit": orbit, "arrows": arrows, "m": p["m"],
            "orbit_labels": p["orbit_labels"], "arc_dirs": arc_dirs,
            "arc_r": p["arc_r"], "arc_ray_len": p["arc_ray_len"],
            "arc_label": p["arc_label"], "callouts": p["callouts"],
            "xlim": p["xlim"], "ylim": p["ylim"], "corner_r": p["corner_r"]}


def build(g):
    x0, x1 = g["xlim"]
    y0, y1 = g["ylim"]
    frame = _rounded_rect(x0, x1, y0, y1, g["corner_r"])
    s = Scene(xlim=g["xlim"], ylim=g["ylim"], clip_pts=frame)

    wash = THEME.ramp(1.0)                     # the darker "grey" of the pair
    for poly in g["grey"]:
        s.add(FilledCurve(poly, color=wash, opacity=0.30, outline=False))
    ink = THEME.ink(Role.CONTENT).color
    for poly in g["orbit"]:
        s.add(FilledCurve(poly, color=ink, opacity=0.95, outline=False))

    # the invariant (Apollonian) family carries the flow: heavier ink than
    # the C1 family, so the two pencils are told apart without the arrows
    for _, run in g["c2"]:
        s.add(Curve(run, role=Role.CONTENT, width_scale=0.72))
    for run in g["c1"]:
        s.add(Curve(run, role=Role.CONTENT, width_scale=0.45))

    for pts, wsc, asc in g["arrows"]:
        s.add(Curve(pts, role=Role.CONTENT, width_scale=wsc,
                    arrows=(1.0,), arrow_scale=asc))

    s.add(Curve(frame, closed=True, role=Role.FRAME))

    # iteration indices: the black cells ARE one orbit, counted 0..5.
    # Style split per label: paper ink on the black cells (ground = fill),
    # haloed annotation ink beside them (ground = paper under curve ink).
    paper = THEME.paper[0]
    # 3 and 4 flank the radially-compressed lens cells and land on the net's
    # curve ink at their pinned anchor; a small vertical nudge (verified
    # ink-free by the collision gate) clears it without moving the anchor.
    orbit_nudges = {3: (0, 3), 4: (0, -3)}
    for k, (lx, ly, pt, on_dark) in enumerate(g["orbit_labels"]):
        s.add(MathLabel(str(k), (lx, ly), role=Role.ANNOTATION,
                        color=paper if on_dark else None,
                        ha="center", va="center", size_pt=pt,
                        halo=not on_dark, pin=True,
                        offset_px=orbit_nudges.get(k, (0, 0))))

    # pi/3 arc at xi+: one advance step, read against the drawn real axis
    d1, d2 = g["arc_dirs"]
    ray = np.array([[1.0, 0.0],
                    [1.0 + g["arc_ray_len"] * d2.real,
                     g["arc_ray_len"] * d2.imag]])
    s.add(Curve(ray, role=Role.ANNOTATION, width_scale=0.9))
    s.add(AngleMark((1.0, 0.0), (d1.real, d1.imag), (d2.real, d2.imag),
                    radius=g["arc_r"]))
    s.add(MathLabel(r"\pi/3", g["arc_label"], role=Role.ANNOTATION,
                    ha="center", va="center", size_pt=9.5,
                    halo=True, pin=True))

    for latex, anchor, target in g["callouts"]:
        s.add(Callout(latex, anchor, target))

    # title card (paper plate, dotted frame outline) and haloed pole labels
    plate = _rounded_rect(x0 + 0.16, x0 + 1.92, y1 - 0.86, y1 - 0.14, 0.08)
    s.add(FilledCurve(plate, role=Role.FRAME, color=THEME.paper[0],
                      opacity=1.0, outline=True))
    s.add(MathLabel(r"\mathbf{Elliptic}", (x0 + 0.31, y1 - 0.26),
                    ha="left", va="top", size_pt=13))
    s.add(MathLabel(r"m = e^{i\pi/3},\;\; M^6 = \mathrm{id}",
                    (x0 + 0.31, y1 - 0.26), ha="left", va="top",
                    offset_px=(0, 26), size_pt=10.5, role=Role.ANNOTATION))
    return s


def assertions(g):
    c = Checks()
    m = g["m"]

    # (1) every plotted C2 point lies on |F(z)| = e^s: the circles ARE the
    # invariant Apollonian family
    for s, run in g["c2"]:
        z = run[:, 0] + 1j * run[:, 1]
        c.check(np.max(np.abs(np.abs(F(z)) - np.exp(s))) < 1e-9,
                f"C2 circle s={s:.3f} leaves |F(z)| = const")

    # (2) M maps each orbit cell's boundary onto the next, pointwise: the
    # black chain IS the multiplier acting
    for k in range(6):
        a = g["orbit"][k]
        b = g["orbit"][(k + 1) % 6]
        za = a[:, 0] + 1j * a[:, 1]
        zb = b[:, 0] + 1j * b[:, 1]
        c.check(np.max(np.abs(mobius(za, m) - zb)) < 1e-9,
                f"M(orbit cell {k}) misses orbit cell {(k + 1) % 6}")

    # (3) C1 perpendicular to C2 at a sampled net intersection
    w0 = np.exp(-2 * D + 1j * 3 * D)
    d = 1e-5
    t2 = zmap(np.log(w0) + 1j * d) - zmap(np.log(w0) - 1j * d)   # along C2
    t1 = zmap(np.log(w0) + d) - zmap(np.log(w0) - d)             # along C1
    t1, t2 = t1 / abs(t1), t2 / abs(t2)
    dot = t1.real * t2.real + t1.imag * t2.imag
    c.check(abs(dot) < 1e-6, f"C1 not orthogonal to C2: tangent dot {dot:.2e}")

    # (4) M has period 6 on 100 random points
    rng = np.random.default_rng(7)
    z = rng.uniform(-3, 3, 100) + 1j * rng.uniform(-2, 2, 100)
    zk = z.copy()
    for _ in range(6):
        zk = mobius(zk, m)
    c.check(np.max(np.abs(zk - z)) < 1e-9, "M^6 is not the identity")

    # (5) the drawn angle arc spans exactly the orbit-advance angle: its
    # bounding directions are net tangents at xi+, and the angle between
    # them is arg m = two angular cells of the grid
    d1, d2 = g["arc_dirs"]
    span = float(np.angle(d2 / d1))
    c.check(abs(span - np.angle(m)) < 1e-3,
            f"angle arc spans {span:.4f}, not arg m = {np.angle(m):.4f}")
    c.check(abs(span - 2 * D) < 1e-3,
            "angle arc does not span two angular cells of the net")

    c.done()
