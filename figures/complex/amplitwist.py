"""The amplitwist: an analytic map's local action is one number, not a field.

Needham's Visual Complex Analysis opens on this device before it names a
single theorem: at a point z0 where f is analytic and f'(z0) != 0, the map
takes an infinitesimal disc to another disc — every radial direction gets
amplified by the SAME factor |f'(z0)| and rotated by the SAME angle
arg f'(z0), regardless of which direction it started in. Direction-
independence is the geometric content of the Cauchy-Riemann equations;
Needham's whole visual apparatus (conformality, the amplitwist concept,
the derivative-as-local-similarity) is built to make that one fact
perceptual instead of algebraic. The natural doubt a reader raises here is
"why should EVERY direction agree?" — a generic smooth R^2 -> R^2 map does
not have this property, so the amplitwist is a real constraint, not a
picture of what any map does locally.

This figure states the constraint by making it fail. Top row: z0 -> f(z0)
= 1/z0, a genuine analytic map with nonzero derivative — eight radial
arrows from a small disc at z0 map to eight arrows from a small disc at
f(z0), all lengthened by |f'(z0)| and turned by arg f'(z0). Bottom row:
the SAME picture under a non-analytic map, g(x, y) = (a x, b y) with
a != b — a real-linear map, so it has no first-order-approximation error
to worry about, only direction-dependence to show. The disc becomes an
ellipse: the arrow along x scales by a, the arrow along y scales by b,
and the right angle between two of the tracked arrows is not preserved.
The same finite-difference construction that reads off the amplification
and twist also reads off the Cauchy-Riemann equations directly from the
drawn arrows (not from a symbolic derivative) — satisfied at top, violated
at bottom. That is the falsifiable content: analyticity is not "smooth
and complex-valued", it is "same amplitwist in every direction", and the
figure is built so that claim can be read off wrong if it is wrong.
"""

import numpy as np

from figlib.correspond import Correspondence
from figlib.figure import Connector, Figure, Panel
from figlib.format import WIDE
from figlib.scene import AngleMark, Curve, MathLabel, Point, Scene, Vector
from figlib.style import Role
from figlib.theme import RISO

CLAIM = (
    "An analytic map with f'(z0) != 0 amplifies and twists an infinitesimal "
    "disc by the SAME factor and angle in every direction (an amplitwist); a "
    "non-analytic R^2 -> R^2 map with unequal singular values does not, and "
    "sends the disc to an ellipse instead."
)

EXPOSITION = """
Chapter 4 of Visual Complex Analysis, "The Amplitwist Concept," is where
Needham finally commits to what "complex differentiable" should mean
geometrically, having spent three chapters building the vocabulary of
maps-as-transformations. The naive route -- treat f as a map R^2 -> R^2
and demand its Jacobian exist -- buries the interesting fact inside a
2x2 matrix. Needham's move is to notice that when f'(z0) exists and is
nonzero, that Jacobian is not an arbitrary linear map: it is
multiplication by the single complex number f'(z0), which stretches
every direction by the same factor |f'(z0)| and rotates it by the same
angle arg f'(z0). He names this one number the amplitwist and only
afterward shows it is equivalent to the Cauchy-Riemann equations -- the
algebra is a corollary of the geometry, not the other way round. The
reader is asked to accept, on the strength of a picture of one arrow
turning into another, that a generic smooth map does NOT have this
property; nothing in the prose forces that belief.

This figure is the falsification the chapter's prose does not attempt:
the same disc-of-arrows construction run once on an analytic map and
once on a real-linear map with unequal axis stretches, so "same
amplitwist in every direction" and "different amplitwist by direction"
sit side by side as two outcomes of one procedure, not as a claim and
its unstated opposite.
"""

THEME = RISO
FORMAT = WIDE

PARAMS = {
    "z0": (1.4, 0.9),          # analytic case: base point in the z-plane
    "r_disc": 0.01,            # domain-disc radius; small enough that the
                               # O(r^2) curvature term of 1/z stays under
                               # ~1% of the O(r) amplitwist term at this z0
                               # (see the honesty note in compute())
    "n_arrows": 8,             # radial arrow directions, evenly spaced
    "accent_deg": (45.0, 135.0),   # the two tracked, non-axis-aligned arrows
    "p0": (-1.1, 0.75),        # non-analytic case: base point in the (x, y) plane
    "a_scale": 1.7,            # x-stretch of g
    "b_scale": 0.65,           # y-stretch of g; a != b is the whole point
    "window_pad": 1.35,        # panel half-width as a multiple of the drawn radius
    "circle_samples": 160,
}


def _f(z: complex) -> complex:
    return 1.0 / z


def _fprime(z: complex) -> complex:
    return -1.0 / z ** 2


def compute(p):
    z0 = complex(*p["z0"])
    r = p["r_disc"]
    n = p["n_arrows"]
    thetas = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    dirs = np.column_stack([np.cos(thetas), np.sin(thetas)])

    tips = z0 + r * (dirs[:, 0] + 1j * dirs[:, 1])
    boundary_theta = np.linspace(0.0, 2 * np.pi, p["circle_samples"])
    boundary = z0 + r * np.exp(1j * boundary_theta)

    w0 = _f(z0)
    tips_w = _f(tips)
    boundary_w = _f(boundary)
    fprime0 = _fprime(z0)

    # ratio and turn angle of each mapped arrow, computed from the actual
    # mapped points (never from fprime0) — this is the load-bearing check
    src_vec = tips - z0
    img_vec = tips_w - w0
    ratios = np.abs(img_vec) / r
    turn = np.angle(img_vec / src_vec)      # wraps to (-pi, pi], constant iff amplitwist holds

    # map accent_deg -> index into thetas (thetas are 0, 360/n, ...)
    accent_idx = [int(round(d / (360.0 / n))) % n for d in p["accent_deg"]]

    # Cauchy-Riemann check straight from the finite-difference Jacobian of
    # the 0deg and 90deg arrows (u_x, v_x) and (u_y, v_y) — not restated
    # from the closed-form derivative.
    i0 = int(round(0.0 / (360.0 / n))) % n
    i90 = int(round(90.0 / (360.0 / n))) % n
    ux, vx = img_vec[i0].real / r, img_vec[i0].imag / r
    uy, vy = img_vec[i90].real / r, img_vec[i90].imag / r

    # --- non-analytic companion: exact real-linear map, no O(r^2) term ---
    p0 = np.array(p["p0"])
    a, b = p["a_scale"], p["b_scale"]
    tips2 = p0 + r * dirs
    boundary2 = p0 + r * np.column_stack(
        [np.cos(boundary_theta), np.sin(boundary_theta)])

    def gmap(pts):
        rel = pts - p0
        return p0 * np.array([a, b]) + rel * np.array([a, b])

    w0_2 = gmap(p0[None, :])[0]
    tips2_w = gmap(tips2)
    boundary2_w = gmap(boundary2)

    src_vec2 = tips2 - p0
    img_vec2 = tips2_w - w0_2
    ratios2 = np.hypot(img_vec2[:, 0], img_vec2[:, 1]) / r
    src_ang2 = np.arctan2(src_vec2[:, 1], src_vec2[:, 0])
    img_ang2 = np.arctan2(img_vec2[:, 1], img_vec2[:, 0])

    ux2, vx2 = img_vec2[i0, 0] / r, img_vec2[i0, 1] / r
    uy2, vy2 = img_vec2[i90, 0] / r, img_vec2[i90, 1] / r

    return {
        "n": n, "r": r, "thetas": thetas, "dirs": dirs,
        "accent_idx": accent_idx, "i0": i0, "i90": i90,
        # analytic case
        "z0": z0, "w0": w0, "tips": tips, "tips_w": tips_w,
        "boundary": boundary, "boundary_w": boundary_w,
        "fprime0": fprime0, "ratios": ratios, "turn": turn,
        "ux": ux, "vx": vx, "uy": uy, "vy": vy,
        # non-analytic case
        "p0": p0, "w0_2": w0_2, "tips2": tips2, "tips2_w": tips2_w,
        "boundary2": boundary2, "boundary2_w": boundary2_w,
        "a": a, "b": b, "ratios2": ratios2,
        "src_ang2": src_ang2, "img_ang2": img_ang2,
        "ux2": ux2, "vx2": vx2, "uy2": uy2, "vy2": vy2,
    }


def _xy(z) -> tuple[float, float]:
    return (float(np.real(z)), float(np.imag(z)))


def _arrow_role(i: int, accent_idx: list[int]) -> Role:
    if i == accent_idx[0]:
        return Role.ACCENT1
    if i == accent_idx[1]:
        return Role.ACCENT2
    return Role.MUTED


def _disc_radius(g, *, analytic: bool, mapped: bool) -> float:
    if analytic:
        center = g["w0"] if mapped else g["z0"]
        boundary = g["boundary_w"] if mapped else g["boundary"]
        return float(np.max(np.abs(np.asarray(boundary) - center)))
    center = g["w0_2"] if mapped else g["p0"]
    boundary = np.asarray(g["boundary2_w"] if mapped else g["boundary2"])
    return float(np.max(np.hypot(boundary[:, 0] - center[0], boundary[:, 1] - center[1])))


# offset_px per (analytic, mapped) for the center label — clear of the
# accent arrows at 45/135deg, tuned against the label-on-ink gate
_CENTER_LABEL_OFFSET = {
    (True, False): (28, -6), (True, True): (59, -6),
    (False, False): (28, -6), (False, True): (6, -26),
}


def _panel(g, *, analytic: bool, mapped: bool, half: float) -> Scene:
    n, accent_idx = g["n"], g["accent_idx"]
    r_disc = _disc_radius(g, analytic=analytic, mapped=mapped)
    if analytic:
        center = g["w0"] if mapped else g["z0"]
        tips = g["tips_w"] if mapped else g["tips"]
        boundary = g["boundary_w"] if mapped else g["boundary"]
        r_center = _xy(center)
    else:
        center = g["w0_2"] if mapped else g["p0"]
        tips = g["tips2_w"] if mapped else g["tips2"]
        boundary = g["boundary2_w"] if mapped else g["boundary2"]
        r_center = tuple(center)

    s = Scene(xlim=(r_center[0] - half, r_center[0] + half),
              ylim=(r_center[1] - half, r_center[1] + half))

    boundary_xy = (np.column_stack([np.real(boundary), np.imag(boundary)])
                   if analytic else np.asarray(boundary))
    s.add(Curve(boundary_xy, role=Role.CONTENT, key="disc"))

    for i in range(n):
        tip_xy = _xy(tips[i]) if analytic else tuple(tips[i])
        role = _arrow_role(i, accent_idx)
        s.add(Vector(r_center, tip_xy, role=role,
                      width_scale=1.4 if role != Role.MUTED else 0.9,
                      key=f"arrow-{i}"))

    s.add(Point(r_center, role=Role.CONTENT, radius_scale=0.9, key="center"))
    if analytic:
        text = r"z_0" if not mapped else r"f(z_0)"
    else:
        text = r"p_0" if not mapped else r"g(p_0)"
    s.add(MathLabel(text, r_center, ha="left", va="bottom",
                     offset_px=_CENTER_LABEL_OFFSET[(analytic, mapped)],
                     halo=True, key="center"))

    # angle mark between the two tracked (accent) arrows, at both ends of
    # the map — its measure is exactly the falsifiable claim: constant for
    # the analytic map, changed for the non-analytic one.
    d0 = g["dirs"][accent_idx[0]]
    d1 = g["dirs"][accent_idx[1]]
    if mapped:
        if analytic:
            v0 = np.array(_xy(tips[accent_idx[0]])) - np.array(r_center)
            v1 = np.array(_xy(tips[accent_idx[1]])) - np.array(r_center)
        else:
            v0 = np.array(tips[accent_idx[0]]) - np.array(r_center)
            v1 = np.array(tips[accent_idx[1]]) - np.array(r_center)
        ang = np.degrees(np.arctan2(v1[1], v1[0]) - np.arctan2(v0[1], v0[0]))
        ang = abs(((ang + 180) % 360) - 180)
        lab = rf"\angle \approx {ang:.0f}^\circ"
    else:
        lab = r"\angle = 90^\circ"
        v0, v1 = d0, d1
    s.add(AngleMark(r_center, tuple(v0 / np.linalg.norm(v0)),
                    tuple(v1 / np.linalg.norm(v1)),
                    radius=0.28 * r_disc, role=Role.ANNOTATION, key="angle"))
    s.add(MathLabel(lab, r_center, ha="left", va="top",
                     offset_px=(10, 58) if mapped else (-20, 58),
                     size_pt=10, role=Role.ANNOTATION))
    return s


def build(g):
    pad = PARAMS["window_pad"]
    # each pair (domain, codomain) shares ONE half-width, so the amplitwist's
    # size change is a visible fact on the page (the codomain disc really is
    # bigger/smaller), and the two panels sit at one page scale — the
    # invariant `correspond.py` polices for any bound (undeclared-changing)
    # key, which here is everything except the center label's text.
    a_half = pad * max(_disc_radius(g, analytic=True, mapped=False),
                       _disc_radius(g, analytic=True, mapped=True))
    n_half = pad * max(_disc_radius(g, analytic=False, mapped=False),
                       _disc_radius(g, analytic=False, mapped=True))
    a_domain = _panel(g, analytic=True, mapped=False, half=a_half)
    a_image = _panel(g, analytic=True, mapped=True, half=a_half)
    n_domain = _panel(g, analytic=False, mapped=False, half=n_half)
    n_image = _panel(g, analytic=False, mapped=True, half=n_half)

    amp = abs(g["fprime0"])
    twist = np.degrees(np.angle(g["fprime0"]))
    a_image.items.append(MathLabel(
        rf"|f'(z_0)| \approx {amp:.2f}\ \ (\text{{amplification}})",
        (0, 0), ha="center", va="top", size_pt=10.5, role=Role.ANNOTATION,
        pin=True, offset_px=(0, -2)))
    a_image.items.append(MathLabel(
        rf"\arg f'(z_0) \approx {twist:.0f}^\circ\ \ (\text{{twist}})",
        (0, 0), ha="center", va="top", size_pt=10.5, role=Role.ANNOTATION,
        pin=True, offset_px=(0, 21)))
    # anchor the pinned pair at the panel's own center in canvas terms by
    # reusing xlim/ylim midpoint (math coords), so it sits at the panel top
    cx = sum(a_image.xlim) / 2
    top_y = a_image.ylim[1]
    a_image.items[-2].anchor = (cx, top_y)
    a_image.items[-1].anchor = (cx, top_y)

    ratio_spread = float(np.ptp(g["ratios2"]))
    n_image.items.append(MathLabel(
        rf"\text{{ratio ranges }} {g['a']:.2f}\ \text{{to}}\ {g['b']:.2f}",
        (0, 0), ha="center", va="top", size_pt=10.5, role=Role.ANNOTATION,
        pin=True, offset_px=(0, -2)))
    cx2 = sum(n_image.xlim) / 2
    top_y2 = n_image.ylim[1]
    n_image.items[-1].anchor = (cx2, top_y2)

    fig = Figure(
        panels=[Panel(a_domain, tag="[a]"), Panel(a_image, tag="[b]"),
                Panel(n_domain, tag="[c]"), Panel(n_image, tag="[d]")],
        grid=(2, 2),
        connectors=[
            Connector(0, 1, kind="map", label=r"f(z) = 1/z"),
            Connector(2, 3, kind="map", label=r"g(x,y) = (ax,\ by)"),
        ],
    )
    return fig


CORRESPONDENCE = [
    Correspondence(
        parts=(0, 1),
        varies="the analytic map f(z) = 1/z applied to the disc and its radii",
        # only the center label's TEXT changes (z_0 -> f(z_0)); the disc,
        # the eight arrows, and the angle mark keep their role/color/dash —
        # only their position moves, which is not part of the fingerprint,
        # so they are correctly left undeclared (bound) here
        changes=("center",),
    ),
    Correspondence(
        parts=(2, 3),
        varies="the non-analytic map g(x,y) = (ax,by) applied to the disc and its radii",
        changes=("center",),
    ),
]


def assertions(g):
    from figlib.gates import Checks
    c = Checks()

    # --- analytic map: the amplitwist itself ---
    ratios, turn = g["ratios"], g["turn"]
    c.check(np.ptp(ratios) / np.mean(ratios) < 0.03,
            f"amplification varies by direction: ratios span "
            f"{ratios.min():.4f}-{ratios.max():.4f} (analytic map should agree)")
    turn_spread = np.ptp(((turn - turn[0] + np.pi) % (2 * np.pi)) - np.pi)
    c.check(turn_spread < 0.03,
            f"twist angle varies by direction: spread {np.degrees(turn_spread):.3f} deg")
    # boundary circle stays (nearly) circular under the map
    bw = g["boundary_w"]
    center_est = np.mean(bw)
    rad = np.abs(bw - center_est)
    c.check(np.ptp(rad) / np.mean(rad) < 0.03,
            f"mapped boundary is not nearly circular: radius spread "
            f"{np.ptp(rad) / np.mean(rad):.4f} (r_disc too large for the "
            f"first-order approximation)")
    # Cauchy-Riemann, read off the finite-difference Jacobian, not restated
    # from the closed-form derivative
    c.check(abs(g["ux"] - g["vy"]) < 0.03 and abs(g["uy"] + g["vx"]) < 0.03,
            f"Cauchy-Riemann violated for an analytic map: "
            f"ux-vy={g['ux'] - g['vy']:.4f}, uy+vx={g['uy'] + g['vx']:.4f}")
    # the right angle between the two tracked arrows survives the map
    v0 = g["tips_w"][g["accent_idx"][0]] - g["w0"]
    v1 = g["tips_w"][g["accent_idx"][1]] - g["w0"]
    ang = abs(np.degrees(np.angle(v1 / v0)))
    c.check(abs(ang - 90.0) < 2.0,
            f"angle between tracked arrows not preserved: {ang:.2f} deg (want ~90)")

    # --- non-analytic companion: the amplitwist genuinely fails ---
    ratios2 = g["ratios2"]
    c.check(np.ptp(ratios2) / np.mean(ratios2) > 0.3,
            f"non-analytic map's ratios did not actually vary by direction: "
            f"span {ratios2.min():.4f}-{ratios2.max():.4f} — companion is not "
            f"falsifying anything")
    c.check(abs(ratios2.min() - g["b"]) < 1e-9 and abs(ratios2.max() - g["a"]) < 1e-9,
            "extreme ratios should be exactly a and b (axis-aligned stretch)")
    c.check(abs(g["ux2"] - g["vy2"]) > 0.3,
            f"Cauchy-Riemann should FAIL for g: ux-vy={g['ux2'] - g['vy2']:.4f} "
            f"is too small to read as a violation")
    v0_2 = g["tips2_w"][g["accent_idx"][0]] - g["w0_2"]
    v1_2 = g["tips2_w"][g["accent_idx"][1]] - g["w0_2"]
    ang2 = np.degrees(np.arccos(
        np.dot(v0_2, v1_2) / (np.linalg.norm(v0_2) * np.linalg.norm(v1_2))))
    c.check(abs(ang2 - 90.0) > 10.0,
            f"the right angle should NOT survive g: measured {ang2:.2f} deg")
    # ellipse identity: the mapped boundary literally satisfies the ellipse
    # equation with semi-axes a*r, b*r about its own center
    rel = g["boundary2_w"] - g["w0_2"]
    val = (rel[:, 0] / (g["a"] * g["r"])) ** 2 + (rel[:, 1] / (g["b"] * g["r"])) ** 2
    c.check(np.max(np.abs(val - 1.0)) < 1e-9,
            "mapped boundary is not the ellipse (x/ar)^2 + (y/br)^2 = 1")

    c.done()
