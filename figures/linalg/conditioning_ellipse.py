"""The condition number, read off a picture: A carries the unit circle to
an ellipse whose semi-axes ARE its singular values.

The unit-length vector v_1 lands on sigma_1 u_1, v_2 lands on sigma_2 u_2,
and a generic unit vector lands strictly between the two -- so
kappa = sigma_1 / sigma_2 is not a property of the matrix in the
abstract, it is the ratio between the two extreme rays of one drawn
picture, attained by name.
"""

import numpy as np

from figlib.figure import Connector, Figure, Panel
from figlib.format import WIDE
from figlib.gates import Checks
from figlib.scene import Curve, MathLabel, Scene, Vector
from figlib.style import Role
from figlib.theme import RISO

CLAIM = (
    "A maps the unit circle to an ellipse with semi-axes sigma_1 >= sigma_2 "
    "along the left singular vectors u_1, u_2: a unit-size input perturbation "
    "is stretched by a factor that ranges over [sigma_2, sigma_1] depending "
    "only on its direction, and the ratio between the best-conditioned "
    "direction (v_1, stretched by sigma_1) and the worst -- the flat "
    "direction v_2, stretched by only sigma_2 -- is exactly "
    "kappa = sigma_1 / sigma_2, attained, not merely bounded."
)

EXPOSITION = """
This is the standard picture behind Trefethen & Bau's treatment of
conditioning (Lecture 12): every linear map's relative sensitivity to
perturbation is governed by the SVD, and the geometry makes the bound
visible instead of algebraic. Feed A the unit circle and read off the
image: it is an ellipse, its semi-axes are the singular values, its axis
directions are the left singular vectors. That picture already contains
the whole conditioning story. A unit perturbation applied along v_1 comes
out as sigma_1 u_1 -- the long axis, the direction A stretches most, the
best-conditioned direction because the map's action there is easiest to
undo. The same-size perturbation applied along v_2 comes out as only
sigma_2 u_2 -- the flat direction, where A compresses instead of
amplifying, so recovering that perturbation from the output means
dividing by the smallest possible gain and any noise in the output gets
amplified the most on the way back. Every other direction lands strictly
between these two extremes (a Rayleigh-quotient fact, not a hope), so the
ratio of worst to best -- kappa = sigma_1/sigma_2 -- is not a loose
worst-case estimate: it is exactly attained by this one pair of rays, and
no direction does worse. The figure draws both the range and its two
attaining endpoints, so the bound and its tightness are the same claim.
"""

THEME = RISO
FORMAT = WIDE

# kappa = sigma1/sigma2 = 5.60: large enough that the ellipse reads as
# visibly flattened (not a near-circle), small enough that the flat axis
# is still a drawable line segment rather than a needle (kappa ~ 50 would
# make sigma2 u2 illegible at the same scale as sigma1 u1).
PARAMS = {
    "sigma1": 2.8,
    "sigma2": 0.5,
    "phi_v_deg": 25.0,     # rotation of the right singular basis (input frame)
    "phi_u_deg": -35.0,    # rotation of the left singular basis (output frame);
                            # deliberately different from phi_v so u_i is not
                            # just a relabelled v_i -- A is a generic matrix,
                            # not a diagonal one in disguise
    "generic_deg": 40.0,   # a third input direction, strictly between v_1
                            # (0 deg) and v_2 (90 deg) in the V-frame, whose
                            # image must land strictly inside [sigma2, sigma1]
    "circle_samples": 240,
    "label_offset_px": 10.0,
}


def _rot(deg: float) -> np.ndarray:
    th = np.deg2rad(deg)
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s], [s, c]])


def compute(p):
    V = _rot(p["phi_v_deg"])
    U = _rot(p["phi_u_deg"])
    sigma1, sigma2 = p["sigma1"], p["sigma2"]
    A = U @ np.diag([sigma1, sigma2]) @ V.T

    v1, v2 = V[:, 0], V[:, 1]
    u1, u2 = U[:, 0], U[:, 1]

    t = np.deg2rad(p["generic_deg"])
    g = V @ np.array([np.cos(t), np.sin(t)])   # unit vector, strictly between v1 and v2
    Ag = A @ g

    tt = np.linspace(0.0, 2.0 * np.pi, p["circle_samples"])
    circle = np.column_stack([np.cos(tt), np.sin(tt)])
    ellipse = circle @ A.T

    Usvd, ssvd, _ = np.linalg.svd(A)   # independent of the construction above

    return {
        "A": A, "sigma1": sigma1, "sigma2": sigma2, "kappa": sigma1 / sigma2,
        "v1": v1, "v2": v2, "u1": u1, "u2": u2, "g": g, "Ag": Ag,
        "circle": circle, "ellipse": ellipse,
        "Usvd": Usvd, "ssvd": ssvd, "params": p,
    }


def _outward(tip: tuple[float, float], pad: float) -> dict:
    """Label placement that grows AWAY from the origin at a vector's tip,
    so text extends outward past the arrowhead instead of back over the
    shaft or across the origin toward the opposite vector."""
    x, y = tip
    return {
        "ha": "left" if x >= 0.0 else "right",
        "va": "bottom" if y >= 0.0 else "top",
        "offset_px": (pad if x >= 0.0 else -pad, pad if y >= 0.0 else -pad),
    }


def _domain_panel(g):
    p = g["params"]
    off = p["label_offset_px"]
    s = Scene()
    s.add(Curve(g["circle"], role=Role.CONTENT, closed=True))
    s.add(Vector((0.0, 0.0), tuple(g["v1"]), role=Role.ACCENT1))
    s.add(Vector((0.0, 0.0), tuple(g["v2"]), role=Role.ACCENT2))
    s.add(Vector((0.0, 0.0), tuple(g["g"]), role=Role.MUTED, width_scale=0.8))

    s.add(MathLabel(r"v_1", tuple(g["v1"]), role=Role.ACCENT1,
                    size_pt=10.5, **_outward(g["v1"], off)))
    s.add(MathLabel(r"v_2", tuple(g["v2"]), role=Role.ACCENT2,
                    size_pt=10.5, **_outward(g["v2"], off)))
    s.add(MathLabel(r"g", tuple(g["g"]), role=Role.MUTED,
                    size_pt=10.5, **_outward(g["g"], off)))
    s.add(MathLabel(r"\|x\| = 1", (0.0, -1.0), role=Role.ANNOTATION,
                    ha="center", va="top", offset_px=(0.0, -10.0), size_pt=9.5))
    return s


def _codomain_panel(g):
    p = g["params"]
    off = p["label_offset_px"]
    sigma1u1 = g["sigma1"] * g["u1"]
    sigma2u2 = g["sigma2"] * g["u2"]
    Ag = g["Ag"]
    s = Scene()
    s.add(Curve(g["ellipse"], role=Role.CONTENT, closed=True))
    s.add(Vector((0.0, 0.0), tuple(sigma1u1), role=Role.ACCENT1))
    s.add(Vector((0.0, 0.0), tuple(sigma2u2), role=Role.ACCENT2))
    s.add(Vector((0.0, 0.0), tuple(Ag), role=Role.MUTED, width_scale=0.8))

    s.add(MathLabel(r"\sigma_1 u_1", tuple(sigma1u1), role=Role.ACCENT1,
                    size_pt=10.5, **_outward(sigma1u1, off)))
    s.add(MathLabel(r"\sigma_2 u_2", tuple(sigma2u2), role=Role.ACCENT2,
                    size_pt=10.5, **_outward(sigma2u2, off)))
    s.add(MathLabel(r"Ag", tuple(Ag), role=Role.MUTED,
                    size_pt=10.5, **_outward(Ag, off)))

    amp_g = float(np.hypot(*Ag))
    caption_y = -(g["sigma1"] + 0.55)
    s.add(MathLabel(
        rf"\times{g['sigma1']:.2f}\ (\mathrm{{best}})\quad"
        rf"\times{amp_g:.2f}\ (\mathrm{{between}})\quad"
        rf"\times{g['sigma2']:.2f}\ (\mathrm{{worst,\ flat}})",
        (0.0, caption_y), role=Role.ANNOTATION, ha="center", va="top",
        size_pt=9.5))
    s.add(MathLabel(
        rf"\kappa = \sigma_1/\sigma_2 = {g['kappa']:.2f}",
        (0.0, caption_y - 0.42), role=Role.ANNOTATION, ha="center", va="top",
        size_pt=9.5))
    return s


def build(g):
    a, b = _domain_panel(g), _codomain_panel(g)
    return Figure(
        panels=[Panel(a, tag=r"[a]"), Panel(b, tag=r"[b]")],
        connectors=[Connector(0, 1, kind="map", label=r"A")],
    )


def assertions(g):
    p = g["params"]
    A, sigma1, sigma2 = g["A"], g["sigma1"], g["sigma2"]
    c = Checks()

    c.check(sigma1 > sigma2 > 0.0,
            "PARAMS do not describe a nondegenerate kappa > 1 matrix")

    # the drawn ellipse's semi-axes equal the singular values of an SVD
    # computed independently from the construction above
    ssvd = np.sort(g["ssvd"])[::-1]
    c.check(abs(float(ssvd[0]) - sigma1) < 1e-9,
            f"independent SVD gives sigma_1={ssvd[0]:.6f}, not {sigma1:.6f}")
    c.check(abs(float(ssvd[1]) - sigma2) < 1e-9,
            f"independent SVD gives sigma_2={ssvd[1]:.6f}, not {sigma2:.6f}")

    # the drawn axis directions are the left singular vectors, up to sign
    Usvd = g["Usvd"]
    c.check(abs(abs(float(np.dot(Usvd[:, 0], g["u1"]))) - 1.0) < 1e-9,
            "drawn u1 axis is not the SVD's first left singular vector")
    c.check(abs(abs(float(np.dot(Usvd[:, 1], g["u2"]))) - 1.0) < 1e-9,
            "drawn u2 axis is not the SVD's second left singular vector")

    # the drawn ellipse really is the image of the drawn circle under A
    c.check(np.allclose(g["ellipse"], g["circle"] @ A.T, atol=1e-12),
            "drawn ellipse is not A applied to the drawn unit circle")

    # the annotated best/worst amplifications are what the drawn vectors show
    amp1 = float(np.linalg.norm(A @ g["v1"]))
    amp2 = float(np.linalg.norm(A @ g["v2"]))
    c.check(abs(amp1 - sigma1) < 1e-9,
            f"amplification along v1 is {amp1:.6f}, annotated as sigma_1={sigma1:.6f}")
    c.check(abs(amp2 - sigma2) < 1e-9,
            f"amplification along v2 is {amp2:.6f}, annotated as sigma_2={sigma2:.6f}")
    # best case, normalized to its own claimed rate: a unit input along v1
    # is stretched by exactly sigma_1, so the ratio to that claimed rate is 1
    c.check(abs(amp1 / sigma1 - 1.0) < 1e-9,
            "best-case amplification does not equal its own annotated sigma_1 rate")
    c.check(abs(amp2 / sigma2 - 1.0) < 1e-9,
            "worst-case amplification does not equal its own annotated sigma_2 rate")

    # the annotated kappa is exactly the ratio of the two drawn amplifications,
    # not merely sigma1/sigma2 read off the parameters
    kappa = g["kappa"]
    c.check(abs(kappa - sigma1 / sigma2) < 1e-12,
            "annotated kappa is not sigma_1/sigma_2")
    c.check(abs(amp1 / amp2 - kappa) < 1e-9,
            "annotated kappa is not the ratio of the two drawn (best, worst) "
            "amplifications -- the bound would not be attained by what is drawn")

    # the generic third direction is neither axis, and its amplification is
    # strictly between sigma_2 and sigma_1 (Rayleigh quotient of a unit
    # vector between the two singular directions) -- this is the "anywhere
    # between sigma_2 and sigma_1" half of the claim, and it could be false
    # if generic_deg were 0 or 90 or the construction were wrong
    t = np.deg2rad(p["generic_deg"])
    expected = float(np.hypot(sigma1 * np.cos(t), sigma2 * np.sin(t)))
    amp_g = float(np.linalg.norm(g["Ag"]))
    c.check(abs(amp_g - expected) < 1e-9,
            "generic direction's amplification does not match the closed-form "
            "Rayleigh quotient sqrt(sigma1^2 cos^2(t) + sigma2^2 sin^2(t))")
    c.check(sigma2 + 1e-6 < amp_g < sigma1 - 1e-6,
            f"generic amplification {amp_g:.6f} is not strictly between "
            f"sigma_2={sigma2:.6f} and sigma_1={sigma1:.6f} -- the range claim "
            f"would be false for this direction")

    c.done()
