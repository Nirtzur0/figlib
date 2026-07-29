"""The Poincare disc model: geodesics are arcs orthogonal to the boundary,
and Euclid's fifth postulate fails there in the strongest possible way.

Needham (Visual Complex Analysis) builds the disc model from Mobius maps
that carry the unit disc to itself: these are exactly the isometries of a
metric in which "straight lines" (geodesics) become circular arcs meeting
the boundary circle at right angles (diameters are the degenerate case,
radius -> infinity). The construction earns its keep only once it is made
to answer Euclid's parallel postulate: given a line L and a point P not on
it, how many lines through P avoid L entirely? In the plane the answer is
exactly one. Here it is not one and not zero -- it is a continuum. Two
special geodesics through P share an ideal endpoint with L (they meet it
only "at infinity," on the bounding circle) and are asymptotically
parallel; every geodesic through P whose direction lies strictly between
these two never touches L at all, anywhere in the disc. That open wedge
is infinite, which is the whole content of the claim: non-Euclidean
geometry is not a technicality about one missing line, it is a wedge of
directions replacing a single ray.

This figure draws L, an external point P, the two limiting (asymptotic)
parallels, and a finite sample from the infinite non-crossing family
between them, so the "infinitely many" is read off a genuine fan rather
than asserted in a caption.
"""

import numpy as np

from figlib.format import COLUMN
from figlib.gates import Checks
from figlib.scene import Callout, Curve, MathLabel, Point, RightAngleMark, Scene
from figlib.style import Role
from figlib.theme import RISO

CLAIM = (
    "Through a point P off a hyperbolic line L there are infinitely many "
    "geodesics that never meet L -- bounded by two limiting parallels "
    "asymptotic to L's own ideal endpoints -- where Euclidean geometry "
    "permits exactly one."
)

EXPOSITION = """
Chapter 6, "Non-Euclidean Geometry," is Needham's answer to a question
the reader has been trained since childhood not to ask: is Euclid's
fifth postulate actually necessary, or is it an assumption that could
fail? Needham builds the Poincare disc concretely, as the fixed domain
of the Mobius automorphisms of the unit disc, specifically so that
"hyperbolic straight line" has a computable meaning -- a circular arc
meeting the boundary at right angles, a diameter in the degenerate case
-- rather than a purely axiomatic one. The chapter earns the model by
making it answer the parallel postulate directly: given a line L and a
point P off it, Euclid says exactly one line through P misses L. In the
disc, Needham shows, that count is not one and not zero -- it is a
continuum, bounded by two special geodesics that share an ideal
endpoint with L, asymptotic to it at the boundary circle, with every
geodesic strictly between them missing L entirely.

This figure draws that count rather than asserting it: L, an external
point P, its two limiting parallels, and a finite sample from the
infinite non-crossing family between them, so "infinitely many" is a
fan the reader can see opening rather than a phrase to take on faith.
"""

THEME = RISO
FORMAT = COLUMN

PARAMS = {
    "alpha_L_deg": 210.0,      # ideal endpoint angles of the given geodesic L
    "beta_L_deg": 330.0,
    "P": (0.0, 0.4),           # the external point
    "n_family": 7,             # sampled non-crossing geodesics through P
    "family_margin": 0.09,     # keep clear of the limiting parallels (frac. of the open arc)
    "n_samples": 240,          # points per drawn arc
    "n_scan": 720,             # angular resolution of the crossing sweep (assertions only)
}

RAMP_LO = 0.35   # floor the family's ramp away from the illegible light end


def _repr(z1, z2):
    """The complete geodesic through z1, z2 as ('circle', center, radius) or
    ('line', theta) -- theta the diameter's angle. Both z1, z2 may be
    interior or boundary points; solving z*c = (|z|^2+1)/2 for both points
    gives the orthogonal-circle center directly (it collapses to the
    tangent-line intersection when z1, z2 are both on |z|=1)."""
    x1, y1 = z1
    x2, y2 = z2
    det = x1 * y2 - x2 * y1
    if abs(det) < 1e-9:
        theta = np.arctan2(y1, x1) if (x1, y1) != (0.0, 0.0) else np.arctan2(y2, x2)
        return ("line", theta)
    A = np.array([[x1, y1], [x2, y2]])
    b = np.array([(x1 * x1 + y1 * y1 + 1.0) / 2.0, (x2 * x2 + y2 * y2 + 1.0) / 2.0])
    c = np.linalg.solve(A, b)
    r = float(np.hypot(x1 - c[0], y1 - c[1]))
    return ("circle", c, r)


def _boundary_pts(rep):
    """The two points where the complete geodesic meets |z| = 1."""
    if rep[0] == "line":
        theta = rep[1]
        u = np.array([np.cos(theta), np.sin(theta)])
        return [u.copy(), -u.copy()]
    _, c, r = rep
    d = float(np.hypot(*c))
    a = (r * r - 1.0 + d * d) / (2.0 * d)
    h = np.sqrt(max(r * r - a * a, 0.0))
    mid = c * (1.0 - a / d)
    perp = np.array([-c[1], c[0]]) / d
    return [mid + h * perp, mid - h * perp]


def _other_endpoint(rep, known):
    pts = _boundary_pts(rep)
    d0 = np.hypot(*(pts[0] - known))
    d1 = np.hypot(*(pts[1] - known))
    return pts[1] if d0 < d1 else pts[0]


def _sample_arc(rep, z1, z2, n):
    """Sample the arc of `rep` from z1 to z2 that stays inside the open disc
    (the other candidate arc/branch exits through |z| > 1)."""
    if rep[0] == "line":
        t = np.linspace(-1.0, 1.0, n)
        theta = rep[1]
        return np.column_stack([t * np.cos(theta), t * np.sin(theta)])
    _, c, r = rep
    a1 = np.arctan2(z1[1] - c[1], z1[0] - c[0])
    a2 = np.arctan2(z2[1] - c[1], z2[0] - c[0])
    d = (a2 - a1) % (2 * np.pi)
    for ang in (a1 + np.linspace(0.0, d, n), a1 - np.linspace(0.0, 2 * np.pi - d, n)):
        mid = c + r * np.array([np.cos(ang[n // 2]), np.sin(ang[n // 2])])
        if np.hypot(*mid) < 1.0 - 1e-6:
            return c + r * np.column_stack([np.cos(ang), np.sin(ang)])
    raise AssertionError("no branch of the geodesic circle stays inside the disc")


def geodesic_between(z1, z2, n):
    rep = _repr(z1, z2)
    return _sample_arc(rep, z1, z2, n), rep


def _pairwise_intersections(repA, repB):
    """Points (possibly none, one tangency, or two) where the two complete
    geodesics meet, in the full plane (not restricted to the disc)."""
    if repA[0] == "circle" and repB[0] == "circle":
        _, c1, r1 = repA
        _, c2, r2 = repB
        d = float(np.hypot(*(c2 - c1)))
        if d < 1e-12 or d > r1 + r2 + 1e-9 or d < abs(r1 - r2) - 1e-9:
            return []
        a = (r1 * r1 - r2 * r2 + d * d) / (2 * d)
        h2 = r1 * r1 - a * a
        if h2 < 0:
            return []
        h = np.sqrt(max(h2, 0.0))
        mid = c1 + a * (c2 - c1) / d
        perp = np.array([-(c2 - c1)[1], (c2 - c1)[0]]) / d
        return [mid + h * perp, mid - h * perp]
    if repA[0] == "line" and repB[0] == "line":
        return [np.array([0.0, 0.0])]
    theta, (kind, c, r) = (repA[1], repB) if repA[0] == "line" else (repB[1], repA)
    u = np.array([np.cos(theta), np.sin(theta)])
    bcoef = -2.0 * np.dot(u, c)
    ccoef = float(np.dot(c, c)) - r * r
    disc = bcoef * bcoef - 4 * ccoef
    if disc < 0:
        return []
    sq = np.sqrt(disc)
    return [((-bcoef + sq) / 2.0) * u, ((-bcoef - sq) / 2.0) * u]


def _crosses_open_disc(repA, repB, tol=1e-6):
    return any(np.hypot(*q) < 1.0 - tol for q in _pairwise_intersections(repA, repB))


def _tangent_dot_at_boundary(rep, z):
    """Dot product of the geodesic's tangent with the unit circle's tangent
    at the boundary point z (|z| = 1): 0 iff the two curves meet orthogonally."""
    t_unit = np.array([-z[1], z[0]])
    if rep[0] == "line":
        t_geo = np.array([np.cos(rep[1]), np.sin(rep[1])])
    else:
        _, c, _ = rep
        radial = z - c
        t_geo = np.array([-radial[1], radial[0]])
    t_geo = t_geo / np.hypot(*t_geo)
    return float(np.dot(t_geo, t_unit))


def compute(p):
    alpha_L = np.radians(p["alpha_L_deg"])
    beta_L = np.radians(p["beta_L_deg"])
    A = np.array([np.cos(alpha_L), np.sin(alpha_L)])
    B = np.array([np.cos(beta_L), np.sin(beta_L)])
    P = np.array(p["P"])
    n = p["n_samples"]

    L_pts, repL = geodesic_between(A, B, n)

    # the two limiting parallels: full geodesics through P sharing exactly
    # one ideal endpoint with L. The endpoint NOT shared is derived, not
    # chosen -- it is what bounds the safe wedge below.
    rep_pa = _repr(tuple(P), tuple(A))
    other_a = _other_endpoint(rep_pa, A)
    pa_pts = _sample_arc(rep_pa, other_a, A, n)

    rep_pb = _repr(tuple(P), tuple(B))
    other_b = _other_endpoint(rep_pb, B)
    pb_pts = _sample_arc(rep_pb, other_b, B, n)

    # the non-crossing wedge, in boundary angle: the open arc from
    # other_b to alpha_L (walking the short way, away from B). Every
    # geodesic through P with its first ideal endpoint in this open arc
    # misses L entirely; the two arc endpoints ARE the limiting parallels.
    ang_other_b = np.arctan2(other_b[1], other_b[0])
    lo = ang_other_b % (2 * np.pi)
    hi = alpha_L % (2 * np.pi)
    if hi < lo:
        hi += 2 * np.pi
    margin = p["family_margin"] * (hi - lo)
    fracs = np.linspace(margin, 1.0 - margin, p["n_family"])
    family_thetas = lo + fracs * (hi - lo)

    family = []
    for th in family_thetas:
        Q1 = np.array([np.cos(th), np.sin(th)])
        rep = _repr(tuple(P), tuple(Q1))
        Q2 = _other_endpoint(rep, Q1)
        pts = _sample_arc(rep, Q1, Q2, n)
        family.append({"theta": float(th), "pts": pts, "rep": rep, "Q1": Q1, "Q2": Q2})

    # a witness geodesic through P that DOES cross L (from inside the
    # "shadow" wedge between the two limiting parallels, as seen from P) --
    # not drawn, only used to certify the family-selection logic is not
    # vacuous (assertions below). midpoint of the excluded wedge
    # (other_a, other_b) mod 2pi, on the far side from L.
    ang_other_a = np.arctan2(other_a[1], other_a[0]) % (2 * np.pi)
    lo2, hi2 = sorted([ang_other_a, ang_other_b])
    witness_theta = 0.5 * (lo2 + hi2)
    Qw = np.array([np.cos(witness_theta), np.sin(witness_theta)])
    rep_witness = _repr(tuple(P), tuple(Qw))

    return {
        "A": A, "B": B, "P": P, "L_pts": L_pts, "repL": repL,
        "pa_pts": pa_pts, "rep_pa": rep_pa, "other_a": other_a,
        "pb_pts": pb_pts, "rep_pb": rep_pb, "other_b": other_b,
        "family": family, "rep_witness": rep_witness,
    }


def build(g):
    s = Scene()
    s.xlim = (-1.18, 1.18)
    s.ylim = (-1.18, 1.18)

    theta = np.linspace(0.0, 2 * np.pi, 240)
    s.add(Curve(np.column_stack([np.cos(theta), np.sin(theta)]),
                closed=True, role=Role.FRAME))

    n_fam = len(g["family"])
    for i, mem in enumerate(g["family"]):
        t = i / (n_fam - 1) if n_fam > 1 else 0.5
        s.add(Curve(mem["pts"], role=Role.CONTENT,
                    color=THEME.ramp(RAMP_LO + (1.0 - RAMP_LO) * t)))

    s.add(Curve(g["pa_pts"], role=Role.ACCENT2, width_scale=1.3))
    s.add(Curve(g["pb_pts"], role=Role.ACCENT2, width_scale=1.3))

    s.add(Curve(g["L_pts"], role=Role.ACCENT1, width_scale=1.5))

    # ground the model's defining property once, at A: the radius OA
    # (construction) meets L at a right angle -- the reason A is an "ideal
    # point" that L, and every asymptotic parallel, is allowed to share.
    A = g["A"]
    s.add(Curve(np.array([[0.0, 0.0], list(A)]), role=Role.CONSTRUCTION, width_scale=0.8))
    tangent_L_at_A = g["L_pts"][1] - g["L_pts"][0]
    s.add(RightAngleMark(corner=tuple(A), dir1=tuple(-A), dir2=tuple(tangent_L_at_A),
                          size=0.1))

    s.add(Point(tuple(g["P"]), role=Role.CONTENT, radius_scale=1.1))
    s.add(MathLabel(r"P", tuple(g["P"]), ha="left", va="bottom",
                     offset_px=(6, -42), size_pt=13))

    s.add(Point(tuple(g["A"]), role=Role.ANNOTATION, radius_scale=0.8))
    s.add(Point(tuple(g["B"]), role=Role.ANNOTATION, radius_scale=0.8))
    s.add(MathLabel(r"A", tuple(g["A"]), ha="right", va="top", offset_px=(-4, 8)))
    s.add(MathLabel(r"B", tuple(g["B"]), ha="left", va="top", offset_px=(4, 8)))

    mid = g["L_pts"][len(g["L_pts"]) // 2]
    s.add(MathLabel(r"L", tuple(mid), ha="center", va="top", offset_px=(0, 10),
                     role=Role.ACCENT1, size_pt=13))

    s.add(Callout(r"\text{asymptotic to } L \text{ at } A",
                  anchor=(-1.05, 0.55),
                  target=tuple(g["pa_pts"][int(0.86 * len(g["pa_pts"]))]),
                  size_pt=9, role=Role.ANNOTATION))
    s.add(Callout(r"\text{asymptotic to } L \text{ at } B",
                  anchor=(1.05, 0.55),
                  target=tuple(g["pb_pts"][int(0.86 * len(g["pb_pts"]))]),
                  size_pt=9, role=Role.ANNOTATION))

    s.add(MathLabel(
        rf"\text{{{n_fam} of infinitely many non-crossing geodesics through }} P",
        (0.0, -1.14), ha="center", va="top", size_pt=9, role=Role.ANNOTATION))

    return s


def assertions(g):
    c = Checks()
    tol_orth = 1e-6
    tol_geo = 1e-6

    def check_orthogonal(rep, z, name):
        c.check(abs(_tangent_dot_at_boundary(rep, z)) < tol_orth,
                f"{name} meets |z|=1 at {z} non-orthogonally")

    # L meets the boundary orthogonally at both A and B.
    check_orthogonal(g["repL"], g["A"], "L")
    check_orthogonal(g["repL"], g["B"], "L")

    # the two limiting parallels meet the boundary orthogonally at BOTH
    # of their ideal endpoints (the shared one and the derived one), and
    # each has no OTHER intersection with L strictly inside the open disc
    # -- only the shared ideal point, which sits on |z|=1, not inside it.
    check_orthogonal(g["rep_pa"], g["A"], "P-A limiting parallel")
    check_orthogonal(g["rep_pa"], g["other_a"], "P-A limiting parallel")
    c.check(not _crosses_open_disc(g["rep_pa"], g["repL"]),
            "P-A limiting parallel crosses L strictly inside the disc")

    check_orthogonal(g["rep_pb"], g["B"], "P-B limiting parallel")
    check_orthogonal(g["rep_pb"], g["other_b"], "P-B limiting parallel")
    c.check(not _crosses_open_disc(g["rep_pb"], g["repL"]),
            "P-B limiting parallel crosses L strictly inside the disc")

    P = g["P"]
    for i, mem in enumerate(g["family"]):
        rep = mem["rep"]
        # genuinely passes through P
        if rep[0] == "line":
            u = np.array([np.cos(rep[1]), np.sin(rep[1])])
            perp_dist = abs(P[0] * u[1] - P[1] * u[0])
        else:
            _, ctr, r = rep
            perp_dist = abs(np.hypot(*(P - ctr)) - r)
        c.check(perp_dist < 1e-6, f"family[{i}] does not pass through P (err {perp_dist:.2e})")
        # orthogonal at both its ideal endpoints
        check_orthogonal(rep, mem["Q1"], f"family[{i}] at Q1")
        check_orthogonal(rep, mem["Q2"], f"family[{i}] at Q2")
        # genuinely does not cross L inside the open disc
        c.check(not _crosses_open_disc(rep, g["repL"]),
                f"family[{i}] (theta={np.degrees(mem['theta']):.1f} deg) crosses L")

    # the family-selection logic is not vacuous: a geodesic through P aimed
    # into the excluded wedge (between the two limiting parallels, on the
    # side facing L) DOES cross L. If this ever stopped failing, the wedge
    # bookkeeping above would be silently picking the whole circle.
    c.check(_crosses_open_disc(g["rep_witness"], g["repL"]),
            "witness geodesic (chosen from the excluded wedge) unexpectedly misses L "
            "-- the non-crossing wedge is mis-bounded")

    c.done()
