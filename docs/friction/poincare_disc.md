# poincare_disc — friction record

## What the claim needed

Hyperbolic geodesics as arcs orthogonal to the unit circle; a fixed
geodesic L; an external point P; the two geodesics through P that share
exactly one ideal endpoint with L (the limiting/asymptotic parallels);
and a finite sample of the genuinely infinite family of geodesics through
P that never meet L at all. The hard part was not drawing an arc — it was
getting the *combinatorics* right: which directions through P actually
avoid L. A first hypothesis (arc-of-the-boundary-circle "same side as P"
via the Euclidean inside/outside test against L's orthogonal circle) is
wrong — it misclassifies a whole wedge of directions, including the
direction straight through L's own closest point to the origin. The
correct criterion only fell out after treating each candidate direction
as a *complete* geodesic (both ideal endpoints), numerically scanning the
full boundary circle for which endpoint-pairs' full lines cross L's
circle inside the open disc, and reading off that the safe set is
literally bounded by the two P-to-{A,B} geodesics.

## What figlib gave for free

Scene/theme/gate machinery end to end: `Role.ACCENT1`/`ACCENT2` for the
distinguished objects, `theme.ramp` for the ordered sweep, `Callout` for
labels that had to leave the crowded interior, `RightAngleMark` for the
one orthogonality witness, `Checks` for a multi-assertion gate report,
and the mechanical/expressivity/color gates catching two real placement
bugs (a label sitting on its own curve; a callout arrow pointing at the
wrong end of an arc) on the first two runs.

## What I hand-rolled

- **Orthogonal-circle geodesic construction** (~90 lines): `_repr`,
  `_boundary_pts`, `_other_endpoint`, `_sample_arc`,
  `_pairwise_intersections`, `_tangent_dot_at_boundary`. Given two points
  in the closed disc (interior or boundary, either or both), find the
  unique circle (or diameter line) through them orthogonal to |z|=1, pick
  the correct one of its two arcs (the one staying inside the disc), and
  test two such geodesics for an interior crossing. None of this is
  disc-specific glue — it is the entire Poincaré-disc geodesic
  primitive, and the same four or five functions would be needed by any
  future hyperbolic figure (ideal triangles, {p,q} tessellations,
  Fuchsian group orbits).
- **The non-crossing wedge logic** (~20 lines): given P and L, derive the
  two limiting-parallel geodesics and use them (not a naive "side of a
  circle" test) to bound the open arc of safe directions. This is the
  correct mental model (an external point's view of a geodesic is
  blocked by a "shadow" wedge bounded by the two asymptotic parallels)
  but nothing in the library suggested it; I only found it by writing a
  disposable numerical sweep script outside the figure to falsify my
  first (wrong) hypothesis before trusting any assertion.

## Gate diagnostics that did NOT contain the fix

None outright wrong, but two were incomplete on their own:
- `label-on-ink: '\text{asymptotic...}' ... no free single-axis nudge;
  nearest ink-free region center (...)` — correct diagnosis, but the fix
  (switch `MathLabel` to `Callout` with an anchor in that region) had to
  be inferred; the gate does not suggest the primitive swap.
- Nothing else — orthogonality/non-crossing bugs are math bugs, not gate
  territory; the numerical assertions were what actually caught them
  (the witness-geodesic assertion exists precisely because the wedge
  logic was wrong once already, silently, with no failing gate to catch
  it — only a by-hand recomputation did).

## Renders to first green: 5

## Proposed primitive

```python
def geodesic_between(z1: tuple[float, float], z2: tuple[float, float],
                      n: int = 200) -> tuple[np.ndarray, GeodesicRep]:
    """Poincare-disc geodesic through z1, z2 (interior or boundary,
    |z| <= 1): sampled arc + its ('circle', center, radius) | ('line',
    theta) representation, for reuse in intersection/orthogonality
    checks. Lives beside cassinian_curves in figlib.geometry — every
    hyperbolic-disc figure (ideal triangles, {p,q} tilings, Mobius
    orbits) needs exactly this and currently has to reimplement it."""
```

`geodesic_between` plus a `geodesics_cross(repA, repB) -> bool` sibling
(the open-disc pairwise-intersection test) would cover the primitive; the
orthogonality tangent-dot check is cheap enough to leave in each figure's
`assertions()`, since what counts as "the boundary point to check" is
figure-specific.
