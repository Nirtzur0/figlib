# mobius_classification — friction record

## What the claim needed

A single algebraic pipeline (matrix -> tau^2 -> multiplier m -> orbit
family in normal coordinates -> pullback through F^{-1}) run four times,
with the fixed-point count/type as the only structural difference
between the parabolic panel and the other three. Needed: a 2x2 grid of
independently-scaled Scenes (no shared-frame correspondence — these are
four instances of one taxonomy, not a mapped pair), a background family
vs. one distinguished orbit (WEIGHT_BG/WEIGHT_ACTOR), fixed-point
callouts, and per-panel title + tau^2 annotation clear of a dense curve
population.

## What figlib gave for free

`Figure(panels=..., grid=(2,2))` handled the whole layout problem —
independent per-panel Transforms, row/col slot geometry, panel frames —
with zero custom code. `Callout` (boxed label + leader) turned out to be
the correct primitive for the fixed-point labels specifically *because*
it is exempt from the label-on-ink check regardless of theme
transparency, unlike `MathLabel(halo=True)`. `WEIGHT_BG/CONTENT/ACTOR`
solved the expressivity "flat hierarchy" gate outright — one line to
pick an actor index, one `if` in the emit loop.

## What I hand-rolled

- **The unifying orbit-family parametrization** — one closed-form
  `t -> (s0 + t ln|m|, phi0 + t arg(m))` in log-polar (bipolar)
  coordinates, normalized so `t=1` is *exactly* one application of M.
  This is the same construction `vca_fig30_elliptic_checkerboard.py`
  already has (its `zmap`/`F`), generalized from the elliptic-only case
  (`ln|m|=0`) to all three two-fixed-point classes by letting `ln|m|`
  vary. ~40 lines. A `mobius.py` builder
  (`two_fixed_orbit_family(xi_plus, xi_minus, m, phi0s, s0s, t_range)` +
  `parabolic_orbit_family(p, t, offsets, u_max)` + `classify_trace2`)
  would let both figures share it instead of each hand-deriving the
  bipolar conjugation.
- **`halo=True` vs. transparent themes, discovered the hard way.** I
  first wrote the title/tau^2/fixed-point labels with `halo=True`
  expecting the corridor exemption `docs/skill.md` implies for haloed
  labels. The gate kept failing them: `gates.py` only grants the
  exemption when `not style.transparent` (a halo with no ground to
  paint is a no-op). Fix was architectural, not cosmetic — move the
  title into a reserved blank margin band (`ylim = (-lim, lim+0.62)`,
  geometry never drawn there) and switch the fixed-point labels to
  `Callout` (exempt unconditionally when `boxed=True`). ~15 lines, but
  cost a full render-inspect cycle to diagnose because the diagnostic
  text ("free: offset_px += ...") reads like autoplace should have
  already applied it, and doesn't say *why* it didn't.

## Gate diagnostics that did NOT contain the fix

- `label-on-ink` on a `halo=True` label suggests a nudge as if the label
  were an ordinary one — it does not surface that the halo declaration
  is being silently ignored because the render is groundless. The
  actual fix (stop relying on halo; use Callout, or reserve blank
  geometry) is not in the diagnostic at all; I only found it by reading
  `gates.py`'s `_label_ink_checks` source.

## Renders to first green: 4

(One further verification run after green, to confirm exit code — 5
`make check` invocations total, 4 of which changed the figure.)

## Proposed primitive

```python
# figlib/mobius.py
def two_fixed_orbit_family(
    xi_plus: complex, xi_minus: complex, m: complex,
    phi0s: Sequence[float], s0s: Sequence[float],
    t_range: tuple[float, float], n_t: int = 400,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Invariant orbit curves of the Mobius map with fixed points
    xi_plus/xi_minus and multiplier m, in the bipolar/log-polar
    conjugation (t=1 == one application of M). Elliptic (|m|=1),
    hyperbolic (m real), loxodromic (general) all fall out of the same
    formula; only t_range and which of {phi0s, s0s} varies differ."""

def parabolic_orbit_family(
    p: complex, t: complex, offsets: Sequence[float], u_max: float,
    n_t: int = 400,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Horocycle pencil at the double fixed point p, tangent direction
    arg(t); offset=0 draws the degenerate line through p."""

def classify_trace2(a, b, c, d) -> tuple[complex, str]:
    """(tau^2, class) from a Mobius matrix; class in
    {elliptic, parabolic, hyperbolic, loxodromic}."""
```

This is `vca_fig30_elliptic_checkerboard.py`'s `F`/`zmap`/orbit-cell
machinery, generalized and named — worth extracting once a second
figure (this one) needed the same conjugation for a case fig30 never
drew.
