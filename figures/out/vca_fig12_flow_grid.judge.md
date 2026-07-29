# Judge record — vca_fig12_flow_grid

**Verdict: RECREATION BETTER**

## Recreation defects (verbatim)

1. truncated streamlines mid-canvas (top over disc; bottom-left squared end) read as rendering errors.
2. No frame; ragged grid edge.
3. Label halos bite the grid (ΔΨ = ΔΦ, Φ = Re Ω).
4. Family labels float unattached; leader ticks needed.
5. Grid breakdown at ±1 shown by absence; draw the critical Φ = ±2 curves branching through ±1 at 45°.
6. Disc interior blacked out (book shows dipole grid; dual-interpretation content lost).
7. "Ω′( ± 1) = 0" spurious spaces.
8. Right-angle marker in a cheap far-field location; earns more at the disc shoulder.

## Book defects fixed by recreation (verbatim)

all mechanism off-figure; stagnation points nearly invisible; ink clogging near disc; blobbing arrowheads; orthogonality implicit; no sampling rule.

## Post-fix disposition

1. **fixed** — the exits were honest (the Ψ = ±1.8 streamlines leave through the
   top/bottom of the window; the bottom-left squared end was a label-halo bite),
   but nothing said so. Now a dotted rounded frame bounds the window, every
   curve terminates on it (corner cuts bisected onto the frame arc, no stubs),
   and the halo that bit the bottom-left streamline is gone with the halos.
2. **fixed** — rounded dotted FRAME-role border at the flow window; all three
   curve families trimmed to it, so the ragged edge reads as a window into the
   infinite flow. (Scene.clip would have clipped the margin labels with the
   curves — the same clip group carries all items — so the clipping is done in
   compute() against the rounded window instead.)
3. **fixed** — no haloed labels remain anywhere. All labels sit in ink-free
   margin bands outside the window or in paper ink on the solid disc; content
   ink is never erased. The new label-on-ink gate (corridor checks) passes.
4. **fixed** — Ψ = Im Ω rides a leader tick continuing a streamline through the
   right frame edge (only streamlines reach that edge — attachment unambiguous
   by position AND style); Φ = Re Ω rides a dashed tick continuing an
   equipotential's tangent through the bottom frame. Both exits are solved on
   the labelled level and asserted.
5. **fixed** — the critical Φ = ±2 level curves are drawn in a finer
   subordinate dash, branching through ±1 at 45°; each branch tip is snapped to
   the exact stagnation point, with the pre-snap gap asserted below the contour
   mask resolution (< 0.08), plus a new assertion that every stagnation point
   is touched exactly. The degeneracy is now shown geometrically, with the red
   dots marking it.
6. **accepted-with-reason** — scope choice, kept deliberate: the dipole grid
   inside the disc is the book's *second* claim, and drawing it faintly inside
   a dark obstacle either fails the contrast gates or lightens the disc and
   weakens the obstacle read. The disc stays solid and now carries the two
   formulas (Ω = z + 1/z, Ω′(±1) = 0) in paper ink — the dead space is used,
   the exterior argument stays clean. Noted in the module docstring.
7. **fixed** — `\Omega'({\pm}1) = 0`: bracing the ± sets it as an ordinary
   atom, so it renders Ω′(±1) = 0 with tight parentheses.
8. **fixed** — marker moved to the (Ψ, Φ) = (0.3, 1.5) crossing on the
   upper-right disc shoulder (a visibly curved cell where orthogonality is the
   surprising claim), placed by Newton solve on both levels with analytic
   tangents; asserted on-crossing to 1e-10 and perpendicular to 1e-12.

Final gate: `figcheck` **PASS** (including the concurrently added
label-on-ink and arrow-on-mark diagnostics); all module assertions pass,
including the new critical-curve-through-±1 checks.
