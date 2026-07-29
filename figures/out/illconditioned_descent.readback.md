# Readback record

**Verdict:** pass

## Intended claim

Gradient descent's iteration count is governed by the Hessian's condition number kappa, not the step size: on the same ill-conditioned quadratic, plain steepest descent needs ~kappa iterations to zig-zag to tolerance, heavy-ball momentum cuts that to ~sqrt(kappa) by damping the oscillation rather than shrinking the step, and Newton's method reaches the optimum in a single step because it rescales by the Hessian and so never sees kappa at all.

## Cold readback

GLANCE: A tilted elliptical bowl (contour lines) with three descent paths racing from a start point to the minimum — the figure is asserting that some optimization methods reach the bottom in far fewer steps than others.

STUDY:
2. The number of iterations an optimizer needs to converge on an ill-conditioned quadratic scales with the condition number kappa — linearly for gradient descent, like sqrt(kappa) for momentum, and O(1) (independent of kappa) for Newton's method.
3. - The path traces (dark blue, ochre, teal) are dense zigzag/spike patterns that overlap heavily near the target, making it hard to visually count or distinguish individual iterations from the legend numbers alone.
   - It's not obvious whether the dashed ellipses are literal level sets of the same quadratic being optimized or a generic schematic backdrop.
   - The teal Newton path looks like a single straight segment, but had to guess whether that's literally one step or a smoothed/simplified rendering of one.
4. Verifiable by inspection: the relative visual iteration counts (gradient descent zigzags far more than momentum, which zigzags far more than Newton's near-straight line), and that all three paths start at x0 and end near x-star, consistent with the stated 58/18/1 counts and the elongated (ill-conditioned) elliptical contours. Taken on trust: that kappa=L/mu=25 is the actual condition number used to generate these specific curves, and the general scaling laws N_GD~kappa, N_mom~sqrt(kappa), N_Newton=1 — the figure shows one instance, not a proof of the asymptotic scaling claim.

## Notes

GLANCE correctly identifies the core claim (methods differ sharply in steps to converge on an ill-conditioned bowl); STUDY recovers the kappa / sqrt(kappa) / O(1) scaling exactly, matching CLAIM. Two accepted minor ambiguities, not fixed: (1) whether the dashed ellipses are literal level sets — they are, by construction and by assertion (7) in the figure program, but no on-figure label says so explicitly; left as accepted since the kappa/valley-shape reading already lands. (2) whether the teal segment is literally one step or simplified — it is literally one step (drawn from the exact computed x_1); the reader guessed correctly. Both are judged non-blocking: the reader recovered the intended claim at both GLANCE and STUDY without needing either clarified.
