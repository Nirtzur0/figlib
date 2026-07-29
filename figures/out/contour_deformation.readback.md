# Readback record

**Verdict:** pass

## Intended claim

The integral of f(z) = Res/(z - z0) around a closed loop depends only on the loop's homotopy class relative to z0, not on its shape: three visibly different loops enclosing z0 all carry the identical value 2 pi i * Res, while dragging that same loop bodily across z0 instead of shrinking it around z0 changes the value discontinuously, by exactly 2 pi i * Res, at the one instant the boundary sweeps over the pole.

## Cold readback

GLANCE

At first glance: three side-by-side loop diagrams, all showing the same closed-loop formula equal to 2πi, plus a fourth panel below showing a circle physically moved to a new position. The figure seems to be asserting that a contour integral's value (2πi) is invariant under continuous deformation of the loop — until panel [d] where something about the drag breaks that invariance.

STUDY

2. Claim: the contour integral ∮ dz/(z−z0) is invariant under continuous deformation of the contour as long as the pole z0 stays enclosed, but the moment the contour is dragged across the pole the value jumps discontinuously by −2πi·Res(f,z0) (i.e., from 2πi to 0).

3. Points of ambiguity:
   - In [d], it's not obvious from the static image which circle is "before" and which is "after" — inferred from the dashed/dotted circle labeled "before: 2πi" versus the solid one labeled "after: 0," plus the arrow direction.
   - The dotted circle in [d] (overlapping the dashed one) is unlabeled beyond the caption "mid-drag: boundary sits ON z0" — its role as the singular transitional configuration (no inside/outside) had to be inferred rather than stated with an explicit value.
   - Panels [a]-[c] show progressively smaller/simpler loops (blob → smoother blob → circle) all enclosing z0 with the same equation underneath — the progression as a demonstration of shape-independence had to be inferred from the sequence, since nothing draws the deformation itself.

4. Verifiable by inspection: the orientation arrows (all counterclockwise) are visible and consistent across [a]-[d]; the enclosure of z0 inside the loop in [a]-[c] is visually clear; the final circle in [d] visibly excludes the marked z0. Taken on trust: the numerical claim that the integral equals 2πi (an assertion, not something the drawing itself proves) and the residue formula Δ = −2πi·Res(f,z0) in [d], asserted as text rather than derived from the picture.

BINDING

5. Same across parts: the pole/marked point z0 (red ×), the integrand and contour-integral notation ∮_γ dz/(z−z0), the counterclockwise orientation convention (arrowheads), and the general "closed loop encircling a marked point" setup restated four times.

6. Differences noticed:
   - Shape of the loop: irregular blob [a] → smoother rounded triangle [b] → circle [c] → circle again [d]
   - Size of the loop: shrinks from [a] to [c], then in [d] two different circle sizes appear (dashed larger, solid smaller)
   - Position: in [a]-[c] the loop is centered on z0; in [d] the loop translates away from z0 entirely
   - Number of loops per panel: one in [a]-[c], but two-to-three overlapping circles in [d] (dashed "before," dotted "mid," solid "after")
   - Presence of a translation arrow (blue) only in [d]
   - The stated integral value: 2πi in [a]-[c] but a transition to 0 (with a labeled jump Δ) in [d]
   - Labeling: [a]-[c] just show the formula; [d] adds prose labels ("dragged across z0," "before: 2πi," "after: 0," "mid-drag: boundary sits ON z0") and an explicit Δ formula not present elsewhere

7. The single difference the figure is arguing about: whether the contour encloses z0 or not — [a]-[c] establish that shape/size deformation doesn't change the integral as long as z0 stays inside, and [d] shows what happens when the deformation instead moves the contour so that z0 crosses from inside to outside the loop.

## Notes

Both GLANCE and STUDY recovered the intended claim, including its two halves (invariance, then the one discontinuous exception) and the correct sign/magnitude of the jump. Binding answers 5-7 correctly identify the pole, the notation, and the orientation convention as the fixed set, and correctly name "enclosure of z0" as the one difference the figure argues about — matching the declared `CORRESPONDENCE` (parts [a]-[c] only; [d] deliberately outside it).

Two residual ambiguities, both accepted rather than fixed:
- Distinguishing [d]'s before/after circles requires reading the labels, not just the geometry. Accepted: the dash/solid role convention (CONSTRUCTION vs CONTENT) is the corpus's standard "state before" vs "the object" contrast, and the labels are exactly the mechanism annotation design step 5 calls for — delegating this one read to text is cheaper than a heavier visual encoding (e.g. color) that the corpus reserves for correspondence, not history.
- The transitional (dotted) loop in [d] is not assigned an explicit numeric value. Accepted: it deliberately has none — at that configuration the integral is not the residue theorem's business (the pole sits ON the contour), so printing a value there would be printing a claim the mathematics does not make.

An earlier draft additionally had a stray "Re" axis label in panel [a] with no matching Im axis anywhere in the figure; a first cold read flagged it as confusing/orphaned. Removed rather than completed into a full axis, since it carried no part of the claim.
