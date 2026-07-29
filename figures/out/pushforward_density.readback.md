# Readback record

**Verdict:** pass

## Intended claim

Mass is conserved under a pushforward, area is not: pushing a uniform
density through a nonlinear map T, each cell's image density equals the
source density divided by that cell's own |det J_T| — the cell that
grows gets visibly lighter, the cell that shrinks visibly darker, read
straight off the drawn polygons' shoelace areas.

## Cold readback

Cold-reader agent (no other context), run against
`figures/out/pushforward_density.png`:

GLANCE: "A grid gets warped by a transformation T, and color (a density)
intensifies where the warped cells shrink — this looks like it's
asserting that squeezing space concentrates density." — claim recovered
at a glance.

STUDY claim: "under a change of variables, probability/mass density
transforms inversely with the local area scaling factor —
rho_Y = rho_X / |det J_T|, so where the map T compresses area
(det J_T < 1) density increases, and where it expands area
(det J_T > 1) density decreases, keeping total mass conserved." —
matches the intended claim exactly.

Confusions raised (pre-fix): whether rho_X was really uniform, since
panel [a] carried no numeric label — fixed by adding the
"rho_X = 1.00 (uniform)" annotation to panel [a]; the reader had
inferred it correctly from the flat color anyway, but no longer has to
guess. Remaining, accepted: the reader verified the rho = 1/det J_T
relation arithmetically only for the two labeled cells, not for the
full 25-cell grid — correct, and exactly what the numerical gate
(assertions()) exists to certify instead of asking the reader to.

Binding (3 parts — panels a, b, colorbar): reader correctly identified
the 5x5 grid as the same 25 regions before/after T, the two
accent-outlined cells as the same tracked cells across panels, and the
colorbar as the shared rho scale. Reader listed grid shape, cell size,
and color as differences, and correctly named the cell-wise Jacobian
determinant as the one difference the figure argues about.

## Notes

One readback iteration: the agent's only confusion (uniform rho_X not
labeled) was fixed by adding a direct annotation rather than trusting
the reader's correct inference — cheap, and removes a guess from the
glance-to-study path per the design step 8 honesty pass.
