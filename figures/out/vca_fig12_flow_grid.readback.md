# Readback record

**Verdict:** pass

## Intended claim

The level curves of Psi = Im Omega and Phi = Re Omega for Omega = z + 1/z form an orthogonal grid of near-squares that flows around the disc and breaks down only at the stagnation points +-1, where the map ceases to be conformal.

## Cold readback

GLANCE: Uniform flow past a circular cylinder — streamlines bending around a black disk with a crossing family of dashed curves, apparently asserting that the flow field around the obstacle forms an orthogonal curvilinear grid (equipotentials perpendicular to streamlines).
STUDY claim: The complex potential Omega = z + 1/z solves ideal flow past the unit cylinder, and because Omega is analytic, its level sets Phi = Re Omega (equipotentials, dashed) and Psi = Im Omega (streamlines, solid) form an everywhere-orthogonal grid with equal spacing (Delta Psi = Delta Phi) — orthogonality failing only at the stagnation points z = +-1 where Omega'(+-1) = 0 and the grid degenerates.
Confusions: (a) both family labels sit near solid curves — a cold reader could attach Phi to the solid lines; (b) Delta Psi = Delta Phi stated but not visually checkable; (c) the single right-angle marker is easy to miss; global vs local orthogonality assertion unclear.
On trust: curves actually level sets of z+1/z; increments equal; disc is unit circle (no scale ticks).

## Notes

Per-confusion disposition after the verification-round fixes:

(a) family labels attachable to the wrong family — FIXED. Both labels moved
outside the flow window and tied to their own curve by a leader tick that
touches it in the family's own line style: Psi = Im Omega sits at a
streamline's right-edge exit (only streamlines reach the right edge, so the
attachment is unambiguous) with a solid tick continuing the streamline
through the dotted frame; Phi = Re Omega sits under an equipotential's
bottom-edge exit with a dashed tick continuing that curve's tangent through
the frame. Tick-to-curve contact is exact (exits solved on the labelled
levels and asserted).

(b) Delta Psi = Delta Phi stated but not visually checkable — ACCEPTED with
reason. The single ladder step is enforced by construction (one dk for both
families) and asserted mechanically; the near-square cells across the whole
window are the visual evidence. A measurement glyph (brace across one cell)
would add annotation load to an already dense grid without adding claim.

(c) right-angle marker easy to miss; global-vs-local unclear — FIXED in
placement, accepted in count. The marker moved from a near-rectilinear
far-field cell to the (Psi, Phi) = (0.3, 1.5) crossing on the disc
shoulder, where the cells are visibly curved and orthogonality is the
surprising claim; it sits at a true drawn crossing with analytic tangents
(asserted on-crossing and perpendicular). It stays a single witness by
design — the global claim is carried by the grid itself, now including the
critical Phi = +-2 curves branching through +-1 at 45 degrees, which show
where the claim fails.

On-trust items: on-level membership, equal increments, and the unit disc
are all asserted mechanically (on-level tolerance against one ladder,
r_min gate at the disc boundary). Scale ticks are omitted as the book
does; the disc itself is the unit of length.
