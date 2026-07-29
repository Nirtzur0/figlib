# Readback record

**Verdict:** pass

## Intended claim

The 2D Ising magnetisation |m(T)| vanishes CONTINUOUSLY at T_c = 2/ln(1+sqrt2) via the exact Onsager law, and only exactly at T_c does a sampled spin configuration show clusters at every scale - the ordered and disordered configurations either side do not.

## Cold readback

GLANCE: A three-panel gallery of blue/gold speckled grids getting
progressively noisier left to right, paired with a curve that drops sharply
from 1 to 0 -- arguing that some order parameter collapses abruptly as a
control parameter crosses a threshold.

STUDY: the 2D Ising magnetization |m| stays near saturation below a sharp
critical temperature T_c = 2/ln(1+sqrt2) and drops to zero above it -- a
genuine non-analytic phase transition, and the three spin configurations are
point-samples on that curve below, at, and above T_c.

Friction:
- the curve looks almost like a step function near T_c; could not tell from
  the image alone whether |m| touches 0 continuously or jumps.
- panel [b]'s dot sits at m~0.73, not at the visual halfway point of the
  drop, which read as a mismatch until re-examining that this is the correct
  finite-size critical magnetisation.
- no legend ties gold/blue to spin sign explicitly (inferred from context).

Binding: same object (one lattice, one order parameter) tracked at three T;
same color encoding across grids. Differences noticed: T (intended),
visual disorder/domain size, dot height on the curve, and non-monotonic
cluster SIZE (panel [b]'s clusters look blobbier despite less total gold
than [c]) -- correctly read as the correlation length peaking at T_c. The
argued difference: T driving |m| through a sharp transition at T_c.

## Notes

Glance recovers the claim. Two accepted defects, not fixed:
1. The curve reading as a near-step at T_c is the true physics (critical
   exponent beta=1/8 makes the Onsager curve genuinely steep near T_c, not
   a drawing choice) -- accepted, not a defect.
2. No in-figure legend maps blue/gold to spin sign; the reader inferred it
   correctly from context without needing one. Accepted: a legend for a
   binary categorical channel already separated by the theme's
   categorical(0)/(1) hues would be redundant ink.

The cold reader independently named the mechanism the figure exists to
show -- cluster size peaking at T_c, non-monotonic in total gold fraction
-- without being told to look for it.
