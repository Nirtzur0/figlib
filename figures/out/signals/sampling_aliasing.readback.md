# Readback record

**Verdict:** pass

## Intended claim

Sampling at rate omega_s convolves a spectrum with an impulse comb of spacing omega_s, replicating the triangle at every multiple of omega_s; the replicas stay separated exactly when omega_s > 2*omega_m, and when omega_s < 2*omega_m adjacent replicas overlap in a band of width 2*omega_m - omega_s where the spectrum is unrecoverable.

## Cold readback

GLANCE: Three stacked frequency-domain spectra: an original band-limited triangular spectrum, then its periodic replicas after sampling — asserting that sampling replicates the spectrum, and that when replicas crowd together they overlap ("alias").

STUDY: sampling a band-limited signal X(jw) (support [-w_m, w_m]) at rate w_s replicates its spectrum at multiples of w_s; when w_s > 2w_m the replicas stay disjoint (panel b) and the original is recoverable, but when w_s < 2w_m adjacent replicas overlap in the shaded regions — aliasing — i.e. the Nyquist condition w_s >= 2w_m.

Confusions raised: (1) dashed box labeled "T" ambiguous (width vs gain); (2) the composite (summed) spectrum in panel c is not drawn, only the overlapping replicas; (3) the red arrows labeled 1/T required a guess — spectrum of the impulse train, superimposed?

## Notes

Glance and study both land the claim (Nyquist condition read geometrically). Fixes applied in response: comb relabeled S(j\omega) (the 1/T weight label was in fact wrong — the impulse-train spectrum has weight 2\pi/T); recovery box relabeled H_r(j\omega) to kill the width-vs-gain ambiguity. Accepted: the summed spectrum in panel c is deliberately not drawn — the overlapping components ARE the argument (the O&S convention); superposing S(j\omega) with the replicas is a pedagogic overlay, now named.
