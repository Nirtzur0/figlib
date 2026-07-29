# Readback record

**Verdict:** pass

## Intended claim

Starting from the same draws of pure noise, the jittery reverse-time SDE paths and the smooth deterministic probability-flow ODE paths of a diffusion model transport the standard-normal prior into the same two-mode data distribution, because both integrate the same score field and share the same marginals at every intermediate time.

## Cold readback

For the same score field, the stochastic reverse-time SDE and the deterministic probability-flow ODE are two different samplers of the same generative process - both transport the Gaussian prior N(0,1) to the same data distribution p0, with the ODE giving smooth non-crossing trajectories and the SDE noisy paths, but identical endpoint marginals (bimodal p0).

## Notes

Cold reader matched the claim. Its three confusions were fixed in revision: dashed branch-mean curves removed (misread as start-end pairing), SDE endpoints marked, and a 70-sample SDE endpoint strip added inside the data lobe so the marginal claim is shown rather than asserted.
