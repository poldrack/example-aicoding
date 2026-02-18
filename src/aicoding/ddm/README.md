# ddm (#4)

Drift diffusion model simulation and EZ-diffusion parameter recovery.

## Approach

- **DDM simulation**: Random walk with drift, stepping by `v*dt + noise*sqrt(dt)*randn` until hitting the upper or lower boundary.
- **EZ-diffusion**: Recovers drift rate (v), boundary separation (a), and non-decision time (t0) from accuracy and RT variance using the closed-form equations from Wagenmakers, van der Maas, & Grasman (2007).

The EZ equations use the logit of accuracy, RT variance of correct responses, and mean RT to estimate all three parameters.
