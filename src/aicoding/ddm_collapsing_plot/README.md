# ddm_collapsing_plot (#17)

Drift diffusion model with collapsing (time-varying) bounds.

## Approach

- Extends the standard DDM with exponentially collapsing boundaries: `bound(t) = a/2 * exp(-collapse_rate * t)`.
- Returns the full timeseries of diffusion steps for each trial (as required by the prompt).
- Symmetric collapse around the midpoint ensures both boundaries shrink equally.
- Includes a plotting function that overlays trajectories with the collapsing bound envelope.
