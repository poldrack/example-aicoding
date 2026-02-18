"""Drift Diffusion Model simulation and EZ-diffusion parameter recovery.

Simulates a DDM with given parameters and recovers them using the
EZ-diffusion method (Wagenmakers, van der Maas, & Grasman, 2007).
"""

import numpy as np


def simulate_ddm(n_trials, drift_rate, boundary, start_point=0.5, dt=0.001,
                 noise_sd=1.0, nondecision_time=0.3, seed=None):
    """Simulate a drift diffusion model.

    Parameters
    ----------
    n_trials : int
        Number of trials to simulate.
    drift_rate : float
        Drift rate (v). Positive values bias toward the upper boundary.
    boundary : float
        Boundary separation (a). The upper boundary is at ``a`` and the
        lower boundary is at 0.
    start_point : float
        Relative starting point (z/a), between 0 and 1. 0.5 is unbiased.
    dt : float
        Time step for the simulation.
    noise_sd : float
        Standard deviation of the within-trial noise.
    nondecision_time : float
        Non-decision time added to RT.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    dict
        'rt': array of response times, 'choice': array of choices (1=upper, 0=lower).
    """
    rng = np.random.RandomState(seed)
    z = boundary * start_point  # absolute starting point

    rts = np.empty(n_trials)
    choices = np.empty(n_trials, dtype=int)

    for i in range(n_trials):
        x = z
        t = 0
        while True:
            x += drift_rate * dt + noise_sd * np.sqrt(dt) * rng.randn()
            t += dt
            if x >= boundary:
                choices[i] = 1
                rts[i] = t + nondecision_time
                break
            elif x <= 0:
                choices[i] = 0
                rts[i] = t + nondecision_time
                break

    return {"rt": rts, "choice": choices}


def ez_diffusion(rt, choices):
    """Recover DDM parameters using the EZ-diffusion model.

    Based on Wagenmakers, van der Maas, & Grasman (2007). Uses mean RT,
    variance of RT, and accuracy to estimate drift rate, boundary separation,
    and non-decision time.

    Parameters
    ----------
    rt : array-like
        Response times.
    choices : array-like
        Binary choices (1 = upper/correct, 0 = lower/incorrect).

    Returns
    -------
    dict
        'drift_rate' (v), 'boundary' (a), 'nondecision_time' (t0).
    """
    rt = np.asarray(rt, dtype=float)
    choices = np.asarray(choices)

    # Proportion of upper-boundary (correct) responses
    pc = np.mean(choices == 1)

    # Edge correction to avoid log(0) issues (Snodgrass & Corwin, 1988)
    n = len(choices)
    if pc == 0.0:
        pc = 0.5 / n
    elif pc == 1.0:
        pc = 1.0 - 0.5 / n

    # Compute the logit-transformed accuracy
    # EZ equations use Pc mapped through the logit
    s = 1.0  # scaling parameter (noise sd), fixed to 1 in EZ-diffusion

    # VRT = variance of RT for correct responses
    correct_mask = choices == 1
    if np.sum(correct_mask) < 2:
        # Not enough correct responses; use all
        mrt = np.mean(rt)
        vrt = np.var(rt)
    else:
        mrt = np.mean(rt[correct_mask])
        vrt = np.var(rt[correct_mask])

    # EZ-diffusion equations (Wagenmakers et al., 2007, Eq. 1-3)
    # Using the logit: L = log(pc / (1 - pc))
    L = np.log(pc / (1 - pc))

    # Sign of drift rate depends on accuracy
    sign_v = 1.0 if pc >= 0.5 else -1.0

    # x = L * (L * pc**2 - L * pc + pc - 0.5) / vrt
    # This is the EZ formula for v^4 / a^2
    x = L * (L * pc**2 - L * pc + pc - 0.5) / vrt if vrt > 0 else 0.01

    # Drift rate: v
    v = sign_v * (x ** 0.25) * s

    # Boundary separation: a
    a = (s**2 * L) / v if abs(v) > 1e-10 else 1.0

    # Non-decision time: t0 = MRT - (a / (2*v)) * (1 - exp(-v*a/s^2)) / (1 + exp(-v*a/s^2))
    exp_term = np.exp(-v * a / s**2)
    t0 = mrt - (a / (2 * v)) * (1 - exp_term) / (1 + exp_term) if abs(v) > 1e-10 else mrt

    return {
        "drift_rate": v,
        "v": v,
        "boundary": abs(a),
        "a": abs(a),
        "nondecision_time": max(t0, 0),
        "t0": max(t0, 0),
    }


if __name__ == "__main__":
    print("Drift Diffusion Model Simulation + EZ-Diffusion Recovery")
    print("=" * 60)

    true_v = 0.5
    true_a = 1.5
    true_z = 0.5
    true_t0 = 0.3

    print(f"\nTrue parameters: v={true_v}, a={true_a}, z={true_z}, t0={true_t0}")
    print("Simulating 1000 trials...")

    sim = simulate_ddm(
        n_trials=5000,
        drift_rate=true_v,
        boundary=true_a,
        start_point=true_z,
        nondecision_time=true_t0,
        seed=42,
    )

    rt = sim["rt"]
    choices = sim["choice"]

    print(f"Mean RT: {np.mean(rt):.3f}s")
    print(f"Accuracy (upper boundary): {np.mean(choices == 1):.3f}")

    recovered = ez_diffusion(rt, choices)
    print(f"\nRecovered parameters:")
    print(f"  Drift rate (v): {recovered['drift_rate']:.3f} (true: {true_v})")
    print(f"  Boundary (a):   {recovered['boundary']:.3f} (true: {true_a})")
    print(f"  Non-decision (t0): {recovered['nondecision_time']:.3f} (true: {true_t0})")
