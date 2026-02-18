"""Drift Diffusion Model with collapsing bounds.

Simulates a DDM where the decision boundaries collapse (shrink) over time,
and returns the full timeseries of diffusion steps for each trial.
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def simulate_ddm_collapsing(n_trials, drift_rate, initial_boundary,
                             collapse_rate=0.01, dt=0.001, noise_sd=1.0,
                             nondecision_time=0.3, start_point=0.5,
                             max_time=10.0, seed=None):
    """Simulate a DDM with collapsing bounds.

    The upper boundary at time t is: initial_boundary * exp(-collapse_rate * t).
    The lower boundary is always at 0 (symmetric collapse around the midpoint).

    Parameters
    ----------
    n_trials : int
        Number of trials.
    drift_rate : float
        Drift rate (v).
    initial_boundary : float
        Initial boundary separation (a).
    collapse_rate : float
        Exponential collapse rate. 0 = fixed bounds.
    dt : float
        Time step.
    noise_sd : float
        Within-trial noise standard deviation.
    nondecision_time : float
        Non-decision time added to RT.
    start_point : float
        Relative starting point (0-1). 0.5 = midpoint.
    max_time : float
        Maximum allowed decision time before forced response.
    seed : int, optional
        Random seed.

    Returns
    -------
    dict
        'rt': array of RTs, 'choice': array (1=upper, 0=lower),
        'timeseries': list of ndarray with full trajectory for each trial.
    """
    rng = np.random.RandomState(seed)

    rts = np.empty(n_trials)
    choices = np.empty(n_trials, dtype=int)
    timeseries = []

    for i in range(n_trials):
        z = initial_boundary * start_point
        x = z
        t = 0
        trajectory = [x]

        while t < max_time:
            x += drift_rate * dt + noise_sd * np.sqrt(dt) * rng.randn()
            t += dt
            trajectory.append(x)

            # Collapsing bounds
            upper = initial_boundary * np.exp(-collapse_rate * t)
            lower = initial_boundary * (1 - np.exp(-collapse_rate * t)) if collapse_rate > 0 else 0.0
            # Symmetric collapse: midpoint stays at a/2
            mid = initial_boundary * 0.5
            upper_bound = mid + (initial_boundary * 0.5) * np.exp(-collapse_rate * t)
            lower_bound = mid - (initial_boundary * 0.5) * np.exp(-collapse_rate * t)

            if x >= upper_bound:
                choices[i] = 1
                rts[i] = t + nondecision_time
                break
            elif x <= lower_bound:
                choices[i] = 0
                rts[i] = t + nondecision_time
                break
        else:
            # Forced response at max_time
            choices[i] = 1 if x >= initial_boundary * 0.5 else 0
            rts[i] = max_time + nondecision_time

        timeseries.append(np.array(trajectory))

    return {"rt": rts, "choice": choices, "timeseries": timeseries}


def plot_ddm_collapsing(result, initial_boundary, collapse_rate, n_trials_to_plot=5,
                        save_path=None):
    """Plot DDM trajectories with collapsing bounds.

    Parameters
    ----------
    result : dict
        Output from simulate_ddm_collapsing.
    initial_boundary : float
        Initial boundary separation.
    collapse_rate : float
        Collapse rate.
    n_trials_to_plot : int
        Number of trajectories to overlay.
    save_path : str, optional
        Path to save the figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    timeseries = result["timeseries"]

    for i in range(min(n_trials_to_plot, len(timeseries))):
        traj = timeseries[i]
        t_vals = np.arange(len(traj)) * 0.001
        ax.plot(t_vals, traj, alpha=0.5, linewidth=0.8)

    # Plot collapsing bounds
    t_max = max(len(ts) for ts in timeseries[:n_trials_to_plot]) * 0.001
    t_bound = np.linspace(0, t_max, 500)
    mid = initial_boundary * 0.5
    upper = mid + (initial_boundary * 0.5) * np.exp(-collapse_rate * t_bound)
    lower = mid - (initial_boundary * 0.5) * np.exp(-collapse_rate * t_bound)
    ax.plot(t_bound, upper, "r--", linewidth=2, label="Upper bound")
    ax.plot(t_bound, lower, "b--", linewidth=2, label="Lower bound")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Evidence")
    ax.set_title("DDM with Collapsing Bounds")
    ax.legend()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


if __name__ == "__main__":
    print("DDM with Collapsing Bounds")
    print("=" * 40)

    result = simulate_ddm_collapsing(
        n_trials=100,
        drift_rate=0.5,
        initial_boundary=2.0,
        collapse_rate=0.05,
        seed=42,
    )

    print(f"Mean RT: {np.mean(result['rt']):.3f}s")
    print(f"Accuracy: {np.mean(result['choice'] == 1):.3f}")

    os.makedirs("outputs", exist_ok=True)
    fig = plot_ddm_collapsing(result, initial_boundary=2.0, collapse_rate=0.05,
                               save_path="outputs/ddm_collapsing.png")
    print("Saved plot to outputs/ddm_collapsing.png")
    plt.close(fig)
