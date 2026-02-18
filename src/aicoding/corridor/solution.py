"""Corridor of stability simulation (Schönbrodt & Perugini, 2013).

Simulates how correlation estimates become less variable (more stable)
as sample size increases, creating the classic "corridor" plot.
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def simulate_corridor_of_stability(true_r=0.3, max_n=500, min_n=10,
                                    n_simulations=500, seed=None):
    """Simulate correlation estimates at increasing sample sizes.

    Parameters
    ----------
    true_r : float
        True population correlation (-1 to 1).
    max_n : int
        Maximum sample size.
    min_n : int
        Minimum sample size (must be >= 3).
    n_simulations : int
        Number of simulation repetitions.
    seed : int, optional
        Random seed.

    Returns
    -------
    dict
        'sample_sizes': list of int, 'correlations': list of arrays.
    """
    rng = np.random.RandomState(seed)
    min_n = max(min_n, 3)

    # Population covariance matrix
    cov = np.array([[1.0, true_r], [true_r, 1.0]])

    sample_sizes = list(range(min_n, max_n + 1))
    all_correlations = []

    for _ in range(n_simulations):
        # Generate max_n bivariate normal samples
        data = rng.multivariate_normal([0, 0], cov, size=max_n)
        corrs = []
        for n in sample_sizes:
            r = np.corrcoef(data[:n, 0], data[:n, 1])[0, 1]
            corrs.append(r)
        all_correlations.append(corrs)

    return {
        "sample_sizes": sample_sizes,
        "correlations": all_correlations,
    }


def plot_corridor_of_stability(result, true_r=0.3, save_path=None):
    """Plot the corridor of stability.

    Parameters
    ----------
    result : dict
        Output from simulate_corridor_of_stability.
    true_r : float
        True population correlation (for reference line).
    save_path : str, optional
        Path to save the figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    sizes = result["sample_sizes"]
    corrs = np.array(result["correlations"])

    # Plot individual simulation trajectories (transparent)
    for i in range(min(len(corrs), 100)):
        ax.plot(sizes, corrs[i], alpha=0.05, color="steelblue", linewidth=0.5)

    # Plot percentile envelope
    lower = np.percentile(corrs, 2.5, axis=0)
    upper = np.percentile(corrs, 97.5, axis=0)
    median = np.median(corrs, axis=0)

    ax.fill_between(sizes, lower, upper, alpha=0.3, color="steelblue", label="95% CI")
    ax.plot(sizes, median, color="navy", linewidth=2, label="Median")

    # True correlation reference line
    ax.axhline(y=true_r, color="red", linestyle="--", linewidth=1.5, label=f"True r = {true_r}")

    ax.set_xlabel("Sample Size")
    ax.set_ylabel("Correlation Estimate")
    ax.set_title("Corridor of Stability")
    ax.legend()
    ax.set_ylim(-1, 1)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


if __name__ == "__main__":
    print("Corridor of Stability Simulation")
    print("=" * 40)

    result = simulate_corridor_of_stability(true_r=0.3, max_n=500, n_simulations=500, seed=42)

    os.makedirs("outputs", exist_ok=True)
    fig = plot_corridor_of_stability(result, true_r=0.3, save_path="outputs/corridor_of_stability.png")
    print("Saved plot to outputs/corridor_of_stability.png")
    plt.close(fig)
