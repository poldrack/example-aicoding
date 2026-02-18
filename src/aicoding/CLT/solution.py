"""Central Limit Theorem simulation across multiple distributions.

Repeatedly generates samples from 6 different distributions (normal,
uniform, chi-squared, Poisson, exponential, and beta), computes the
mean of each sample, and plots the distribution of sample means to
demonstrate the Central Limit Theorem.
"""

import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

DISTRIBUTIONS = [
    "normal",
    "uniform",
    "chi-squared",
    "poisson",
    "exponential",
    "beta",
]


def simulate_sample_means(distribution, n_samples=1000, sample_size=30, seed=42):
    """Draw repeated samples from a distribution and return their means.

    For each of n_samples iterations, draws a sample of size sample_size
    from the specified distribution and computes the sample mean.

    Args:
        distribution: Name of the distribution. One of: "normal",
            "uniform", "chi-squared", "poisson", "exponential", "beta".
        n_samples: Number of samples to draw (default 1000).
        sample_size: Number of observations per sample (default 30).
        seed: Random seed for reproducibility (default 42).

    Returns:
        numpy array of shape (n_samples,) containing the sample means.

    Raises:
        ValueError: If distribution is not one of the supported names.
    """
    if distribution not in DISTRIBUTIONS:
        raise ValueError(
            f"Unsupported distribution: '{distribution}'. "
            f"Must be one of {DISTRIBUTIONS}."
        )

    rng = np.random.default_rng(seed)
    means = np.empty(n_samples)

    for i in range(n_samples):
        if distribution == "normal":
            sample = rng.standard_normal(sample_size)
        elif distribution == "uniform":
            sample = rng.uniform(0.0, 1.0, sample_size)
        elif distribution == "chi-squared":
            sample = rng.chisquare(df=2, size=sample_size)
        elif distribution == "poisson":
            sample = rng.poisson(lam=3, size=sample_size)
        elif distribution == "exponential":
            sample = rng.exponential(scale=1.0, size=sample_size)
        elif distribution == "beta":
            sample = rng.beta(a=2, b=5, size=sample_size)

        means[i] = np.mean(sample)

    return means


def run_clt_simulation(n_samples=1000, sample_size=30, seed=42):
    """Run the CLT simulation for all 6 distributions.

    Calls simulate_sample_means for each distribution, using
    deterministic sub-seeds derived from the base seed so that each
    distribution gets its own independent random stream.

    Args:
        n_samples: Number of samples per distribution (default 1000).
        sample_size: Number of observations per sample (default 30).
        seed: Base random seed (default 42).

    Returns:
        Dictionary mapping distribution name (str) to numpy array of
        sample means.
    """
    base_rng = np.random.default_rng(seed)
    results = {}

    for dist in DISTRIBUTIONS:
        # Generate a unique sub-seed for each distribution
        sub_seed = base_rng.integers(0, 2**31)
        results[dist] = simulate_sample_means(
            dist, n_samples=n_samples, sample_size=sample_size, seed=sub_seed
        )

    return results


def plot_clt(results):
    """Create a figure with 6 subplots showing the distribution of sample means.

    Each subplot shows a histogram of the sample means for one
    distribution, demonstrating how the sampling distribution of the
    mean approaches normality (Central Limit Theorem).

    Args:
        results: Dictionary mapping distribution name to numpy array
            of sample means (as returned by run_clt_simulation).

    Returns:
        matplotlib Figure containing 6 subplots arranged in a 2x3 grid.
    """
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle("Central Limit Theorem: Distribution of Sample Means", fontsize=14)
    axes = axes.flatten()

    for ax, dist in zip(axes, DISTRIBUTIONS):
        means = results[dist]
        ax.hist(means, bins=40, edgecolor="black", alpha=0.7, density=True)
        ax.set_title(dist)
        ax.set_xlabel("Sample Mean")
        ax.set_ylabel("Density")

        # Annotate with mean and std
        mu = np.mean(means)
        sigma = np.std(means)
        ax.axvline(mu, color="red", linestyle="--", linewidth=1.5, label=f"mean={mu:.3f}")
        ax.legend(fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


if __name__ == "__main__":
    print("=== Central Limit Theorem Simulation ===\n")

    n_samples = 1000
    sample_size = 30
    seed = 42

    print(f"Parameters: n_samples={n_samples}, sample_size={sample_size}, seed={seed}\n")

    results = run_clt_simulation(n_samples=n_samples, sample_size=sample_size, seed=seed)

    for dist, means in results.items():
        print(
            f"  {dist:15s}  |  grand mean = {np.mean(means):.4f}  |  "
            f"std of means = {np.std(means):.4f}"
        )

    print("\nGenerating plot...")
    os.makedirs("outputs", exist_ok=True)
    fig = plot_clt(results)
    fig.savefig("outputs/clt_simulation.png", dpi=150)
    print("Saved figure to outputs/clt_simulation.png")
    plt.close(fig)
