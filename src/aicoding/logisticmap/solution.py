"""Logistic map simulation and bifurcation plot.

Implements the logistic map x_{n+1} = r * x_n * (1 - x_n) and creates
a bifurcation diagram showing x vs r.
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def logistic_map(r, x0=0.5, n_steps=1000):
    """Iterate the logistic map.

    Parameters
    ----------
    r : float
        Growth rate parameter.
    x0 : float
        Initial value (between 0 and 1).
    n_steps : int
        Number of iterations.

    Returns
    -------
    ndarray
        Array of x values.
    """
    x = np.empty(n_steps)
    x[0] = x0
    for i in range(1, n_steps):
        x[i] = r * x[i - 1] * (1 - x[i - 1])
    return x


def bifurcation_data(r_min=2.5, r_max=4.0, r_steps=1000, n_iterations=1000,
                     n_last=100, x0=0.5):
    """Generate bifurcation diagram data.

    Parameters
    ----------
    r_min, r_max : float
        Range of r values.
    r_steps : int
        Number of r values to sample.
    n_iterations : int
        Total iterations per r value (includes transient).
    n_last : int
        Number of final iterations to keep (after transient).
    x0 : float
        Initial x value.

    Returns
    -------
    r_vals : ndarray
        r values (repeated for each kept iteration).
    x_vals : ndarray
        Corresponding x values.
    """
    r_values = np.linspace(r_min, r_max, r_steps)
    all_r = []
    all_x = []

    for r in r_values:
        trajectory = logistic_map(r, x0=x0, n_steps=n_iterations)
        last = trajectory[-n_last:]
        all_r.extend([r] * n_last)
        all_x.extend(last)

    return np.array(all_r), np.array(all_x)


def plot_bifurcation(r_min=2.5, r_max=4.0, r_steps=1000, save_path=None):
    """Create a bifurcation diagram of the logistic map.

    Parameters
    ----------
    r_min, r_max : float
        Range of r values.
    r_steps : int
        Number of r values to sample.
    save_path : str, optional
        Path to save the figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    r_vals, x_vals = bifurcation_data(r_min=r_min, r_max=r_max, r_steps=r_steps)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(r_vals, x_vals, s=0.01, c="black", alpha=0.5)
    ax.set_xlabel("r")
    ax.set_ylabel("x")
    ax.set_title("Logistic Map Bifurcation Diagram")
    ax.set_xlim(r_min, r_max)
    ax.set_ylim(0, 1)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


if __name__ == "__main__":
    print("Logistic Map Bifurcation Diagram")
    print("=" * 40)

    os.makedirs("outputs", exist_ok=True)
    fig = plot_bifurcation(r_min=2.5, r_max=4.0, r_steps=2000, save_path="outputs/logistic_map.png")
    print("Saved bifurcation plot to outputs/logistic_map.png")
    plt.close(fig)
