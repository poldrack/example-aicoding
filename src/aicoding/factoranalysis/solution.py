"""Exploratory factor analysis with BIC-based model selection.

Loads survey data, selects impulsivity-related variables, performs EFA
with 1–5 factors, selects the best model via BIC, and reports dominant
loadings per factor.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import FactorAnalysis


# Variable prefixes to select
PREFIXES = [
    "upps_impulsivity_survey",
    "sensation_seeking_survey",
    "bis11_survey",
    "dickman_survey",
]

DATA_URL = (
    "https://raw.githubusercontent.com/IanEisenberg/Self_Regulation_Ontology/"
    "master/Data/Complete_02-16-2019/meaningful_variables.csv"
)


def load_and_select_variables(data=None, url=DATA_URL):
    """Load data and select variables matching the target survey prefixes.

    Parameters
    ----------
    data : DataFrame, optional
        Pre-loaded DataFrame. If None, loads from URL.
    url : str
        URL to download data from.

    Returns
    -------
    DataFrame
        Subset of columns matching the target prefixes.
    """
    if data is None:
        data = pd.read_csv(url, index_col=0)

    selected_cols = [
        col for col in data.columns
        if any(col.startswith(prefix) for prefix in PREFIXES)
    ]
    return data[selected_cols]


def varimax_rotation(loadings, max_iter=100, tol=1e-6):
    """Apply varimax rotation to a loadings matrix.

    Parameters
    ----------
    loadings : ndarray of shape (n_variables, n_factors)
        Unrotated factor loadings.
    max_iter : int
        Maximum iterations.
    tol : float
        Convergence tolerance.

    Returns
    -------
    ndarray of shape (n_variables, n_factors)
        Rotated loadings.
    """
    n_vars, n_factors = loadings.shape
    if n_factors < 2:
        return loadings.copy()

    rotated = loadings.copy()
    for _ in range(max_iter):
        old = rotated.copy()
        for i in range(n_factors):
            for j in range(i + 1, n_factors):
                # Extract columns i and j
                x = rotated[:, i]
                y = rotated[:, j]
                # Varimax criterion rotation angle
                u = x**2 - y**2
                v = 2 * x * y
                a = np.sum(u)
                b = np.sum(v)
                c = np.sum(u**2 - v**2)
                d = 2 * np.sum(u * v)
                num = d - 2 * a * b / n_vars
                den = c - (a**2 - b**2) / n_vars
                angle = 0.25 * np.arctan2(num, den)
                cos_a = np.cos(angle)
                sin_a = np.sin(angle)
                rotated[:, i] = x * cos_a + y * sin_a
                rotated[:, j] = -x * sin_a + y * cos_a
        if np.max(np.abs(rotated - old)) < tol:
            break
    return rotated


def fit_factor_analysis(df, n_factors=3):
    """Fit a factor analysis model and compute BIC.

    Parameters
    ----------
    df : DataFrame
        Data with variables as columns.
    n_factors : int
        Number of factors.

    Returns
    -------
    dict
        'loadings': ndarray (n_vars, n_factors),
        'bic': float,
        'model': FactorAnalysis instance.
    """
    # Drop rows with NaN
    clean_df = df.dropna()
    X = clean_df.values

    model = FactorAnalysis(n_components=n_factors, random_state=42)
    model.fit(X)

    loadings = model.components_.T  # (n_variables, n_factors)
    loadings = varimax_rotation(loadings)

    # Compute BIC
    # BIC = -2 * log_likelihood + k * log(n)
    n = X.shape[0]
    p = X.shape[1]
    log_likelihood = model.score(X) * n  # score returns per-sample avg
    k = n_factors * p + p  # loadings + noise variances
    bic = -2 * log_likelihood + k * np.log(n)

    return {
        "loadings": loadings,
        "bic": bic,
        "model": model,
    }


def select_best_n_factors(df, max_factors=5):
    """Select the number of factors that minimizes BIC.

    Parameters
    ----------
    df : DataFrame
        Data with variables as columns.
    max_factors : int
        Maximum number of factors to try.

    Returns
    -------
    int
        Optimal number of factors.
    """
    best_bic = np.inf
    best_n = 1

    for n in range(1, max_factors + 1):
        result = fit_factor_analysis(df, n_factors=n)
        if result["bic"] < best_bic:
            best_bic = result["bic"]
            best_n = n

    return best_n


def get_dominant_loadings(loadings, variable_names, top_n=3):
    """For each factor, return the variables with the highest absolute loadings.

    Parameters
    ----------
    loadings : ndarray of shape (n_variables, n_factors)
    variable_names : list or Index of str
    top_n : int
        Number of top-loading variables to return per factor (default 3).

    Returns
    -------
    dict
        factor_index -> list of variable names (sorted by absolute loading).
    """
    n_vars, n_factors = loadings.shape
    abs_loadings = np.abs(loadings)

    dominant = {}
    for i in range(n_factors):
        ranked = np.argsort(abs_loadings[:, i])[::-1][:top_n]
        dominant[i] = [variable_names[v] for v in ranked]

    return dominant


if __name__ == "__main__":
    print("Exploratory Factor Analysis with BIC Selection")
    print("=" * 60)

    print("Loading data from GitHub...")
    df = load_and_select_variables()
    print(f"Selected {df.shape[1]} variables, {df.shape[0]} observations.")
    print(f"Variables: {list(df.columns)[:5]}... ")

    print("\nFitting models with 1-5 factors...")
    for n in range(1, 6):
        result = fit_factor_analysis(df, n_factors=n)
        print(f"  {n} factors: BIC = {result['bic']:.1f}")

    best_n = select_best_n_factors(df, max_factors=5)
    print(f"\nBest model: {best_n} factors (minimum BIC)")

    result = fit_factor_analysis(df, n_factors=best_n)
    dominant = get_dominant_loadings(result["loadings"], df.columns)

    print(f"\nTop 3 loadings for {best_n}-factor solution:")
    for factor, variables in dominant.items():
        print(f"\n  Factor {factor + 1}:")
        for var in variables:
            loading = result["loadings"][list(df.columns).index(var), factor]
            print(f"    {var}: {loading:.3f}")
