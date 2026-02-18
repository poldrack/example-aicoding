"""Wrapper for statsmodels ttest_ind that produces R-style t.test output.

This module provides a ``ttest_ind`` function whose interface mirrors
``statsmodels.stats.weightstats.ttest_ind`` while returning a rich
``TTestResult`` object whose ``__str__`` mimics the output of R's ``t.test``.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as scipy_stats
from statsmodels.stats.weightstats import ttest_ind as _sm_ttest_ind


class TTestResult:
    """Container for the results of an independent two-sample t-test.

    Attributes
    ----------
    t_statistic : float
        The t test statistic.
    p_value : float
        The p-value of the test.
    df : float
        Degrees of freedom used in the test.
    conf_int : tuple[float, float]
        95 % confidence interval for the difference in means (x1 - x2).
    mean_x1 : float
        Sample mean of the first group.
    mean_x2 : float
        Sample mean of the second group.
    alternative : str
        The alternative hypothesis used ('two-sided', 'larger', or 'smaller').
    usevar : str
        Whether pooled or unequal variance was used.
    """

    def __init__(
        self,
        t_statistic: float,
        p_value: float,
        df: float,
        conf_int: tuple[float, float],
        mean_x1: float,
        mean_x2: float,
        alternative: str = "two-sided",
        usevar: str = "pooled",
    ) -> None:
        self.t_statistic = float(t_statistic)
        self.p_value = float(p_value)
        self.df = float(df)
        self.conf_int = (float(conf_int[0]), float(conf_int[1]))
        self.mean_x1 = float(mean_x1)
        self.mean_x2 = float(mean_x2)
        self.alternative = alternative
        self.usevar = usevar

    # ---- R-style string representation ----------------------------------- #

    def __str__(self) -> str:
        """Format the result to resemble R's ``t.test`` console output."""
        if self.usevar == "unequal":
            title = "\tWelch Two Sample t-test\n"
        else:
            title = "\tTwo Sample t-test\n"

        data_line = "data:  x and y"

        stats_line = f"t = {self.t_statistic:.4f}, df = {self.df:.4g}, p-value = {self._format_p()}"

        alt_map = {
            "two-sided": "true difference in means is not equal to 0",
            "larger": "true difference in means is greater than 0",
            "smaller": "true difference in means is less than 0",
        }
        alt_line = f"alternative hypothesis: {alt_map[self.alternative]}"

        ci_line = (
            "95 percent confidence interval:\n"
            f" {self.conf_int[0]:.6f} {self.conf_int[1]:.6f}"
        )

        estimates = (
            "sample estimates:\n"
            "mean of x mean of y \n"
            f" {self.mean_x1:8g}  {self.mean_x2:8g} "
        )

        return "\n".join([title, data_line, stats_line, alt_line, ci_line, estimates])

    def __repr__(self) -> str:
        return (
            f"TTestResult(t={self.t_statistic:.4f}, p={self.p_value:.4g}, "
            f"df={self.df:.4g})"
        )

    # ---- helpers --------------------------------------------------------- #

    def _format_p(self) -> str:
        """Format p-value: use scientific notation for very small values."""
        if self.p_value < 2.2e-16:
            return "< 2.2e-16"
        return f"{self.p_value:.4g}"


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def ttest_ind(
    x1,
    x2,
    alternative: str = "two-sided",
    usevar: str = "pooled",
) -> TTestResult:
    """Independent two-sample t-test with R-style output.

    This is a thin wrapper around ``statsmodels.stats.weightstats.ttest_ind``
    that computes the confidence interval for the difference in means and
    returns a ``TTestResult`` whose ``__str__`` mimics R's ``t.test``.

    Parameters
    ----------
    x1 : array_like
        First sample.
    x2 : array_like
        Second sample.
    alternative : str, optional
        One of ``'two-sided'`` (default), ``'larger'``, or ``'smaller'``.
    usevar : str, optional
        ``'pooled'`` (default) for equal-variance or ``'unequal'`` for Welch.

    Returns
    -------
    TTestResult
        Object containing t-statistic, p-value, df, confidence interval,
        and sample means.
    """
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)

    # Compute core test via statsmodels
    t_stat, p_value, df = _sm_ttest_ind(x1, x2, alternative=alternative, usevar=usevar)

    # Sample means
    mean_x1 = float(np.mean(x1))
    mean_x2 = float(np.mean(x2))

    # Compute 95% confidence interval for the difference (x1 - x2)
    diff = mean_x1 - mean_x2
    se = _standard_error(x1, x2, usevar)
    ci = _confidence_interval(diff, se, df, alpha=0.05)

    return TTestResult(
        t_statistic=t_stat,
        p_value=p_value,
        df=df,
        conf_int=ci,
        mean_x1=mean_x1,
        mean_x2=mean_x2,
        alternative=alternative,
        usevar=usevar,
    )


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #

def _standard_error(x1: np.ndarray, x2: np.ndarray, usevar: str) -> float:
    """Compute the standard error of the difference in means."""
    n1, n2 = len(x1), len(x2)

    if usevar == "pooled":
        s1_sq = np.var(x1, ddof=1)
        s2_sq = np.var(x2, ddof=1)
        sp_sq = ((n1 - 1) * s1_sq + (n2 - 1) * s2_sq) / (n1 + n2 - 2)
        return float(np.sqrt(sp_sq * (1 / n1 + 1 / n2)))
    else:
        # Welch: separate variances
        s1_sq = np.var(x1, ddof=1)
        s2_sq = np.var(x2, ddof=1)
        return float(np.sqrt(s1_sq / n1 + s2_sq / n2))


def _confidence_interval(
    diff: float, se: float, df: float, alpha: float = 0.05
) -> tuple[float, float]:
    """Compute a (1 - alpha) confidence interval for the difference."""
    t_crit = scipy_stats.t.ppf(1 - alpha / 2, df)
    return (diff - t_crit * se, diff + t_crit * se)


# --------------------------------------------------------------------------- #
# Main block
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    # Quick demo with sample data
    x = [1, 2, 3, 4, 5]
    y = [2, 4, 6, 8, 10]

    print("=== Pooled (equal variance) ===")
    result_pooled = ttest_ind(x, y, usevar="pooled")
    print(result_pooled)
    print()

    print("=== Welch (unequal variance) ===")
    result_welch = ttest_ind(x, y, usevar="unequal")
    print(result_welch)
    print()

    print("=== One-sided (smaller) ===")
    result_smaller = ttest_ind(x, y, alternative="smaller")
    print(result_smaller)
