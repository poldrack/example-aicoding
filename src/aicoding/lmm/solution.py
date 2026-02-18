"""Combine CSV files and fit a linear mixed model.

Reads CSV files with rt/correct/compatible columns, combines them
with a subject identifier, and fits a linear mixed model with
random intercept and slope for the compatible factor.
"""

import os
import tempfile
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def combine_csv_files(directory):
    """Combine CSV files from a directory into a single DataFrame.

    Each file should have columns: rt, correct, compatible.
    The filename (without extension) is used as the subject identifier.

    Args:
        directory: Path to directory containing CSV files.

    Returns:
        Combined DataFrame with an added 'subject' column.
    """
    frames = []
    for filename in sorted(os.listdir(directory)):
        if filename.endswith(".csv"):
            filepath = os.path.join(directory, filename)
            df = pd.read_csv(filepath)
            df["subject"] = os.path.splitext(filename)[0]
            frames.append(df)

    return pd.concat(frames, ignore_index=True)


def fit_lmm(df):
    """Fit a linear mixed model with random intercept and slope.

    Model: rt ~ compatible, with random intercept and slope for
    compatible grouped by subject.

    Args:
        df: DataFrame with columns rt, compatible, subject.

    Returns:
        Fitted MixedLMResults object.
    """
    model = smf.mixedlm(
        "rt ~ compatible",
        df,
        groups=df["subject"],
        re_formula="~compatible",
    )
    result = model.fit()
    return result


def generate_test_data(n_subjects=5, n_trials=100, effect_size=-50, seed=42):
    """Generate synthetic CSV files with known effect for testing.

    Args:
        n_subjects: Number of subject files to create.
        n_trials: Number of trials per subject.
        effect_size: True effect of compatible condition on RT (ms).
        seed: Random seed for reproducibility.

    Returns:
        Path to temporary directory containing the generated CSV files.
    """
    tmpdir = tempfile.mkdtemp()
    rng = np.random.default_rng(seed)

    for i in range(n_subjects):
        compatible = np.array([0] * (n_trials // 2) + [1] * (n_trials // 2))
        rng.shuffle(compatible)
        base_rt = 500 + rng.normal(0, 30)
        slope = effect_size + rng.normal(0, 10)
        rt = base_rt + slope * compatible + rng.normal(0, 50, n_trials)
        correct = rng.binomial(1, 0.85, n_trials)
        df = pd.DataFrame({"rt": rt, "correct": correct, "compatible": compatible})
        df.to_csv(os.path.join(tmpdir, f"subject_{i}.csv"), index=False)

    return tmpdir


if __name__ == "__main__":
    # Generate test data and run the analysis
    data_dir = generate_test_data()
    print(f"Generated test data in: {data_dir}")

    df = combine_csv_files(data_dir)
    print(f"Combined {len(df)} rows from {df['subject'].nunique()} subjects")

    result = fit_lmm(df)
    print("\nLinear Mixed Model Results:")
    print(result.summary())

    # Cleanup
    for f in os.listdir(data_dir):
        os.remove(os.path.join(data_dir, f))
    os.rmdir(data_dir)
