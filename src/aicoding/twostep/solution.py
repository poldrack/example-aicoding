"""Daw two-step reinforcement learning task analysis.

Generates synthetic behavioral data for the two-step task and analyzes
it to extract model-based and model-free indices of behavior.

In the Daw two-step task:
  - Participants make a step-1 choice between two options.
  - Each step-1 choice leads to one of two step-2 states with either a
    *common* (70%) or *rare* (30%) transition probability.
  - In the step-2 state, participants make another choice and receive a
    binary reward.
  - Model-free learners tend to repeat previously rewarded step-1 actions
    regardless of the transition type.
  - Model-based learners account for transition structure: they repeat a
    step-1 action after a common-transition reward OR after a rare-
    transition non-reward (because the non-rewarding step-2 state reached
    via a rare transition is the *other* state that is commonly reached
    by the other step-1 action).

The key analysis uses logistic regression to predict stay/switch behavior
at step 1 as a function of:
  - previous reward (main effect) --> model-free index
  - previous reward x transition type interaction --> model-based index
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

def generate_synthetic_data(n_trials=200, model_based_weight=0.5, seed=42):
    """Generate synthetic two-step task data as a DataFrame.

    The agent's behavior is a mixture of model-based and model-free
    strategies controlled by ``model_based_weight``:
      - weight = 0.0 --> purely model-free (repeat if rewarded)
      - weight = 1.0 --> purely model-based (account for transitions)

    Args:
        n_trials: Number of trials to simulate.
        model_based_weight: Weight between 0 and 1 controlling the
            balance between model-based (1) and model-free (0) behavior.
        seed: Random seed for reproducibility.

    Returns:
        pandas DataFrame with columns:
            trial, step1_choice, step1_state, step2_state,
            step2_choice, reward, previous_reward
    """
    rng = np.random.default_rng(seed)

    # Transition probability: 70% common, 30% rare
    p_common = 0.7

    # Reward probabilities for each of the 4 step-2 actions
    # (2 states x 2 choices) -- drift slowly over trials
    reward_probs = _generate_drifting_reward_probs(n_trials, rng)

    # Storage
    records = []
    prev_reward = np.nan
    prev_step1_choice = None
    prev_transition = None

    for t in range(n_trials):
        # --- Step 1 choice ---
        if t == 0 or np.isnan(prev_reward):
            step1_choice = rng.integers(0, 2)
        else:
            step1_choice = _agent_step1_choice(
                prev_step1_choice,
                prev_reward,
                prev_transition,
                model_based_weight,
                rng,
            )

        step1_state = 0  # always start in state 0

        # --- Transition ---
        is_common = rng.random() < p_common
        if is_common:
            # Common: choice 0 -> state A (0), choice 1 -> state B (1)
            step2_state_idx = step1_choice
        else:
            # Rare: choice 0 -> state B (1), choice 1 -> state A (0)
            step2_state_idx = 1 - step1_choice
        transition_type = "common" if is_common else "rare"

        # --- Step 2 choice ---
        step2_choice = rng.integers(0, 2)

        # --- Reward ---
        rp = reward_probs[t, step2_state_idx, step2_choice]
        reward = int(rng.random() < rp)

        records.append(
            {
                "trial": t,
                "step1_choice": int(step1_choice),
                "step1_state": int(step1_state),
                "step2_state": transition_type,
                "step2_choice": int(step2_choice),
                "reward": int(reward),
                "previous_reward": prev_reward,
            }
        )

        # Update history for next trial
        prev_reward = reward
        prev_step1_choice = step1_choice
        prev_transition = transition_type

    df = pd.DataFrame(records)
    # Convert previous_reward: first trial is NaN, rest are int-like
    df["previous_reward"] = df["previous_reward"].astype("Int64")
    # Convert back to float so NaN is preserved as np.nan
    df["previous_reward"] = df["previous_reward"].astype("float")
    return df


def _generate_drifting_reward_probs(n_trials, rng):
    """Generate slowly drifting reward probabilities for step-2 actions.

    Returns:
        Array of shape (n_trials, 2, 2) -- [trial, state, choice]
    """
    probs = np.zeros((n_trials, 2, 2))
    # Initialize around 0.25-0.75
    probs[0] = rng.uniform(0.25, 0.75, size=(2, 2))
    for t in range(1, n_trials):
        noise = rng.normal(0, 0.025, size=(2, 2))
        probs[t] = np.clip(probs[t - 1] + noise, 0.1, 0.9)
    return probs


def _agent_step1_choice(prev_choice, prev_reward, prev_transition,
                         mb_weight, rng):
    """Decide step-1 choice based on a mixture of MB and MF strategies.

    Model-free component: stay if previously rewarded, switch otherwise.
    Model-based component: stay if (common + rewarded) or (rare + not
    rewarded); switch otherwise.

    Args:
        prev_choice: Previous step-1 choice (0 or 1).
        prev_reward: Previous reward (0 or 1).
        prev_transition: Previous transition type ("common" or "rare").
        mb_weight: Weight for model-based strategy (0 to 1).
        rng: numpy random generator.

    Returns:
        Step-1 choice (0 or 1).
    """
    mf_weight = 1.0 - mb_weight

    # Model-free: stay if rewarded
    mf_stay = prev_reward == 1

    # Model-based: stay if (common + rewarded) or (rare + not rewarded)
    if prev_transition == "common":
        mb_stay = prev_reward == 1
    else:  # rare
        mb_stay = prev_reward == 0

    # Combined stay probability
    stay_prob_raw = mb_weight * float(mb_stay) + mf_weight * float(mf_stay)
    # Map to a probability between 0.2 and 0.8 to add noise
    stay_prob = 0.2 + 0.6 * stay_prob_raw

    if rng.random() < stay_prob:
        return int(prev_choice)
    else:
        return 1 - int(prev_choice)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze_twostep(df):
    """Analyze two-step task data using logistic regression.

    Predicts stay/switch behavior at step 1 as a function of:
      - previous reward (coded as -0.5 / +0.5)
      - transition type (coded as +0.5 common / -0.5 rare)
      - reward x transition interaction

    Args:
        df: DataFrame produced by ``generate_synthetic_data`` (or
            equivalent). Must contain columns: trial, step1_choice,
            step2_state, reward, previous_reward.

    Returns:
        dict with keys:
            model_free_index: Coefficient for previous reward main effect.
                A positive value means the agent tends to stay after
                reward (model-free signature).
            model_based_index: Coefficient for reward x transition
                interaction.  A positive value means the agent accounts
                for transition structure (model-based signature).
            coefficients: dict mapping regressor names to their
                estimated coefficients.
    """
    # Work on a copy to avoid mutating the input
    data = df.copy()

    # Drop the first trial (no previous reward)
    data = data.dropna(subset=["previous_reward"]).reset_index(drop=True)

    # Compute "stay": did the agent repeat the same step-1 choice as the
    # previous trial?
    data["prev_step1_choice"] = data["step1_choice"].shift(1)
    data = data.dropna(subset=["prev_step1_choice"]).reset_index(drop=True)
    data["stay"] = (data["step1_choice"] == data["prev_step1_choice"]).astype(int)

    # Code predictors with effect coding (-0.5 / +0.5)
    data["reward_coded"] = data["previous_reward"].map({0: -0.5, 1: 0.5})
    data["transition_coded"] = data["step2_state"].shift(1)
    data = data.dropna(subset=["transition_coded"]).reset_index(drop=True)
    data["transition_coded"] = data["transition_coded"].map(
        {"common": 0.5, "rare": -0.5}
    )

    # Interaction
    data["reward_x_transition"] = (
        data["reward_coded"] * data["transition_coded"]
    )

    # Logistic regression
    X = data[["reward_coded", "transition_coded", "reward_x_transition"]]
    X = sm.add_constant(X)
    y = data["stay"]

    model = sm.Logit(y, X)
    result = model.fit(disp=False, maxiter=200)

    coefficients = {
        "intercept": float(result.params["const"]),
        "reward": float(result.params["reward_coded"]),
        "transition": float(result.params["transition_coded"]),
        "reward_x_transition": float(result.params["reward_x_transition"]),
    }

    return {
        "model_free_index": coefficients["reward"],
        "model_based_index": coefficients["reward_x_transition"],
        "coefficients": coefficients,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Daw Two-Step Reinforcement Learning Task Analysis ===\n")

    # Generate synthetic data with balanced MB/MF behavior
    print("Generating synthetic data (n=200, MB weight=0.5) ...")
    df = generate_synthetic_data(n_trials=200, model_based_weight=0.5, seed=42)
    print(f"  Trials: {len(df)}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Transition counts:\n{df['step2_state'].value_counts().to_string()}\n")

    # Analyze
    result = analyze_twostep(df)
    print("Logistic regression results (stay ~ reward * transition):")
    for name, val in result["coefficients"].items():
        print(f"  {name:25s} = {val:+.4f}")
    print(f"\n  Model-free index  (reward main effect):      {result['model_free_index']:+.4f}")
    print(f"  Model-based index (reward x transition):     {result['model_based_index']:+.4f}")

    # Compare purely MF vs purely MB agents
    print("\n--- Comparison across agent types ---")
    for label, w in [("Model-free (w=0.0)", 0.0),
                      ("Balanced   (w=0.5)", 0.5),
                      ("Model-based(w=1.0)", 1.0)]:
        df_agent = generate_synthetic_data(n_trials=500, model_based_weight=w, seed=99)
        res = analyze_twostep(df_agent)
        print(f"  {label}:  MF index = {res['model_free_index']:+.4f},  "
              f"MB index = {res['model_based_index']:+.4f}")
