# Two-Step Reinforcement Learning Task Analysis

## Problem

Analyze human performance on the Daw two-step reinforcement learning task to extract indices of model-based and model-free behavior.

## Background

The Daw two-step task (Daw et al., 2011) is a sequential decision-making paradigm that dissociates model-based from model-free reinforcement learning:

- **Step 1:** Participants choose between two options.
- **Transition:** Each step-1 choice leads to one of two step-2 states via either a *common* (70%) or *rare* (30%) transition.
- **Step 2:** Participants make a second choice and receive a binary reward.

The critical analysis examines **stay/switch** behavior at step 1 as a function of the *previous* trial's reward and transition type:

- **Model-free** learners repeat previously rewarded actions regardless of transition type (main effect of reward on stay probability).
- **Model-based** learners account for the transition structure: they stay after common+rewarded trials and rare+unrewarded trials, but switch after common+unrewarded and rare+rewarded trials (reward x transition interaction).

## Approach

### Synthetic Data Generation

`generate_synthetic_data()` simulates an agent whose behavior is a mixture of model-based and model-free strategies, controlled by `model_based_weight` (0 = purely model-free, 1 = purely model-based). Reward probabilities drift slowly across trials to mimic the original task design.

### Analysis

`analyze_twostep()` fits a logistic regression predicting stay/switch as a function of:

- **Previous reward** (effect-coded: -0.5 unrewarded, +0.5 rewarded)
- **Previous transition type** (effect-coded: +0.5 common, -0.5 rare)
- **Reward x transition interaction**

The returned indices are:

| Index | Coefficient | Interpretation |
|-------|------------|----------------|
| `model_free_index` | Reward main effect | Positive = tends to repeat rewarded actions |
| `model_based_index` | Reward x transition | Positive = accounts for transition structure |

## Usage

```python
from aicoding.twostep.solution import generate_synthetic_data, analyze_twostep

df = generate_synthetic_data(n_trials=200, model_based_weight=0.5, seed=42)
result = analyze_twostep(df)
print(result["model_free_index"])   # reward main effect
print(result["model_based_index"])  # reward x transition interaction
```

## References

Daw, N. D., Gershman, S. J., Seymour, B., Dayan, P., & Dolan, R. J. (2011). Model-based influences on humans' choices and striatal prediction errors. *Neuron*, 69(6), 1204-1215.
