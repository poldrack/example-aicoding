# Project Requirements: AI-Driven Data Science Problem Solver

## Overview

This project uses Claude Code to solve a set of data science problems using test-driven development (TDD). Each problem is defined in `problems.tsv` and implemented as an independent module within a shared project structure.

## Project Structure

```
project-root/
├── REQUIREMENTS.md
├── problems.tsv
├── pyproject.toml
├── src/
│   └── aicoding/
│       ├── __init__.py
│       ├── pubmed/
│       │   ├── __init__.py
│       │   ├── solution.py
│       │   └── README.md
│       ├── lmm/
│       │   ├── ...
│       └── ...
├── tests/
│   ├── __init__.py
│   ├── pubmed/
│   │   ├── __init__.py
│   │   └── test_solution.py
│   └── ...
└── data/              # any input datasets referenced by problems
```

- Each problem gets its own subdirectory under `src/aicoding/`, using the module name derived from the `File` column (strip the leading number prefix and `.py` suffix).
- The corresponding tests live under `tests/` in a mirrored directory structure.

## Problem Definition Format

Problems are defined in `problems.tsv` (tab-separated). Columns:

| Column | Description |
|--------|-------------|
| `File` | Source filename. Derive the module slug by stripping the leading `N_` prefix and `.py` suffix (e.g., `2_pubmed.py` → `pubmed`). |
| `Prompt` | Full problem statement and requirements. This is the specification to implement against. |
| `NumPromptsToSuccess` | Historical difficulty rating (informational only — not used at runtime). |

## Problem Inventory

The following 32 problems are defined in the TSV. Each becomes a subdirectory under `src/aicoding/`.

### Tier 1 — Previously solved in 1 prompt

| # | Module | Summary |
|---|--------|---------|
| 2 | `pubmed` | Abstract base class for publications, Biopython/Entrez PubMed search |
| 3 | `LDA` | PubMed abstract download, text cleaning, LDA topic modeling |
| 5 | `lmm` | Combine CSV files, linear mixed model (random intercept + slope) |
| 7 | `download_github_code` | GitHub API — download 100 most recent Python files |
| 10 | `mk_classification_data` | Generate synthetic classification test data |
| 15 | `DAG` | Directed graph class, d-separation test, validate against NetworkX |
| 20 | `imgsmooth` | Spatial smoothing on NIfTI images using median filter, plot comparison |
| 25 | `overfitting` | Simulate overfitting with polynomial regression, train/test error, plots |
| 28 | `ascii_art` | Read an image, render as ASCII art |
| 30 | `PCA_poweriteration` | PCA via power iteration (from scratch), compare to sklearn PCA |
| 31 | `ttest_wrapper` | Wrapper for statsmodels `ttest_ind` with R-style output |
| 32 | `randomization` | Permutation test simulation for correlation hypothesis testing |

### Tier 2 — Previously solved in 2–3 prompts

| # | Module | Summary |
|---|--------|---------|
| 9 | `bad_cv` | Demonstrate incorrect cross-validation (feature selection outside loop) |
| 16 | `CLT` | Central limit theorem simulation across 6 distributions, plot means |
| 21 | `PC` | PC causal inference algorithm from scratch, partial correlation test |
| 22 | `textanalysis` | Sentiment analysis, linguistic complexity (L2SCA, Coh-Metrix measures) |
| 23 | `twostep` | Daw two-step reinforcement learning task analysis |
| 24 | `balanced_cv` | Balanced cross-validation class (sklearn-style), minimize split variance |
| 29 | `pdf_generation` | Markdown to PDF renderer with header image, argparse CLI |

### Tier 3 — Previously solved in 4+ prompts

| # | Module | Summary |
|---|--------|---------|
| 1 | `linear_regression` | Extend sklearn LinearRegression with t-stats and p-values |
| 18 | `hurdle` | Hurdle regression model with sklearn-style interface |

### Tier 4 — Previously difficult / unresolved

| # | Module | Summary | Known Issues |
|---|--------|---------|--------------|
| 6 | `pubmed_emails` | Scrape author emails from institutional sites via PubMed | Web scraping fragile |
| 8 | `code_complexity` | Cyclomatic complexity, maintainability index, Halstead metrics via radon | Needed many iterations |
| 11 | `mk_animation` | Text scrolling animation, record to AVI | System library issues |
| 4 | `ddm` | Drift diffusion model simulation + EZ-diffusion parameter recovery | Hallucinated equations |
| 27 | `PAweather` | NOAA weather API, monthly max temp timeseries 1960–2000 | API label issues |
| 13 | `logisticmap` | Logistic map simulation, bifurcation plot (x vs r) | Plot rendering issues |
| 17 | `ddm_collapsing_plot` | DDM with collapsing bounds, return full timeseries | Incomplete output |
| 12 | `transformer` | Three-layer transformer model, train on small text corpus | torchtext compat |
| 14 | `GES` | Greedy equivalence search for causal discovery | Persistent KeyError |
| 19 | `corridor` | Schönbrodt "corridor of stability" — correlation variance vs sample size | Plot issues |
| 26 | `factoranalysis` | Exploratory factor analysis with BIC selection on survey data | BIC computation |

## Execution Order

Work through problems by tier, completing all problems in a tier before moving to the next:

1. **Tier 1** (12 problems) — all previously solved in 1 prompt.
2. **Tier 2** (7 problems) — previously solved in 2–3 prompts.
3. **Tier 3** (2 problems) — previously solved in 4+ prompts.
4. **Tier 4** (11 problems) — previously difficult or unresolved. Consult the `notes` column for known failure modes.

Within each tier, work in the order listed in the inventory tables above.

**Dependency note:** Problem #10 (`mk_classification_data`) generates test data for problem #9 (`bad_cv`). Since #10 is in Tier 1 and #9 is in Tier 2, this dependency is satisfied naturally by the tier ordering.

## Development Workflow (Per Problem)

Follow strict TDD for each problem in this order:

1. **Read the problem** from the corresponding row in `problems.tsv`.
2. **Create the subdirectory** under `src/aicoding/<module>/` and `tests/<module>/`.
3. **Write tests first** in `tests/<module>/test_solution.py` that encode the expected behavior described in the `Prompt` column. Tests should cover:
   - Correctness on representative inputs (happy path).
   - Edge cases (empty inputs, boundary values, type errors where applicable).
   - Output format and type validation.
4. **Run the tests** — confirm they fail.
5. **Implement the solution** in `src/aicoding/<module>/solution.py` to make the tests pass.  
6. **Doublecheck against the problem description**: After the code is generated, reread the problem description and make sure that the code follows the instructions completely.  Edit if necessary to achieve this.
6. **Run the tests again** — confirm they all pass.
7. **Write a brief README.md** in the problem subdirectory summarizing the approach.
8. **Evaluation harness**: Create an evaluation harness that runs each of the solution scripts and generates an html report with the outputs (text and images) from each script.

## Technical Specifications

- **Language:** Python 3.11+
- **Test framework:** pytest
- **Package manager:** pip with pyproject.toml
- **Dependencies:** Use well-established libraries (e.g., pandas, numpy, scikit-learn, scipy, statsmodels, matplotlib, biopython, nibabel, radon). Pin versions in `pyproject.toml`.
- **Code style:** Clean, modular, simple. Functions over classes where possible (unless a class is explicitly requested in the prompt). Docstrings on all public functions.
- **APIs**: Code must use real external API calls.

## Test Requirements

Each problem's test file must include:

- At least one test per documented behavior in the problem prompt.
- At least one edge-case test.
- Tests should be independent and not rely on execution order.
- Use pytest fixtures for shared setup where appropriate.
- Tests should run fast (< 5 seconds each) unless the problem inherently requires longer computation.
- For problems that involve external APIs (PubMed, GitHub, NOAA), it is essential to ensure that the code runs with the external API.  Mock tests may be used to test the interfaces, but the true API also needs to be validated.
- For problems that require synthetic data, generate it within the test fixtures.

## Execution

Run all tests from the project root:

```bash
pytest tests/ -v
```

Run tests for a single problem:

```bash
pytest tests/<module>/ -v
```

## Deliverables

For each problem in the TSV:

1. `src/aicoding/<module>/solution.py` — working implementation. MUST HAVE __main__ block.
2. `tests/<module>/test_solution.py` — passing test suite.
3. `src/aicoding/<module>/README.md` — brief summary of the approach.

## Notes

- If a problem is ambiguous, document assumptions in the README and encode them in the tests.
- If a problem requires external data, place it in `data/` and reference it by relative path.
- Each problem should be solvable independently — no cross-dependencies between problem modules, with the exception of #10 (`mk_classification_data`) which generates test data for #9 (`bad_cv`).
- Problems with known historical issues (Tier 4) may require extra care. 
- For problems that request plots or visual output, tests should validate that the figure is created (e.g., check that a file exists or that matplotlib axes contain expected data), not the visual appearance.
