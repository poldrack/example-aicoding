# Project Tasks — AI-Driven Data Science Problem Solver

## Legend
- [ ] Not started
- [~] In progress
- [x] Completed

---

## Phase 0: Project Setup

- [x] **0.1** Restructure `src/` from `example_aicoding` to `aicoding` package with `__init__.py`
- [x] **0.2** Create `tests/` directory with `__init__.py`
- [x] **0.3** Create `data/` directory (already existed)
- [x] **0.4** ~~Locate or create `problems.tsv`~~ — Found at `data/coding_problems.tsv`
- [x] **0.5** Update `pyproject.toml`: set package name, add core dependencies, pin versions
- [x] **0.6** Install the project in dev mode and verify pytest runs

---

## Phase 1: Tier 1 Problems (12 problems — previously solved in 1 prompt)

For each problem: read prompt → create dirs → write tests → confirm fail → implement → recheck prompt → confirm pass → write README.

- [x] **1.1** `pubmed` (#2) — Abstract base class for publications, Biopython/Entrez PubMed search (10 tests)
- [x] **1.2** `LDA` (#3) — PubMed abstract download, text cleaning, LDA topic modeling (10 tests)
- [x] **1.3** `lmm` (#5) — Combine CSV files, linear mixed model (random intercept + slope) (10 tests)
- [x] **1.4** `download_github_code` (#7) — GitHub API — download 100 most recent Python files (5 tests)
- [x] **1.5** `mk_classification_data` (#10) — Generate synthetic classification test data (20 tests)
- [x] **1.6** `DAG` (#15) — Directed graph class, d-separation test, validate against NetworkX (44 tests)
- [x] **1.7** `imgsmooth` (#20) — Spatial smoothing on NIfTI images using median filter, plot comparison (19 tests)
- [x] **1.8** `overfitting` (#25) — Simulate overfitting with polynomial regression, train/test error, plots (22 tests)
- [x] **1.9** `ascii_art` (#28) — Read an image, render as ASCII art (11 tests)
- [x] **1.10** `PCA_poweriteration` (#30) — PCA via power iteration (from scratch), compare to sklearn PCA (28 tests)
- [x] **1.11** `ttest_wrapper` (#31) — Wrapper for statsmodels `ttest_ind` with R-style output (37 tests)
- [x] **1.12** `randomization` (#32) — Permutation test simulation for correlation hypothesis testing (23 tests)

---

## Phase 2: Tier 2 Problems (7 problems — previously solved in 2–3 prompts)

- [x] **2.1** `bad_cv` (#9) — Demonstrate incorrect cross-validation (feature selection outside loop). Depends on `mk_classification_data` (#10) from Tier 1.
- [x] **2.2** `CLT` (#16) — Central limit theorem simulation across 6 distributions, plot means
- [x] **2.3** `PC` (#21) — PC causal inference algorithm from scratch, partial correlation test
- [x] **2.4** `textanalysis` (#22) — Sentiment analysis, linguistic complexity (L2SCA, Coh-Metrix measures)
- [x] **2.5** `twostep` (#23) — Daw two-step reinforcement learning task analysis
- [x] **2.6** `balanced_cv` (#24) — Balanced cross-validation class (sklearn-style), minimize split variance
- [x] **2.7** `pdf_generation` (#29) — Markdown to PDF renderer with header image, argparse CLI

---

## Phase 3: Tier 3 Problems (2 problems — previously solved in 4+ prompts)

- [x] **3.1** `linear_regression` (#1) — Extend sklearn LinearRegression with t-stats and p-values (27 tests)
- [x] **3.2** `hurdle` (#18) — Hurdle regression model with sklearn-style interface (20 tests)

---

## Phase 4: Tier 4 Problems (11 problems — previously difficult / unresolved)

Each has known issues noted from prior attempts. Extra care required.

- [x] **4.1** `pubmed_emails` (#6) — Scrape author emails from institutional sites via PubMed (14 tests)
- [x] **4.2** `code_complexity` (#8) — Cyclomatic complexity, maintainability index, Halstead metrics via radon (19 tests)
- [x] **4.3** `mk_animation` (#11) — Text scrolling animation, record to AVI (7 tests)
- [x] **4.4** `ddm` (#4) — Drift diffusion model simulation + EZ-diffusion parameter recovery (17 tests)
- [x] **4.5** `PAweather` (#27) — NOAA weather API, monthly max temp timeseries 1960–2000 (8 tests)
- [x] **4.6** `logisticmap` (#13) — Logistic map simulation, bifurcation plot (x vs r) (14 tests)
- [x] **4.7** `ddm_collapsing_plot` (#17) — DDM with collapsing bounds, return full timeseries (11 tests)
- [x] **4.8** `transformer` (#12) — Three-layer transformer model, train on small text corpus (12 tests)
- [x] **4.9** `GES` (#14) — Greedy equivalence search for causal discovery (13 tests)
- [x] **4.10** `corridor` (#19) — Schönbrodt "corridor of stability" — correlation variance vs sample size (12 tests)
- [x] **4.11** `factoranalysis` (#26) — Exploratory factor analysis with BIC selection on survey data (15 tests)

---

## Phase 5: Evaluation Harness & Final Deliverables

- [x] **5.1** Build evaluation harness that runs each solution script and generates an HTML report with outputs (text and images)
- [x] **5.2** Run full test suite (`pytest tests/ -v`) — 592 passed, 11 skipped, 0 failed
- [x] **5.3** All README.md files present for all 32 modules
- [x] **5.4** All 32 `solution.py` files have `__main__` blocks

---

## Notes

- **TDD workflow per problem:** tests first → confirm fail → implement → confirm pass → README
- **Dependency:** `mk_classification_data` (1.5) must complete before `bad_cv` (2.1)
- **External APIs:** `pubmed`, `LDA`, `download_github_code`, `pubmed_emails`, `PAweather` need real API validation plus mock tests
- **`problems.tsv`** is located at `data/coding_problems.tsv`
- **Python version:** REQUIREMENTS.md says 3.11+, but `.python-version` and `pyproject.toml` specify 3.13 — will use 3.13 as configured
