# factoranalysis (#26)

Exploratory factor analysis with BIC-based model selection on survey data.

## Approach

- Loads data from the Self-Regulation Ontology dataset on GitHub.
- Selects variables starting with `upps_impulsivity_survey`, `sensation_seeking_survey`, `bis11_survey`, or `dickman_survey`.
- Fits factor analysis models with 1–5 factors using sklearn's `FactorAnalysis`.
- Computes BIC as `-2 * log_likelihood + k * log(n)` to select the best number of factors.
- For the preferred solution, identifies which variables load most strongly on each factor.
