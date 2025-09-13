# Weights in Population Analysis

This guide explains how to use observation weights in `population_margins`, how weighted averaging and delta‑method standard errors are computed, and how weights interact with groups and scenarios.

## Scope and Policy

- Weights are supported in `population_margins` via the `weights` keyword.
- `profile_margins` does not accept weights — profiles evaluate scenarios at reference points without averaging over the sample.
- Statistical correctness: Weighted quantities use proper normalization and delta‑method SEs use the averaged gradient with the model’s full covariance matrix Σ.

## Supported Forms

- `weights = nothing` (default): Unweighted analysis.
- `weights = :colname` (Symbol): Column in `data` with weights (sampling or frequency).
- `weights = vector::AbstractVector{<:Real}`: Vector of weights with `length == nrow(data)`.

## Weighted Computation

Let `w_i ≥ 0` be weights for observation `i` in the current context (after grouping filters). Then:

- Weighted mean effect: `AME = (∑ w_i · Δ_i) / (∑ w_i)`
- Weighted averaged gradient: `ḡ = (∑ w_i · g_i) / (∑ w_i)`
- Standard error (delta method): `se = sqrt(ḡ' · Σ · ḡ)`

Where `Δ_i` is the per‑row effect (continuous derivative or categorical contrast) and `g_i` is the corresponding per‑row parameter gradient; `Σ` is the model covariance matrix.

These formulas are used consistently in:
- Ungrouped population effects and predictions
- Grouped analyses (applied within each subgroup)
- Scenario analyses (applied within each scenario × subgroup context)

## Examples

```julia
using Random
using DataFrames, CategoricalArrays, GLM, Margins

Random.seed!(123)
n = 200
df = DataFrame(
    y = randn(n),
    x = randn(n),
    z = randn(n),
    group = categorical(rand(["A","B"], n)),
    samp_w = rand(0.5:0.1:2.0, n),             # sampling weights
    freq_w = rand([1,2,3,4], n)                 # frequency weights
)

model = lm(@formula(y ~ x + z + group), df)

# 1) Unweighted population AME
ame_unw = population_margins(model, df; type=:effects, vars=[:x, :z])

# 2) Sampling weights via column name
ame_samp = population_margins(model, df; type=:effects, vars=[:x, :z], weights=:samp_w)

# 3) Frequency weights via column name
ame_freq = population_margins(model, df; type=:effects, vars=[:x, :z], weights=:freq_w)

# 4) Explicit weight vector
wvec = Float64.(df.samp_w)
ame_vec = population_margins(model, df; type=:effects, vars=[:x, :z], weights=wvec)

# 5) Grouped weighted analysis
grp_samp = population_margins(model, df; type=:effects, vars=[:x], groups=:group, weights=:samp_w)

# 6) Scenarios with weights (counterfactual z values)
scen_w = population_margins(model, df; type=:effects, vars=[:x], scenarios=(z=[-1.0, 0.0, 1.0]), weights=:samp_w)
```

All results use weighted averaging with proper normalization by the total weight in each context and delta‑method SEs computed from the averaged gradient and full covariance Σ.

## Best Practices

- Provide non‑negative weights; zero weights effectively drop observations.
- For grouped analyses, ensure the weight column/vector aligns with the original data (the implementation indexes weights by original row indices).
- Confirm units/interpretation: sampling vs frequency weights may yield different magnitudes depending on the empirical distribution they imply.
- Use stable data types (Float64 for weight vectors) to avoid implicit conversions.

## Error Handling

- Length mismatch for `weights::Vector` vs `nrow(data)` → error.
- Invalid weight column name → error.
- Using a variable as both a weight and a simultaneous effect variable or grouping key may error if it creates an internal contradiction; prefer distinct columns.

