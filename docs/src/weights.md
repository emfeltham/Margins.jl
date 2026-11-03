# Weights in Population Analysis

This guide explains how to use observation weights in `population_margins` (and the model-level `second_differences`), how weighted averaging and delta‑method standard errors are computed, and how weights interact with groups and scenarios.

## Scope and Policy

- Weights are supported in `population_margins` and the model-level `second_differences` via the `weights` keyword. The default, `weights = :model`, uses the weights the model was fit with.
- `profile_margins` does not accept weights, and its typical values are unweighted (see [Known limitation: profile margins](#Known-limitation:-profile-margins)).
- Weighted quantities use proper normalization, and delta‑method SEs use the averaged gradient with the covariance matrix Σ given by `vcov` (see [What weights do not change](#What-weights-do-not-change)).

## Supported Forms

- `weights = :model` (default): the model's estimation weights if it has any (`glm(…; wts = w)`, `fit(MixedModel, …; wts = w)`), otherwise unweighted.
- `weights = :none`: unweighted analysis (the default before 3.0). `weights = nothing` means the same.
- `weights = :colname` (Symbol): column in `data` with weights. `:model` and `:none` are reserved; pass a column with one of those names as a vector.
- `weights = vector::AbstractVector{<:Real}`: vector of weights with `length == nrow(data)`.

`metadata[:weights_source]` records which applied: `:model`, `:user` (a column or vector) or `:none`. A second-differences table records it as table metadata, `metadata(sd, "weights_source")`.

## Model weights by default

The standard errors come from the model's covariance matrix, and that matrix was computed with the estimation weights (GLM.jl 1.x treats `wts` as frequency weights). An unweighted average with that covariance would describe one sample in the point estimate and another in its SE, so averages use the model's weights unless told otherwise. This also makes the model's own identities hold: for a canonical-link GLM with an intercept, Σ wᵢ(yᵢ − μ̂ᵢ) = 0, so the weighted average prediction equals the weighted mean of y.

```julia
m = glm(@formula(y ~ x + z), df, Binomial(); wts = df.w)

population_margins(m, df)                  # weighted by df.w   (weights_source = :model)
population_margins(m, df; weights = :w)    # the same numbers   (weights_source = :user)
population_margins(m, df; weights = :none) # unweighted, as in 2.x

aap = population_margins(m, df; type = :predictions)
aap.estimates[1] ≈ sum(df.w .* df.y) / sum(df.w)    # true
```

Explicit weights on an unweighted model are a legitimate analysis (averaging over a reweighted population) and give no warning.

How each model class is read (`Margins._model_weights`):

| model | weights used by `:model` |
|:------|:-------------------------|
| GLM.jl `lm`, `glm` | the `wts` of the fit (none for an unweighted fit) |
| MixedModels.jl `LinearMixedModel`, `GeneralizedLinearMixedModel` | the `wts` of the fit |
| FixedEffectModels.jl `reg(…)` | none, with a warning (once per session): a `FixedEffectModel` does not store its weights, so whether it was weighted is unknown; pass `weights = :w` for a weighted `reg`, or `weights = :none` to silence the warning |
| other model classes | none |

The model's weight vector must have one entry per row of `data`. If the fit dropped rows (for example rows with missing values), the lengths differ and `:model` raises an error asking you to pass `weights=` for the rows of `data` explicitly; Margins does not try to match rows. If a model carries StatsBase `AnalyticWeights` (precision weights, which set the estimation precision but not the population), `:model` is an error asking for an explicit choice; frequency and probability weights are used as they are.

## Weighted Computation

Let `w_i ≥ 0` be weights for observation `i` in the current context (after grouping filters). Then:

- Weighted mean effect: `AME = (∑ w_i · Δ_i) / (∑ w_i)`
- Weighted averaged gradient: `ḡ = (∑ w_i · g_i) / (∑ w_i)`
- Standard error (delta method): `se = sqrt(ḡ' · Σ · ḡ)`

Where `Δ_i` is the per‑row effect (continuous derivative or categorical contrast) and `g_i` is the corresponding per‑row parameter gradient; `Σ` is the covariance matrix from `vcov`.

These formulas are used consistently in:
- Ungrouped population effects and predictions
- Grouped analyses (applied within each subgroup)
- Scenario analyses (applied within each scenario × subgroup context)

## What weights do not change

Weights set the point estimates. The standard errors come from `Σ`, which is `vcov(model)` by default: for GLM.jl that is a frequency-weight covariance (multiplying `wts` by 10 divides every SE by √10). These are **not design-based survey standard errors**: there is no adjustment for strata, clusters or the sampling design. For design-based inference, pass an appropriate estimator or covariance matrix as `vcov` (for example a CovarianceMatrices.jl cluster-robust estimator, or a matrix you computed).

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

# 1) Unweighted population AME (the model has no weights, so :model means none)
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

# 7) A weighted fit: its weights are used by default
wmodel = lm(@formula(y ~ x + z + group), df; wts=Float64.(df.freq_w))
ame_w = population_margins(wmodel, df; vars=[:x, :z])                  # weighted by freq_w
ame_0 = population_margins(wmodel, df; vars=[:x, :z], weights=:none)   # unweighted
```

All results use weighted averaging with proper normalization by the total weight in each context and delta‑method SEs computed from the averaged gradient and Σ.

## Known limitation: profile margins

`profile_margins` evaluates at reference points rather than averaging over the sample, and it has no `weights` keyword. The typical values it uses are **unweighted**, even for a weighted model: the means (or `typical` function) of `means_grid`, `balanced_grid` and `hierarchical_grid`, the fill-in values for variables a grid leaves out, and the frequency mixtures of categorical variables. Results report `weights_source = :none`. To evaluate at weighted typical values, compute them and give them in the grid, e.g. `cartesian_grid(x = [sum(df.w .* df.x) / sum(df.w)])`.

## Best Practices

- Provide non‑negative weights; zero weights effectively drop observations.
- For grouped analyses, ensure the weight column/vector aligns with the original data (the implementation indexes weights by original row indices).
- Confirm units/interpretation: sampling vs frequency weights may yield different magnitudes depending on the empirical distribution they imply.
- Pass the data the model was fit to (its estimation sample) when relying on `weights = :model`.
- Use stable data types (Float64 for weight vectors) to avoid implicit conversions.

## Error Handling

- Length mismatch for `weights::Vector` vs `nrow(data)` → error.
- Model weights whose length differs from `nrow(data)` (with `weights = :model`) → error asking for explicit `weights=`.
- Invalid weight column name → error.
- Using a variable as both a weight and a simultaneous effect variable or grouping key may error if it creates an internal contradiction; prefer distinct columns.
