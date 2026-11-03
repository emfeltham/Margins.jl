# Centralized statistical computation utilities for Margins.jl.
#
# This file consolidates the repeated delta-method SE computation patterns
# while maintaining **ZERO ALLOCATION** characteristics. All functions are
# designed to work with pre-allocated buffers and existing data structures.
#
# **CRITICAL**: No new allocations are introduced. All functions either:
# 1. Return scalar values (zero allocation)
# 2. Write to pre-allocated output buffers
# 3. Work with existing gradient vectors without copying

"""
    compute_delta_method_se(gradient::Vector{Float64}, Σ::Matrix{Float64}) -> Float64

Compute delta-method standard error from gradient and covariance matrix.

This is a zero-allocation wrapper around delta_method_se()
that provides a consistent API point for all SE computations in Margins.jl.

# Arguments
- `gradient::Vector{Float64}`: Pre-allocated gradient vector (not copied)
- `Σ::Matrix{Float64}`: Model covariance matrix

# Returns
- `Float64`: Standard error (zero allocation)

# Performance
- Zero allocations (direct delegation to FormulaCompiler)
- No gradient copying or modification
- Identical performance to direct FormulaCompiler calls
"""
@inline function compute_delta_method_se(gradient::AbstractVector{Float64}, Σ::Matrix{Float64})::Float64
    return delta_method_se(gradient, Σ)
end

# =============================================================================
# Inference distribution: t with residual df for linear models, z otherwise
# =============================================================================

"""
    _inference_df(model, vcov) -> Float64

Degrees of freedom of the reference distribution used for every test statistic, p-value
and confidence interval that Margins reports for `model` with covariance estimator `vcov`.
This is the single place where the rule is defined:

- **Linear models with the model-based covariance** — a GLM.jl `LinearModel` (from `lm`) or
  a `GeneralizedLinearModel` with a `Normal` family and an `IdentityLink`, with `vcov` the
  default `GLM.vcov` (or a matrix equal to `vcov(model)`): `dof_residual(model)`. Marginal
  effects, predictions and their contrasts are then linear combinations of β, so
  `estimate / se` has an exact t distribution with the residual df (as in Stata's `margins`
  after `regress`, R's `emmeans`, and `GLM.coeftable` for `lm`).
- **Everything else** — other GLMs, mixed models, FixedEffectModels, and *any* robust,
  clustered or user-supplied covariance (a CovarianceMatrices estimator such as `HC1()` or
  `CR1(...)`, a custom function, or a matrix that differs from `vcov(model)`): `Inf`,
  i.e. the standard Normal (z). The covariance argument does not carry the information
  needed for Stata's small-sample conventions (residual df after `vce(robust)`,
  `n_clusters - 1` after `vce(cluster)`), so these estimators use asymptotic z inference.

The rule depends only on the model and the covariance estimator, not on the quantity: with an
identity link the link and response scales coincide, and for a linear model all quantities
(including elasticities) use the same t reference, as Stata does after `regress`.
"""
function _inference_df(model, vcov)::Float64
    _is_linear_regression(model) || return Inf
    _is_model_based_vcov(model, vcov) || return Inf
    d = Float64(GLM.dof_residual(model))
    return (isfinite(d) && d > 0) ? d : Inf
end

# A GLM.jl linear regression: `LinearModel`, or a Normal-family identity-link GLM (bare or
# wrapped in a `TableRegressionModel`). Purely type-based, so it never errors on other models.
function _is_linear_regression(model)::Bool
    inner = model isa StatsModels.TableRegressionModel ? model.model : model
    inner isa GLM.LinearModel && return true
    if inner isa GLM.GeneralizedLinearModel
        return inner.rr.d isa Normal && GLM.Link(inner) isa GLM.IdentityLink
    end
    return false
end

# Is `vcov` the model's own (classical) covariance estimator?
_is_model_based_vcov(model, vcov::Function) = vcov === GLM.vcov
_is_model_based_vcov(model, vcov::AbstractMatrix) = vcov == GLM.vcov(model)
_is_model_based_vcov(model, vcov) = false  # CovarianceMatrices estimators and anything else

"""
    _inference_dist(df::Real)

Reference distribution for `df` degrees of freedom: `TDist(df)` when `df` is finite,
`Normal()` when `df == Inf`.
"""
_inference_dist(df::Real) = isfinite(df) ? TDist(df) : Normal()

# Degrees of freedom stored on a result (`Inf` for results built without one).
_result_df(mr)::Float64 = Float64(get(mr.metadata, :df, Inf))

# Store the inference df on a result's metadata (called once by the public entry points).
function _attach_inference_df!(result, model, vcov, Σ=nothing)
    result.metadata[:df] = _inference_df(model, vcov)
    # The covariance matrix behind this result's standard errors, so that second differences
    # can default to it (`_stored_vcov`).
    isnothing(Σ) || (result.metadata[:vcov] = Matrix{Float64}(Σ))
    # For a linear regression also keep the model-based covariance and residual df, so that
    # second differences can decide t vs z from the covariance matrix they are *given*
    # (see `_second_difference_dof`), not from the one this result was computed with.
    if _is_linear_regression(model)
        d = Float64(GLM.dof_residual(model))
        if isfinite(d) && d > 0
            result.metadata[:model_df] = d
            result.metadata[:model_vcov] = Matrix{Float64}(GLM.vcov(model))
        end
    end
    return result
end

# The covariance matrix a result was computed with (second-difference functions default to it).
function _stored_vcov(ame_result)::Matrix{Float64}
    V = get(ame_result.metadata, :vcov, nothing)
    isnothing(V) && error("This result does not record the covariance matrix it was computed " *
                          "with; pass `vcov` explicitly.")
    return V
end

# Inference df for a second difference computed with covariance `Σ`: t with the residual df
# when `Σ` is the linear model's own covariance, z (`Inf`) for any other matrix or when the
# result carries no linear-model information (same rule as `_inference_df`).
function _second_difference_dof(ame_result, Σ::AbstractMatrix)::Float64
    d = get(ame_result.metadata, :model_df, Inf)
    V = get(ame_result.metadata, :model_vcov, nothing)
    (isfinite(d) && V !== nothing && size(V) == size(Σ) && Σ == V) ? Float64(d) : Inf
end

# Make the second-difference df rule agree with `_inference_df(model, vcov)` for the covariance the
# user gave: drop the recorded linear-model information when that rule says z (e.g. `vcov = m -> vcov(m)`).
function _align_second_difference_df!(ame_result, model, vcov)
    isfinite(_inference_df(model, vcov)) && return ame_result
    delete!(ame_result.metadata, :model_df)
    delete!(ame_result.metadata, :model_vcov)
    return ame_result
end

"""
    compute_statistic(estimate::Float64, se::Float64) -> Float64

Test statistic `estimate / se` for H₀: estimate = 0 (a t statistic when the inference
df is finite, a z statistic when it is `Inf`). Zero allocation scalar computation.
"""
@inline function compute_statistic(estimate::Float64, se::Float64)::Float64
    return estimate / se
end

"""
    compute_p_value(statistic::Float64, df::Real=Inf) -> Float64

Two-tailed p-value of `statistic` under the t distribution with `df` degrees of
freedom (Normal when `df == Inf`). Uses `ccdf`, so tiny p-values do not underflow to
`1 - 1`.
"""
@inline function compute_p_value(statistic::Float64, df::Real=Inf)::Float64
    return 2 * ccdf(_inference_dist(df), abs(statistic))
end

"""
    _critical_value(α::Real, df::Real=Inf) -> Float64

Two-sided critical value `quantile(dist, 1 - α/2)` of the t(`df`) distribution (Normal
when `df == Inf`).
"""
_critical_value(α::Real, df::Real=Inf) = quantile(_inference_dist(df), 1 - α / 2)

"""
    compute_confidence_interval(estimate::Float64, se::Float64, α::Float64=0.05, df::Real=Inf) -> (Float64, Float64)

Compute confidence interval bounds `estimate ± c·se`, where `c` is the t(`df`) critical
value (Normal when `df == Inf`). Zero allocation, returns tuple of scalars.
"""
@inline function compute_confidence_interval(estimate::Float64, se::Float64, α::Float64=0.05, df::Real=Inf)
    margin = _critical_value(α, df) * se
    return (estimate - margin, estimate + margin)
end

"""
    compute_statistical_summary_scalars(estimate, gradient, Σ, α=0.05, df=Inf) -> (se, statistic, p_value, ci_lower, ci_upper)

Compute complete statistical summary returning scalar tuple. `df` is the inference
degrees of freedom (see [`_inference_df`](@ref)); `Inf` gives z inference.
"""
@inline function compute_statistical_summary_scalars(
    estimate::Float64, gradient::AbstractVector{Float64}, Σ::AbstractMatrix{Float64}, α::Float64=0.05, df::Real=Inf
)
    se = compute_delta_method_se(gradient, Σ)
    statistic = compute_statistic(estimate, se)
    p_value = compute_p_value(statistic, df)
    ci_lower, ci_upper = compute_confidence_interval(estimate, se, α, df)
    
    return (se, statistic, p_value, ci_lower, ci_upper)
end

"""
    compute_se_only(gradient::Vector{Float64}, Σ::Matrix{Float64}) -> Float64

Compute only the standard error (most common use case).
Identical to compute_delta_method_se but with clearer naming for single-purpose use.
"""
@inline function compute_se_only(gradient::AbstractVector{Float64}, Σ::Matrix{Float64})::Float64
    return delta_method_se(gradient, Σ)
end

"""
    _is_gaussian_identity(model) -> Bool

Return `true` only when `model` is a Gaussian-family model with an identity link
(i.e. an ordinary linear model). This is the sole model family for which a
symmetric prediction interval `ŷ ± z·√(SE² + σ²)` is well-defined: GLMs have no
additive residual variance on the response scale (the outcome is Bernoulli/count
and bounded), so a symmetric band there would be statistically meaningless and
could escape the valid range. Handles both bare and `TableRegressionModel`-wrapped
models. Mixed models are excluded (their predictive variance involves the random
effects, which is out of scope here).
"""
function _is_gaussian_identity(model)::Bool
    (_auto_link(model) isa GLM.IdentityLink) || return false
    inner = hasfield(typeof(model), :model) ? getfield(model, :model) : model
    hasfield(typeof(inner), :rr) || return false
    rr = getfield(inner, :rr)
    rr isa GLM.LmResp && return true
    if rr isa GLM.GlmResp && hasfield(typeof(rr), :d)
        return getfield(rr, :d) isa Normal
    end
    return false
end

"""
    _prediction_residual_variance(model) -> Float64

Residual variance σ̂² used to widen a confidence interval into a prediction
interval. For an ordinary linear model this is `RSS / (n − p)` via
`deviance(model) / dof_residual(model)`, matching R's
`predict(lm, interval = "prediction")`.

Throws an `ArgumentError` for any model that is not Gaussian with an identity
link, rather than emitting a misleading symmetric interval.

Note: this is the homoskedastic residual variance. The resulting prediction
interval assumes constant error variance; if a heteroskedasticity-robust `vcov`
is supplied to widen the coefficient SEs, the residual term here is still the
single pooled σ̂², so the two pieces rest on different variance assumptions.
"""
function _prediction_residual_variance(model)::Float64
    if !_is_gaussian_identity(model)
        throw(ArgumentError(
            "Prediction intervals (interval=:prediction) are only supported for " *
            "Gaussian models with an identity link (ordinary linear models). " *
            "Model of type $(typeof(model)) is not eligible — use " *
            "interval=:confidence instead."))
    end
    dof_resid = GLM.dof_residual(model)
    if dof_resid <= 0
        throw(ArgumentError(
            "Cannot compute a prediction interval: the model has " *
            "dof_residual = $dof_resid (no residual degrees of freedom to " *
            "estimate σ²). A saturated model has no residual variance to add."))
    end
    return GLM.deviance(model) / dof_resid
end

"""
    contrast(result::MarginsResult, row1::Int, row2::Int, Σ::Matrix{Float64}; alpha::Float64=0.05) -> ContrastResult

Compute contrast between two result rows with delta-method inference.

This function works for both `EffectsResult` and `PredictionsResult` objects,
computing the difference between estimates at two rows with proper standard errors.

# Arguments
- `result`: EffectsResult or PredictionsResult object
- `row1`, `row2`: Row numbers to contrast
- `Σ`: Model covariance matrix from `vcov(model)`
- `alpha`: Significance level (default 0.05)

# Returns
`ContrastResult` object with fields: contrast, se, statistic, df, p_value, ci_lower, ci_upper, estimate1, estimate2, gradient, row1, row2, metadata

# Inference
A contrast is a linear combination of the same estimates, so it uses the result's own
reference distribution: `df = result.metadata[:df]` (t with the residual df for a linear
model with the model-based covariance, `Inf` = z otherwise; see the `population_margins`
docstring).

# Statistical Correctness
The standard error accounts for the covariance between the two estimates. The gradient-based delta method
automatically handles scale transformations (_e.g._, log-odds → probability for logistic models).
See `src/inference/delta_method.jl` for detailed mathematical justification of why
`vcov(model)` is always on the right scale.

# Examples
```julia
# Prediction contrast
pred_result = profile_margins(model, data, cartesian_grid(x = [0, 1]); type = :predictions)
cr = contrast(pred_result, 1, 2, vcov(model))
println(cr) # Formatted output

# Effect contrast
eff_result = profile_margins(model, data, cartesian_grid(x = [0, 1]); type = :effects, vars = [:z])
cr = contrast(eff_result, 1, 2, vcov(model))

# Convert to DataFrame
df = DataFrame(cr)
```
"""
function contrast(result::MarginsResult, row1::Int, row2::Int,
                 Σ::Matrix{Float64}; alpha::Float64=0.05)
    n = length(result.estimates)
    (row1 < 1 || row1 > n) && throw(ArgumentError("row1 = $row1 out of bounds (1:$n)"))
    (row2 < 1 || row2 > n) && throw(ArgumentError("row2 = $row2 out of bounds (1:$n)"))

    contrast_val = result.estimates[row1] - result.estimates[row2]
    grad_contrast = result.gradients[row1, :] .- result.gradients[row2, :]

    dof = _result_df(result)
    se, statistic, p_value, ci_lower, ci_upper = compute_statistical_summary_scalars(
        contrast_val, grad_contrast, Σ, alpha, dof
    )

    return ContrastResult(
        contrast_val,
        se,
        statistic,
        dof,
        p_value,
        ci_lower,
        ci_upper,
        result.estimates[row1],
        result.estimates[row2],
        grad_contrast,
        row1,
        row2,
        Dict{Symbol, Any}(:alpha => alpha, :source => :result_object)
    )
end

"""
    contrast(df::AbstractDataFrame, row1::Int, row2::Int, Σ::Matrix{Float64}; alpha::Float64=0.05) -> ContrastResult

Compute contrast between two rows in a margins DataFrame using row indices.

This method provides a simple interface for computing contrasts directly from DataFrames
produced by `DataFrame(profile_margins(...))` or `DataFrame(population_margins(...))`.

# Arguments
- `df`: DataFrame from `DataFrame(result; include_gradients=true)`
- `row1`, `row2`: Row numbers to contrast
- `Σ`: Model covariance matrix from `vcov(model)`
- `alpha`: Significance level (default 0.05)

# Returns
`ContrastResult` object with fields: contrast, se, statistic, df, p_value, ci_lower, ci_upper, estimate1, estimate2, gradient, row1, row2, metadata

# Examples
```julia
using Margins, GLM, DataFrames

# Get predictions DataFrame
pred_result = profile_margins(model, data, cartesian_grid(treatment=[0, 1], age=[30, 40]))
df = DataFrame(pred_result; include_gradients=true)
println(df)  # View the structure

# Contrast rows 2 and 1 directly
cr = contrast(df, 2, 1, vcov(model))
println(cr) # Formatted output

# Works with any margins result
eff_result = profile_margins(model, data, cartesian_grid(education=["HS", "College"]);
                             type=:effects, vars=[:x1])
df = DataFrame(eff_result; include_gradients=true)
cr = contrast(df, 2, 1, vcov(model))  # Compare row 2 vs row 1
```

# Inference
Uses the `df` column of the two rows (t with that df when finite, z when `Inf`); a
DataFrame without a `df` column gets z inference.

# Error Handling
- Throws `ArgumentError` if gradient column is not present (use `include_gradients=true`)
- Throws `ArgumentError` if row indices are out of bounds

# See Also
- `contrast(::MarginsResult, ::Int, ::Int, ...)`: Direct result object method
- `contrast(::DataFrame, ::NamedTuple, ::NamedTuple, ...)`: Column-based row selection
"""
function contrast(
    df::AbstractDataFrame,
    row1::Int,
    row2::Int,
    Σ::Matrix{Float64};
    alpha::Float64=0.05,
    estimatename = :estimate
)
    # Validate gradient column exists
    if !hasproperty(df, :gradient)
        error("DataFrame must include gradient column. Use: DataFrame(result; include_gradients=true)")
    end

    # Validate row indices
    n = nrow(df)
    (row1 < 1 || row1 > n) && throw(ArgumentError("row1=$row1 out of bounds (1:$n)"))
    (row2 < 1 || row2 > n) && throw(ArgumentError("row2=$row2 out of bounds (1:$n)"))

    # Extract estimates and gradients
    est1 = df[row1, estimatename]
    est2 = df[row2, estimatename]
    grad1 = df.gradient[row1]
    grad2 = df.gradient[row2]

    # Compute contrast
    contrast_val = est1 - est2
    grad_contrast = grad1 .- grad2

    # Statistical inference
    dof = _dataframe_contrast_df(df, row1, row2)
    se, statistic, p_value, ci_lower, ci_upper = compute_statistical_summary_scalars(
        contrast_val, grad_contrast, Σ, alpha, dof
    )

    return ContrastResult(
        contrast_val,
        se,
        statistic,
        dof,
        p_value,
        ci_lower,
        ci_upper,
        est1,
        est2,
        grad_contrast,
        row1,
        row2,
        Dict{Symbol, Any}(:alpha => alpha, :source => :dataframe_integer)
    )
end

"""
    contrast(df::AbstractDataFrame, row1_spec::NamedTuple, row2_spec::NamedTuple, Σ::Matrix{Float64}; alpha::Float64=0.05) -> ContrastResult

Compute contrast between two rows identified by column values in a margins DataFrame.

This method provides a convenient interface for computing contrasts using column-based
row selection, similar to DataFrame filtering. Works with DataFrames produced by
`DataFrame(profile_margins(...))` or `DataFrame(population_margins(...))`.

# Arguments
- `df`: DataFrame from `DataFrame(result; include_gradients=true)`
- `row1_spec`: NamedTuple specifying column values for first row (e.g., `(x=1, treatment=0)`)
- `row2_spec`: NamedTuple specifying column values for second row (e.g., `(x=1, treatment=1)`)
- `Σ`: Model covariance matrix from `vcov(model)`
- `alpha`: Significance level (default 0.05)

# Column Specifications
The NamedTuple keys must match column names in the DataFrame:
- **Profile columns**: Bare variable names for `profile_margins()` results (e.g., `treatment`, `age`)
- **Scenario columns**: `at_` prefixed for population scenarios (e.g., `at_education`)
- **Effect columns**: `variable`, `contrast` for identifying specific effects
- **Group columns**: `over_` prefixed for population groups (e.g., `over_region`)

# Returns
`ContrastResult` object with fields: contrast, se, statistic, df, p_value, ci_lower, ci_upper, estimate1, estimate2, gradient, row1, row2, metadata

# Examples
```julia
using Margins, GLM, DataFrames

# Profile predictions across treatment levels
pred_result = profile_margins(model, data,
                              cartesian_grid(treatment=[0, 1], age=[30, 40, 50]))
df = DataFrame(pred_result; include_gradients=true)

# Contrast predictions at age=30 across treatment levels
cr = contrast(df, (treatment=1, age=30), (treatment=0, age=30), vcov(model))
println(cr)

# Effect of x1 across modifier levels
eff_result = profile_margins(model, data,
                             cartesian_grid(education=["HS", "College", "Grad"]),
                             type=:effects, vars=[:x1, :x2])
df = DataFrame(eff_result; include_gradients=true)

# Compare x1 effect between education levels
cr = contrast(df, (variable="x1", education="College"), (variable="x1", education="HS"), vcov(model))

# Population margins with scenarios
pop_result = population_margins(model, data; scenarios=(age=[30, 60],), type=:effects)
df = DataFrame(pop_result; include_gradients=true)

# Contrast using at_ prefix (population scenarios)
cr = contrast(df, (variable="x1", at_age=60), (variable="x1", at_age=30), vcov(model))
```

# Error Handling
- Throws `ArgumentError` if specified column does not exist (lists available columns)
- Throws `ArgumentError` if no rows match the specification
- Throws `ArgumentError` if multiple rows match (ambiguous specification - be more specific)
- Throws `ArgumentError` if gradient column is not present (use `include_gradients=true`)

# See Also
- `contrast(::MarginsResult, ::Int, ::Int, ...)`: Direct result object method
- `contrast(::AbstractDataFrame, ::Int, ::Int, ...)`: Simple integer-based row selection
"""
function contrast(
    df::AbstractDataFrame,
    row1_spec::NamedTuple,
    row2_spec::NamedTuple,
    Σ::Matrix{Float64};
    alpha::Float64=0.05,
    estimatename = :estimate
)
    # Validate gradient column exists
    if !hasproperty(df, :gradient)
        error("DataFrame must include gradient column. Use: DataFrame(result; include_gradients=true)")
    end

    # Find matching rows
    row1_idx = _find_unique_row(df, row1_spec)
    row2_idx = _find_unique_row(df, row2_spec)

    # Extract estimates and gradients
    est1 = df[row1_idx, estimatename]
    est2 = df[row2_idx, estimatename]
    grad1 = df[row1_idx, :gradient]
    grad2 = df[row2_idx, :gradient]

    # Compute contrast
    contrast_val = est1 - est2
    grad_contrast = grad1 .- grad2

    # Statistical inference
    dof = _dataframe_contrast_df(df, row1_idx, row2_idx)
    se, statistic, p_value, ci_lower, ci_upper = compute_statistical_summary_scalars(
        contrast_val, grad_contrast, Σ, alpha, dof
    )

    return ContrastResult(
        contrast_val,
        se,
        statistic,
        dof,
        p_value,
        ci_lower,
        ci_upper,
        est1,
        est2,
        grad_contrast,
        row1_idx,
        row2_idx,
        Dict{Symbol, Any}(:alpha => alpha, :source => :dataframe_namedtuple,
                          :row1_spec => row1_spec, :row2_spec => row2_spec)
    )
end

# Inference df for a contrast between two rows of a margins DataFrame: the rows' `df`
# column (they come from one result, so share it); `Inf` (z) if the column is absent.
function _dataframe_contrast_df(df::AbstractDataFrame, row1::Int, row2::Int)::Float64
    hasproperty(df, :df) || return Inf
    d1 = Float64(df[row1, :df]); d2 = Float64(df[row2, :df])
    return d1 == d2 ? d1 : min(d1, d2)
end

"""
    _find_unique_row(df::DataFrame, spec::NamedTuple) -> Int

Internal utility to find a unique row matching column specifications.

Applies all column filters specified in the NamedTuple and returns the
unique matching row index. Throws informative errors if no match or
multiple matches are found.

# Arguments
- `df`: DataFrame to search
- `spec`: NamedTuple of (column_name => value) pairs

# Returns
- `Int`: Row index of the unique match

# Errors
- Column not found: Lists available columns
- No matches: Reports the specification that failed
- Multiple matches: Indicates ambiguous specification
"""
function _find_unique_row(df::DataFrame, spec::NamedTuple)
    # Start with all rows
    mask = trues(nrow(df))

    # Apply each filter
    for (col, val) in pairs(spec)
        if !hasproperty(df, col)
            available_cols = join(names(df), ", ")
            throw(ArgumentError("Column :$col not found in DataFrame. Available columns: $available_cols"))
        end

        # Handle different comparison types (supports strings, numbers, symbols, etc.)
        mask .&= (df[!, col] .== val)
    end

    # Count matches
    matching_rows = findall(mask)
    n_matches = length(matching_rows)

    if n_matches == 0
        throw(ArgumentError("No rows match specification: $spec"))
    elseif n_matches > 1
        throw(ArgumentError("Multiple rows ($n_matches) match specification: $spec. " *
                          "Provide more specific column values to identify a unique row."))
    end

    return matching_rows[1]
end

"""
    DataFrame(cr::ContrastResult; include_gradient::Bool=false, include_metadata::Bool=false)

Convert ContrastResult to a single-row DataFrame.

# Arguments
- `cr`: ContrastResult returned by `contrast()` functions
- `include_gradient`: Whether to include the gradient vector column (default: false)
- `include_metadata`: Whether to include metadata fields as columns (default: false)

# Returns
- `DataFrame`: Single-row DataFrame with contrast statistics

# Columns
The resulting DataFrame contains:
- `contrast`: The estimated contrast (difference)
- `se`: Standard error
- `statistic`: Test statistic `contrast / se` (t when `df` is finite, z when `df == Inf`)
- `df`: Inference degrees of freedom (`Inf` for z)
- `p_value`: Two-tailed p-value
- `ci_lower`: Lower confidence interval bound
- `ci_upper`: Upper confidence interval bound
- `estimate1`: First estimate value
- `estimate2`: Second estimate value
- `row1`: First row index (if available)
- `row2`: Second row index (if available)
- `gradient`: Contrast gradient vector (only if `include_gradient=true`)
- `alpha`: Significance level (only if `include_metadata=true`)
- `source`: Source of contrast computation (only if `include_metadata=true`)
- `row1_spec`: First row specification as string (only if `include_metadata=true` and available)
- `row2_spec`: Second row specification as string (only if `include_metadata=true` and available)

# Examples
```julia
using Margins, GLM, DataFrames

# Profile predictions
pred_result = profile_margins(model, data, cartesian_grid(x=[0, 1]))
df_pred = DataFrame(pred_result; include_gradients=true)

# Compute contrast
cr = contrast(df_pred, 1, 2, vcov(model))

# Convert to DataFrame
df_contrast = DataFrame(cr)
# → Single row with: contrast, se, statistic, df, p_value, ci_lower, ci_upper, estimate1, estimate2

# Include gradient for further analysis
df_contrast_full = DataFrame(cr; include_gradient=true)

# Include metadata for complete information
df_contrast_complete = DataFrame(cr; include_gradient=true, include_metadata=true)
# → Includes alpha, source, and row specifications (if available)
```

# See Also
- `contrast(::MarginsResult, ...)`: Compute contrasts from result objects
- `contrast(::AbstractDataFrame, ...)`: Compute contrasts from DataFrames
"""
function DataFrames.DataFrame(
    cr::ContrastResult;
    include_gradient::Bool=false, include_metadata::Bool=false
)
    # Build DataFrame with all required fields (columns in documented order)
    df_dict = Pair{Symbol, Any}[
        :contrast => [cr.contrast],
        :se => [cr.se],
        :statistic => [cr.statistic],
        :df => [cr.df],
        :p_value => [cr.p_value],
        :ci_lower => [cr.ci_lower],
        :ci_upper => [cr.ci_upper],
        :estimate1 => [cr.estimate1],
        :estimate2 => [cr.estimate2]
    ]

    # Add row indices if available
    if !isnothing(cr.row1)
        push!(df_dict, :row1 => [cr.row1])
    end
    if !isnothing(cr.row2)
        push!(df_dict, :row2 => [cr.row2])
    end

    # Add gradient if requested
    if include_gradient
        push!(df_dict, :gradient => [cr.gradient])
    end

    # Add metadata fields if requested
    if include_metadata
        # Always include alpha if available
        if haskey(cr.metadata, :alpha)
            push!(df_dict, :alpha => [cr.metadata[:alpha]])
        end

        # Include source if available
        if haskey(cr.metadata, :source)
            push!(df_dict, :source => [string(cr.metadata[:source])])
        end

        # Include row specifications if available (for NamedTuple-based contrasts)
        if haskey(cr.metadata, :row1_spec)
            push!(df_dict, :row1_spec => [string(cr.metadata[:row1_spec])])
        end
        if haskey(cr.metadata, :row2_spec)
            push!(df_dict, :row2_spec => [string(cr.metadata[:row2_spec])])
        end
    end

    return DataFrame(df_dict...)
end

# End of statistics.jl