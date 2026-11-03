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

# The df rule is a property of the model class: `_inference_df(model)` in
# core/model_interface.jl, overridden by the entry points' `df=` keyword.

"""
    _inference_dist(df::Real)

Reference distribution for `df` degrees of freedom: `TDist(df)` when `df` is finite,
`Normal()` when `df == Inf`.
"""
_inference_dist(df::Real) = isfinite(df) ? TDist(df) : Normal()

# The inference of a request's result: the engine's covariance matrix (a fresh copy of the
# request's Σ, so results never share it), the request's df and alpha.
_request_inference(req, engine) = Inference(engine.Σ, req.df, req.alpha)

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
    (_link(model) isa GLM.IdentityLink) || return false
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

With estimation weights (`lm(...; wts=w)`), GLM.jl treats `w` as frequency weights:
σ̂² = Σwᵢrᵢ²/(Σwᵢ − p), with t(Σwᵢ − p), the same convention as `vcov(model)` and the
model's df. The interval is for one new observation of weight 1 (as R's `predict.lm`
assumes by default). It equals R's `predict(lm(...), interval = "prediction")` on the data
with each row repeated wᵢ times, and differs from R's `lm(..., weights = w)`, which treats `w`
as precision weights (σ̂² = Σwᵢrᵢ²/(n − p), df n − p). A weighted fit warns once per session.

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
    _model_weights(model) isa AbstractVector && _warn_weighted_prediction_interval()
    return GLM.deviance(model) / dof_resid
end

# Once per session, like R's "Assuming constant prediction variance even though model fit is weighted"
@noinline _warn_weighted_prediction_interval() =
    @warn "interval=:prediction with a weighted fit: the interval is for a new observation of weight 1, " *
          "with GLM.jl's frequency-weight residual variance Σwᵢrᵢ²/(Σwᵢ − p) and df Σwᵢ − p (the convention " *
          "of vcov(model)). This equals R's predict.lm on the data with each row repeated wᵢ times; R's " *
          "lm(..., weights = w) treats w as precision weights and gives a different interval." maxlog=1

"""
    contrast(result::MarginsResult, row1::Int, row2::Int; alpha=result.inference.alpha) -> ContrastResult
    contrast(result::MarginsResult, spec1::NamedTuple, spec2::NamedTuple; alpha=result.inference.alpha) -> ContrastResult

Contrast between two rows of a result (`estimate[row1] − estimate[row2]`) with delta-method
inference.

This function works for both `EffectsResult` and `PredictionsResult` objects. The rows are
given by number, or by the values of their `DataFrame(result)` columns.

# Arguments
- `result`: EffectsResult or PredictionsResult object
- `row1`, `row2`: Row numbers to contrast
- `spec1`, `spec2`: Column values that identify one row each, with the column names of
  `DataFrame(result)`: bare grid columns for `profile_margins` results (e.g. `(treatment=1, age=30)`),
  `over_<var>` for `groups=` and `at_<var>` for `scenarios=` columns of `population_margins`
  results, and `variable` / `contrast` to pick an effect (e.g. `(variable="x1", at_age=60)`).
  Each spec must match exactly one row.
- `alpha`: Significance level, a real number in (0, 1) (default: the `alpha` the result was
  computed with)

# Returns
`ContrastResult` object with fields: contrast, se, statistic, df, p_value, ci_lower, ci_upper, estimate1, estimate2, gradient, row1, row2, metadata

# Inference
Everything comes from the result (`result.inference`): the standard error is
`√((g₁ − g₂)'Σ(g₁ − g₂))` with `gᵢ = result.gradients[rowᵢ, :]` and `Σ = result.inference.vcov`,
the covariance matrix the result's standard errors were computed with, and the reference
distribution is the result's (`df = result.inference.df`: t when finite, z when `Inf`). To
contrast with a different covariance estimator, recompute the result with `vcov=`.

# Statistical Correctness
The standard error accounts for the covariance between the two estimates. The gradient-based delta method
automatically handles scale transformations (_e.g._, log-odds → probability for logistic models).
See `src/inference/delta_method.jl` for detailed mathematical justification of why
the parameter covariance matrix is always on the right scale.

# Examples
```julia
# Prediction contrast
pred_result = profile_margins(model, data, cartesian_grid(x = [0, 1]); type = :predictions)
cr = contrast(pred_result, 2, 1)
println(cr) # Formatted output

# The same rows, by their column values
cr = contrast(pred_result, (x = 1,), (x = 0,))

# Effect contrast across scenarios, at 90%
eff_result = population_margins(model, data; scenarios = (age = [30, 60],), vars = [:x1])
cr = contrast(eff_result, (variable = "x1", at_age = 60), (variable = "x1", at_age = 30); alpha = 0.10)

# Convert to DataFrame
df = DataFrame(cr)
```
"""
function contrast(result::MarginsResult, row1::Int, row2::Int; alpha::Real=result.inference.alpha)
    n = length(result.estimates)
    (row1 < 1 || row1 > n) && throw(ArgumentError("row1 = $row1 out of bounds (1:$n)"))
    (row2 < 1 || row2 > n) && throw(ArgumentError("row2 = $row2 out of bounds (1:$n)"))
    return _contrast_rows(result, row1, row2, _validate_alpha(alpha),
                          Dict{Symbol, Any}(:source => :result_object))
end

function contrast(result::MarginsResult, spec1::NamedTuple, spec2::NamedTuple;
                  alpha::Real=result.inference.alpha)
    alpha = _validate_alpha(alpha)
    table = DataFrame(result)
    row1 = _find_unique_row(table, spec1)
    row2 = _find_unique_row(table, spec2)
    return _contrast_rows(result, row1, row2, alpha,
                          Dict{Symbol, Any}(:source => :result_namedtuple,
                                            :row1_spec => spec1, :row2_spec => spec2))
end

# The contrast of two rows of a result, with the result's covariance matrix and df
function _contrast_rows(result::MarginsResult, row1::Int, row2::Int, alpha::Float64,
                        metadata::Dict{Symbol, Any})
    contrast_val = result.estimates[row1] - result.estimates[row2]
    grad_contrast = result.gradients[row1, :] .- result.gradients[row2, :]

    (; vcov, df) = result.inference
    se, statistic, p_value, ci_lower, ci_upper = compute_statistical_summary_scalars(
        contrast_val, grad_contrast, vcov, alpha, df
    )
    metadata[:alpha] = alpha

    return ContrastResult(
        contrast_val,
        se,
        statistic,
        df,
        p_value,
        ci_lower,
        ci_upper,
        result.estimates[row1],
        result.estimates[row2],
        grad_contrast,
        row1,
        row2,
        metadata
    )
end

# Margins 2.x forms of `contrast`, removed in 3.0: say what to call instead of a bare MethodError.
function contrast(::MarginsResult, ::Union{Int,NamedTuple}, ::Union{Int,NamedTuple},
                  ::AbstractMatrix; kwargs...)
    throw(ArgumentError(
        "contrast(result, row1, row2, Σ) was removed in Margins 3.0: a contrast uses the " *
        "covariance matrix the result was computed with (result.inference.vcov). Call " *
        "contrast(result, row1, row2). For a different covariance estimator, recompute the " *
        "result with vcov=… (population_margins / profile_margins)."))
end
function contrast(::AbstractDataFrame, args...; kwargs...)
    throw(ArgumentError(
        "contrast(df::DataFrame, …) was removed in Margins 3.0: contrast the result itself, " *
        "contrast(result, row1, row2) or contrast(result, (col=value, …), (col=value, …)), with " *
        "the same row numbers and DataFrame(result) column names. It uses the result's own " *
        "covariance matrix, df and alpha (result.inference)."))
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
pred_result = profile_margins(model, data, cartesian_grid(x=[0, 1]); type=:predictions)

# Compute contrast
cr = contrast(pred_result, 2, 1)

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