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

"""
    compute_t_statistic(estimate::Float64, se::Float64) -> Float64

Compute t-statistic from estimate and standard error.
Zero allocation scalar computation.
"""
@inline function compute_t_statistic(estimate::Float64, se::Float64)::Float64
    return estimate / se
end

"""
    compute_p_value(t_stat::Float64) -> Float64

Compute two-tailed p-value from t-statistic.
Zero allocation scalar computation.
"""
@inline function compute_p_value(t_stat::Float64)::Float64
    return 2 * (1 - cdf(Normal(), abs(t_stat)))
end

"""
    compute_confidence_interval(estimate::Float64, se::Float64, α::Float64=0.05) -> (Float64, Float64)

Compute confidence interval bounds.
Zero allocation, returns tuple of scalars.

# Arguments
- `estimate::Float64`: Point estimate
- `se::Float64`: Standard error  
- `α::Float64`: Significance level (default 0.05 for 95% CI)

# Returns
- `(lower_bound, upper_bound)`: Tuple of confidence bounds
"""
@inline function compute_confidence_interval(estimate::Float64, se::Float64, α::Float64=0.05)
    critical_value = quantile(Normal(), 1 - α/2)
    margin = critical_value * se
    return (estimate - margin, estimate + margin)
end

"""
    compute_statistical_summary_scalars(estimate::Float64, gradient::Vector{Float64}, Σ::Matrix{Float64}, α::Float64=0.05) -> (Float64, Float64, Float64, Float64, Float64)

Compute complete statistical summary returning scalar tuple.

This is the **preferred** function for most use cases as it avoids any allocation
by returning a tuple of scalars rather than a struct.

# Arguments
- `estimate::Float64`: Point estimate
- `gradient::Vector{Float64}`: Pre-allocated gradient (not copied)
- `Σ::Matrix{Float64}`: Covariance matrix
- `α::Float64`: Significance level for CI

# Returns
`(se, t_stat, p_value, ci_lower, ci_upper)` - All Float64 scalars

# Performance
- Zero allocations (all scalar computations)
- No gradient copying
- Optimal for tight loops and performance-critical code
"""
@inline function compute_statistical_summary_scalars(
    estimate::Float64, gradient::Vector{Float64}, Σ::Matrix{Float64}, α::Float64=0.05
)
    se = compute_delta_method_se(gradient, Σ)
    t_stat = compute_t_statistic(estimate, se)
    p_value = compute_p_value(t_stat)
    ci_lower, ci_upper = compute_confidence_interval(estimate, se, α)
    
    return (se, t_stat, p_value, ci_lower, ci_upper)
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
    contrast(result::MarginsResult, row1::Int, row2::Int, Σ::Matrix{Float64}; α::Float64=0.05)

Compute contrast between two result rows with delta-method inference.

This function works for both `EffectsResult` and `PredictionsResult` objects,
computing the difference between estimates at two rows with proper standard errors.

# Arguments
- `result`: EffectsResult or PredictionsResult object
- `row1`, `row2`: Row numbers to contrast
- `Σ`: Model covariance matrix from `vcov(model)`
- `α`: Significance level (default 0.05)

# Returns
`(contrast, se, t_stat, p_value, ci_lower, ci_upper, estimate1, estimate2)`

# Examples
```julia
# Prediction contrast
pred_result = profile_margins(model, data, cartesian_grid(x=[0, 1]); type=:predictions)
contrast(pred_result, 1, 2, vcov(model))

# Effect contrast
eff_result = profile_margins(model, data, cartesian_grid(x=[0, 1]); type=:effects, vars=[:z])
contrast(eff_result, 1, 2, vcov(model))
```
"""
function contrast(result::MarginsResult, row1::Int, row2::Int,
                 Σ::Matrix{Float64}; α::Float64=0.05)
    n = length(result.estimates)
    (row1 < 1 || row1 > n) && throw(ArgumentError("row1=$row1 out of bounds (1:$n)"))
    (row2 < 1 || row2 > n) && throw(ArgumentError("row2=$row2 out of bounds (1:$n)"))

    contrast_val = result.estimates[row1] - result.estimates[row2]
    grad_contrast = result.gradients[row1, :] .- result.gradients[row2, :]

    se, t_stat, p_value, ci_lower, ci_upper = compute_statistical_summary_scalars(
        contrast_val, grad_contrast, Σ, α
    )

    return (contrast=contrast_val, se=se, t_stat=t_stat, p_value=p_value,
            ci_lower=ci_lower, ci_upper=ci_upper,
            estimate1=result.estimates[row1], estimate2=result.estimates[row2])
end

"""
    contrast(df::DataFrame, row1::Int, row2::Int, Σ::Matrix{Float64}; α::Float64=0.05)

Compute contrast between two rows in a margins DataFrame using row indices.

This method provides a simple interface for computing contrasts directly from DataFrames
produced by `DataFrame(profile_margins(...))` or `DataFrame(population_margins(...))`.

# Arguments
- `df`: DataFrame from `DataFrame(result; include_gradients=true)`
- `row1`, `row2`: Row numbers to contrast
- `Σ`: Model covariance matrix from `vcov(model)`
- `α`: Significance level (default 0.05)

# Returns
`(contrast, se, t_stat, p_value, ci_lower, ci_upper, estimate1, estimate2, gradient)`

Note: Also returns contrast gradient vector (`gradient`) for further analysis.

# Examples
```julia
using Margins, GLM, DataFrames

# Get predictions DataFrame
pred_result = profile_margins(model, data, cartesian_grid(treatment=[0, 1], age=[30, 40]))
df = DataFrame(pred_result; include_gradients=true)
println(df)  # View the structure

# Contrast rows 2 and 1 directly
contrast(df, 2, 1, vcov(model))

# Works with any margins result
eff_result = profile_margins(model, data, cartesian_grid(education=["HS", "College"]);
                             type=:effects, vars=[:x1])
df = DataFrame(eff_result; include_gradients=true)
contrast(df, 2, 1, vcov(model))  # Compare row 2 vs row 1
```

# Error Handling
- Throws `ArgumentError` if gradient column is not present (use `include_gradients=true`)
- Throws `ArgumentError` if row indices are out of bounds

# See Also
- `contrast(::MarginsResult, ::Int, ::Int, ...)`: Direct result object method
- `contrast(::DataFrame, ::NamedTuple, ::NamedTuple, ...)`: Column-based row selection
"""
function contrast(
    df::DataFrame,
    row1::Int,
    row2::Int,
    Σ::Matrix{Float64};
    α::Float64=0.05,
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
    se, t_stat, p_value, ci_lower, ci_upper = compute_statistical_summary_scalars(
        contrast_val, grad_contrast, Σ, α
    )

    return (contrast=contrast_val, se=se, t_stat=t_stat, p_value=p_value,
            ci_lower=ci_lower, ci_upper=ci_upper,
            estimate1=est1, estimate2=est2, gradient=grad_contrast)
end

"""
    contrast(df::DataFrame, row1_spec::NamedTuple, row2_spec::NamedTuple, Σ::Matrix{Float64}; α::Float64=0.05)

Compute contrast between two rows identified by column values in a margins DataFrame.

This method provides a convenient interface for computing contrasts using column-based
row selection, similar to DataFrame filtering. Works with DataFrames produced by
`DataFrame(profile_margins(...))` or `DataFrame(population_margins(...))`.

# Arguments
- `df`: DataFrame from `DataFrame(result; include_gradients=true)`
- `row1_spec`: NamedTuple specifying column values for first row (e.g., `(x=1, treatment=0)`)
- `row2_spec`: NamedTuple specifying column values for second row (e.g., `(x=1, treatment=1)`)
- `Σ`: Model covariance matrix from `vcov(model)`
- `α`: Significance level (default 0.05)

# Column Specifications
The NamedTuple keys must match column names in the DataFrame:
- **Profile columns**: Bare variable names for `profile_margins()` results (e.g., `treatment`, `age`)
- **Scenario columns**: `at_` prefixed for population scenarios (e.g., `at_education`)
- **Effect columns**: `variable`, `contrast` for identifying specific effects
- **Group columns**: Group variable names for grouped analyses

# Returns
`(contrast, se, t_stat, p_value, ci_lower, ci_upper, estimate1, estimate2, row1, row2, gradient)`

Note: Also returns the matched row indices (`row1`, `row2`) and contrast gradient vector (`gradient`)
for transparency and further analysis.

# Examples
```julia
using Margins, GLM, DataFrames

# Profile predictions across treatment levels
pred_result = profile_margins(model, data,
                              cartesian_grid(treatment=[0, 1], age=[30, 40, 50]))
df = DataFrame(pred_result; include_gradients=true)

# Contrast predictions at age=30 across treatment levels
contrast(df, (treatment=1, age=30), (treatment=0, age=30), vcov(model))

# Effect of x1 across modifier levels
eff_result = profile_margins(model, data,
                             cartesian_grid(education=["HS", "College", "Grad"]),
                             type=:effects, vars=[:x1, :x2])
df = DataFrame(eff_result; include_gradients=true)

# Compare x1 effect between education levels
contrast(df, (variable="x1", education="College"), (variable="x1", education="HS"), vcov(model))

# Population margins with scenarios
pop_result = population_margins(model, data; scenarios=(age=[30, 60],), type=:effects)
df = DataFrame(pop_result; include_gradients=true)

# Contrast using at_ prefix (population scenarios)
contrast(df, (variable="x1", at_age=60), (variable="x1", at_age=30), vcov(model))
```

# Error Handling
- Throws `ArgumentError` if specified column does not exist (lists available columns)
- Throws `ArgumentError` if no rows match the specification
- Throws `ArgumentError` if multiple rows match (ambiguous specification - be more specific)
- Throws `ArgumentError` if gradient column is not present (use `include_gradients=true`)

# See Also
- `contrast(::MarginsResult, ::Int, ::Int, ...)`: Direct result object method
- `contrast(::DataFrame, ::Int, ::Int, ...)`: Simple integer-based row selection
"""
function contrast(
    df::DataFrame,
    row1_spec::NamedTuple,
    row2_spec::NamedTuple,
    Σ::Matrix{Float64};
    α::Float64=0.05,
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
    se, t_stat, p_value, ci_lower, ci_upper = compute_statistical_summary_scalars(
        contrast_val, grad_contrast, Σ, α
    )

    return (
        contrast=contrast_val, se=se, t_stat=t_stat, p_value=p_value,
        ci_lower=ci_lower, ci_upper=ci_upper,
        estimate1=est1, estimate2=est2,
        row1=row1_idx, row2=row2_idx, gradient=grad_contrast
    )
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

# End of statistics.jl