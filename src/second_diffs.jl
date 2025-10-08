# second_diffs.jl
# Second differences (interaction contrasts)

using LinearAlgebra: dot

"""
    second_difference(ame_result, variable::Symbol, modifier::Symbol, vcov::Matrix{Float64}; contrast="derivative")

Calculate the second difference (interaction effect) of a variable's AME across a modifier
using proper delta-method standard errors with gradient information.

This measures whether the effect of `variable` differs across levels of the `modifier`,
i.e., the interaction ∂²P / (∂variable ∂modifier).

# Arguments
- `ame_result`: Result from population_margins with scenarios over the modifier
- `variable`: The focal variable to calculate second difference for
- `modifier`: The moderating variable. Must have exactly 2 levels in profile_values.
- `vcov`: Parameter covariance matrix from the model (e.g., GLM.vcov(model))
- `contrast`: The contrast type to filter on (default "derivative" for continuous vars)

# Returns
A NamedTuple with:
- `variable`: Variable name
- `modifier`: modifier variable name
- `second_diff`: The second difference estimate (AME_level2 - AME_level1)
- `se`: Standard error of the second difference (using proper delta method)
- `z_stat`: Z-statistic
- `p_value`: P-value for test that second difference ≠ 0
- `ame_at_level1`: AME at first level of modifier
- `ame_at_level2`: AME at second level of modifier
- `modifier_level1`: Value of first modifier level
- `modifier_level2`: Value of second modifier level

# Example
For socio4 as modifier (measuring accuracy/discrimination):
- Positive: Variable increases accuracy (improves sensitivity more than FPR)
- Negative: Variable decreases accuracy (increases FPR more than sensitivity)
- Zero: Variable doesn't affect discrimination ability
"""
function second_difference(
    ame_result::EffectsResult, variable::Symbol, modifier::Symbol, vcov::Matrix{Float64};
    contrast="derivative"
)
    # Check that modifier exists in profile_values
    if isnothing(ame_result.profile_values) || !haskey(ame_result.profile_values, modifier)
        error("AME result does not have modifier $modifier in profile_values")
    end

    # Find indices for this variable and contrast at each modifier level
    var_str = string(variable)
    indices = Int[]
    modifier_levels = []

    for i in 1:length(ame_result.estimates)
        if ame_result.variables[i] == var_str && ame_result.terms[i] == contrast
            push!(indices, i)
            push!(modifier_levels, ame_result.profile_values[modifier][i])
        end
    end

    if length(indices) != 2
        error("Expected 2 rows (one for each $modifier level), got $(length(indices))")
    end

    # Sort by modifier level for consistency
    perm = sortperm(modifier_levels)
    idx1, idx2 = indices[perm[1]], indices[perm[2]]
    level1, level2 = modifier_levels[perm[1]], modifier_levels[perm[2]]

    # Get estimates
    est_level1 = ame_result.estimates[idx1]
    est_level2 = ame_result.estimates[idx2]
    second_diff = est_level2 - est_level1

    # Get gradients for each AME
    g1 = ame_result.gradients[idx1, :]
    g2 = ame_result.gradients[idx2, :]

    # Gradient of the difference: g_diff = g2 - g1
    g_diff = g2 .- g1

    # Delta method SE: sqrt((g2-g1)' * Σ * (g2-g1))
    # This properly accounts for the covariance between the two AMEs
    variance = dot(g_diff, vcov, g_diff)
    se_diff = sqrt(max(0.0, variance))  # Ensure non-negative due to numerical precision

    # Calculate z-statistic and p-value
    z_stat = second_diff / se_diff
    p_value = 2 * (1 - cdf(Normal(), abs(z_stat)))

    return (
        variable = variable,
        modifier = modifier,
        second_diff = second_diff,
        se = se_diff,
        z_stat = z_stat,
        p_value = p_value,
        ame_at_level1 = est_level1,
        ame_at_level2 = est_level2,
        modifier_level1 = level1,
        modifier_level2 = level2
    )
end

"""
    second_differences_table(ame_result, variables::Vector{Symbol}, modifier::Symbol, vcov::Matrix{Float64}; contrast="derivative")

Calculate second differences for multiple variables and return as DataFrame.
"""
function second_differences_table(
    ame_result::EffectsResult, variables::Vector{Symbol}, modifier::Symbol, vcov::Matrix{Float64}; contrast="derivative"
)
    results = [second_difference(ame_result, v, modifier, vcov; contrast=contrast) for v in variables]
    df = DataFrame(results)
    # Add significance indicator
    df.significant = df.p_value .< 0.05
    return df
end

"""
    second_differences_all_contrasts(ame_result, variable::Symbol, modifier::Symbol, vcov::Matrix{Float64})

Calculate second differences for all contrasts of a variable (useful for categorical variables).

For continuous variables with only "derivative" contrast, returns a single-row DataFrame.
For categorical variables, returns one row per contrast level.

# Arguments
- `ame_result`: Result from population_margins with scenarios over the modifier
- `variable`: The variable to calculate second differences for
- `modifier`: The moderating variable
- `vcov`: Parameter covariance matrix from the model

# Returns
DataFrame with one row per contrast, containing second difference statistics.

# Example
```julia
# For continuous variable - returns 1 row (derivative contrast)
sd_age = second_differences_all_contrasts(ame_result1, :age_h, :socio4, Σ)

# For categorical variable - returns multiple rows (one per contrast)
sd_relation = second_differences_all_contrasts(ame_result1, :relation, :socio4, Σ)
```
"""
function second_differences_all_contrasts(
    ame_result::EffectsResult, variable::Symbol, modifier::Symbol, vcov::Matrix{Float64}
)
    # Get all unique contrasts for this variable
    var_str = string(variable)
    contrasts = unique([ame_result.terms[i] for i in 1:length(ame_result.estimates)
                        if ame_result.variables[i] == var_str])

    # Compute second difference for each contrast
    results = NamedTuple[]
    for contrast in contrasts
        try
            result = second_difference(ame_result, variable, modifier, vcov; contrast=contrast)
            push!(results, result)
        catch e
            # Skip contrasts that don't have exactly 2 modifier levels
            if !occursin("Expected 2 rows", string(e))
                rethrow(e)
            end
        end
    end

    if isempty(results)
        error("No valid contrasts found for variable $variable with modifier $modifier")
    end

    df = DataFrame(results)
    # Add significance indicator and contrast column for clarity
    df.contrast = [r.modifier_level2 == r.modifier_level1 ? "derivative" :
                   ame_result.terms[findfirst(i -> ame_result.variables[i] == var_str &&
                                               ame_result.profile_values[modifier][i] == r.modifier_level1,
                                               1:length(ame_result.estimates))]
                   for r in results]
    df.significant = df.p_value .< 0.05

    return df
end
