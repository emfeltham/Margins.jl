# contrasts.jl
# Second differences: discrete contrasts between pre-computed AME values
#
# This module contains functions for computing second differences (interaction effects)
# from pre-computed AME results. For local derivatives at specific points, see at_point.jl

using LinearAlgebra: dot

"""
    second_differences(ame_result, variables, modifier::Symbol, vcov::Matrix{Float64}; kwargs...)

Unified interface for computing second differences (interaction effects) across all modifier types.

This is the recommended function that automatically handles:
- Binary modifiers (2 levels)
- Categorical modifiers (>2 levels) with all pairwise contrasts
- Continuous modifiers with slope scaling
- Multiple focal variables (compute derivatives for several effects simultaneously)
- Multiple focal variable contrasts (for categorical variables)

Automatically routes to the appropriate specialized function based on:
1. Number of focal variables (single vs multiple)
2. Number of modifier levels (2 vs >2)
3. Number of focal variable contrasts (1 vs multiple)
4. Specified modifier type (auto-detected by default)

# Arguments
- `ame_result::EffectsResult`: Result from population_margins with scenarios over the modifier
- `variables::Union{Symbol,Vector{Symbol}}`: The focal variable(s) to calculate second differences for
  - Single Symbol: One focal variable
  - Vector of Symbols: Multiple focal variables (all analyzed with same modifier)
- `modifier::Symbol`: The moderating variable
- `vcov::Matrix{Float64}`: Parameter covariance matrix from the model

# Keyword Arguments
- `contrast::String="dy/dx"`: Focal variable contrast to analyze (for categorical variables; "derivative" is accepted as an alias of "dy/dx")
- `modifier_type::Symbol=:auto`: Modifier type (:auto, :binary, :categorical, :continuous)
- `all_contrasts::Bool=true`: Compute for all focal variable contrasts (if applicable)

# Returns
- DataFrame with all applicable second differences (one row per variable × contrast × modifier
  pair). The columns are the same for every modifier type (see `Margins.SECOND_DIFFERENCE_COLUMNS`):
  `variable, contrast, modifier, modifier_type, modifier_level1, modifier_level2, ame_at_level1,
  ame_at_level2, second_diff, se, statistic, df, p_value, significant, gradient`.

# Inference
`statistic = second_diff / se`, referred to t(`df`) with `df` inherited from `ame_result`
(`dof_residual(model)` for a linear model with the model-based covariance, `Inf` = z otherwise).

# Examples
```julia
using Margins, GLM, DataFrames

# Single variable, binary modifier → single contrast
ames = population_margins(model, data; scenarios=(treated=[0,1],), type=:effects)
sd = second_differences(ames, :age, :treated, vcov(model))
# → DataFrame with 1 row

# Multiple variables, binary modifier → compare effects
ames = population_margins(model, data; scenarios=(treated=[0,1],), type=:effects)
sd = second_differences(ames, [:age, :education, :experience], :treated, vcov(model))
# → DataFrame with 3 rows (one per variable)
# → Compare which effects vary most with treatment

# Single variable, categorical modifier → all pairwise
ames = population_margins(model, data; scenarios=(education=[:hs, :college, :grad],), type=:effects)
sd = second_differences(ames, :income, :education, vcov(model))
# → DataFrame with 3 rows (college-hs, grad-hs, grad-college)

# Multiple variables, categorical modifier → full comparison
ames = population_margins(model, data; scenarios=(region=[:north, :south, :west],), type=:effects)
sd = second_differences(ames, [:x1, :x2, :x3], :region, vcov(model))
# → DataFrame with 9 rows (3 variables × 3 region pairs)

# Multiple variables, continuous modifier → slopes
ames = population_margins(model, data; scenarios=(age=[30, 45, 60],), type=:effects)
sd = second_differences(ames, [:education_yrs, :experience], :age, vcov(model); modifier_type=:continuous)
# → DataFrame with 6 rows (2 variables × 3 age pairs)

# Categorical variable, categorical modifier → full matrix
ames = population_margins(model, data; scenarios=(region=[:north, :south, :west],),
                         type=:effects, vars=[:education], contrasts=:pairwise)
sd = second_differences(ames, :education, :region, vcov(model))
# → DataFrame with 9 rows (3 education contrasts × 3 region pairs)

# Single contrast only (disable all_contrasts)
sd = second_differences(ames, :education, :region, vcov(model);
                       contrast="college - hs", all_contrasts=false)
# → DataFrame with 3 rows (just college-hs contrast across region pairs)
```

# See Also
- `second_difference()`: Original function for binary modifiers only (backward compatibility)
- `second_differences_pairwise()`: All pairwise modifier comparisons for single contrast
- `second_differences_all_contrasts()`: All contrasts × all modifier pairs
"""
function second_differences(
    ame_result::EffectsResult,
    variables::Union{Symbol,Vector{Symbol}},
    modifier::Symbol,
    vcov::Matrix{Float64};
    contrast::String="dy/dx",
    modifier_type::Symbol=:auto,
    all_contrasts::Bool=true
)
    # Normalize variables to Vector{Symbol}
    vars_vec = variables isa Symbol ? [variables] : variables

    # Compute second differences for each variable
    results = DataFrame[]

    for variable in vars_vec
        # Get information about the variable's contrasts
        var_str = string(variable)
        variable_contrasts = unique([
            ame_result.terms[i] for i in 1:length(ame_result.estimates)
            if ame_result.variables[i] == var_str
        ])

        has_multiple_contrasts = length(variable_contrasts) > 1

        # Determine if user wants all contrasts or just one
        if has_multiple_contrasts && all_contrasts
            # Use second_differences_all_contrasts for comprehensive results
            df = second_differences_all_contrasts(
                ame_result, variable, modifier, vcov;
                modifier_type=modifier_type
            )
        else
            # Use second_differences_pairwise for single contrast
            # Auto-detect the contrast if it is the default "dy/dx" (or its alias "derivative") and there is only one contrast
            contrast_to_use = contrast
            if _contrast_label(contrast) == "dy/dx" && length(variable_contrasts) == 1
                contrast_to_use = variable_contrasts[1]
            end

            df = second_differences_pairwise(
                ame_result, variable, modifier, vcov;
                contrast=contrast_to_use,
                modifier_type=modifier_type
            )
        end

        push!(results, df)
    end

    # Combine all results
    return vcat(results..., cols=:union)
end

"""
    second_difference(ame_result, variable::Symbol, modifier::Symbol, vcov::Matrix{Float64}; contrast="dy/dx")

Calculate the second difference (interaction effect) of a variable's AME across a modifier
using proper delta-method standard errors with gradient information.

This measures whether the effect of `variable` differs across levels of the `modifier`,
i.e., the interaction ∂²P / (∂variable ∂modifier).

# Arguments
- `ame_result`: Result from population_margins with scenarios over the modifier
- `variable`: The focal variable to calculate second difference for
- `modifier`: The moderating variable. Must have exactly 2 levels in profile_values.
- `vcov`: Parameter covariance matrix from the model (e.g., GLM.vcov(model))
- `contrast`: The contrast type to filter on (default "dy/dx" for continuous vars; "derivative" is accepted as an alias)

# Returns
A NamedTuple holding one row of the unified second-difference schema
(`Margins.SECOND_DIFFERENCE_COLUMNS`): `variable, contrast, modifier, modifier_type (= :binary),
modifier_level1, modifier_level2, ame_at_level1, ame_at_level2, second_diff (= AME_level2 -
AME_level1), se, statistic, df, p_value, significant, gradient`. Levels are sorted, so
`modifier_level1 < modifier_level2`. `df` is inherited from `ame_result` (t when finite, z when `Inf`).

# Example
For socio4 as modifier (measuring accuracy/discrimination):
- Positive: Variable increases accuracy (improves sensitivity more than FPR)
- Negative: Variable decreases accuracy (increases FPR more than sensitivity)
- Zero: Variable doesn't affect discrimination ability
"""
function second_difference(
    ame_result::EffectsResult, variable::Symbol, modifier::Symbol, vcov::Matrix{Float64};
    contrast="dy/dx"
)
    # Check that modifier exists in profile_values
    if isnothing(ame_result.profile_values) || !haskey(ame_result.profile_values, modifier)
        error("AME result does not have modifier $modifier in profile_values")
    end

    # Validate dimensions: vcov must match gradient dimensions
    n_params_vcov = size(vcov, 1)
    n_params_gradients = size(ame_result.gradients, 2)
    if n_params_vcov != n_params_gradients
        error("Dimension mismatch: vcov has $n_params_vcov parameters but gradients has $n_params_gradients")
    end

    # Validate vcov is square
    if size(vcov, 1) != size(vcov, 2)
        error("vcov must be square matrix, got size $(size(vcov))")
    end

    # Find indices for this variable and contrast at each modifier level
    var_str = string(variable)
    indices = Int[]
    modifier_levels = []

    for i in 1:length(ame_result.estimates)
        if ame_result.variables[i] == var_str && ame_result.terms[i] == _contrast_label(contrast)
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

    return _second_difference_row(;
        variable, contrast = contrast, modifier, modifier_type = :binary,
        level1, level2, est1 = est_level1, est2 = est_level2,
        second_diff, se = se_diff, gradient = g_diff, dof = _result_df(ame_result),
    )
end

"""
    second_differences_all_contrasts(ame_result, variable::Symbol, modifier::Symbol, vcov::Matrix{Float64}; modifier_type=:auto)

Calculate second differences for all contrasts of a variable across all modifier levels.

This is the comprehensive function that handles:
- **Categorical focal variable**: Iterates over all variable contrasts
- **Binary modifier** (2 levels): Uses `second_difference()` for each contrast
- **Categorical/Continuous modifier** (>2 levels): Uses `second_differences_pairwise()` for each contrast

For continuous variables with only the "dy/dx" contrast, returns results for all modifier pairs.
For categorical variables, returns results for all contrasts × all modifier pairs.

# Arguments
- `ame_result`: Result from population_margins with scenarios over the modifier
- `variable`: The variable to calculate second differences for
- `modifier`: The moderating variable
- `vcov`: Parameter covariance matrix from the model
- `modifier_type`: Modifier type (:auto, :binary, :categorical, :continuous)

# Returns
DataFrame with one row per (contrast × modifier pair), with the unified second-difference
columns (`Margins.SECOND_DIFFERENCE_COLUMNS`, identical for every modifier type).

# Examples
```julia
# Continuous focal variable, binary modifier - returns 1 row
ames = population_margins(model, data; scenarios=(treated=[0,1],), type=:effects)
sd = second_differences_all_contrasts(ames, :age, :treated, vcov)

# Categorical focal variable, binary modifier - returns rows per contrast
ames = population_margins(model, data; scenarios=(treated=[0,1],),
                         type=:effects, vars=[:education], contrasts=:pairwise)
sd = second_differences_all_contrasts(ames, :education, :treated, vcov)
# Returns: college-hs, grad-hs, grad-college (each compared across treated=0 vs 1)

# Categorical focal variable, categorical modifier (>2 levels) - full matrix
ames = population_margins(model, data; scenarios=(region=[:north, :south, :west],),
                         type=:effects, vars=[:education], contrasts=:pairwise)
sd = second_differences_all_contrasts(ames, :education, :region, vcov)
# Returns: (college-hs, grad-hs, grad-college) × (south-north, west-north, west-south)
#          = 9 rows total (3 contrasts × 3 modifier pairs)
```
"""
function second_differences_all_contrasts(
    ame_result::EffectsResult, variable::Symbol, modifier::Symbol,
    vcov::Matrix{Float64};
    modifier_type::Symbol=:auto
)
    # Get all unique contrasts for this variable
    var_str = string(variable)
    contrasts = unique(
        [
            ame_result.terms[i] for i in 1:length(ame_result.estimates)
            if ame_result.variables[i] == var_str
        ]
    )

    # Check number of modifier levels to decide which approach
    modifier_levels = unique([
        ame_result.profile_values[modifier][i]
        for i in 1:length(ame_result.estimates)
        if ame_result.variables[i] == var_str
    ])
    n_modifier_levels = length(modifier_levels)

    # For binary modifier (2 levels), use original approach with second_difference()
    if n_modifier_levels == 2 && modifier_type ∈ (:auto, :binary)
        results = NamedTuple[]
        skipped_contrasts = String[]

        for contrast in contrasts
            try
                result = second_difference(
                    ame_result, variable, modifier, vcov; contrast = contrast
                )
                push!(results, result)
            catch e
                if occursin("Expected 2 rows", string(e))
                    push!(skipped_contrasts, contrast)
                else
                    rethrow(e)
                end
            end
        end

        if !isempty(skipped_contrasts)
            @warn "Skipped $(length(skipped_contrasts)) contrast(s) for variable $variable: $(join(skipped_contrasts, ", "))"
        end

        if isempty(results)
            error("No valid contrasts found for variable $variable with modifier $modifier")
        end

        return _second_difference_frame(results)
    else
        # For >2 modifier levels, use second_differences_pairwise() for each contrast
        all_results = DataFrame[]

        for contrast in contrasts
            try
                df_contrast = second_differences_pairwise(
                    ame_result, variable, modifier, vcov;
                    contrast = contrast,
                    modifier_type = modifier_type
                )
                push!(all_results, df_contrast)
            catch e
                if occursin("Need at least 2 modifier levels", string(e))
                    @warn "Skipping contrast '$contrast' for variable $variable: insufficient modifier levels"
                else
                    rethrow(e)
                end
            end
        end

        if isempty(all_results)
            error("No valid contrasts found for variable $variable with modifier $modifier")
        end

        # Combine all contrast results
        df = vcat(all_results..., cols=:union)

        return df
    end
end

"""
    second_differences_pairwise(ame_result, variable::Symbol, modifier::Symbol, vcov::Matrix{Float64}; contrast="dy/dx", modifier_type=:auto)

Calculate second differences (interaction effects) for all pairwise combinations of modifier levels.

Handles binary, categorical (>2 levels), and continuous modifiers. For categorical modifiers with K levels,
computes all K(K-1)/2 pairwise contrasts. For continuous modifiers, scales differences by modifier distance.

# Arguments
- `ame_result::EffectsResult`: Result from population_margins with scenarios over the modifier
- `variable::Symbol`: The focal variable to calculate second differences for
- `modifier::Symbol`: The moderating variable
- `vcov::Matrix{Float64}`: Parameter covariance matrix from the model
- `contrast::String`: The contrast type to filter on (default "dy/dx" for continuous vars; "derivative" is accepted as an alias)
- `modifier_type::Symbol`: Modifier type (:binary, :categorical, :continuous, :auto for auto-detection)

# Returns
DataFrame with one row per pairwise contrast and the unified second-difference columns
(`Margins.SECOND_DIFFERENCE_COLUMNS`), the same for every modifier type:
- `variable`, `contrast`, `modifier`: focal variable, its contrast, the modifier
- `modifier_type`: detected or specified type (`:binary`, `:categorical`, `:continuous`)
- `modifier_level1`, `modifier_level2`: modifier levels compared
- `ame_at_level1`, `ame_at_level2`: focal effect at each level
- `second_diff`: second difference (binary/categorical) or slope (continuous)
- `se`: standard error via delta method
- `statistic`, `df`: `second_diff / se` and its degrees of freedom (t when finite, z when `Inf`;
  inherited from `ame_result`)
- `p_value`: p-value for H₀: second difference = 0
- `significant`: `p_value < 0.05`
- `gradient`: gradient vector (∂second_diff/∂β) for custom variance calculations

# Modifier Types
- **Binary** (2 levels): Single contrast
- **Categorical** (>2 levels): All pairwise comparisons (K choose 2)
- **Continuous**: Pairwise slopes scaled by modifier difference: (AME₂ - AME₁)/(m₂ - m₁)

# Examples
```julia
using Margins, GLM, DataFrames

# Binary modifier (2 levels) - single contrast
ames = population_margins(model, data; scenarios=(treated=[0, 1],), type=:effects)
sd = second_differences_pairwise(ames, :age, :treated, vcov(model))
# Returns 1 row

# Categorical modifier (3 levels) - all pairwise
ames = population_margins(model, data; scenarios=(education=[:hs, :college, :grad],), type=:effects)
sd = second_differences_pairwise(ames, :income, :education, vcov(model))
# Returns 3 rows: college-hs, grad-hs, grad-college

# Continuous modifier - slopes per unit change
ames = population_margins(model, data; scenarios=(age=[30, 40, 50],), type=:effects)
sd = second_differences_pairwise(ames, :education, :age, vcov(model); modifier_type=:continuous)
# Returns 3 rows with slopes: (AME₄₀-AME₃₀)/10, (AME₅₀-AME₃₀)/20, (AME₅₀-AME₄₀)/10
```

# Statistical Notes
- Uses proper delta method: SE = sqrt((g₂-g₁)' Σ (g₂-g₁)) for categorical
- For continuous modifiers: SE(slope) = SE(AME₂-AME₁) / |m₂-m₁|
- All pairwise comparisons are independent tests (no multiple testing correction applied)
- For categorical modifiers with many levels, consider Bonferroni or other adjustments
"""
function second_differences_pairwise(
    ame_result::EffectsResult,
    variable::Symbol,
    modifier::Symbol,
    vcov::Matrix{Float64};
    contrast::String="dy/dx",
    modifier_type::Symbol=:auto
)
    # Check that modifier exists in profile_values
    if isnothing(ame_result.profile_values) || !haskey(ame_result.profile_values, modifier)
        error("AME result does not have modifier $modifier in profile_values")
    end

    # Validate dimensions
    n_params_vcov = size(vcov, 1)
    n_params_gradients = size(ame_result.gradients, 2)
    if n_params_vcov != n_params_gradients
        error("Dimension mismatch: vcov has $n_params_vcov parameters but gradients has $n_params_gradients")
    end
    if size(vcov, 1) != size(vcov, 2)
        error("vcov must be square matrix, got size $(size(vcov))")
    end

    # Find indices for this variable and contrast at each modifier level
    var_str = string(variable)
    indices = Int[]
    modifier_levels = []

    for i in 1:length(ame_result.estimates)
        if ame_result.variables[i] == var_str && ame_result.terms[i] == _contrast_label(contrast)
            push!(indices, i)
            push!(modifier_levels, ame_result.profile_values[modifier][i])
        end
    end

    if length(indices) < 2
        error("Need at least 2 modifier levels, found $(length(indices)) for variable $variable with contrast $contrast")
    end

    # Auto-detect modifier type if requested
    unique_levels = unique(modifier_levels)
    n_levels = length(unique_levels)

    if modifier_type == :auto
        if n_levels == 2
            modifier_type = :binary
        else
            # Check if values are numeric and could be continuous
            if all(x -> x isa Number, unique_levels)
                modifier_type = :continuous
            else
                modifier_type = :categorical
            end
        end
    end

    # Generate all pairwise combinations (i < j to avoid duplicates)
    level_pairs = [(i, j) for i in 1:length(indices), j in 1:length(indices) if i < j]

    if isempty(level_pairs)
        error("No pairwise combinations found for modifier $modifier")
    end

    # Compute second difference for each pair
    results = []

    for (i, j) in level_pairs
        idx1, idx2 = indices[i], indices[j]
        level1, level2 = modifier_levels[i], modifier_levels[j]

        # Ensure consistent ordering (level1 < level2)
        if modifier_type == :continuous && level2 < level1
            idx1, idx2 = idx2, idx1
            level1, level2 = level2, level1
        end

        # Get estimates
        est_level1 = ame_result.estimates[idx1]
        est_level2 = ame_result.estimates[idx2]

        # Get gradients
        g1 = ame_result.gradients[idx1, :]
        g2 = ame_result.gradients[idx2, :]
        g_diff = g2 .- g1

        # Delta method variance
        variance = dot(g_diff, vcov, g_diff)
        se_raw = sqrt(max(0.0, variance))

        # For continuous modifiers, compute slope and scale SE
        if modifier_type == :continuous
            if !isa(level1, Number) || !isa(level2, Number)
                error("Continuous modifier requires numeric levels, got $(typeof(level1)) and $(typeof(level2))")
            end

            modifier_diff = abs(level2 - level1)
            if modifier_diff ≈ 0.0
                @warn "Modifier levels are nearly identical ($level1 ≈ $level2), skipping"
                continue
            end

            # Slope: change in AME per unit change in modifier
            second_diff = (est_level2 - est_level1) / modifier_diff
            se_diff = se_raw / modifier_diff
            # Scale gradient to match scaled second_diff
            g_diff_scaled = g_diff ./ modifier_diff
        else
            # Categorical/binary: simple difference
            second_diff = est_level2 - est_level1
            se_diff = se_raw
            g_diff_scaled = g_diff
        end

        push!(results, _second_difference_row(;
            variable, contrast, modifier, modifier_type,
            level1, level2, est1 = est_level1, est2 = est_level2,
            second_diff, se = se_diff, gradient = g_diff_scaled, dof = _result_df(ame_result),
        ))
    end

    if isempty(results)
        error("No valid pairwise contrasts computed for variable $variable with modifier $modifier")
    end

    return _second_difference_frame(results)
end
