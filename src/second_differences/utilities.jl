# utilities.jl
# Shared utilities for second differences computations

"""
    _expand_profile_to_cartesian(profile::NamedTuple) -> Vector{NamedTuple}

Expand a profile NamedTuple with vector-valued entries into a vector of NamedTuples
representing the Cartesian product of all values.

# Arguments
- `profile::NamedTuple`: Profile specification where values can be scalars or vectors

# Returns
- `Vector{NamedTuple}`: Vector of profile points (Cartesian product)

# Examples
```julia
# Single scalar
_expand_profile_to_cartesian((age=40,))
# → [(age=40,)]

# Single vector
_expand_profile_to_cartesian((socio4=[false, true],))
# → [(socio4=false,), (socio4=true,)]

# Multiple vectors
_expand_profile_to_cartesian((socio4=[false, true], region=["north", "south"]))
# → [(socio4=false, region="north"),
#    (socio4=true,  region="north"),
#    (socio4=false, region="south"),
#    (socio4=true,  region="south")]
```
"""
function _expand_profile_to_cartesian(profile::NamedTuple)
    # If empty, return single empty NamedTuple
    if length(profile) == 0
        return [NamedTuple()]
    end

    # Get keys and values
    keys_vec = keys(profile)

    # Normalize all values to vectors
    value_vecs = []
    for k in keys_vec
        val = profile[k]
        if val isa AbstractVector
            push!(value_vecs, collect(val))
        else
            # Scalar - wrap in vector
            push!(value_vecs, [val])
        end
    end

    # Compute Cartesian product
    # For example: [[:a, :b], [1, 2]] → [(:a, 1), (:b, 1), (:a, 2), (:b, 2)]
    cartesian_indices = Iterators.product(value_vecs...)

    # Build NamedTuples from the product
    result = NamedTuple[]
    for combo in cartesian_indices
        nt = NamedTuple{keys_vec}(combo)
        push!(result, nt)
    end

    return result
end

"""
    second_differences_table(ame_result, variables::Vector{Symbol}, modifier::Symbol, vcov::Matrix{Float64}; kwargs...)

Calculate second differences for multiple variables and return as DataFrame.

This function is now a convenience wrapper around `second_differences()` with multiple variables.
Consider using `second_differences()` directly for more flexibility.

# Arguments
- `ame_result::EffectsResult`: Result from population_margins with scenarios over the modifier
- `variables::Vector{Symbol}`: Multiple focal variables to analyze
- `modifier::Symbol`: The moderating variable
- `vcov::Matrix{Float64}`: Parameter covariance matrix

# Keyword Arguments
- `contrast::String="dy/dx"`: Focal variable contrast to analyze
- `modifier_type::Symbol=:auto`: Modifier type (:auto, :binary, :categorical, :continuous)
- `all_contrasts::Bool=true`: Compute for all focal variable contrasts (if applicable)

# Returns
DataFrame with second differences for all variables, with `significant` column added

# Example
```julia
ames = population_margins(model, data; scenarios=(treated=[0,1],), type=:effects)
sd = second_differences_table(ames, [:age, :education, :income], :treated, vcov(model))
# → DataFrame with 3 rows + significance indicator
```
"""
function second_differences_table(
    ame_result::EffectsResult,
    variables::Vector{Symbol},
    modifier::Symbol,
    vcov::Matrix{Float64};
    contrast::String="dy/dx",
    modifier_type::Symbol=:auto,
    all_contrasts::Bool=true
)
    # Use the main second_differences function with multiple variables
    df = second_differences(
        ame_result, variables, modifier, vcov;
        contrast=contrast,
        modifier_type=modifier_type,
        all_contrasts=all_contrasts
    )

    # Add significance indicator if not already present
    if !hasproperty(df, :significant)
        df.significant = df.p_value .< 0.05
    end

    return df
end

# Continuous effects are labelled "dy/dx" (v2.3.0; the scenarios/groups path used to say
# "derivative"). "derivative" is still accepted wherever a contrast label is passed in.
_contrast_label(c::AbstractString) = c == "derivative" ? "dy/dx" : String(c)

# =============================================================================
# Unified output schema
# =============================================================================

"""
Columns of every second-difference DataFrame, in order (`second_differences`,
`second_differences_pairwise`, `second_differences_all_contrasts`,
`second_differences_table`; `second_difference` returns one such row as a NamedTuple):

| column            | meaning                                                                  |
|:------------------|:-------------------------------------------------------------------------|
| `variable`        | focal variable (`Symbol`)                                                |
| `contrast`        | focal contrast (`"dy/dx"` for a continuous variable, `"B - A"` levels)   |
| `modifier`        | modifier variable (`Symbol`)                                             |
| `modifier_type`   | `:binary`, `:categorical` or `:continuous`                               |
| `modifier_level1` | first modifier value compared                                            |
| `modifier_level2` | second modifier value compared                                           |
| `ame_at_level1`   | focal effect at `modifier_level1`                                        |
| `ame_at_level2`   | focal effect at `modifier_level2`                                        |
| `second_diff`     | `ame_at_level2 - ame_at_level1`; for `:continuous`, divided by the modifier distance (a slope) |
| `se`              | delta-method standard error                                              |
| `statistic`       | `second_diff / se`                                                       |
| `df`              | inference degrees of freedom (t when finite, z when `Inf`)               |
| `p_value`         | two-sided p-value for H₀: `second_diff = 0`                              |
| `significant`     | `p_value < 0.05`                                                         |
| `gradient`        | ∂`second_diff`/∂β, for custom variance calculations                      |

`second_differences_at` returns the same columns (with `modifier_type = :continuous`,
`modifier_level1/2 = eval_point ∓ delta_used`, and `second_diff` the local derivative
∂AME/∂modifier), followed by `eval_point`, `delta_used` and one column per `profile`
variable.

`df` is inherited from the effects result (`ame_result.metadata[:df]`): a second difference is
a linear combination of the same estimates, so it follows the same rule — the residual df for a
linear model with the model-based covariance, `Inf` (z) otherwise.
"""
const SECOND_DIFFERENCE_COLUMNS = (
    :variable, :contrast, :modifier, :modifier_type, :modifier_level1, :modifier_level2,
    :ame_at_level1, :ame_at_level2, :second_diff, :se, :statistic, :df, :p_value,
    :significant, :gradient,
)

# One row of the unified second-difference schema. A zero standard error (no interaction
# in the model) gives statistic 0 / p 1 for a zero estimate and Inf / 0 otherwise.
function _second_difference_row(;
    variable::Symbol, contrast::AbstractString, modifier::Symbol, modifier_type::Symbol,
    level1, level2, est1::Real, est2::Real, second_diff::Real, se::Real,
    gradient::AbstractVector, dof::Real,
)
    if se ≈ 0.0
        statistic = second_diff ≈ 0.0 ? 0.0 : Inf
        p_value = second_diff ≈ 0.0 ? 1.0 : 0.0
    else
        statistic = second_diff / se
        p_value = compute_p_value(Float64(statistic), dof)
    end
    return (
        variable = variable,
        contrast = _contrast_label(contrast),
        modifier = modifier,
        modifier_type = modifier_type,
        modifier_level1 = level1,
        modifier_level2 = level2,
        ame_at_level1 = Float64(est1),
        ame_at_level2 = Float64(est2),
        second_diff = Float64(second_diff),
        se = Float64(se),
        statistic = Float64(statistic),
        df = Float64(dof),
        p_value = Float64(p_value),
        significant = p_value < 0.05,
        gradient = gradient,
    )
end

# DataFrame with exactly the unified columns (plus any trailing extras), in schema order
function _second_difference_frame(rows)
    df = DataFrame(rows)
    extras = [c for c in propertynames(df) if !(c in SECOND_DIFFERENCE_COLUMNS)]
    return select(df, [SECOND_DIFFERENCE_COLUMNS..., extras...])
end
