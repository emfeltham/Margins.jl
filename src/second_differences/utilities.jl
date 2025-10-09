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
- `contrast::String="derivative"`: Focal variable contrast to analyze
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
    contrast::String="derivative",
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
