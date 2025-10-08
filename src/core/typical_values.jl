# Core typical value computation functions
# Simplified architecture with 3 functions for statistical correctness

"""
    _create_frequency_mixture(col) -> Union{Float64, CategoricalMixture}

Create frequency-weighted mixture for categorical variables.
Generates proportional representation based on empirical frequencies in the data.

# Arguments
- `col`: Data column (CategoricalArray, Vector{Bool}, or other categorical data)

# Returns
- For Boolean: `Float64` representing P(true)
- For Categorical: `CategoricalMixture` with levels and empirical weights

# Frequency Calculation
Computes empirical frequencies for each unique value:
- **Boolean variables**: P(true) = count(true) / total_count
- **Categorical variables**: P(level) = count(level) / total_count for each level

# Mixture Construction
For categorical variables, creates CategoricalMixture with:
- **Levels**: All unique values in the data
- **Weights**: Empirical proportions summing to 1.0

# Usage in Reference Grids
When constructing reference grids for profile margins, unspecified categorical
variables use frequency mixtures to represent typical compositions rather than
arbitrary single values.

# Statistical Interpretation
The mixture represents the marginal distribution of the categorical variable
in the dataset, enabling profile margins that account for realistic categorical
compositions while holding other variables fixed at profile values.

# Performance
- Single pass through data for frequency counting
- Efficient dictionary-based counting
- Minimal memory allocation for level storage

# Example
```julia
# Boolean variable
flags = [true, false, true, true, false]  # 60% true
mixture = _create_frequency_mixture(flags)  # Returns 0.6

# Categorical variable
education = categorical(["HS", "College", "HS", "Graduate", "College"])
mixture = _create_frequency_mixture(education)
# Returns CategoricalMixture with levels ["HS", "College", "Graduate"]
# and weights [0.4, 0.4, 0.2]
```
"""
function _create_frequency_mixture(col)
    if eltype(col) <: Bool
        # For Boolean variables, return probability of true
        return mean(col)
    end

    # For categorical variables, compute frequency distribution
    level_counts = Dict()
    total_count = length(col)

    for value in col
        level_counts[value] = get(level_counts, value, 0) + 1
    end

    # Convert to levels and weights
    levels = collect(keys(level_counts))
    weights = [level_counts[level] / total_count for level in levels]

    return CategoricalMixture(levels, weights)
end

"""
    _get_typical_value(col, typical) -> Any

Get typical value for a variable based on its type and distribution.
Always uses frequency mixtures for categorical variables (never mode).

# Arguments
- `col`: Data column of any supported type

# Returns
- Continuous variables: mean(col)
- Categorical variables: CategoricalMixture with empirical frequencies
- Boolean variables: P(true) as Float64
- String variables: CategoricalMixture with empirical frequencies

# Statistical Correctness
Never use mode or arbitrary defaults for categorical variables.
Always uses frequency mixtures to maintain population representativeness.
"""
function _get_typical_value(col, typical)
    if _is_continuous_variable(col)
        return typical(col)
    elseif col isa CategoricalArray
        return _create_frequency_mixture(col)
    elseif eltype(col) <: Bool
        return _create_frequency_mixture(col)  # Returns P(true) as Float64
    elseif eltype(col) <: AbstractString
        return _create_frequency_mixture(col)  # Returns CategoricalMixture
    else
        throw(MarginsError("Unsupported data type $(eltype(col)) for variable. " *
                          "Statistical correctness cannot be guaranteed for unknown data types. " *
                          "Supported types: numeric (Int64, Float64), Bool, CategoricalArray, AbstractString."))
    end
end

"""
    get_typical_values(data; typical=mean) -> Dict{Symbol, Any}

Compute typical values for all variables in a dataset.
Provides simple workflow for reference grid construction.

# Arguments
- `data`: DataFrame or NamedTuple containing the data
- `typical`: Function to compute typical values for continuous variables (default: mean)

# Returns
- `Dict{Symbol, Any}`: Mapping of variable names to typical values
  - Continuous variables: result of `typical` function (mean, median, etc.)
  - Categorical variables: CategoricalMixture with empirical frequencies
  - Boolean variables: P(true) as Float64

# Usage
```julia
# Get typical values once
typical_values = get_typical_values(data)

# Use in reference grid construction
ref_grid = build_reference_grid(typical_values, user_specifications)
result = profile_margins(model, data, ref_grid)
```

# Statistical Correctness
All categorical variables use frequency mixtures based on actual population composition.
No arbitrary defaults or mode-based selection.
"""
function get_typical_values(data; typical = mean)
    data_nt = data isa NamedTuple ? data : Tables.columntable(data)
    typical_values = Dict{Symbol, Any}()

    for (name, col) in pairs(data_nt)
        typical_values[name] = _get_typical_value(col, typical)
    end

    return typical_values
end
