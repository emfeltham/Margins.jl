# inference/mixture_utilities.jl
# Mixture creation utilities for reference grid construction

"""
    validate_mixture_weights(weights; atol=1e-10)

Validate that mixture weights are non-negative and sum to 1.0.

# Arguments
- `weights::AbstractVector{<:Real}`: Vector of mixture weights
- `atol::Real=1e-10`: Absolute tolerance for sum check

# Throws
- `ArgumentError`: If weights are negative or don't sum to 1.0

# Example
```julia
validate_mixture_weights([0.3, 0.7])           # ✓ Valid
validate_mixture_weights([0.3, 0.8])           # ✗ Doesn't sum to 1.0
validate_mixture_weights([0.3, -0.7, 1.4])     # ✗ Negative weight
```
"""
function validate_mixture_weights(weights::AbstractVector{<:Real}; atol::Real=1e-10)
    if any(w < 0 for w in weights)
        throw(ArgumentError("Mixture weights must be non-negative: $weights"))
    end

    weight_sum = sum(weights)
    if !isapprox(weight_sum, 1.0, atol=atol)
        throw(ArgumentError("Mixture weights must sum to 1.0 (±$atol): got $weights (sum = $weight_sum)"))
    end
end

"""
    validate_mixture_levels(levels::AbstractVector)

Validate that mixture levels are unique and non-empty.

# Arguments
- `levels`: Vector of level identifiers

# Throws
- `ArgumentError`: If levels are invalid

# Example
```julia
validate_mixture_levels(["A", "B", "C"])    # ✓ Valid
validate_mixture_levels(["A", "A", "B"])    # ✗ Duplicate levels
validate_mixture_levels(String[])           # ✗ Empty levels
```
"""
function validate_mixture_levels(levels::AbstractVector)
    if isempty(levels)
        throw(ArgumentError("Mixture levels cannot be empty"))
    end

    if length(unique(levels)) != length(levels)
        duplicates = [level for level in unique(levels) if count(==(level), levels) > 1]
        throw(ArgumentError("Mixture levels must be unique. Duplicates found: $duplicates"))
    end
end

"""
    create_mixture_column(mixture_spec, n_rows::Int)

Create a column of identical mixture specifications for use in reference grids.

# Arguments
- `mixture_spec`: A mixture object with `levels` and `weights` properties
- `n_rows`: Number of rows to create

# Returns
Vector of mixture objects, all identical to the input specification

# Example
```julia
mixture = mix("A" => 0.3, "B" => 0.7)
col = create_mixture_column(mixture, 1000)  # 1000 rows of identical mixture
```

This is more efficient than `fill(mixture, n_rows)` for large datasets as it
avoids potential copying issues with complex mixture objects.
"""
function create_mixture_column(mixture_spec, n_rows::Int)
    if n_rows < 0
        throw(ArgumentError("Number of rows must be non-negative, got $n_rows"))
    end
    return fill(mixture_spec, n_rows)
end

"""
    expand_mixture_grid(base_data, mixture_specs::Dict{Symbol, Any})

Create all combinations of base data with mixture specifications for systematic
marginal effects computation.

# Arguments
- `base_data`: Base data as NamedTuple or DataFrame-compatible structure
- `mixture_specs`: Dictionary mapping column names to mixture specifications

# Returns
Vector of NamedTuple data structures, each representing one combination

# Example
```julia
base_data = (x = [1.0, 2.0], y = [0.1, 0.2])
mixtures = Dict(
    :group => mix("A" => 0.5, "B" => 0.5),
    :treatment => mix("Control" => 0.3, "Treatment" => 0.7)
)

expanded = expand_mixture_grid(base_data, mixtures)
# Returns data with mixture columns added to each row
```

# Use Cases
- Reference grid creation for marginal effects
- Counterfactual analysis with multiple mixture variables
- Systematic sensitivity analysis across mixture specifications
"""
function expand_mixture_grid(base_data, mixture_specs::Dict{Symbol, <:Any})
    if isempty(mixture_specs)
        return [base_data]  # No mixtures to expand
    end

    # Ensure base_data is a NamedTuple
    if !(base_data isa NamedTuple)
        throw(ArgumentError("base_data must be a NamedTuple. Use Tables.columntable() to convert DataFrames."))
    end

    n_rows = length(first(values(base_data)))

    # Create expanded data with mixture columns
    expanded_data = Dict{Symbol, Any}()

    # Copy all base columns
    for (col_name, col_data) in pairs(base_data)
        expanded_data[col_name] = col_data
    end

    # Add mixture columns
    for (col_name, mixture_spec) in mixture_specs
        if haskey(expanded_data, col_name)
            @warn "Overriding existing column $col_name with mixture specification"
        end
        expanded_data[col_name] = create_mixture_column(mixture_spec, n_rows)
    end

    return [NamedTuple(expanded_data)]
end

"""
    create_balanced_mixture(levels::AbstractVector)

Create a balanced (equal weight) mixture from a vector of levels.

# Arguments
- `levels`: Vector of level identifiers

# Returns
Dictionary suitable for creating mixture objects: `Dict(level => weight, ...)`

# Example
```julia
balanced = create_balanced_mixture(["A", "B", "C"])
# Returns: Dict("A" => 0.333..., "B" => 0.333..., "C" => 0.333...)

# Use with mixture constructor:
mixture = mix(balanced...)  # Splat the dictionary
```

This is useful for creating reference mixtures where all levels should be
equally weighted for marginal effects computation.
"""
function create_balanced_mixture(levels::AbstractVector)
    if isempty(levels)
        throw(ArgumentError("Cannot create balanced mixture from empty levels"))
    end

    # Use FormulaCompiler's validation function
    validate_mixture_levels(levels)

    n_levels = length(levels)
    weight = 1.0 / n_levels

    return Dict(string(level) => weight for level in levels)
end
