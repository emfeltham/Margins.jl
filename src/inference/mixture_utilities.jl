# inference/mixture_utilities.jl
# Mixture creation utilities for reference grid construction

# `validate_mixture_weights` and `validate_mixture_levels` are FormulaCompiler's (imported
# and re-exported in src/Margins.jl); Margins' duplicate copies were removed because they
# clashed with FormulaCompiler's exports under `using FormulaCompiler, Margins`.

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

    # FormulaCompiler's validation (errors on duplicate levels)
    validate_mixture_levels(levels)

    # Delegate the construction to FormulaCompiler, keeping Margins' Dict return value
    m = FormulaCompiler.create_balanced_mixture(levels)
    return Dict{String, Float64}(string(l) => w for (l, w) in zip(m.levels, m.weights))
end
