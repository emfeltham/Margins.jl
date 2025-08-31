# profile/refgrids.jl - Reference grid builders for profile_margins

"""
    _build_reference_grid(at_spec, data_nt) -> DataFrame

Build reference grid from at specification.

Supports various input formats:
- `:means` - Use sample means for continuous, first levels for categorical
- `Dict` - Cartesian product of specified values, means for others  
- `Vector{Dict}` - Explicit list of profiles
- `DataFrame` - Use directly

# Examples
```julia
# At sample means
_build_reference_grid(:means, data_nt)

# Cartesian product
_build_reference_grid(Dict(:x1 => [0, 1], :x2 => [2, 3]), data_nt) 

# Explicit profiles
_build_reference_grid([Dict(:x1 => 0, :x2 => 2), Dict(:x1 => 1, :x2 => 3)], data_nt)
```
"""
function _build_reference_grid(at_spec, data_nt::NamedTuple)
    if at_spec === :means
        return _build_means_refgrid(data_nt)
    elseif at_spec isa Dict
        return _build_cartesian_refgrid(at_spec, data_nt)
    elseif at_spec isa Vector
        return _build_explicit_refgrid(at_spec, data_nt)
    elseif at_spec isa DataFrame
        return at_spec  # Use directly
    else
        error("Invalid at specification: $at_spec. Must be :means, Dict, Vector{Dict}, or DataFrame")
    end
end

"""
    _build_means_refgrid(data_nt) -> DataFrame

Build reference grid with sample means for continuous variables and first levels for categorical.
"""
function _build_means_refgrid(data_nt::NamedTuple)
    row = Dict{Symbol,Any}()
    
    for (name, col) in pairs(data_nt)
        if eltype(col) <: Real && !(eltype(col) <: Bool)
            # Continuous variable: use mean
            row[name] = mean(col)
        elseif col isa CategoricalArray
            # Categorical array: use first level
            row[name] = levels(col)[1] 
        elseif eltype(col) <: Bool
            # Boolean: use false as reference
            row[name] = false
        elseif eltype(col) <: AbstractString
            # String categorical: use first unique value
            row[name] = first(unique(col))
        else
            # Fallback: use first value
            row[name] = first(col)
        end
    end
    
    return DataFrame([row])
end

"""
    _build_cartesian_refgrid(at_spec::Dict, data_nt) -> DataFrame

Build Cartesian product reference grid from Dict specification.
Variables not specified use typical values (means/first levels).
"""
function _build_cartesian_refgrid(at_spec::Dict, data_nt::NamedTuple)
    # Start with base row using means/typical values
    base_grid = _build_means_refgrid(data_nt)
    base_dict = Dict(pairs(base_grid[1, :]))
    
    # Extract specified variables and their values
    var_names = collect(keys(at_spec))
    var_values = []
    
    for var in var_names
        vals = at_spec[var]
        if vals isa Vector
            push!(var_values, vals)
        else
            push!(var_values, [vals])  # Convert single value to vector
        end
    end
    
    # Generate Cartesian product
    grid_rows = []
    for combo in Iterators.product(var_values...)
        row = copy(base_dict)
        for (i, var) in enumerate(var_names)
            row[var] = combo[i]
        end
        push!(grid_rows, row)
    end
    
    return DataFrame(grid_rows)
end

"""
    _build_explicit_refgrid(at_spec::Vector, data_nt) -> DataFrame

Build reference grid from explicit vector of profiles (Dicts or NamedTuples).
Missing variables are filled with typical values.
"""
function _build_explicit_refgrid(at_spec::Vector, data_nt::NamedTuple)
    # Get base row with typical values
    base_grid = _build_means_refgrid(data_nt)
    base_dict = Dict(pairs(base_grid[1, :]))
    
    grid_rows = []
    for profile in at_spec
        row = copy(base_dict)
        # Override with profile-specific values
        for (k, v) in pairs(profile)
            row[k] = v
        end
        push!(grid_rows, row)
    end
    
    return DataFrame(grid_rows)
end

# _get_typical_value is already defined in engine/utilities.jl