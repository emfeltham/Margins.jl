# profile/refgrids.jl - Reference grid builders for profile_margins

# Cache for typical values computation (performance optimization)
const TYPICAL_VALUES_CACHE = Dict{UInt64, Dict{Symbol, Any}}()

"""
    _build_reference_grid(at_spec, data_nt) -> DataFrame

Build unified reference grid from at specification with efficient single-pass approach.

Supports various input formats:
- `:means` - Use sample means for continuous, first levels for categorical
- `Dict` - Cartesian product of specified values, typical values for others  
- `Vector{Dict}` - Explicit list of profiles
- `DataFrame` - Use directly (most efficient)

The unified approach ensures consistent typical value computation and efficient memory usage.

# Arguments
- `at_spec`: Profile specification in any supported format
- `data_nt::NamedTuple`: Original data in columntable format

# Returns
- `DataFrame`: Reference grid ready for FormulaCompiler processing

# Examples
```julia
# At sample means (most common case)
_build_reference_grid(:means, data_nt)

# Cartesian product specification
_build_reference_grid(Dict(:x1 => [0, 1], :x2 => [2, 3]), data_nt) 

# Explicit profiles for complex scenarios
_build_reference_grid([Dict(:x1 => 0, :x2 => 2), Dict(:x1 => 1, :x2 => 3)], data_nt)

# Pre-built DataFrame (zero additional processing)
_build_reference_grid(existing_grid, data_nt)
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
        return at_spec  # Use directly (most efficient path)
    else
        throw(ArgumentError("Invalid at specification: $(typeof(at_spec)). Must be :means, Dict, Vector{Dict}, or DataFrame"))
    end
end

"""
    _build_means_refgrid(data_nt) -> DataFrame

Build reference grid with sample means for continuous variables and first levels for categorical.
Uses unified typical value computation for consistency.
"""
function _build_means_refgrid(data_nt::NamedTuple)
    # Use unified typical value computation
    typical_values = _get_typical_values_dict(data_nt)
    return DataFrame([typical_values])
end

"""
    _build_cartesian_refgrid(at_spec::Dict, data_nt) -> DataFrame

Build Cartesian product reference grid from Dict specification with optimized memory usage.
Variables not specified use typical values (means/first levels).
Uses unified approach for consistent typical value computation.
"""
function _build_cartesian_refgrid(at_spec::Dict, data_nt::NamedTuple)
    # Use unified typical value computation for base values
    base_dict = _get_typical_values_dict(data_nt)
    
    # Extract specified variables and their values (optimized)
    var_names = collect(keys(at_spec))
    var_values = Vector{Vector{Any}}(undef, length(var_names))
    
    for (i, var) in enumerate(var_names)
        vals = at_spec[var]
        var_values[i] = vals isa Vector ? vals : [vals]  # Convert single value to vector
    end
    
    # Pre-calculate total number of combinations for efficient allocation
    n_combinations = prod(length(vals) for vals in var_values)
    
    # Pre-allocate result vector for better performance
    grid_rows = Vector{Dict{Symbol,Any}}(undef, n_combinations)
    
    # Generate Cartesian product with optimized allocation
    combo_idx = 1
    for combo in Iterators.product(var_values...)
        row = copy(base_dict)  # Start with base values
        for (i, var) in enumerate(var_names)
            row[var] = combo[i]
        end
        grid_rows[combo_idx] = row
        combo_idx += 1
    end
    
    return DataFrame(grid_rows)
end

"""
    _build_explicit_refgrid(at_spec::Vector, data_nt) -> DataFrame

Build reference grid from explicit vector of profiles (Dicts or NamedTuples).
Missing variables are filled with typical values using unified approach.
"""
function _build_explicit_refgrid(at_spec::Vector, data_nt::NamedTuple)
    # Use unified typical value computation for base values
    base_dict = _get_typical_values_dict(data_nt)
    
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

"""
    _get_typical_values_dict(data_nt) -> Dict{Symbol, Any}

Compute typical values for all variables with caching for performance.
Delegates to the main _get_typical_value() function for consistency.

This ensures consistent typical value computation across all reference grid methods
and eliminates code duplication. Uses caching to avoid recomputing typical values
for the same data.
"""
function _get_typical_values_dict(data_nt::NamedTuple)
    # Create cache key based on data structure and content hash
    cache_key = hash((keys(data_nt), [hash(col) for col in values(data_nt)]))
    
    # Check cache first
    if haskey(TYPICAL_VALUES_CACHE, cache_key)
        return TYPICAL_VALUES_CACHE[cache_key]
    end
    
    # Compute typical values
    typical_values = Dict{Symbol, Any}()
    for (name, col) in pairs(data_nt)
        typical_values[name] = _get_typical_value(col)
    end
    
    # Cache result for future use
    TYPICAL_VALUES_CACHE[cache_key] = typical_values
    
    return typical_values
end