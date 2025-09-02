# profile/refgrids.jl - Reference grid builders for profile_margins

using Statistics

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

Compute typical values for all variables with optimized caching for performance.
Uses a lightweight cache key to avoid O(n) hashing of data columns.

This ensures consistent typical value computation across all reference grid methods
and eliminates code duplication. Uses caching to avoid recomputing typical values
for the same data. Converts CategoricalMixture objects to scenario values.
"""
function _get_typical_values_dict(data_nt::NamedTuple)
    # Create lightweight cache key based on data structure only (not content)
    # This avoids O(n) hash computation on large columns
    cache_key = hash((keys(data_nt), [length(col) for col in values(data_nt)], [eltype(col) for col in values(data_nt)]))
    
    # Check cache first
    if haskey(TYPICAL_VALUES_CACHE, cache_key)
        return TYPICAL_VALUES_CACHE[cache_key]
    end
    
    # Compute typical values with optimized implementations
    typical_values = Dict{Symbol, Any}()
    for (name, col) in pairs(data_nt)
        typical_val = _get_typical_value_optimized(col)
        
        # Store CategoricalMixture objects for later processing into override vectors
        if typical_val isa CategoricalMixture
            typical_values[name] = typical_val
        elseif eltype(col) <: Bool && typical_val isa Float64
            # For Bool columns, typical_val is already P(true) - use directly
            typical_values[name] = typical_val
        else
            typical_values[name] = typical_val
        end
    end
    
    # Cache result for future use
    TYPICAL_VALUES_CACHE[cache_key] = typical_values
    
    return typical_values
end

"""
    _get_typical_value_optimized(col) -> Any

Optimized version of _get_typical_value that avoids unnecessary O(n) computations
by using simplified heuristics for typical values in profile scenarios.
"""
function _get_typical_value_optimized(col)
    if _is_continuous_variable(col)
        # For continuous variables, use sample mean (O(n) unavoidable for statistical correctness)
        return mean(col)
    elseif col isa CategoricalArray
        # For CategoricalArray, use simplified frequency mixture
        return _create_frequency_mixture_optimized(col)
    elseif eltype(col) <: Bool  
        # For Bool, use mean (O(n) but necessary for statistical correctness)
        return mean(col)
    elseif eltype(col) <: AbstractString
        # For strings, use first value as heuristic (O(1))
        return first(col)
    else
        # Fallback to first value (O(1))
        return first(col)
    end
end

"""
    _create_frequency_mixture_optimized(col) -> CategoricalMixture

Optimized frequency mixture computation that may use sampling for very large datasets.
"""
function _create_frequency_mixture_optimized(col)
    n = length(col)
    
    # For smaller datasets, compute exact frequencies
    if n <= 10000
        return _create_frequency_mixture_exact(col)
    else
        # For larger datasets, use sampling to estimate frequencies (O(1) sample size)
        return _create_frequency_mixture_sampled(col)
    end
end

"""
    _create_frequency_mixture_exact(col) -> CategoricalMixture

Exact frequency computation (original implementation).
"""
function _create_frequency_mixture_exact(col)
    # Special handling for Bool: return probability of true
    if eltype(col) <: Bool
        p_true = mean(col)  # Proportion of true values
        return p_true
    end
    
    # General categorical handling
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
    _create_frequency_mixture_sampled(col) -> CategoricalMixture

Sample-based frequency estimation for large datasets to achieve O(1) scaling.
Uses a fixed sample size for frequency estimation with deterministic sampling.
"""
function _create_frequency_mixture_sampled(col)
    # Special handling for Bool: use deterministic sampling
    if eltype(col) <: Bool
        # Sample fixed number of values for O(1) scaling using deterministic indices
        sample_size = min(1000, length(col))
        step = max(1, length(col) ÷ sample_size)
        sample_values = [col[i] for i in 1:step:length(col)]
        p_true = mean(sample_values)
        return p_true
    end
    
    # General categorical handling with deterministic sampling
    sample_size = min(1000, length(col))
    step = max(1, length(col) ÷ sample_size)
    
    level_counts = Dict()
    sampled_count = 0
    for i in 1:step:length(col)
        value = col[i]
        level_counts[value] = get(level_counts, value, 0) + 1
        sampled_count += 1
    end
    
    # Convert to levels and weights
    levels = collect(keys(level_counts))
    weights = [level_counts[level] / sampled_count for level in levels]
    
    return CategoricalMixture(levels, weights)
end