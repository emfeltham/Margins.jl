# profile/refgrids.jl - Reference grid builders for profile_margins

using Statistics
using FormulaCompiler

"""
    _filter_data_to_model_variables(data_nt::NamedTuple, model) -> NamedTuple

Filter data to only include variables that are actually used in the model.
This is the cleanest approach to prevent non-model variables from leaking into computations.

# Arguments
- `data_nt::NamedTuple`: Full data in columntable format
- `model`: Statistical model (GLM, MixedModel, etc.)

# Returns
- `NamedTuple`: Filtered data containing only model variables

# Notes
- Works with GLM.jl models and MixedModels.jl models
- Excludes outcome variables (not needed for margin computation)
- Could be extended for mixed models to handle fixed vs random effects
"""
function _filter_data_to_model_variables(data_nt::NamedTuple, model)
    # Get model variables using FormulaCompiler
    compiled = FormulaCompiler.compile_formula(model, data_nt)
    
    # Extract all variables used in model operations
    model_vars = Set{Symbol}()
    for op in compiled.ops
        if op isa FormulaCompiler.LoadOp
            Col = typeof(op).parameters[1]
            push!(model_vars, Col)
        elseif op isa FormulaCompiler.ContrastOp
            Col = typeof(op).parameters[1]
            push!(model_vars, Col)
        end
    end
    
    # Filter data to only include predictor variables (exclude response)
    filtered_data = Dict{Symbol, Any}()
    for var in model_vars
        if haskey(data_nt, var)
            filtered_data[var] = getproperty(data_nt, var)
        end
    end
    
    return NamedTuple(filtered_data)
end

# Cache for typical values computation (performance optimization)
const TYPICAL_VALUES_CACHE = Dict{UInt64, Dict{Symbol, Any}}()

"""
    _parse_numlist(str::String) -> Vector{Float64}

Parse numlist notation like "-2(2)2" into vector [-2, 0, 2].
Format: start(step)end where step is the increment.
"""
function _parse_numlist(str::String)
    # Match pattern like "-2(2)2" or "1(0.5)3"
    m = match(r"^(-?\d*\.?\d+)\((-?\d*\.?\d+)\)(-?\d*\.?\d+)$", str)
    if m === nothing
        throw(ArgumentError("Invalid numlist format: $str. Expected format: start(step)end"))
    end
    
    start_val = parse(Float64, m.captures[1])
    step_val = parse(Float64, m.captures[2])
    end_val = parse(Float64, m.captures[3])
    
    # Generate sequence
    if step_val == 0
        throw(ArgumentError("Step size cannot be zero in numlist: $str"))
    end
    
    # Handle both positive and negative steps
    result = Float64[]
    current = start_val
    if step_val > 0
        while current <= end_val + 1e-10  # Small tolerance for floating point
            push!(result, current)
            current += step_val
        end
    else
        while current >= end_val - 1e-10  # Small tolerance for floating point
            push!(result, current)
            current += step_val
        end
    end
    
    return result
end

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
function _build_reference_grid(at_spec, data_nt::NamedTuple, model, typical)
    if isnothing(at_spec)
        return _build_means_refgrid(data_nt, model, typical)
    elseif at_spec isa Dict
        # Validate that all variables in at_spec are in the model (via filtered data)
        for var in keys(at_spec)
            if !haskey(data_nt, var)
                throw(ArgumentError("Variable :$var specified in 'at' is not in the model. Model variables: $(sort(collect(keys(data_nt))))"))
            end
        end
        return _build_cartesian_refgrid(at_spec, data_nt, typical)
    elseif at_spec isa Vector
        return _build_explicit_refgrid(at_spec, data_nt, typical)
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
function _build_means_refgrid(data_nt::NamedTuple, model, typical)
    # Data is already filtered to model variables, so we can use it directly
    typical_values = _get_typical_values_dict(data_nt, typical)
    
    # Convert to DataFrame by creating columns with proper types
    cols = Dict{Symbol, Vector}()
    for (k, v) in typical_values
        # Preserve type information by creating typed vectors
        if v isa Real && !(v isa Bool)
            cols[k] = Float64[v]
        elseif v isa Bool  
            cols[k] = Bool[v]
        else
            cols[k] = [v]  # Generic for strings, categoricals, etc.
        end
    end
    return DataFrame(cols)
end

"""
    _build_cartesian_refgrid(at_spec::Dict, data_nt) -> DataFrame

Build Cartesian product reference grid from Dict specification with optimized memory usage.
Variables not specified use typical values (means/first levels).
Uses unified approach for consistent typical value computation.
"""
function _build_cartesian_refgrid(at_spec::Dict, data_nt::NamedTuple, typical)
    # Use unified typical value computation for base values
    base_dict = _get_typical_values_dict(data_nt, typical)
    
    # Extract specified variables and their values (optimized)
    var_names = collect(keys(at_spec))
    var_values = Vector{Vector{Any}}(undef, length(var_names))
    
    for (i, var) in enumerate(var_names)
        vals = at_spec[var]
        # Handle numlist parsing for strings like "-2(2)2" -> [-2, 0, 2]
        if vals isa String && occursin(r"^-?\d+\(\d+\)-?\d+$", vals)
            vals = _parse_numlist(vals)
        end
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
    
    # Convert vector of dicts to DataFrame using columntable approach
    if isempty(grid_rows)
        return DataFrame()
    end
    # Convert to column table format for efficient DataFrame construction, preserving types
    cols = Dict{Symbol, Vector}()
    for key in keys(first(grid_rows))
        # Get all values for this column
        values = [row[key] for row in grid_rows]
        # Create vector with proper type based on first value type
        first_val = first(values)
        if all(v -> typeof(v) == typeof(first_val), values)
            # All same type - use specific type
            cols[key] = typeof(first_val)[values...]
        else
            # Mixed types - fall back to Any
            cols[key] = Vector{Any}(values)
        end
    end
    return DataFrame(cols)
end

"""
    _build_explicit_refgrid(at_spec::Vector, data_nt) -> DataFrame

Build reference grid from explicit vector of profiles (Dicts or NamedTuples).
Missing variables are filled with typical values using unified approach.
"""
function _build_explicit_refgrid(at_spec::Vector, data_nt::NamedTuple, typical)
    # Use unified typical value computation for base values
    base_dict = _get_typical_values_dict(data_nt, typical)
    
    grid_rows = []
    for profile in at_spec
        row = copy(base_dict)
        # Override with profile-specific values
        for (k, v) in pairs(profile)
            row[k] = v
        end
        push!(grid_rows, row)
    end
    
    # Convert vector of dicts to DataFrame using columntable approach
    if isempty(grid_rows)
        return DataFrame()
    end
    # Convert to column table format for efficient DataFrame construction, preserving types
    cols = Dict{Symbol, Vector}()
    for key in keys(first(grid_rows))
        # Get all values for this column
        values = [row[key] for row in grid_rows]
        # Create vector with proper type based on first value type
        first_val = first(values)
        if all(v -> typeof(v) == typeof(first_val), values)
            # All same type - use specific type
            cols[key] = typeof(first_val)[values...]
        else
            # Mixed types - fall back to Any
            cols[key] = Vector{Any}(values)
        end
    end
    return DataFrame(cols)
end

"""
    _get_typical_values_dict(data_nt) -> Dict{Symbol, Any}

Compute typical values for all variables with optimized caching for performance.
Uses a lightweight cache key to avoid O(n) hashing of data columns.

This ensures consistent typical value computation across all reference grid methods
and eliminates code duplication. Uses caching to avoid recomputing typical values
for the same data. Converts CategoricalMixture objects to scenario values.
"""
function _get_typical_values_dict(data_nt::NamedTuple, typical)
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
        typical_val = _get_typical_value_optimized(col, typical)
        
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
function _get_typical_value_optimized(col, typical)
    if _is_continuous_variable(col)
        # For continuous variables, use the specified typical function (mean, median, etc.)
        return typical(col)
    elseif col isa CategoricalArray
        # For CategoricalArray, use simplified frequency mixture
        return _create_frequency_mixture_optimized(col)
    elseif eltype(col) <: Bool  
        # For Bool, use mean (O(n) but necessary for statistical correctness)
        return mean(col)
    elseif eltype(col) <: AbstractString
        throw(MarginsError("String variables are not supported in optimized reference grids. " *
                          "Statistical correctness cannot be guaranteed for arbitrary string data types. " *
                          "Consider using CategoricalArray for categorical string variables."))
    else
        throw(MarginsError("Unsupported data type $(eltype(col)) for variable in reference grid. " *
                          "Statistical correctness cannot be guaranteed for unknown data types. " *
                          "Supported types: numeric (Int64, Float64), Bool, CategoricalArray."))
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

# =============================================================================  
# Hierarchical Reference Grid Grammar Support
# =============================================================================

"""
    _parse_reference_grid_specification(spec, data_nt) -> Vector{Dict}

Adapt group parsing logic for reference grid construction context.
This function processes hierarchical specifications using the `=>` operator grammar
to generate systematic reference grids instead of data filtering.

# Arguments
- `spec`: Hierarchical specification (Symbol, Vector, Tuple, or Pair with `=>`)
- `data_nt`: NamedTuple containing the data for computing representative values

# Returns
- `Vector{Dict}`: Each Dict represents a reference grid row specification

# Examples
```julia
# Simple categorical: all observed levels
spec = :education
result = _parse_reference_grid_specification(spec, data_nt)

# Cross-tabulation: all combinations  
spec = [:region, :education]
result = _parse_reference_grid_specification(spec, data_nt)

# Continuous representatives: data-driven values
spec = (:income, :quartiles)
result = _parse_reference_grid_specification(spec, data_nt)

# Hierarchical specifications
spec = :region => :education
result = _parse_reference_grid_specification(spec, data_nt)
```
"""
function _parse_reference_grid_specification(spec, data_nt)
    # Simple categorical specification: :education
    if spec isa Symbol
        return _parse_categorical_reference_spec([spec], data_nt)
    end
    
    # Vector specification: may be all symbols or mixed specs  
    if spec isa AbstractVector
        if all(x -> x isa Symbol, spec)
            # All symbols - simple categorical cross-tabulation
            return _parse_categorical_reference_spec(spec, data_nt)
        else
            # Mixed vector - handle each spec and create cross-product
            all_specs = []
            for single_spec in spec
                spec_results = _parse_reference_grid_specification(single_spec, data_nt)
                if isempty(all_specs)
                    all_specs = spec_results
                else
                    # Cross-product with existing specifications
                    new_specs = []
                    for existing_spec in all_specs
                        for new_spec in spec_results
                            combined_spec = merge(existing_spec, new_spec)
                            push!(new_specs, combined_spec)
                        end
                    end
                    all_specs = new_specs
                end
            end
            return all_specs
        end
    end
    
    # Tuple specifications: Check mixture first, then continuous
    if spec isa Tuple && length(spec) == 2
        var, spec_val = spec
        if var isa Symbol
            # Check if it's a mixture specification first
            if _is_mixture_specification(spec_val)
                return _parse_mixture_reference_spec(var, spec_val, data_nt)
            else
                # Otherwise treat as continuous representative specification
                return _parse_continuous_representative_spec(var, spec_val, data_nt)
            end
        end
    end
    
    # Hierarchical specification: :region => :education  
    if spec isa Pair
        outer_spec = spec.first
        inner_spec = spec.second
        return _create_hierarchical_reference_specs(outer_spec, inner_spec, data_nt)
    end
    
    error("Invalid reference grid specification. Supported syntax: Symbol, Vector{Symbol}, (Symbol, representative_type), outer => inner, or (Symbol, mixture_spec)")
end

"""
    _parse_categorical_reference_spec(vars, data_nt) -> Vector{Dict}

Parse categorical reference specification for cross-tabulation.
Returns all combinations of categorical levels or mixture specifications.

Supports:
- Regular categorical variables: all unique levels
- Mixture specifications: mix_proportional, direct CategoricalMixture objects
"""
function _parse_categorical_reference_spec(vars, data_nt)
    if isempty(vars)
        return [Dict()]
    end
    
    # Generate all combinations of categorical levels or mixtures
    specs = []
    
    for var in vars
        if !haskey(data_nt, var)
            error("Variable $var not found in data")
        end
        
        col = data_nt[var]
        if _is_continuous_variable(col)
            error("Variable $var is continuous but used in categorical specification. Use (:$var, :mean) or (:$var, :quartiles) instead.")
        end
        
        # Get all unique levels (no mixture support here - use hierarchical specs for mixtures)
        levels = unique(col)
        var_specs = [Dict(var => level) for level in levels]
        
        if isempty(specs)
            specs = var_specs
        else
            # Cross-product with existing specs
            new_specs = []
            for existing_spec in specs
                for var_spec in var_specs
                    combined_spec = merge(existing_spec, var_spec)
                    push!(new_specs, combined_spec)
                end
            end
            specs = new_specs
        end
    end
    
    return isempty(specs) ? [Dict()] : specs
end

"""
    _parse_continuous_representative_spec(var, rep_spec, data_nt) -> Vector{Dict}

Parse continuous variable representative specification.
Computes representative values (quartiles, mean, median, etc.) from data.
"""
function _parse_continuous_representative_spec(var, rep_spec, data_nt)
    if !haskey(data_nt, var)
        error("Variable $var not found in data")
    end
    
    col = data_nt[var]
    if !_is_continuous_variable(col)
        error("Variable $var is not continuous but used with representative specification $rep_spec")
    end
    
    representatives = []
    
    if rep_spec === :quartiles
        quartile_values = [quantile(col, q) for q in [0.25, 0.50, 0.75, 1.0]]
        representatives = [Dict(var => val) for val in quartile_values]
    elseif rep_spec === :quintiles
        quintile_values = [quantile(col, q) for q in [0.2, 0.4, 0.6, 0.8, 1.0]]
        representatives = [Dict(var => val) for val in quintile_values]
    elseif rep_spec === :deciles
        decile_values = [quantile(col, q) for q in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]]
        representatives = [Dict(var => val) for val in decile_values]
    elseif rep_spec === :terciles || rep_spec === :tertiles
        tercile_values = [quantile(col, q) for q in [1/3, 2/3, 1.0]]
        representatives = [Dict(var => val) for val in tercile_values]
    elseif rep_spec === :mean
        representatives = [Dict(var => mean(col))]
    elseif rep_spec === :median
        representatives = [Dict(var => median(col))]
    elseif rep_spec === :min
        representatives = [Dict(var => minimum(col))]
    elseif rep_spec === :max
        representatives = [Dict(var => maximum(col))]
    elseif rep_spec isa Vector{<:Real} && length(rep_spec) > 0 && all(0 ≤ q ≤ 1 for q in rep_spec)
        # Custom percentiles (values between 0 and 1)
        percentile_values = [quantile(col, q) for q in rep_spec]
        representatives = [Dict(var => val) for val in percentile_values]
    elseif rep_spec isa Vector{<:Real}
        # Fixed values specified
        representatives = [Dict(var => val) for val in rep_spec]
    elseif rep_spec isa Tuple && length(rep_spec) == 2 && rep_spec[1] === :percentiles
        # Custom percentiles specification: (:percentiles, [0.1, 0.5, 0.9])
        percentiles = rep_spec[2]
        if percentiles isa Vector{<:Real} && all(0 ≤ q ≤ 1 for q in percentiles)
            percentile_values = [quantile(col, q) for q in percentiles]
            representatives = [Dict(var => val) for val in percentile_values]
        else
            error("Percentiles must be a vector of values between 0 and 1: $percentiles")
        end
    elseif rep_spec isa Tuple && length(rep_spec) == 2 && rep_spec[1] === :quantiles
        # N-quantiles specification: (:quantiles, 7) for septiles
        n_quantiles = rep_spec[2]
        if n_quantiles isa Integer && n_quantiles > 1
            quantile_values = [quantile(col, i/n_quantiles) for i in 1:n_quantiles]
            representatives = [Dict(var => val) for val in quantile_values]
        else
            error("Number of quantiles must be an integer > 1: $n_quantiles")
        end
    elseif rep_spec isa Tuple && length(rep_spec) == 2 && rep_spec[1] === :range
        # Range specification: (:range, (min_val, max_val)) or (:range, n_points)
        range_spec = rep_spec[2]
        if range_spec isa Tuple && length(range_spec) == 2
            min_val, max_val = range_spec
            representatives = [Dict(var => min_val), Dict(var => max_val)]
        elseif range_spec isa Integer && range_spec > 1
            # Create n evenly spaced points from min to max
            min_val, max_val = extrema(col)
            range_values = range(min_val, max_val, length=range_spec)
            representatives = [Dict(var => val) for val in range_values]
        else
            error("Range specification must be (min, max) tuple or integer number of points: $range_spec")
        end
    else
        error("Unsupported representative specification: $rep_spec. Supported: :quartiles, :quintiles, :deciles, :terciles, :mean, :median, :min, :max, Vector of values, (:percentiles, [0.1, 0.5, 0.9]), (:quantiles, n), (:range, (min, max)), or (:range, n_points)")
    end
    
    return representatives
end

"""
    _parse_deep_hierarchical_spec(spec, data_nt, max_depth) -> Vector{Dict}

Parse hierarchical specifications with support for 3+ level nesting.
Includes grid size estimation and safety checks.
"""
function _parse_deep_hierarchical_spec(spec, data_nt, max_depth)
    # Estimate total grid size before parsing
    estimated_size = _estimate_grid_size(spec, data_nt)
    _validate_grid_size(estimated_size)
    
    # Parse with depth tracking
    return _parse_hierarchical_recursive(spec, data_nt, 1, max_depth)
end

"""
    _parse_hierarchical_recursive(spec, data_nt, current_depth, max_depth) -> Vector{Dict}

Recursively parse hierarchical specifications with depth tracking.
"""
function _parse_hierarchical_recursive(spec, data_nt, current_depth, max_depth)
    if current_depth > max_depth
        error("Maximum nesting depth ($max_depth) exceeded at level $current_depth")
    end
    
    if spec isa Pair
        outer_spec, inner_spec = spec.first, spec.second
        
        # Get outer combinations
        outer_combinations = _parse_reference_grid_specification(outer_spec, data_nt)
        
        # Check if inner_spec contains more nesting
        if inner_spec isa Pair
            # Recursive case: more nesting levels
            result_combinations = []
            for outer_combo in outer_combinations
                # Filter data to this outer group
                filtered_data = _filter_data_to_group(data_nt, outer_combo)
                
                # Recursively parse inner specifications
                inner_combinations = _parse_hierarchical_recursive(
                    inner_spec, filtered_data, current_depth + 1, max_depth
                )
                
                # Merge outer and inner combinations
                for inner_combo in inner_combinations
                    merged_combo = merge(outer_combo, inner_combo)
                    push!(result_combinations, merged_combo)
                end
            end
            return result_combinations
        else
            # Base case: use existing 2-level logic
            return _create_hierarchical_reference_specs(outer_spec, inner_spec, data_nt)
        end
    else
        # Not hierarchical - use standard parsing
        return _parse_reference_grid_specification(spec, data_nt)
    end
end

"""
    _filter_data_to_group(data_nt, group_spec) -> NamedTuple

Filter data to a specific group defined by group_spec for recursive parsing.
"""
function _filter_data_to_group(data_nt, group_spec)
    # Get indices of observations that match the group specification
    group_indices = _get_group_indices(group_spec, data_nt)
    
    if isempty(group_indices)
        @warn "Empty group found for specification $group_spec - using full data"
        return data_nt
    end
    
    # Create filtered data for this group
    filtered_data = NamedTuple()
    for (var, col) in pairs(data_nt)
        filtered_col = col[group_indices]
        filtered_data = merge(filtered_data, NamedTuple{(var,)}((filtered_col,)))
    end
    
    return filtered_data
end

"""
    _create_hierarchical_reference_specs(outer_spec, inner_spec, data_nt) -> Vector{Dict}

Create hierarchical reference specifications using => syntax for reference grids.
Unlike group filtering, this computes representative values within each outer group.
"""
function _create_hierarchical_reference_specs(outer_spec, inner_spec, data_nt)
    # Parse outer specification
    outer_specs = _parse_reference_grid_specification(outer_spec, data_nt)
    
    # Handle inner specification - can be single spec or Vector of specs
    if inner_spec isa AbstractVector
        # Multiple inner specifications - create parallel representatives within each outer
        hierarchical_specs = []
        for outer_spec_dict in outer_specs
            for inner_single_spec in inner_spec
                inner_specs = _compute_representative_values_within_group(inner_single_spec, outer_spec_dict, data_nt)
                for inner_spec_dict in inner_specs
                    # Merge outer and inner specifications
                    combined_spec = merge(outer_spec_dict, inner_spec_dict)
                    push!(hierarchical_specs, combined_spec)
                end
            end
        end
        return hierarchical_specs
    else
        # Single inner specification
        hierarchical_specs = []
        for outer_spec_dict in outer_specs
            inner_specs = _compute_representative_values_within_group(inner_spec, outer_spec_dict, data_nt)
            for inner_spec_dict in inner_specs
                # Merge outer and inner specifications
                combined_spec = merge(outer_spec_dict, inner_spec_dict)
                push!(hierarchical_specs, combined_spec)
            end
        end
        return hierarchical_specs
    end
end

"""
    _compute_representative_values_within_group(spec, group_spec, data_nt) -> Vector{Dict}

Compute representative values within a specific group defined by group_spec.
This enables hierarchical reference grid construction where inner representatives
are computed within each outer group.
"""
function _compute_representative_values_within_group(spec, group_spec, data_nt)
    # Filter data to the specified group
    group_indices = _get_group_indices(group_spec, data_nt)
    
    if isempty(group_indices)
        @warn "Empty group found for specification $group_spec - skipping"
        return []
    end
    
    # Create filtered data for this group
    filtered_data = NamedTuple()
    for (var, col) in pairs(data_nt)
        filtered_col = col[group_indices]
        filtered_data = merge(filtered_data, NamedTuple{(var,)}((filtered_col,)))
    end
    
    # Parse the inner specification using the filtered group data
    return _parse_reference_grid_specification(spec, filtered_data)
end

"""
    _get_group_indices(group_spec, data_nt) -> Vector{Int}

Get indices of observations that match the group specification.
"""
function _get_group_indices(group_spec, data_nt)
    n_obs = length(first(data_nt))
    indices = collect(1:n_obs)
    
    for (var, val) in group_spec
        if haskey(data_nt, var)
            var_indices = findall(==(val), data_nt[var])
            indices = intersect(indices, var_indices)
        end
    end
    
    return indices
end

# =============================================================================
# Deep Hierarchical Nesting Support (Phase 1)
# =============================================================================

"""
    _estimate_grid_size(spec, data_nt) -> Int

Estimate total number of combinations without full computation.
Used for warnings and memory planning.
"""
function _estimate_grid_size(spec, data_nt)
    return _estimate_size_recursive(spec, data_nt)
end

"""
    _estimate_size_recursive(spec, data_nt) -> Int

Recursively estimate grid size for hierarchical specifications.
"""
function _estimate_size_recursive(spec, data_nt)
    if spec isa Pair
        outer_spec, inner_spec = spec.first, spec.second
        outer_size = _estimate_size_recursive(outer_spec, data_nt)
        
        if inner_spec isa Pair
            # Nested pair - multiplicative worst case
            inner_size = _estimate_size_recursive(inner_spec, data_nt)
            return outer_size * inner_size
        elseif inner_spec isa Vector
            # Multiple parallel specifications
            total_inner = sum(_estimate_size_recursive(s, data_nt) for s in inner_spec)
            return outer_size * total_inner
        else
            inner_size = _estimate_size_recursive(inner_spec, data_nt)
            return outer_size * inner_size
        end
    elseif spec isa Symbol
        # Categorical variable - count unique levels
        if haskey(data_nt, spec)
            col = data_nt[spec]
            if _is_continuous_variable(col)
                return 1  # Continuous variables typically produce single representative
            else
                return length(unique(col))
            end
        else
            return 1
        end
    elseif spec isa Vector
        if all(x -> x isa Symbol, spec)
            # Cross-tabulation of symbols
            total_size = 1
            for var in spec
                if haskey(data_nt, var)
                    col = data_nt[var]
                    if !_is_continuous_variable(col)
                        total_size *= length(unique(col))
                    end
                end
            end
            return total_size
        else
            # Mixed vector - multiplicative
            total_size = 1
            for single_spec in spec
                total_size *= _estimate_size_recursive(single_spec, data_nt)
            end
            return total_size
        end
    elseif spec isa Tuple && length(spec) == 2
        var, rep_spec = spec
        if _is_mixture_specification(rep_spec)
            return 1  # Mixtures produce single reference point
        elseif rep_spec === :quartiles
            return 4
        elseif rep_spec === :quintiles  
            return 5
        elseif rep_spec === :deciles
            return 10
        elseif rep_spec === :terciles || rep_spec === :tertiles
            return 3
        elseif rep_spec isa Vector
            return length(rep_spec)
        elseif rep_spec === :mean || rep_spec === :median || rep_spec === :min || rep_spec === :max
            return 1
        elseif rep_spec isa Tuple && length(rep_spec) == 2
            spec_type, spec_value = rep_spec
            if spec_type === :percentiles && spec_value isa Vector
                return length(spec_value)
            elseif spec_type === :quantiles && spec_value isa Integer
                return spec_value
            elseif spec_type === :range
                if spec_value isa Integer
                    return spec_value
                elseif spec_value isa Tuple
                    return 2  # min and max
                end
            end
        else
            return 1
        end
    else
        return 1
    end
end

"""
    _validate_grid_size(estimated_size) -> Bool

Check if estimated grid size is reasonable and warn user.
"""
function _validate_grid_size(estimated_size)
    if estimated_size > 1_000_000
        error("Estimated grid size too large: $(estimated_size) combinations. " *
              "Consider reducing nesting depth or using more selective specifications.")
    elseif estimated_size > 100_000
        @warn "Large grid estimated: $(estimated_size) combinations. " *
              "This may consume significant memory and time."
    elseif estimated_size > 10_000
        @info "Medium grid estimated: $(estimated_size) combinations."
    end
    
    return true
end

"""
    _validate_nesting_depth(spec, max_depth=5) -> Int

Count and validate nesting depth of specification.
"""
function _validate_nesting_depth(spec, max_depth=5)
    depth = _count_nesting_depth(spec)
    if depth > max_depth
        error("Nesting depth ($depth) exceeds maximum ($max_depth). " *
              "Use max_depth parameter to increase limit if needed.")
    end
    return depth
end

"""
    _count_nesting_depth(spec) -> Int

Count the maximum nesting depth in a hierarchical specification.
"""
function _count_nesting_depth(spec)
    if spec isa Pair
        outer_depth = _count_nesting_depth(spec.first)
        inner_depth = _count_nesting_depth(spec.second)
        return 1 + max(outer_depth, inner_depth)
    elseif spec isa Vector
        return maximum(_count_nesting_depth(s) for s in spec; init=0)
    else
        return 0
    end
end

# =============================================================================
# Phase 2.1: Mixture Integration Support  
# =============================================================================

"""
    _is_mixture_specification(spec) -> Bool

Check if a specification is a mixture specification.
Supports mix_proportional symbol and CategoricalMixture objects.
"""
function _is_mixture_specification(spec)
    return spec === :mix_proportional || spec isa CategoricalMixture
end

"""
    _parse_mixture_reference_spec(var, mixture_spec, data_nt) -> Vector{Dict}

Parse mixture reference specification for a categorical variable.
Creates a single reference point with the mixture specification.

Supports:
- :mix_proportional - Use observed frequency proportions from data
- CategoricalMixture objects - Direct mixture specification
"""
function _parse_mixture_reference_spec(var, mixture_spec, data_nt)
    if !haskey(data_nt, var)
        error("Variable $var not found in data")
    end
    
    col = data_nt[var]
    
    # Validate this is a categorical variable
    if _is_continuous_variable(col)
        error("Variable $var is continuous but used with mixture specification. Mixtures are only supported for categorical variables.")
    end
    
    if mixture_spec === :mix_proportional
        # Use existing frequency mixture computation
        mixture = _create_frequency_mixture_exact(col)
        return [Dict(var => mixture)]
    elseif mixture_spec isa CategoricalMixture
        # Validate mixture against data
        validate_mixture_against_data(mixture_spec, col, var)
        return [Dict(var => mixture_spec)]
    else
        error("Unsupported mixture specification: $mixture_spec. Supported: :mix_proportional or CategoricalMixture object")
    end
end

# =============================================================================
# Enhanced Validation and Error Handling (Phase 1.2)
# =============================================================================

"""
    _validate_reference_specification(spec, data_nt, max_depth) -> Bool

Validate reference specification against data before parsing.
Provides comprehensive error checking with informative messages.
"""
function _validate_reference_specification(spec, data_nt, max_depth)
    _validate_data_not_empty(data_nt)
    _validate_specification_structure(spec, data_nt)
    _validate_nesting_depth(spec, max_depth)
    return true
end

"""
    _validate_data_not_empty(data_nt) -> Bool

Ensure data is not empty for meaningful representative value computation.
"""
function _validate_data_not_empty(data_nt)
    if isempty(data_nt) || (length(data_nt) > 0 && length(first(data_nt)) == 0)
        @warn "Empty dataset provided to hierarchical_grid(). Representative values may be undefined (NaN/missing)."
    end
    return true
end

"""
    _validate_specification_structure(spec, data_nt) -> Bool

Validate the structure and content of the specification.
"""
function _validate_specification_structure(spec, data_nt)
    if spec isa Symbol
        _validate_variable_exists(spec, data_nt)
    elseif spec isa AbstractVector
        for item in spec
            _validate_specification_structure(item, data_nt)
        end
    elseif spec isa Tuple && length(spec) == 2
        var, rep_spec = spec
        if var isa Symbol
            _validate_variable_exists(var, data_nt)
            _validate_representative_specification(var, rep_spec, data_nt)
        else
            error("First element of tuple specification must be a Symbol (variable name), got: $(typeof(var))")
        end
    elseif spec isa Pair
        _validate_specification_structure(spec.first, data_nt)
        _validate_specification_structure(spec.second, data_nt)
    else
        error("Invalid specification type: $(typeof(spec)). Must be Symbol, Vector, Tuple, or Pair.")
    end
    return true
end

"""
    _validate_variable_exists(var, data_nt) -> Bool

Check that variable exists in data.
"""
function _validate_variable_exists(var, data_nt)
    if !haskey(data_nt, var)
        available_vars = collect(keys(data_nt))
        error("Variable :$var not found in data. Available variables: $available_vars")
    end
    return true
end

"""
    _validate_representative_specification(var, rep_spec, data_nt) -> Bool

Validate representative value specification against variable type and data.
"""
function _validate_representative_specification(var, rep_spec, data_nt)
    col = data_nt[var]
    
    # Check if variable is continuous for statistical specifications
    continuous_specs = [:quartiles, :quintiles, :deciles, :terciles, :tertiles, 
                       :mean, :median, :min, :max]
    
    if rep_spec in continuous_specs && !_is_continuous_variable(col)
        error("Representative specification :$rep_spec requires continuous variable, but :$var is $(eltype(col)). Use categorical cross-tabulation syntax instead.")
    end
    
    # Validate percentiles specifications
    if rep_spec isa Vector{<:Real} && length(rep_spec) > 0 && all(0 ≤ q ≤ 1 for q in rep_spec)
        # This is percentiles - validate they make sense
        if any(q < 0 || q > 1 for q in rep_spec)
            error("Percentile values must be between 0 and 1, got: $rep_spec")
        end
    end
    
    # Validate tuple specifications  
    if rep_spec isa Tuple && length(rep_spec) == 2
        spec_type, spec_value = rep_spec
        
        if spec_type === :percentiles
            if !(spec_value isa Vector{<:Real}) || any(q < 0 || q > 1 for q in spec_value)
                error("Percentiles specification requires vector of values in [0,1], got: $spec_value")
            end
        elseif spec_type === :quantiles
            if !(spec_value isa Integer) || spec_value <= 1
                error("Quantiles specification requires integer > 1, got: $spec_value")
            end
        elseif spec_type === :range
            if spec_value isa Integer
                if spec_value <= 1
                    error("Range specification with integer requires value > 1, got: $spec_value")
                end
                # Valid integer specification
            elseif spec_value isa Tuple && length(spec_value) == 2
                min_val, max_val = spec_value
                if !(min_val isa Real && max_val isa Real)
                    error("Range specification with tuple requires (min, max) numeric values, got: $spec_value")
                end
                if min_val > max_val
                    @warn "Range specification has min > max: ($min_val, $max_val). This will create a descending sequence."
                end
            else
                error("Range specification must be integer (number of points) or (min, max) tuple, got: $spec_value")
            end
        else
            error("Unknown tuple specification type: $spec_type. Supported: :percentiles, :quantiles, :range")
        end
    end
    
    return true
end

"""
    _validate_grid_combinations(combinations, data_nt) -> Bool

Validate that grid combinations are reasonable and warn about empty groups.
Only validates categorical combinations - continuous representatives are expected not to match exactly.
"""
function _validate_grid_combinations(combinations, data_nt)
    if isempty(combinations)
        @warn "Empty combination list generated. Check specification validity."
        return true
    end
    
    # Only validate categorical combinations - continuous values aren't expected to match exactly
    categorical_combinations = filter(combinations) do combo
        # Check if this combination has only categorical variables (no continuous representative values)
        has_only_categorical = true
        for (var, val) in combo
            if haskey(data_nt, var) && (_is_continuous_variable(data_nt[var]) || val isa Real)
                has_only_categorical = false
                break
            end
        end
        has_only_categorical
    end
    
    if !isempty(categorical_combinations)
        empty_combinations = 0
        
        for combo in categorical_combinations
            # Only check categorical parts of the combination
            categorical_parts = Dict(var => val for (var, val) in combo if haskey(data_nt, var) && !_is_continuous_variable(data_nt[var]))
            if !isempty(categorical_parts)
                indices = _get_group_indices(categorical_parts, data_nt)
                if isempty(indices)
                    empty_combinations += 1
                    if empty_combinations <= 3  # Only show first few to avoid spam
                        @warn "Categorical combination $categorical_parts matches no observations in data"
                    end
                end
            end
        end
        
        if empty_combinations > 0
            total_combinations = length(categorical_combinations)
            percentage_empty = round(100 * empty_combinations / total_combinations, digits=1)
            @warn "$(empty_combinations) out of $(total_combinations) categorical combinations ($(percentage_empty)%) match no data. Consider reviewing specification."
        end
    end
    
    # Warn about very large grids
    if length(combinations) > 1000
        @warn "Large reference grid generated ($(length(combinations)) combinations). This may impact performance or memory usage."
    elseif length(combinations) > 100
        @info "Medium reference grid generated ($(length(combinations)) combinations)."
    end
    
    return true
end

"""
    _get_typical_value_for_reference_grid(var, data_nt)

Get typical value for a variable when it's not specified in a reference point.
Uses same logic as means_grid() for consistency.
"""
function _get_typical_value_for_reference_grid(var, data_nt)
    if !haskey(data_nt, var)
        error("Variable $var not found in data when computing typical value")
    end
    
    col = data_nt[var]
    
    if _is_continuous_variable(col)
        return mean(col)  # Use mean for continuous variables
    else
        # For categorical, use most frequent level (mode)
        level_counts = Dict()
        for val in col
            level_counts[val] = get(level_counts, val, 0) + 1
        end
        # Return the most frequent level
        return first(sort(collect(level_counts), by=x->x[2], rev=true))[1]
    end
end

"""
    _build_hierarchical_reference_grid(grid_combinations, data_nt) -> DataFrame

Build reference grid DataFrame from parsed specifications.
Converts vector of Dict specifications to a proper DataFrame structure.
Uses typical values instead of missing for unspecified variables.
"""
function _build_hierarchical_reference_grid(grid_combinations, data_nt)
    if isempty(grid_combinations)
        return DataFrame()
    end
    
    # Convert to DataFrame format
    cols = Dict{Symbol, Vector}()
    
    # Get all possible columns from all combinations AND all data variables
    all_keys = Set{Symbol}()
    for combo in grid_combinations
        union!(all_keys, keys(combo))
    end
    # Ensure all data variables are included (for filling with typical values)
    union!(all_keys, keys(data_nt))
    
    # Initialize columns
    for key in all_keys
        cols[key] = Any[]
    end
    
    # Fill in values - Phase 1.2: Use typical values instead of missing
    for combo in grid_combinations
        for key in all_keys
            if haskey(combo, key)
                push!(cols[key], combo[key])
            else
                # For missing keys, fill with typical values from data
                typical_value = _get_typical_value_for_reference_grid(key, data_nt)
                push!(cols[key], typical_value)
            end
        end
    end
    
    # Convert to proper types where possible - enhanced for Phase 1.2
    # Handle missing values properly by using Union types
    for (key, values) in cols
        non_missing_values = filter(!ismissing, values)
        has_missing = any(ismissing, values)
        
        if isempty(non_missing_values)
            # All missing - keep as Any with Missing
            cols[key] = Vector{Union{Missing, Any}}(values)
        elseif all(val -> val isa Real && !(val isa Bool), non_missing_values)
            # Numeric values (possibly with missing)
            if has_missing
                cols[key] = Vector{Union{Missing, Float64}}([ismissing(val) ? missing : convert(Float64, val) for val in values])
            else
                cols[key] = Float64[convert(Float64, val) for val in values]
            end
        elseif all(val -> val isa Bool, non_missing_values)
            # Boolean values (possibly with missing)
            if has_missing
                cols[key] = Vector{Union{Missing, Bool}}([ismissing(val) ? missing : convert(Bool, val) for val in values])
            else
                cols[key] = Bool[convert(Bool, val) for val in values]
            end
        elseif all(val -> val isa AbstractString, non_missing_values)
            # String values (possibly with missing)
            if has_missing
                cols[key] = Vector{Union{Missing, String}}([ismissing(val) ? missing : string(val) for val in values])
            else
                cols[key] = String[string(val) for val in values]
            end
        elseif all(val -> val isa CategoricalValue, non_missing_values)
            # CategoricalValues (possibly with missing)
            first_val = first(non_missing_values)
            T = typeof(first_val)
            if has_missing
                cols[key] = Vector{Union{Missing, T}}(values)
            else
                cols[key] = T[val for val in values]
            end
        else
            # Mixed types - keep as Any
            if has_missing
                cols[key] = Vector{Union{Missing, Any}}(values)
            else
                cols[key] = Vector{Any}(values)
            end
        end
    end
    
    return DataFrame(cols)
end

"""
    hierarchical_grid(data, spec; max_depth=5, warn_large=true) -> DataFrame

Enhanced hierarchical reference grid builder with deep nesting support.
Uses group nesting grammar for systematic reference grid construction with
support for 3+ level hierarchical specifications.

# Arguments
- `data`: DataFrame or NamedTuple containing the data
- `spec`: Hierarchical specification using `=>` operator grammar
- `max_depth::Int=5`: Maximum allowed nesting depth
- `warn_large::Bool=true`: Show warnings for large grids

# Returns
- `DataFrame`: Reference grid ready for use with profile_margins()

# Examples
```julia
# Simple categorical: all observed levels
reference_grid = hierarchical_grid(data, :education)

# Cross-tabulation: all combinations
reference_grid = hierarchical_grid(data, [:region, :education])

# Continuous representatives: data-driven values
reference_grid = hierarchical_grid(data, (:income, :quartiles))

# 2-level hierarchical reference construction
reference_grid = hierarchical_grid(data, :region => :education)
reference_grid = hierarchical_grid(data, :region => (:income, :quartiles))

# 3-level hierarchical nesting (NEW)
reference_grid = hierarchical_grid(data, :country => (:region => :education))

# 4-level hierarchical nesting (NEW)
reference_grid = hierarchical_grid(data, :country => (:region => (:city => :education)))

# Mixed depth nesting (NEW)
reference_spec = :country => (:region => [
    :education,                              # Simple categorical within region
    (:income, :quartiles),                  # Statistical representatives within region
    (:city => (:age, :mean))               # 3-level nested specification
])
reference_grid = hierarchical_grid(data, reference_spec)

# Complex hierarchical with mixed types
reference_spec = :region => [
    :education,                    # All education levels within each region
    (:income, :quartiles),        # Income quartiles within each region  
    (:age, :mean)                 # Mean age within each region
]
reference_grid = hierarchical_grid(data, reference_spec)

# Phase 2.1: Mixture integration examples
reference_spec = :region => [
    (:education, :mix_proportional),         # Population-proportion education mixtures within each region
    (:employment, mix("full_time" => 0.6, "part_time" => 0.4)),  # Custom employment mixture within each region
    (:income, :quartiles)                    # Income quartiles within each region
]
reference_grid = hierarchical_grid(data, reference_spec)

# Control nesting depth and warnings
reference_grid = hierarchical_grid(data, very_deep_spec; max_depth=10, warn_large=false)

# Use with profile_margins
result = profile_margins(model, data, reference_grid; vars=[:treatment])
```

# Grammar Support

## Basic References
- `:education` - All observed levels of categorical variable
- `[:region, :education]` - Cross-tabulation of categorical variables
- `(:income, :quartiles)` - Income quartiles (Q1, Q2, Q3, Q4)
- `(:income, :quintiles)` - Income quintiles (P1, P2, P3, P4, P5)
- `(:income, :deciles)` - Income deciles (D1, D2, ..., D10)
- `(:income, :terciles)` - Income terciles/tertiles (T1, T2, T3)
- `(:age, :mean)` - Overall mean age
- `(:age, :median)` - Overall median age
- `(:age, :min)` - Minimum age
- `(:age, :max)` - Maximum age
- `(:age, [30, 50, 70])` - Fixed representative ages
- `(:income, [0.1, 0.5, 0.9])` - Custom percentiles (10th, 50th, 90th)
- `(:income, (:percentiles, [0.25, 0.75]))` - Explicit percentiles specification
- `(:income, (:quantiles, 7))` - Custom n-quantiles (septiles)
- `(:income, (:range, 5))` - 5 evenly spaced points from min to max
- `(:income, (:range, (25000, 75000)))` - Fixed range endpoints

## Hierarchical References (Enhanced)
- `:region => :education` - Education levels within each region (2-level)
- `:region => (:income, :quartiles)` - Income quartiles within each region (2-level)
- `:region => [(:income, :deciles), (:age, :mean)]` - Multiple representatives within each region (2-level)
- `:country => (:region => :education)` - Education levels within region within country (**3-level**)
- `:country => (:region => (:city => :education))` - City education within region within country (**4-level**)
- `:country => (:region => (:city => [(:income, :quartiles), (:age, :mean)]))` - **Complex 4-level nesting**

## Mixture Integration (Phase 2.1)
- `(:education, :mix_proportional)` - Use observed frequency proportions from data
- `(:treatment, mix("control" => 0.4, "treatment" => 0.6))` - Custom mixture specification
- `:region => (:education, :mix_proportional)` - Population-proportion mixtures within each region
- `:region => [(:education, :mix_proportional), (:income, :quartiles)]` - Combined mixture and statistical representatives

## Performance and Safety Features (NEW)
- Automatic grid size estimation with warnings for large combinations
- Maximum nesting depth protection (configurable)
- Informative error messages for malformed deep specifications
- Memory usage warnings and recommendations

This provides the most sophisticated reference grid system in any statistical software,
enabling complex multi-dimensional policy analysis that is currently impossible elsewhere.
"""
function hierarchical_grid(data, spec; max_depth=5, warn_large=true)
    data_nt = data isa NamedTuple ? data : Tables.columntable(data)
    
    # Enhanced validation with depth checking
    _validate_reference_specification(spec, data_nt, max_depth)
    
    # Determine which parser to use based on nesting depth
    actual_depth = _validate_nesting_depth(spec, max_depth)
    
    if actual_depth <= 2
        # Use optimized 2-level path for performance
        parsed_specs = _parse_reference_grid_specification(spec, data_nt)
    else
        # Use new deep nesting parser
        parsed_specs = _parse_deep_hierarchical_spec(spec, data_nt, max_depth)
    end
    
    # Grid size validation and warnings
    if warn_large
        _validate_grid_combinations(parsed_specs, data_nt)
    end
    
    # Build the reference grid DataFrame
    return _build_hierarchical_reference_grid(parsed_specs, data_nt)
end

# =============================================================================
# Reference Grid Builder Functions (NEW - AsBalanced Support)  
# =============================================================================

"""
    means_grid(data; typical=mean) -> DataFrame

Build reference grid with sample means for continuous variables and frequency-weighted 
mixtures for categorical variables. This is the standard "at means" behavior.

# Arguments
- `data`: DataFrame or NamedTuple containing the data
- `typical`: Function to compute typical values for continuous variables (default: mean)

# Returns
- `DataFrame`: Single-row reference grid with typical values

# Examples
```julia
# Standard "at means" grid
ref_grid = means_grid(data)
result = profile_margins(model, data, ref_grid)

# Using median instead of mean for continuous variables
ref_grid = means_grid(data; typical=median)
```
"""
function means_grid(data; typical=mean)
    data_nt = data isa NamedTuple ? data : Tables.columntable(data)
    
    # Use existing infrastructure with frequency weighting
    typical_values = _get_typical_values_dict(data_nt, typical)
    
    # Convert to DataFrame format
    cols = Dict{Symbol, Vector}()
    for (k, v) in typical_values
        if v isa Real && !(v isa Bool)
            cols[k] = Float64[v]
        elseif v isa Bool  
            cols[k] = Bool[v]
        else
            cols[k] = [v]  # Generic for CategoricalMixture, strings, etc.
        end
    end
    
    return DataFrame(cols)
end

"""
    balanced_grid(data; vars...) -> DataFrame

Build reference grid with balanced (equal weight) mixtures for specified categorical variables.
This creates orthogonal factorial designs for AsBalanced analysis.

Continuous variables use sample means. Categorical variables specified in `vars` get
equal probability weights. Other categorical variables use frequency-weighted mixtures.

# Arguments
- `data`: DataFrame or NamedTuple containing the data
- `vars...`: Keyword arguments specifying which variables to balance
  - Use `:all` to balance all categorical variables: `education=:all`
  - Use specific levels: `education=["high_school", "college"]`

# Returns  
- `DataFrame`: Single-row reference grid with balanced categorical mixtures

# Examples
```julia
# Balance all levels of education variable
ref_grid = balanced_grid(data; education=:all)

# Balance specific education levels 
ref_grid = balanced_grid(data; education=["high_school", "college"])

# Balance multiple variables
ref_grid = balanced_grid(data; education=:all, region=:all)

# Use with profile_margins
result = profile_margins(model, data, ref_grid)
```
"""
function balanced_grid(data; vars...)
    data_nt = data isa NamedTuple ? data : Tables.columntable(data)
    
    # Start with frequency-weighted typical values
    typical_values = _get_typical_values_dict(data_nt, mean)
    
    # Override specified variables with balanced mixtures
    for (var, spec) in vars
        if !haskey(data_nt, var)
            throw(ArgumentError("Variable :$var not found in data. Available variables: $(collect(keys(data_nt)))"))
        end
        
        col = getproperty(data_nt, var)
        
        if spec === :all
            # Balance all levels of this categorical variable
            typical_values[var] = create_balanced_mixture(col)
        elseif spec isa Vector
            # Balance only specified levels
            if eltype(col) <: Bool && spec isa Vector{String}
                # Handle Bool with string levels
                if !issubset(spec, ["true", "false"])
                    throw(ArgumentError("Bool variable :$var can only specify levels 'true' and/or 'false'"))
                end
                equal_weights = fill(1.0/length(spec), length(spec))
                typical_values[var] = CategoricalMixture(spec, equal_weights)
            else
                # Validate levels exist in data
                actual_levels = if col isa CategoricalArray
                    string.(CategoricalArrays.levels(col))
                else
                    unique(string.(col))
                end
                
                spec_str = string.(spec)
                missing_levels = setdiff(spec_str, actual_levels) 
                if !isempty(missing_levels)
                    throw(ArgumentError("Variable :$var contains levels not found in data: $missing_levels. Available: $actual_levels"))
                end
                
                equal_weights = fill(1.0/length(spec), length(spec))
                typical_values[var] = CategoricalMixture(spec_str, equal_weights)
            end
        else
            throw(ArgumentError("Balanced specification for :$var must be :all or Vector of levels"))
        end
    end
    
    # Convert to DataFrame format
    cols = Dict{Symbol, Vector}()
    for (k, v) in typical_values
        if v isa Real && !(v isa Bool)
            cols[k] = Float64[v]
        elseif v isa Bool  
            cols[k] = Bool[v]
        else
            cols[k] = [v]  # CategoricalMixture objects, strings, etc.
        end
    end
    
    return DataFrame(cols)
end

"""
    cartesian_grid(; vars...) -> DataFrame

Build Cartesian product reference grid from variable specifications.
This is a pure grid constructor that creates combinations from provided values
without needing reference data. When used with `profile_margins()`, missing 
model variables are automatically completed with typical values.

# Arguments
- `vars...`: Keyword arguments mapping variables to values
  - Single values: `age=45`
  - Multiple values: `age=[25, 45, 65]`
  - Numlist notation: `age="25(10)65"` (25, 35, 45, 55, 65)

# Returns
- `DataFrame`: Reference grid with Cartesian product of specified values

# Examples
```julia
# Simple cartesian product
ref_grid = cartesian_grid(age=[25, 45, 65], treatment=[true, false])

# Using range construction
ref_grid = cartesian_grid(
    x1 = collect(range(extrema(data.x1)...; length = 5)),
    x2 = collect(range(extrema(data.x2)...; length = 5))
)

# Mixed specifications
ref_grid = cartesian_grid(age="25(10)65", education=["college", "graduate"])

# Use with profile_margins (automatically completed with typical values for missing variables)
result = profile_margins(model, data, ref_grid)
```
"""
function cartesian_grid(; vars...)
    at_spec = Dict{Symbol, Any}(vars)
    
    if isempty(at_spec)
        return DataFrame()
    end
    
    # Extract specified variables and their values
    var_names = collect(keys(at_spec))
    var_values = Vector{Vector{Any}}(undef, length(var_names))
    
    for (i, var) in enumerate(var_names)
        vals = at_spec[var]
        # Handle numlist parsing for strings like "-2(2)2" -> [-2, 0, 2]
        if vals isa String && occursin(r"^-?\d+\(\d+\)-?\d+$", vals)
            vals = _parse_numlist(vals)
        end
        var_values[i] = vals isa Vector ? vals : [vals]  # Convert single value to vector
    end
    
    # Pre-calculate total number of combinations for efficient allocation
    n_combinations = prod(length(vals) for vals in var_values)
    
    # Pre-allocate result vector for better performance
    grid_rows = Vector{Dict{Symbol,Any}}(undef, n_combinations)
    
    # Generate Cartesian product
    combo_idx = 1
    for combo in Iterators.product(var_values...)
        row = Dict{Symbol,Any}()
        for (i, var) in enumerate(var_names)
            row[var] = combo[i]
        end
        grid_rows[combo_idx] = row
        combo_idx += 1
    end
    
    # Convert to DataFrame
    if isempty(grid_rows)
        return DataFrame()
    end
    
    cols = Dict{Symbol, Vector}()
    for key in keys(first(grid_rows))
        values = [row[key] for row in grid_rows]
        # Create vector with proper type based on first value type
        first_val = first(values)
        if all(v -> typeof(v) == typeof(first_val), values)
            cols[key] = typeof(first_val)[values...]
        else
            cols[key] = Vector{Any}(values)
        end
    end
    
    return DataFrame(cols)
end

"""
    complete_reference_grid(reference_grid, model, data; typical=mean) -> DataFrame

Complete a partial reference grid by adding typical values for missing model variables.
This function takes a reference grid that may be missing some model variables and
adds appropriate typical values for those missing variables.

# Arguments
- `reference_grid`: DataFrame with partial reference grid (e.g., from cartesian_grid)
- `model`: Fitted statistical model
- `data`: Original data (DataFrame or NamedTuple) to compute typical values
- `typical`: Function to compute typical values for continuous variables (default: mean)

# Returns
- `DataFrame`: Complete reference grid with all model variables

# Examples
```julia
# Create partial grid with only x1 and x2
partial_grid = cartesian_grid(x1=[0, 1, 2], x2=[10, 20])

# Complete it with typical values for other model variables
complete_grid = complete_reference_grid(partial_grid, model, data)

# Use with profile_margins
result = profile_margins(model, data, complete_grid)
```
"""
function complete_reference_grid(reference_grid::DataFrame, model, data; typical=mean)
    # Convert data to NamedTuple for consistency
    data_nt = data isa NamedTuple ? data : Tables.columntable(data)
    
    # Filter data to model variables only
    filtered_data = _filter_data_to_model_variables(data_nt, model)
    
    # Get typical values for all model variables
    typical_values = _get_typical_values_dict(filtered_data, typical)
    
    # Create complete reference grid
    n_rows = nrow(reference_grid)
    complete_cols = Dict{Symbol, Vector}()
    
    # First, add all columns from the original reference grid
    for col_name in names(reference_grid)
        complete_cols[Symbol(col_name)] = reference_grid[!, col_name]
    end
    
    # Then, add typical values for missing model variables
    for (var, typical_val) in typical_values
        if !haskey(complete_cols, var)
            # This variable is missing from reference grid, add typical values
            if typical_val isa Real && !(typical_val isa Bool)
                complete_cols[var] = fill(Float64(typical_val), n_rows)
            elseif typical_val isa Bool
                complete_cols[var] = fill(typical_val, n_rows)
            else
                # CategoricalMixture or other types
                complete_cols[var] = fill(typical_val, n_rows)
            end
        end
    end
    
    return DataFrame(complete_cols)
end

"""
    quantile_grid(data; vars...) -> DataFrame

Build reference grid using quantiles of continuous variables.

# Arguments
- `data`: DataFrame or NamedTuple containing the data  
- `vars...`: Keyword arguments mapping variables to quantile specifications
  - Vector of quantiles: `income=[0.1, 0.5, 0.9]` (10th, 50th, 90th percentiles)
  - Number of quantiles: `age=4` (quartiles: 25th, 50th, 75th percentiles)

# Returns
- `DataFrame`: Reference grid with quantile-based values

# Examples
```julia
# Specific quantiles
ref_grid = quantile_grid(data; income=[0.1, 0.5, 0.9])

# Standard quantiles (quartiles, quintiles, etc.)
ref_grid = quantile_grid(data; age=4, income=5)

# Use with profile_margins
result = profile_margins(model, data, ref_grid)
```
"""
function quantile_grid(data; vars...)
    data_nt = data isa NamedTuple ? data : Tables.columntable(data)
    
    # Convert quantile specifications to actual values
    at_dict = Dict{Symbol, Any}()
    for (var, spec) in vars
        if !haskey(data_nt, var)
            throw(ArgumentError("Variable :$var not found in data"))
        end
        
        col = getproperty(data_nt, var)
        if !_is_continuous_variable(col)
            throw(ArgumentError("quantile_grid() only supports continuous variables. Variable :$var is $(eltype(col))"))
        end
        
        if spec isa Number && spec > 1
            # Number of quantiles (e.g., 4 for quartiles)
            n_quantiles = Int(spec)
            quantiles = [(i)/(n_quantiles+1) for i in 1:n_quantiles]
            at_dict[var] = [quantile(col, q) for q in quantiles]
        elseif spec isa Vector{<:Real} && all(0 .≤ spec .≤ 1)
            # Explicit quantiles
            at_dict[var] = [quantile(col, q) for q in spec]
        else
            throw(ArgumentError("Quantile specification for :$var must be number of quantiles (>1) or vector of quantile values in [0,1]"))
        end
    end
    
    return _build_cartesian_refgrid(at_dict, data_nt, mean)
end

# =============================================================================
# String-to-Categorical Conversion for Reference Grids
# =============================================================================

"""
    process_reference_grid(data::DataFrame, grid::DataFrame) -> DataFrame

Process reference grid to convert string values to proper categorical values.

When users specify categorical levels using strings in reference grids
(e.g., `cartesian_grid(education=["High School", "College"])`), this function
automatically converts those strings to proper `CategoricalValue` objects
that match the categorical structure of the original data.

# Arguments
- `data::DataFrame`: Original data containing categorical variables
- `grid::DataFrame`: Reference grid potentially containing string specifications

# Returns
- `DataFrame`: Processed grid with string values converted to categorical values

# Algorithm
1. For each column in the grid:
   - Check if corresponding data column is categorical
   - If grid column contains strings AND data column is categorical:
     - Validate all strings exist as valid levels in the data
     - Convert strings to proper `CategoricalValue` objects using CategoricalArrays.jl
   - Otherwise, pass through unchanged

# Error Handling
- Throws informative error if string doesn't match any categorical level
- Suggests available levels when conversion fails

# Examples
```julia
# Original data has categorical education variable
data = DataFrame(education = categorical(["High School", "College", "Graduate"]))

# User specifies grid with strings
grid = DataFrame(education = ["High School", "College"])

# Automatic conversion to proper categorical values
processed_grid = process_reference_grid(data, grid)
```
"""
function process_reference_grid(data::DataFrame, grid::DataFrame)
    processed_grid = copy(grid)
    
    for col_name in names(grid)
        # Convert to Symbol for property access
        col_symbol = Symbol(col_name)
        
        if hasproperty(data, col_symbol) && isa(data[!, col_symbol], CategoricalVector)
            data_cat_vec = data[!, col_symbol]
            data_levels = levels(data_cat_vec)
            grid_values = grid[!, col_symbol]
            
            if eltype(grid_values) <: AbstractString
                # Validate all strings exist in categorical levels
                for str_val in grid_values
                    if !(str_val in data_levels)
                        error("Level '$str_val' not found in column '$col_name'. Available levels: $data_levels")
                    end
                end
                
                # Use CategoricalArrays.jl to create proper categorical vector
                # This preserves the same levels, ordering, and structure as original
                converted_vec = CategoricalArrays.categorical(grid_values, levels=data_levels, ordered=isordered(data_cat_vec))
                processed_grid[!, col_symbol] = converted_vec
            end
        end
    end
    
    return processed_grid
end
