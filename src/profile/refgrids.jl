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
    
    # Filter data to only include model variables
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
    
    # Continuous representative specification: (:income, :quartiles) or (:age, :mean)
    if spec isa Tuple && length(spec) == 2
        var, rep_spec = spec
        if var isa Symbol
            return _parse_continuous_representative_spec(var, rep_spec, data_nt)
        end
    end
    
    # Hierarchical specification: :region => :education  
    if spec isa Pair
        outer_spec = spec.first
        inner_spec = spec.second
        return _create_hierarchical_reference_specs(outer_spec, inner_spec, data_nt)
    end
    
    error("Invalid reference grid specification. Supported syntax: Symbol, Vector{Symbol}, (Symbol, representative_type), or outer => inner")
end

"""
    _parse_categorical_reference_spec(vars, data_nt) -> Vector{Dict}

Parse categorical reference specification for cross-tabulation.
Returns all combinations of categorical levels.
"""
function _parse_categorical_reference_spec(vars, data_nt)
    if isempty(vars)
        return [Dict()]
    end
    
    # Generate all combinations of categorical levels
    specs = []
    
    for var in vars
        if !haskey(data_nt, var)
            error("Variable $var not found in data")
        end
        
        col = data_nt[var]
        if _is_continuous_variable(col)
            error("Variable $var is continuous but used in categorical specification. Use (:$var, :mean) or (:$var, :quartiles) instead.")
        end
        
        # Get all unique levels
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
    elseif rep_spec === :mean
        representatives = [Dict(var => mean(col))]
    elseif rep_spec === :median
        representatives = [Dict(var => median(col))]
    elseif rep_spec isa Vector{<:Real}
        # Fixed values specified
        representatives = [Dict(var => val) for val in rep_spec]
    else
        error("Unsupported representative specification: $rep_spec. Supported: :quartiles, :quintiles, :mean, :median, or Vector of values")
    end
    
    return representatives
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

"""
    _build_hierarchical_reference_grid(grid_combinations) -> DataFrame

Build reference grid DataFrame from parsed specifications.
Converts vector of Dict specifications to a proper DataFrame structure.
"""
function _build_hierarchical_reference_grid(grid_combinations)
    if isempty(grid_combinations)
        return DataFrame()
    end
    
    # Convert to DataFrame format
    cols = Dict{Symbol, Vector}()
    
    # Get all possible columns from all combinations
    all_keys = Set{Symbol}()
    for combo in grid_combinations
        union!(all_keys, keys(combo))
    end
    
    # Initialize columns
    for key in all_keys
        cols[key] = Any[]
    end
    
    # Fill in values
    for combo in grid_combinations
        for key in all_keys
            if haskey(combo, key)
                push!(cols[key], combo[key])
            else
                # This should not happen with proper parsing, but handle gracefully
                push!(cols[key], missing)
            end
        end
    end
    
    # Convert to proper types where possible
    for (key, values) in cols
        if all(val -> !ismissing(val) && val isa Real && !(val isa Bool), values)
            cols[key] = Float64[val for val in values]
        elseif all(val -> !ismissing(val) && val isa Bool, values)
            cols[key] = Bool[val for val in values]
        # Leave other types as-is (Any[] for mixed types)
        end
    end
    
    return DataFrame(cols)
end

"""
    hierarchical_grid(data, spec) -> DataFrame

Build hierarchical reference grid using group nesting grammar for systematic 
reference grid construction. Uses the same `=>` operator grammar as data stratification
but generates reference grids instead of filtering data.

# Arguments
- `data`: DataFrame or NamedTuple containing the data
- `spec`: Hierarchical specification using `=>` operator grammar

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

# Hierarchical reference construction
reference_grid = hierarchical_grid(data, :region => :education)
reference_grid = hierarchical_grid(data, :region => (:income, :quartiles))

# Complex hierarchical with mixed types
reference_spec = :region => [
    :education,                    # All education levels within each region
    (:income, :quartiles),        # Income quartiles within each region  
    (:age, :mean)                 # Mean age within each region
]
reference_grid = hierarchical_grid(data, reference_spec)

# Use with profile_margins
result = profile_margins(model, data, reference_grid; vars=[:treatment])
```

# Grammar Support

## Basic References
- `:education` - All observed levels of categorical variable
- `[:region, :education]` - Cross-tabulation of categorical variables
- `(:income, :quartiles)` - Income quartiles (Q1, Q2, Q3, Q4)
- `(:income, :quintiles)` - Income quintiles (P1, P2, P3, P4, P5) 
- `(:age, :mean)` - Overall mean age
- `(:age, :median)` - Overall median age
- `(:age, [30, 50, 70])` - Fixed representative ages

## Hierarchical References
- `:region => :education` - Education levels within each region
- `:region => (:income, :quartiles)` - Income quartiles within each region
- `:region => [(:income, :quartiles), (:age, :mean)]` - Multiple representatives within each region

This provides unprecedented ease of use for complex reference grid construction with
methodological rigor through data-driven representative value selection.
"""
function hierarchical_grid(data, spec)
    data_nt = data isa NamedTuple ? data : Tables.columntable(data)
    
    # Parse the hierarchical specification
    parsed_specs = _parse_reference_grid_specification(spec, data_nt)
    
    # Build the reference grid DataFrame
    return _build_hierarchical_reference_grid(parsed_specs)
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
    cartesian_grid(data; vars...) -> DataFrame

Build Cartesian product reference grid from variable specifications.
This provides a cleaner builder approach for reference grid specification.

# Arguments
- `data`: DataFrame or NamedTuple containing the data
- `vars...`: Keyword arguments mapping variables to values
  - Single values: `age=45`
  - Multiple values: `age=[25, 45, 65]`
  - Numlist notation: `age="25(10)65"` (25, 35, 45, 55, 65)

# Returns
- `DataFrame`: Reference grid with Cartesian product of specified values

# Examples
```julia
# Simple cartesian product
ref_grid = cartesian_grid(data; age=[25, 45, 65], treatment=[true, false])

# Mixed specifications
ref_grid = cartesian_grid(data; age="25(10)65", education=["college", "graduate"])

# Use with profile_margins
result = profile_margins(model, data, ref_grid)
```
"""
function cartesian_grid(data; vars...)
    data_nt = data isa NamedTuple ? data : Tables.columntable(data)
    at_dict = Dict{Symbol, Any}(vars)
    return _build_cartesian_refgrid(at_dict, data_nt, mean)
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