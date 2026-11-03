# profile/reference_grids.jl

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
    # Get model variables using FormulaCompiler (using cache for performance)
    compiled = _shared_compiled(model, data_nt)

    # Extract all variables used in model operations
    model_vars = extract_variable_names(compiled)   # every data column the formula reads
    
    # Filter data to only include predictor variables (exclude response), in the data's column order
    keep = Tuple(k for k in keys(data_nt) if k in model_vars)
    return NamedTuple{keep}(map(k -> getproperty(data_nt, k), keep))
end


"""
    _parse_numlist(str::String) -> Vector{Float64}

Parse numlist notation like "-2(2)2" into vector [-2, 0, 2].
Format: start(step)end where step is the increment.
"""
const _NUMLIST_RE = r"^(-?\d*\.?\d+)\((-?\d*\.?\d+)\)(-?\d*\.?\d+)$"

# Single source of truth for "is this string a numlist": the callers' gate and the parser
# must agree, or a valid numlist (e.g. a decimal step) falls through as a literal String.
_is_numlist(str::AbstractString) = occursin(_NUMLIST_RE, str)

function _parse_numlist(str::String)
    # Match pattern like "-2(2)2" or "1(0.5)3"
    m = match(_NUMLIST_RE, str)
    if isnothing(m)
        throw(ArgumentError("Invalid numlist format: $str. Expected format: start(step)end"))
    end
    
    start_val = parse(Float64, m.captures[1])
    step_val = parse(Float64, m.captures[2])
    end_val = parse(Float64, m.captures[3])
    
    # Generate sequence
    if step_val == 0
        throw(ArgumentError("Step size cannot be zero in numlist: $str"))
    end
    
    # Compute each point as start + k*step rather than accumulating, so rounding error
    # does not grow with sequence length. Works for positive and negative steps; the
    # tolerance admits an endpoint that is off by floating-point noise.
    nsteps = floor(Int, (end_val - start_val) / step_val + 1e-10 / abs(step_val))
    return [start_val + k * step_val for k in 0:nsteps]  # empty if the range is backwards
end

# ---- shared grid helpers: weights, column types, order ----------------------------------------

"""
    _grid_weights(weights, data, data_nt, builder; model=nothing) -> Union{Nothing, Vector{Float64}}

Resolve the `weights` keyword of a reference-grid builder to one weight per row of `data_nt`
(`nothing`: unweighted). The forms and checks are those of `population_margins`' `weights=`
(`_resolve_weights`): a column name (`Symbol`) or a vector of real, finite, non-negative weights,
not all zero; `nothing` and `:none` mean unweighted. `:model` (the fitted model's estimation
weights) needs a model, so only `complete_reference_grid` accepts it.
"""
function _grid_weights(weights, data, data_nt, builder; model=nothing)
    (isnothing(weights) || weights === :none) && return nothing
    if weights === :model
        isnothing(model) && throw(ArgumentError(
            "$builder(…; weights=:model): this builder has no model to read weights from. Pass the " *
            "weights as a column name (weights=:w) or a vector, or complete a partial grid with " *
            "complete_reference_grid(grid, model, data; weights=:model)."))
        return first(_resolve_weights(:model, model, data, data_nt))
    end
    weights isa Union{Symbol, AbstractVector} || throw(ArgumentError(
        "$builder: weights must be a column name (Symbol), a vector of weights, :none or nothing; got $(typeof(weights))"))
    return first(_resolve_weights(weights, nothing, data, data_nt))
end

# The data with only the positively weighted rows, and their weights: a row with weight 0 is
# dropped by every weighted builder (also where it would otherwise appear: observed levels,
# :min/:max/:range, levels of a hierarchical group)
function _drop_zero_weight_rows(data_nt::NamedTuple, w)
    isnothing(w) && return (data_nt, nothing)
    pos = findall(>(0), w)
    length(pos) == length(w) && return (data_nt, w)
    return (map(c -> c[pos], data_nt), w[pos])
end

# A grid column from the values of its rows: typed as the values when they share one type
function _typed_grid_column(values::AbstractVector)
    isempty(values) && return Any[]
    first_val = first(values)
    return all(v -> typeof(v) == typeof(first_val), values) ? typeof(first_val)[values...] : Vector{Any}(values)
end

# A one-row grid column holding a typical value (a real number as Float64)
_typical_grid_column(v) = v isa Real && !(v isa Bool) ? Float64[v] : v isa Bool ? Bool[v] : [v]

"""
    _nested_product(value_lists) -> Vector{Tuple}

The Cartesian product of `value_lists` in the reference grids' row order: the first list varies
slowest and the last fastest (the order of nested `for` loops written in the same order, i.e.
`Iterators.product` with the lists reversed), each list in its own order.
"""
_nested_product(value_lists) = vec([reverse(c) for c in Iterators.product(reverse(Tuple(value_lists))...)])

"""
    _ordered_values(col) -> Vector

The distinct values of `col` in a deterministic order: categorical level order for a
`CategoricalArray` (observed levels only), otherwise ascending (`false < true`, numbers, strings
lexicographically). Values that cannot be compared keep their order of first appearance.
"""
function _ordered_values(col)
    u = unique(col)
    if col isa CategoricalArray
        return sort!(u; by=v -> ismissing(v) ? typemax(Int) : CategoricalArrays.levelcode(v))
    end
    try
        return sort!(u)
    catch err
        err isa MethodError || rethrow()
        return u
    end
end

"""
    _build_cartesian_refgrid(at_spec, data_nt, typical; weights=nothing) -> DataFrame

The Cartesian product of the values in `at_spec` (an ordered collection of `var => values`
pairs; a value that is not a vector is one value, a numlist string is expanded), with every other
column of `data_nt` at its typical value. Columns in the data's order; rows nested by
`_nested_product` in `at_spec`'s order.
"""
function _build_cartesian_refgrid(at_spec, data_nt::NamedTuple, typical; weights=nothing)
    spec = [Symbol(k) => v for (k, v) in at_spec]
    spec_vars = first.(spec)
    others = Tuple(k for k in keys(data_nt) if !(k in spec_vars))
    base = get_typical_values(NamedTuple{others}(map(k -> data_nt[k], others)); typical, weights)
    value_lists = [_grid_values(v) for (_, v) in spec]
    combos = _nested_product(value_lists)
    isempty(combos) && return DataFrame()
    n = length(combos)
    cols = Pair{Symbol, AbstractVector}[]
    for k in keys(data_nt)
        i = findfirst(==(k), spec_vars)
        if isnothing(i)
            push!(cols, k => _typed_grid_column(fill(base[k], n)))
        else
            push!(cols, k => _typed_grid_column([c[i] for c in combos]))
        end
    end
    df = DataFrame(cols)
    return _mark_typical_values!(df, [String(k) for k in others])
end

# The values of one grid variable: a vector as given, a numlist string expanded, anything else one value
function _grid_values(vals)
    if vals isa String && _is_numlist(vals)
        vals = _parse_numlist(vals)
    end
    return vals isa AbstractVector ? collect(vals) : [vals]
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
function _parse_reference_grid_specification(spec, data_nt; weights=nothing)
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
                spec_results = _parse_reference_grid_specification(single_spec, data_nt; weights)
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
                return _parse_mixture_reference_spec(var, spec_val, data_nt; weights)
            else
                # Otherwise treat as continuous representative specification
                return _parse_continuous_representative_spec(var, spec_val, data_nt; weights)
            end
        end
    end
    
    # Hierarchical specification: :region => :education  
    if spec isa Pair
        outer_spec = spec.first
        inner_spec = spec.second
        return _create_hierarchical_reference_specs(outer_spec, inner_spec, data_nt; weights)
    end
    
    error("Invalid reference grid specification. Supported syntax: Symbol, Vector{Symbol}, (Symbol, representative_type), outer => inner, or (Symbol, mixture_spec)")
end

"""
    _parse_categorical_reference_spec(vars, data_nt) -> Vector{Dict}

Parse categorical reference specification for cross-tabulation.
Returns all combinations of categorical levels or mixture specifications.

Supports:
- Regular categorical variables: all observed values, in level order (`_ordered_values`: categorical
  level order, otherwise ascending); several variables are crossed with the first varying slowest
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
        
        # Observed levels in level order (no mixture support here - use hierarchical specs for mixtures)
        levels = _ordered_values(col)
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

# The one quantile convention of the grid builders: `n` quantiles are the `n − 1` interior cut
# points that divide the distribution into `n` equal-probability groups, probabilities `i/n` for
# `i = 1, …, n − 1` (`quantile_grid(data; x = n)`, `hierarchical_grid`'s `(:x, :quartiles)` and
# `(:x, (:quantiles, n))`). The maximum (probability 1) is not a cut point.
_cut_probabilities(n::Integer) = [i / n for i in 1:(n - 1)]

# Named quantile specifications of `hierarchical_grid` and their number of groups
const _NAMED_QUANTILES = Dict{Symbol,Int}(:terciles => 3, :tertiles => 3, :quartiles => 4,
                                          :quintiles => 5, :deciles => 10)

"""
    _parse_continuous_representative_spec(var, rep_spec, data_nt) -> Vector{Dict}

Parse continuous variable representative specification.
Computes representative values (quartiles, mean, median, etc.) from data.
"""
function _parse_continuous_representative_spec(var, rep_spec, data_nt; weights=nothing)
    if !haskey(data_nt, var)
        error("Variable $var not found in data")
    end
    
    col = data_nt[var]
    if !_is_continuous_variable(col)
        error("Variable $var is not continuous but used with representative specification $rep_spec")
    end
    
    representatives = []
    # Quantiles (weighted with `weights`: `_typical_quantiles`)
    q(ps) = _typical_quantiles(col, ps; weights)
    
    if haskey(_NAMED_QUANTILES, rep_spec)
        # the n − 1 interior cut points, as `quantile_grid(data; x = n)`: :quartiles → 0.25, 0.5, 0.75
        representatives = [Dict(var => val) for val in q(_cut_probabilities(_NAMED_QUANTILES[rep_spec]))]
    elseif rep_spec === :mean
        representatives = [Dict(var => typical_value(col, mean; weights))]
    elseif rep_spec === :median
        representatives = [Dict(var => typical_value(col, median; weights))]
    elseif rep_spec === :min
        representatives = [Dict(var => minimum(col))]
    elseif rep_spec === :max
        representatives = [Dict(var => maximum(col))]
    elseif rep_spec isa Vector{<:Real} && length(rep_spec) > 0 && all(0 ≤ q ≤ 1 for q in rep_spec)
        # Custom percentiles (values between 0 and 1)
        percentile_values = q(rep_spec)
        representatives = [Dict(var => val) for val in percentile_values]
    elseif rep_spec isa Vector{<:Real}
        # Fixed values specified
        representatives = [Dict(var => val) for val in rep_spec]
    elseif rep_spec isa Tuple && length(rep_spec) == 2 && rep_spec[1] === :percentiles
        # Custom percentiles specification: (:percentiles, [0.1, 0.5, 0.9])
        percentiles = rep_spec[2]
        if percentiles isa Vector{<:Real} && all(0 ≤ q ≤ 1 for q in percentiles)
            percentile_values = q(percentiles)
            representatives = [Dict(var => val) for val in percentile_values]
        else
            error("Percentiles must be a vector of values between 0 and 1: $percentiles")
        end
    elseif rep_spec isa Tuple && length(rep_spec) == 2 && rep_spec[1] === :quantiles
        # N-quantiles specification: (:quantiles, 7), the 6 cut points of the septiles
        n_quantiles = rep_spec[2]
        if n_quantiles isa Integer && n_quantiles > 1
            quantile_values = q(_cut_probabilities(n_quantiles))
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
function _parse_deep_hierarchical_spec(spec, data_nt, max_depth; weights=nothing)
    # Estimate total grid size before parsing
    estimated_size = _estimate_grid_size(spec, data_nt)
    _validate_grid_size(estimated_size)
    
    # Parse with depth tracking
    return _parse_hierarchical_recursive(spec, data_nt, 1, max_depth; weights)
end

"""
    _parse_hierarchical_recursive(spec, data_nt, current_depth, max_depth) -> Vector{Dict}

Recursively parse hierarchical specifications with depth tracking.
"""
function _parse_hierarchical_recursive(spec, data_nt, current_depth, max_depth; weights=nothing)
    if current_depth > max_depth
        error("Maximum nesting depth ($max_depth) exceeded at level $current_depth")
    end
    
    if spec isa Pair
        outer_spec, inner_spec = spec.first, spec.second
        
        # Get outer combinations
        outer_combinations = _parse_reference_grid_specification(outer_spec, data_nt; weights)
        
        # Check if inner_spec contains more nesting
        if inner_spec isa Pair
            # Recursive case: more nesting levels
            result_combinations = []
            for outer_combo in outer_combinations
                # Filter data to this outer group
                (filtered_data, filtered_w) = _filter_data_to_group(data_nt, outer_combo; weights)
                
                # Recursively parse inner specifications
                inner_combinations = _parse_hierarchical_recursive(
                    inner_spec, filtered_data, current_depth + 1, max_depth; weights=filtered_w
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
            return _create_hierarchical_reference_specs(outer_spec, inner_spec, data_nt; weights)
        end
    else
        # Not hierarchical - use standard parsing
        return _parse_reference_grid_specification(spec, data_nt; weights)
    end
end

"""
    _filter_data_to_group(data_nt, group_spec; weights=nothing) -> (NamedTuple, weights)

Filter data (and the row weights, if any) to a specific group defined by group_spec for
recursive parsing.
"""
function _filter_data_to_group(data_nt, group_spec; weights=nothing)
    # Get indices of observations that match the group specification
    group_indices = _get_group_indices(group_spec, data_nt)
    
    if isempty(group_indices)
        @warn "Empty group found for specification $group_spec - using full data"
        return (data_nt, weights)
    end
    
    # Create filtered data for this group
    filtered_data = map(col -> col[group_indices], data_nt)
    return (filtered_data, isnothing(weights) ? nothing : weights[group_indices])
end

"""
    _create_hierarchical_reference_specs(outer_spec, inner_spec, data_nt) -> Vector{Dict}

Create hierarchical reference specifications using => syntax for reference grids.
Unlike group filtering, this computes representative values within each outer group.
"""
function _create_hierarchical_reference_specs(outer_spec, inner_spec, data_nt; weights=nothing)
    # Parse outer specification
    outer_specs = _parse_reference_grid_specification(outer_spec, data_nt; weights)
    
    # Handle inner specification - can be single spec or Vector of specs
    if inner_spec isa AbstractVector
        # Multiple inner specifications - create parallel representatives within each outer
        hierarchical_specs = []
        for outer_spec_dict in outer_specs
            for inner_single_spec in inner_spec
                inner_specs = _compute_representative_values_within_group(inner_single_spec, outer_spec_dict, data_nt; weights)
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
            inner_specs = _compute_representative_values_within_group(inner_spec, outer_spec_dict, data_nt; weights)
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
function _compute_representative_values_within_group(spec, group_spec, data_nt; weights=nothing)
    # Filter data to the specified group
    group_indices = _get_group_indices(group_spec, data_nt)
    
    if isempty(group_indices)
        @warn "Empty group found for specification $group_spec - skipping"
        return []
    end
    
    # Create filtered data (and weights) for this group
    filtered_data = map(col -> col[group_indices], data_nt)
    filtered_w = isnothing(weights) ? nothing : weights[group_indices]
    
    # Parse the inner specification using the filtered group data
    return _parse_reference_grid_specification(spec, filtered_data; weights=filtered_w)
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
        elseif haskey(_NAMED_QUANTILES, rep_spec)
            return _NAMED_QUANTILES[rep_spec] - 1
        elseif rep_spec isa Vector
            return length(rep_spec)
        elseif rep_spec === :mean || rep_spec === :median || rep_spec === :min || rep_spec === :max
            return 1
        elseif rep_spec isa Tuple && length(rep_spec) == 2
            spec_type, spec_value = rep_spec
            if spec_type === :percentiles && spec_value isa Vector
                return length(spec_value)
            elseif spec_type === :quantiles && spec_value isa Integer
                return spec_value - 1
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
function _parse_mixture_reference_spec(var, mixture_spec, data_nt; weights=nothing)
    if !haskey(data_nt, var)
        error("Variable $var not found in data")
    end
    
    col = data_nt[var]
    
    # Validate this is a categorical variable
    if _is_continuous_variable(col)
        error("Variable $var is continuous but used with mixture specification. Mixtures are only supported for categorical variables.")
    end
    
    if mixture_spec === :mix_proportional
        # The (weighted) frequency mixture, as for a typical value
        mixture = typical_value(col; weights)
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
            # A frequency mixture (`:mix_proportional`, a `CategoricalMixture`) is a weighted
            # combination of levels, not a level, so no row "matches" it: leave it out
            categorical_parts = Dict(var => val for (var, val) in combo
                                     if haskey(data_nt, var) && !_is_continuous_variable(data_nt[var]) &&
                                        !(val isa CategoricalMixture))
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
    _build_hierarchical_reference_grid(grid_combinations, data_nt; weights=nothing) -> DataFrame

Build reference grid DataFrame from parsed specifications.
Converts vector of Dict specifications to a proper DataFrame structure.
Uses typical values (weighted with `weights`) instead of missing for unspecified variables.
Columns are in the data's column order (any specified variable not in the data follows);
rows are in the order of `grid_combinations`.
"""
function _build_hierarchical_reference_grid(grid_combinations, data_nt; weights=nothing)
    if isempty(grid_combinations)
        return DataFrame()
    end
    
    # All data variables (filled with typical values where a combination does not set them),
    # in the data's column order, then any other variable a combination sets
    all_keys = collect(Symbol, keys(data_nt))
    for combo in grid_combinations, k in keys(combo)
        k in all_keys || push!(all_keys, k)
    end
    # (`AbstractVector`, so a categorical column is not converted to a plain `Vector` on assignment)
    cols = Dict{Symbol, AbstractVector}(key => Any[] for key in all_keys)
    
    # Positions filled with a typical value (rather than specified by a combination)
    typical_mask = Dict{Symbol, BitVector}(key => falses(length(grid_combinations)) for key in all_keys)
    # Each column's typical value, computed once (on the whole data, weighted with `weights`)
    typical_cache = Dict{Symbol, Any}()

    # Fill in values - Phase 1.2: Use typical values instead of missing
    for (row, combo) in enumerate(grid_combinations)
        for key in all_keys
            if haskey(combo, key)
                push!(cols[key], combo[key])
            else
                typical_mask[key][row] = true
                # For missing keys, fill with typical values from data
                if !haskey(data_nt, key)
                    error("Variable $key not found in data when computing typical value")
                end
                tv = get!(() -> typical_value(data_nt[key], mean; weights), typical_cache, key)
                push!(cols[key], tv)
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
        elseif all(val -> val isa CategoricalValue, non_missing_values)
            # Levels of a categorical column (possibly with missing): a categorical column with
            # the data column's levels and ordering, not their strings (checked before the
            # string case: a `CategoricalValue{String}` is an `AbstractString`)
            first_val = first(non_missing_values)
            cols[key] = categorical([ismissing(val) ? missing : unwrap(val) for val in values];
                                    levels=levels(first_val), ordered=isordered(first_val))
        elseif all(val -> val isa AbstractString, non_missing_values)
            # String values (possibly with missing)
            if has_missing
                cols[key] = Vector{Union{Missing, String}}([ismissing(val) ? missing : string(val) for val in values])
            else
                cols[key] = String[string(val) for val in values]
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
    
    df = DataFrame([key => cols[key] for key in all_keys])
    # Record which cells hold typical values: columns no combination specifies are marked
    # whole; columns some combinations specify and others leave to the typical value (e.g.
    # `:g => [(:k, [1, 3]), (:x, [0, 1])]`) are marked per row.
    for (k, mask) in typical_mask
        c = string(k)
        c in names(df) || continue
        if all(mask)
            _mark_typical_values!(df, [c])
        elseif any(mask)
            _mark_typical_value_rows!(df, c, mask)
        end
    end
    return df
end

"""
    hierarchical_grid(data, spec; max_depth=5, warn_large=true, weights=nothing) -> DataFrame

Enhanced hierarchical reference grid builder with deep nesting support.
Uses group nesting grammar for systematic reference grid construction with
support for 3+ level hierarchical specifications.

# Arguments
- `data`: DataFrame or NamedTuple containing the data
- `spec`: Hierarchical specification using `=>` operator grammar
- `max_depth::Int=5`: Maximum allowed nesting depth
- `warn_large::Bool=true`: Show warnings for large grids
- `weights=nothing`: weights for the typical values: a column name (`weights=:w`) or a vector with
  one weight per row, with the same checks as `population_margins`' `weights=` (real, finite,
  non-negative, not all zero). `nothing` or `:none`: unweighted. A `Symbol` names a column of
  `data`; a vector is used as given (it is not looked up in `data`). Rows with weight 0 are
  dropped. With `weights`, `(:x, :mean)` / `(:x, :median)` are the weighted mean / median,
  `:mix_proportional` uses weighted shares, the quantile specifications (`:quartiles`,
  `(:percentiles, …)`, `(:quantiles, n)`, a vector of probabilities, …) are weighted quantiles
  (as for `quantile_grid`), and the variables a combination leaves out are filled with weighted
  typical values (see `means_grid`). Within a group (`outer => inner`), the group's rows and
  weights are used.

# Order
Columns follow the data's column order. Rows follow the specification: a categorical variable
takes its observed levels in level order (`levels(col)`; ascending for Bool and string columns),
a vector of specifications is crossed with the first varying slowest, and `outer => inner` lists
each outer value's inner rows together.

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

Named quantiles (`:terciles`, `:quartiles`, `:quintiles`, `:deciles`) and `(:quantiles, n)` follow
`quantile_grid(data; x = n)`: the n − 1 interior cut points, probabilities i/n for i = 1, …, n − 1,
so `(:x, :quartiles)` and `quantile_grid(data; x = 4)` give the same three values. (Before 3.0 they
gave n values including the maximum, probability 1.0.)

## Basic References
- `:education` - All observed levels of categorical variable
- `[:region, :education]` - Cross-tabulation of categorical variables
- `(:income, :quartiles)` - Income quartiles: the 3 cut points at probabilities 0.25, 0.5, 0.75
- `(:income, :quintiles)` - Income quintiles: the 4 cut points 0.2, 0.4, 0.6, 0.8
- `(:income, :deciles)` - Income deciles: the 9 cut points 0.1, …, 0.9
- `(:income, :terciles)` (or `:tertiles`) - Income terciles: the 2 cut points 1/3, 2/3
- `(:age, :mean)` - Overall mean age
- `(:age, :median)` - Overall median age
- `(:age, :min)` - Minimum age
- `(:age, :max)` - Maximum age
- `(:age, [30, 50, 70])` - Fixed representative ages
- `(:income, [0.1, 0.5, 0.9])` - Custom percentiles (10th, 50th, 90th), as given (0 and 1 give the min and max)
- `(:income, (:percentiles, [0.25, 0.75]))` - Explicit percentiles specification
- `(:income, (:quantiles, 7))` - n-quantiles: the n − 1 cut points i/n, i = 1, …, n − 1 (septiles: 6 values)
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
function hierarchical_grid(data, spec; max_depth=5, warn_large=true, weights=nothing)
    data_nt = _narrow_missing_columns(data isa NamedTuple ? data : Tables.columntable(data))
    # Weighted: the positively weighted rows only, and their weights
    (data_nt, w) = _drop_zero_weight_rows(data_nt, _grid_weights(weights, data, data_nt, "hierarchical_grid"))
    
    # Enhanced validation with depth checking
    _validate_reference_specification(spec, data_nt, max_depth)
    
    # Determine which parser to use based on nesting depth
    actual_depth = _validate_nesting_depth(spec, max_depth)
    
    if actual_depth <= 2
        # Use optimized 2-level path for performance
        parsed_specs = _parse_reference_grid_specification(spec, data_nt; weights=w)
    else
        # Use new deep nesting parser
        parsed_specs = _parse_deep_hierarchical_spec(spec, data_nt, max_depth; weights=w)
    end
    
    # Grid size validation and warnings
    if warn_large
        _validate_grid_combinations(parsed_specs, data_nt)
    end
    
    # Build the reference grid DataFrame
    return _build_hierarchical_reference_grid(parsed_specs, data_nt; weights=w)
end

# =============================================================================
# Reference Grid Builder Functions (NEW - AsBalanced Support)  
# =============================================================================

const _TYPICAL_VALUE_KEY = "Margins.typical_value"

"""
    _mark_typical_values!(df, cols) -> df

Record (as DataFrames column metadata) that `cols` of a reference grid hold typical values
computed from the data alone, rather than values the user chose. `complete_reference_grid`
may replace such a column — and only such a column — when the model treats the variable as
categorical and the typical value is not one of its levels (e.g. the mean of an integer code).
"""
function _mark_typical_values!(df::DataFrame, cols)
    for c in cols
        colmetadata!(df, c, _TYPICAL_VALUE_KEY, true; style=:note)
    end
    return df
end

_is_typical_value_column(df::DataFrame, col) = colmetadata(df, col, _TYPICAL_VALUE_KEY, false) === true

const _TYPICAL_VALUE_ROWS_KEY = "Margins.typical_value_rows"

"""
    _mark_typical_value_rows!(df, col, mask) -> df

Record that the cells `mask` (a `BitVector`, one entry per row) of column `col` hold typical
values computed from the data, while the other cells hold values the user specified. Used
by `hierarchical_grid` for columns only some combinations specify.

Stored as `:default`-style column metadata, which DataFrames keeps through `copy` and
`DataFrame(df)` but drops on every operation that can move, add, remove or overwrite rows
or cells (sorting, subsetting, `vcat`, `push!`, assignment). A grid edited that way is
therefore treated as fully user-specified (its non-level values are rejected) rather than
having the wrong cells replaced. The mask's length is checked against the grid as well.
"""
function _mark_typical_value_rows!(df::DataFrame, col, mask::AbstractVector{Bool})
    length(mask) == nrow(df) || throw(ArgumentError("typical-value mask length $(length(mask)) ≠ nrow $(nrow(df))"))
    colmetadata!(df, col, _TYPICAL_VALUE_ROWS_KEY, BitVector(mask); style=:default)
    return df
end

"""
    _typical_value_rows(df, col) -> Union{BitVector, Nothing}

The rows of `col` that hold typical values: all rows of a column marked whole by
`_mark_typical_values!`, the recorded rows of a column marked by `_mark_typical_value_rows!`,
or `nothing` when no cell is marked (every value was specified by the user).
"""
function _typical_value_rows(df::DataFrame, col)
    _is_typical_value_column(df, col) && return trues(nrow(df))
    mask = colmetadata(df, col, _TYPICAL_VALUE_ROWS_KEY, nothing)
    (mask isa AbstractVector{Bool} && length(mask) == nrow(df)) || return nothing
    return BitVector(mask)
end

"""
    means_grid(data; typical=mean, weights=nothing) -> DataFrame

A one-row reference grid with every column of `data` at its typical value: the mean (or
`typical`) of numeric columns, the share of `true` for `Bool` columns, and the frequency mixture
of categorical and string columns. This is the standard "at means" grid (MEM).

# Arguments
- `data`: DataFrame or any Tables.jl table
- `typical`: function giving the typical value of a numeric column (default `mean`)
- `weights=nothing`: weights for the typical values: a column name (`weights=:w`) or a vector with
  one weight per row, with the same checks as `population_margins`' `weights=` (real, finite,
  non-negative, not all zero). `nothing` or `:none`: unweighted. A `Symbol` names a column of
  `data`; a vector is used as given (it is not looked up in `data`). Rows with weight 0 are
  dropped. See *Typical values* below.

# Typical values
Numeric columns get `typical` (default `mean`; `median` also has a weighted version), `Bool`
columns the share of `true`, and categorical and string columns a frequency mixture of their
observed values. With `weights`, these are the weighted mean or median and weighted shares
(`Margins.typical_value` defines them; the weighted median is the smallest value whose cumulative
weight reaches half the total, the midpoint of two values on an exact tie, so equal weights give
`median`). A custom `typical` function cannot be weighted: `weights` with a `typical` other than
`mean`/`median` is an `ArgumentError`.

Columns follow the data's column order.

# Examples
```julia
ref_grid = means_grid(data)                     # unweighted means and shares
result = profile_margins(model, data, ref_grid)

ref_grid = means_grid(data; typical=median)     # medians for numeric columns
ref_grid = means_grid(data; weights=:w)         # weighted means and shares (e.g. survey weights)
```

`profile_margins` evaluates the grid it is given: for a MEM at weighted means, build the grid
with `weights` (`profile_margins` has no `weights` keyword).
"""
function means_grid(data; typical=mean, weights=nothing)
    data_nt = _narrow_missing_columns(data isa NamedTuple ? data : Tables.columntable(data))
    w = _grid_weights(weights, data, data_nt, "means_grid")
    
    # Typical values (frequency mixtures for categorical columns), weighted with `w`
    typical_values = get_typical_values(data_nt; typical=typical, weights=w)
    
    # One column per data column, in the data's column order
    df = DataFrame([k => _typical_grid_column(typical_values[k]) for k in keys(data_nt)])
    return _mark_typical_values!(df, names(df))
end

"""
    balanced_grid(data; typical=mean, weights=nothing, vars...) -> DataFrame

Build reference grid with balanced (equal weight) mixtures for specified categorical variables.
This creates orthogonal factorial designs for AsBalanced analysis.

Continuous variables use sample means. Categorical variables specified in `vars` get
equal probability weights. Other categorical variables use frequency-weighted mixtures.

# Arguments
- `data`: DataFrame or NamedTuple containing the data
- `typical`: function giving the typical value of a numeric column (default `mean`)
- `weights=nothing`: weights for the typical values: a column name (`weights=:w`) or a vector with
  one weight per row, with the same checks as `population_margins`' `weights=` (real, finite,
  non-negative, not all zero). `nothing` or `:none`: unweighted. A `Symbol` names a column of
  `data`; a vector is used as given (it is not looked up in `data`). Rows with weight 0 are
  dropped. `weights` applies to the typical values of the columns that are not balanced (as in
  `means_grid`); the balanced mixtures keep equal shares. (`typical` and `weights` are keywords
  of `balanced_grid`, so a data column with one of these names cannot be balanced.)
- `vars...`: Keyword arguments specifying which variables to balance
  - Use `:all` to balance all categorical variables: `education=:all`
  - Use specific levels: `education=["high_school", "college"]`

Columns follow the data's column order.

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
function balanced_grid(data; typical=mean, weights=nothing, vars...)
    data_nt = _narrow_missing_columns(data isa NamedTuple ? data : Tables.columntable(data))
    # Weighted: the positively weighted rows only (so a value seen only in zero-weight rows is not
    # a level to balance over), and their weights
    (data_nt, w) = _drop_zero_weight_rows(data_nt, _grid_weights(weights, data, data_nt, "balanced_grid"))

    # Start with frequency-weighted typical values
    typical_values = get_typical_values(data_nt; typical=typical, weights=w)
    
    # Override specified variables with balanced mixtures
    for (var, spec) in vars
        if !haskey(data_nt, var)
            throw(ArgumentError("Variable :$var not found in data. Available variables: $(collect(keys(data_nt)))"))
        end
        
        col = getproperty(data_nt, var)
        
        if spec === :all
            # Balance all levels of this categorical variable
            lvls = if col isa CategoricalArray
                string.(CategoricalArrays.levels(col))
            else
                string.(unique(col))
            end
            equal_weights = fill(1.0/length(lvls), length(lvls))
            typical_values[var] = CategoricalMixture(lvls, equal_weights)
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
    
    # One column per data column, in the data's column order
    df = DataFrame([k => _typical_grid_column(typical_values[k]) for k in keys(data_nt)])
    balanced = Set(Symbol(k) for (k, _) in vars)
    return _mark_typical_values!(df, [c for c in names(df) if !(Symbol(c) in balanced)])
end

"""
    cartesian_grid(; vars...) -> DataFrame
    cartesian_grid(spec::Dict) -> DataFrame

Build Cartesian product reference grid from variable specifications.
This is a pure grid constructor that creates combinations from provided values
without needing reference data. When used with `profile_margins()`, missing
model variables are automatically completed with (unweighted) typical values; for weighted ones,
complete the grid first with `complete_reference_grid(grid, model, data; weights=…)`.

# Order
Columns follow the order of the keywords; rows are nested with the first variable varying slowest
and the last fastest (`cartesian_grid(a=[1, 2], b=[3, 4])` has rows `(1,3), (1,4), (2,3), (2,4)`),
each variable's values in the order given. A `Dict` has no order, so its variables are taken in
alphabetical order; use keywords to choose the order.

# Arguments
- `vars...`: Keyword arguments mapping variables to values
  - Single values: `age=45`
  - Multiple values: `age=[25, 45, 65]` (any vector, e.g. a range, is a list of values)
  - Numlist notation: `age="25(10)65"` (25, 35, 45, 55, 65)
- `spec::Dict`: Dictionary mapping variable symbols to values
  - Example: `Dict(:age => [25, 45, 65], :treatment => [true, false])`

# Returns
- `DataFrame`: Reference grid with Cartesian product of specified values

# Examples
```julia
# Simple cartesian product (keyword syntax)
ref_grid = cartesian_grid(age=[25, 45, 65], treatment=[true, false])

# Dict syntax (equivalent)
ref_grid = cartesian_grid(Dict(:age => [25, 45, 65], :treatment => [true, false]))

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
    return _cartesian_grid_impl([Symbol(k) => v for (k, v) in vars])
end

# A Dict has no order of its own: its variables are taken in sorted (alphabetical) order
function cartesian_grid(spec::AbstractDict)
    return _cartesian_grid_impl(sort!([Symbol(k) => v for (k, v) in spec]; by=p -> String(first(p))))
end

# Internal implementation shared by both methods: `spec` is an ordered vector of `var => values`.
# Columns in that order; rows nested with the first variable varying slowest (`_nested_product`).
function _cartesian_grid_impl(spec::Vector{<:Pair{Symbol}})
    isempty(spec) && return DataFrame()
    combos = _nested_product([_grid_values(v) for (_, v) in spec])
    isempty(combos) && return DataFrame()
    return DataFrame([var => _typed_grid_column([c[i] for c in combos]) for (i, (var, _)) in enumerate(spec)])
end

"""
    complete_reference_grid(reference_grid, model, data; typical=mean, weights=nothing) -> DataFrame

Complete a partial reference grid by adding typical values for missing model variables.
This function takes a reference grid that may be missing some model variables and
adds appropriate typical values for those missing variables. `profile_margins` does this itself,
unweighted, for a grid that leaves model variables out; call it yourself for weighted fill-ins.

# Arguments
- `reference_grid`: DataFrame with partial reference grid (e.g., from cartesian_grid)
- `model`: Fitted statistical model
- `data`: Original data (DataFrame or NamedTuple) to compute typical values
- `typical`: Function to compute typical values for continuous variables (default: mean)
- `weights=nothing`: weights for the typical values: a column name (`weights=:w`) or a vector with
  one weight per row, with the same checks as `population_margins`' `weights=` (real, finite,
  non-negative, not all zero). `nothing` or `:none`: unweighted. A `Symbol` names a column of
  `data`; a vector is used as given (it is not looked up in `data`). Rows with weight 0 are
  dropped. `weights=:model` uses the model's estimation weights (as `population_margins`' default;
  unweighted for an unweighted model). The weights apply to the filled-in values and to the
  frequency mixture that replaces a typical-value cell of a variable the model treats as
  categorical (see `means_grid`, *Typical values*).

Columns: the grid's model-variable columns in the grid's order, then the filled-in variables in
the data's column order. Rows: those of `reference_grid`.

# Returns
- `DataFrame`: Complete reference grid with all model variables

# Examples
```julia
# Create partial grid with only x1 and x2
partial_grid = cartesian_grid(x1=[0, 1, 2], x2=[10, 20])

# Complete it with typical values for other model variables
complete_grid = complete_reference_grid(partial_grid, model, data)

# ... with weighted typical values (the model's estimation weights, or a column)
complete_grid = complete_reference_grid(partial_grid, model, data; weights=:model)

# Use with profile_margins
result = profile_margins(model, data, complete_grid)
```
"""
function complete_reference_grid(reference_grid::DataFrame, model, data; typical=mean, weights=nothing)
    # Convert data to NamedTuple for consistency
    data_nt = _narrow_missing_columns(data isa NamedTuple ? data : Tables.columntable(data))
    # Weights for the typical values (`nothing`: unweighted, the default); `:model` reads the model's
    w = _grid_weights(weights, data, data_nt, "complete_reference_grid"; model)
    isnothing(w) || _check_weighted_typical(typical)
    
    # Columns the MODEL treats as categorical (e.g. integer codes fitted with
    # `contrasts = Dict(:k => DummyCoding())`) become CategoricalArrays, so their typical
    # value is a frequency mixture over the model's levels rather than the mean code.
    data_nt = _categorize_model_categoricals(data_nt, model)

    # Filter data to model variables only
    filtered_data = _filter_data_to_model_variables(data_nt, model)
    
    # Typical values are needed only for model variables the grid does not specify;
    # values for grid columns would be discarded. Skipping them avoids O(n) passes
    # (including categorical frequency tables) over every grid-specified column.
    grid_syms = Set(Symbol.(names(reference_grid)))
    unspecified = Tuple(k for k in keys(filtered_data) if !(k in grid_syms))
    typical_values = get_typical_values(NamedTuple{unspecified}(map(k -> filtered_data[k], unspecified));
                                        typical=typical, weights=w)

    # Create complete reference grid
    n_rows = nrow(reference_grid)
    # NOTE: must be AbstractVector, not Vector. A CategoricalVector is not a `Vector`
    # subtype, so assigning one into a `Dict{Symbol,Vector}` forces convert(Vector, ::CategoricalVector).
    # Under CategoricalArrays >= 1.0 that conversion returns a plain Array of unwrapped values
    # (PR #420), silently stripping the categorical pool/levels and collapsing non-baseline
    # levels to the baseline downstream. Using AbstractVector preserves the categorical column.
    complete_cols = Pair{Symbol, AbstractVector}[]

    # First, add only MODEL columns from the original reference grid
    # This prevents non-model variables from leaking through
    for col_name in names(reference_grid)
        col_sym = Symbol(col_name)
        if haskey(filtered_data, col_sym)  # Only include if it's a model variable
            gcol = reference_grid[!, col_name]
            dcol = filtered_data[col_sym]
            # Numeric grid values (codes, Bools) for a variable the model treats as
            # categorical must be model levels.
            if dcol isa CategoricalArray && eltype(gcol) <: Real
                lv = CategoricalArrays.unwrap.(levels(dcol))
                is_level(v) = any(l -> isequal(l, v) || (l isa Real && l == v), lv)
                # Cells a data-only builder (`means_grid`, `balanced_grid`,
                # `hierarchical_grid`, ...) filled with a typical value (e.g. the mean code
                # k = 1.94) without knowing the model treats the variable as categorical. They
                # take the frequency mixture over the model's levels, as for any categorical
                # variable. Every other cell was specified by the user and must be a level.
                trows = _typical_value_rows(reference_grid, col_name)
                is_typical(i) = !isnothing(trows) && trows[i]
                bad = unique([gcol[i] for i in eachindex(gcol) if !is_typical(i) && !is_level(gcol[i])])
                if !isempty(bad)
                    throw(MarginsError(
                        "Reference grid values $(bad) for :$col_sym are not levels of " *
                        "this variable, which the fitted model treats as categorical. " *
                        "Valid levels: $(lv). (Use a mixture, e.g. mix(...), for a " *
                        "weighted combination of levels.)"))
                end
                if !isnothing(trows) && any(trows)
                    mixture = typical_value(dcol; weights=w)
                    @debug "Variable $col_sym is categorical in the model; using its frequency mixture instead of the typical value(s) $(unique(gcol[trows]))."
                    # A column that mixes typical and user-specified cells holds the
                    # mixture in some rows and levels in others; `profile_margins` evaluates
                    # such grids in homogeneous row blocks (`_heterogeneous_grid_blocks`).
                    gcol = all(trows) ? fill(mixture, length(gcol)) :
                        Any[trows[i] ? mixture : gcol[i] for i in eachindex(gcol)]
                end
            end
            push!(complete_cols, col_sym => gcol)
        end
    end

    # Then, add typical values for missing model variables, in the data's column order
    for var in unspecified
        # This variable is missing from reference grid: set it to its typical value
        # (a real number as Float64; a Bool, CategoricalMixture or level as is)
        push!(complete_cols, var => _counterfactual_column(SetTo(typical_values[var]), n_rows))
    end
    
    return DataFrame(complete_cols)
end

"""
    quantile_grid(data; weights=nothing, vars...) -> DataFrame

Build reference grid using quantiles of continuous variables. Every other column of `data` is
at its typical value (as in `means_grid`).

# Arguments
- `data`: DataFrame or NamedTuple containing the data  
- `weights=nothing`: weights for the typical values: a column name (`weights=:w`) or a vector with
  one weight per row, with the same checks as `population_margins`' `weights=` (real, finite,
  non-negative, not all zero). `nothing` or `:none`: unweighted. A `Symbol` names a column of
  `data`; a vector is used as given (it is not looked up in `data`). Rows with weight 0 are
  dropped. With `weights`, the quantiles are weighted quantiles and the other columns weighted typical
  values. The weighted quantile is the weighted analogue of `Statistics.quantile`'s (type 7)
  that `groups=(:x, n)` uses for weighted quantile bins: with equal weights it is
  `Statistics.quantile`, and it does not depend on the weights' scale. It interpolates, so at
  probability 0.5 it can differ from the weighted median of `means_grid(…; typical=median)`,
  which does not. (`weights` is a keyword, so a data column named `weights` cannot be given a
  quantile specification.)
- `vars...`: Keyword arguments mapping variables to quantile specifications
  - Vector of quantiles: `income=[0.1, 0.5, 0.9]` (10th, 50th, 90th percentiles)
  - Number of quantiles `n` (an integer > 1): the `n - 1` cut points that divide the distribution
    into `n` equal-probability groups, probabilities `i/n` for `i = 1, …, n-1`: `age=4` gives the
    quartiles (25th, 50th, 75th percentiles), `income=5` the quintiles (20th, …, 80th), `x=2` the
    median. (Before 3.0, `n` gave `n` values at probabilities `i/(n+1)`.)

Columns follow the data's column order; rows are nested with the first keyword varying slowest
(as in `cartesian_grid`).

# Returns
- `DataFrame`: Reference grid with quantile-based values

# Examples
```julia
# Specific quantiles
ref_grid = quantile_grid(data; income=[0.1, 0.5, 0.9])

# Standard quantiles: age quartiles (3 values) × income quintiles (4 values) = 12 rows
ref_grid = quantile_grid(data; age=4, income=5)

# Use with profile_margins
result = profile_margins(model, data, ref_grid)
```
"""
function quantile_grid(data; weights=nothing, vars...)
    data_nt = _narrow_missing_columns(data isa NamedTuple ? data : Tables.columntable(data))
    w = _grid_weights(weights, data, data_nt, "quantile_grid")
    
    # Convert quantile specifications to actual values (in the order of the keywords)
    at_spec = Pair{Symbol, Any}[]
    for (var, spec) in vars
        if !haskey(data_nt, var)
            throw(ArgumentError("Variable :$var not found in data"))
        end
        
        col = getproperty(data_nt, var)
        if !_is_continuous_variable(col)
            throw(ArgumentError("quantile_grid() only supports continuous variables. Variable :$var is $(eltype(col))"))
        end
        
        if spec isa Integer && spec > 1
            # n-quantiles: the n - 1 cut points that divide the distribution into n groups of
            # equal probability (4 → quartiles 0.25, 0.5, 0.75; 2 → the median)
            push!(at_spec, var => _typical_quantiles(col, _cut_probabilities(Int(spec)); weights=w))
        elseif spec isa Vector{<:Real} && all(0 .≤ spec .≤ 1)
            # Explicit quantiles
            push!(at_spec, var => _typical_quantiles(col, spec; weights=w))
        else
            throw(ArgumentError("Quantile specification for :$var must be an integer number of quantiles n > 1 (the n - 1 cut points, e.g. 4 for quartiles) or a vector of probabilities in [0, 1]"))
        end
    end
    
    return _build_cartesian_refgrid(at_spec, data_nt, mean; weights=w)
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

# Any other Tables.jl source (a NamedTuple of columns, ...): the same, on its columns
process_reference_grid(data, grid::DataFrame) =
    process_reference_grid(DataFrame(Tables.columntable(data); copycols=false), grid)
