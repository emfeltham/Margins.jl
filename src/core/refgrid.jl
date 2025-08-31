# refgrid.jl - Phase 3 Direct Iterator Builders

"""
Phase 3: Direct Iterator Builders for Unified Profile API

All builders return `Iterator{Dict{Symbol,Any}}` for direct consumption by `_profile_margins_core()`.
This eliminates the DataFrame materialization step and provides a composable, type-safe API.
"""

# ========================================================================================
# Core Builder Functions
# ========================================================================================

"""
    refgrid_means(data_nt::NamedTuple; vars=:all, over=nothing)

Create iterator yielding a single profile with means for continuous variables and first levels for categoricals.

# Arguments
- `data_nt`: Data as NamedTuple
- `vars`: Variables to include (`:all` for all, or collection of symbols)
- `over`: Grouping variables (optional)

# Returns
`Iterator{Dict{Symbol,Any}}` yielding profile dictionaries
"""
function refgrid_means(data_nt::NamedTuple; vars=:all, over=nothing)
    # Build single means profile
    prof = Dict{Symbol,Any}()
    included_vars = vars === :all ? keys(data_nt) : vars
    
    for k in included_vars
        col = getproperty(data_nt, k)
        if eltype(col) <: Real && !(eltype(col) <: Bool)
            prof[k] = mean(col)
        elseif Base.find_package("CategoricalArrays") !== nothing && (col isa CategoricalArrays.CategoricalArray)
            prof[k] = levels(col)[1]
        elseif eltype(col) <: Bool
            prof[k] = false
        elseif eltype(col) <: AbstractString
            prof[k] = first(unique(col))
        else
            prof[k] = first(col)
        end
    end
    
    if over === nothing
        # Single profile
        return (prof for _ in 1:1)
    else
        # Grouped means - one profile per group
        return _apply_grouping((prof for _ in 1:1), data_nt, over)
    end
end

"""
    refgrid_cartesian(spec::NamedTuple, data_nt::NamedTuple; over=nothing)

Create iterator yielding Cartesian product profiles from specification.

# Arguments
- `spec`: NamedTuple specification like `(x=[1,2,3], z=["A","B"])`
- `data_nt`: Data as NamedTuple  
- `over`: Grouping variables (optional)

# Returns
`Iterator{Dict{Symbol,Any}}` yielding profile dictionaries in deterministic order
"""
function refgrid_cartesian(spec::NamedTuple, data_nt::NamedTuple; over=nothing)
    # Build expanded values for each key in NamedTuple order
    keys_vec = collect(keys(spec))
    vals_vec = [_expand_at_values(data_nt, k, getproperty(spec, k)) for k in keys_vec]
    
    # Create base iterator
    base_iter = (Dict{Symbol,Any}(k => combo[i] for (i, k) in enumerate(keys_vec)) 
                 for combo in Iterators.product(vals_vec...))
    
    if over === nothing
        return base_iter
    else
        return _apply_grouping(base_iter, data_nt, over)
    end
end

"""
    refgrid_sequence(focal_var::Symbol, values, data_nt::NamedTuple; others=:means, over=nothing)

Create iterator with focal variable varying and others fixed.

# Arguments
- `focal_var`: The variable to vary
- `values`: Values for the focal variable
- `data_nt`: Data as NamedTuple
- `others`: How to set other variables (`:means` or Dict of specific values)
- `over`: Grouping variables (optional)

# Returns
`Iterator{Dict{Symbol,Any}}` yielding profile dictionaries
"""
function refgrid_sequence(focal_var::Symbol, values, data_nt::NamedTuple; others=:means, over=nothing)
    # Build base profile with others fixed
    base_prof = Dict{Symbol,Any}()
    
    if others === :means
        for k in keys(data_nt)
            k === focal_var && continue
            col = getproperty(data_nt, k)
            if eltype(col) <: Real && !(eltype(col) <: Bool)
                base_prof[k] = mean(col)
            elseif Base.find_package("CategoricalArrays") !== nothing && (col isa CategoricalArrays.CategoricalArray)
                base_prof[k] = levels(col)[1]
            elseif eltype(col) <: Bool
                base_prof[k] = false
            elseif eltype(col) <: AbstractString
                base_prof[k] = first(unique(col))
            else
                base_prof[k] = first(col)
            end
        end
    elseif others isa Dict
        merge!(base_prof, others)
    else
        error("others must be :means or Dict")
    end
    
    # Create iterator varying focal variable
    expanded_values = collect(values)
    base_iter = (merge(base_prof, Dict{Symbol,Any}(focal_var => val)) for val in expanded_values)
    
    if over === nothing
        return base_iter
    else
        return _apply_grouping(base_iter, data_nt, over)
    end
end

"""
    refgrid_quantiles(data_nt::NamedTuple; specs, over=nothing)

Create iterator with variables at specified quantiles.

# Arguments
- `data_nt`: Data as NamedTuple
- `specs`: NamedTuple like `(income=[:p10,:p50,:p90], age=[:p25,:p75])`
- `over`: Grouping variables (optional)

# Returns
`Iterator{Dict{Symbol,Any}}` yielding profile dictionaries
"""
function refgrid_quantiles(data_nt::NamedTuple; specs, over=nothing)
    # Convert quantile specs to actual values
    expanded_spec = NamedTuple{keys(specs)}([_expand_at_values(data_nt, k, v) for (k, v) in pairs(specs)])
    
    # Delegate to cartesian
    return refgrid_cartesian(expanded_spec, data_nt; over=over)
end

"""
    refgrid_levels(data_nt::NamedTuple; var::Symbol, levels, others=:means, over=nothing)

Create iterator with categorical variable at specific levels, others fixed.

# Arguments
- `data_nt`: Data as NamedTuple
- `var`: Categorical variable to vary
- `levels`: Specific levels to use
- `others`: How to set other variables (`:means` or Dict)
- `over`: Grouping variables (optional)

# Returns
`Iterator{Dict{Symbol,Any}}` yielding profile dictionaries
"""
function refgrid_levels(data_nt::NamedTuple; var::Symbol, levels, others=:means, over=nothing)
    # Validate levels exist in data
    col = getproperty(data_nt, var)
    available_levels = if Base.find_package("CategoricalArrays") !== nothing && (col isa CategoricalArrays.CategoricalArray)
        CategoricalArrays.levels(col)
    else
        unique(col)
    end
    
    for level in levels
        if level ∉ available_levels
            error("Level '$level' not found in variable '$var'. Available levels: $available_levels")
        end
    end
    
    # Delegate to sequence
    return refgrid_sequence(var, levels, data_nt; others=others, over=over)
end

# ========================================================================================
# Grouping Support
# ========================================================================================

"""
    _apply_grouping(base_iter, data_nt::NamedTuple, over)

Apply grouping to a base profile iterator, yielding profiles with group metadata.
"""
function _apply_grouping(base_iter, data_nt::NamedTuple, over)
    # Convert over to collection
    group_vars = over isa Symbol ? [over] : collect(over)
    
    # Get unique group combinations
    group_data = NamedTuple{Tuple(group_vars)}(getproperty(data_nt, var) for var in group_vars)
    unique_groups = unique(zip(values(group_data)...))
    
    # For each group, yield all base profiles with group metadata
    return Iterators.flatten(
        (merge(prof, Dict{Symbol,Any}(group_vars[i] => group_vals[i] for i in 1:length(group_vars)))
         for prof in base_iter)
        for group_vals in unique_groups
    )
end

# ========================================================================================
# Advanced Builders (Future Extension Points)
# ========================================================================================

"""
    refgrid_slice(data_nt::NamedTuple; where, builder_func, builder_args...)

Create iterator on a data slice using another builder.

# Arguments
- `data_nt`: Data as NamedTuple
- `where`: Filtering conditions as NamedTuple
- `builder_func`: Builder function to apply (e.g., `refgrid_means`)
- `builder_args...`: Arguments to pass to builder function

# Returns
`Iterator{Dict{Symbol,Any}}` yielding profile dictionaries from filtered data
"""
function refgrid_slice(data_nt::NamedTuple; where, builder_func, builder_args...)
    # TODO: Implement data filtering and delegate to builder_func
    # For now, delegate directly (Phase 3 extension point)
    return builder_func(data_nt, builder_args...)
end